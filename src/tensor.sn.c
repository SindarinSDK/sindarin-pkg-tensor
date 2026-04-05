/* ==============================================================================
 * sindarin-pkg-tensor/src/tensor.sn.c — ggml tensor bridge for Sindarin
 * ==============================================================================
 * Implements tensor operations via the ggml C API.
 * Tensors are managed as host-side float arrays in a global pool.
 * Compute-intensive ops (matmul, add, relu, softmax) run through ggml graphs
 * which dispatch to the best available backend (CPU SIMD / GPU).
 * GNN-specific ops (scatter, attention aggregate) are implemented directly in C.
 *
 * Phase 1: inference operations only — no autograd / optimizer.
 *
 * The compiler's generated sn_types.h (force-included) provides:
 *   SnArray, sn_array_new, sn_array_push, sn_array_get, sn_array_length
 * ============================================================================== */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-opt.h>

/* ======================================================================
 * Tensor pool — maps integer handles to host-side float arrays
 * ====================================================================== */

#define SN_TENSOR_MAX 65536

typedef struct {
    float  *data;
    int64_t ne[4];    /* shape: ne[0]=cols, ne[1]=rows, ne[2], ne[3] */
    int     n_dims;
    int64_t n_elem;
} TPool;

static TPool   g_pool[SN_TENSOR_MAX];
static int     g_pool_count = 0;

/* RtTensor is __sn__Tensor, defined in the compiler-generated sn_types.h
 * (force-included). It has fields: int __rc__, long long __sn___handle. */
typedef __sn__Tensor RtTensor;

/* Allocate a new pool slot and return its index */
static int pool_alloc(int64_t ne0, int64_t ne1, int n_dims)
{
    if (g_pool_count >= SN_TENSOR_MAX) {
        fprintf(stderr, "tensor pool exhausted\n");
        abort();
    }
    int idx = g_pool_count++;
    TPool *s = &g_pool[idx];
    s->ne[0] = ne0;
    s->ne[1] = ne1;
    s->ne[2] = 1;
    s->ne[3] = 1;
    s->n_dims = n_dims;
    s->n_elem = ne0 * ne1;
    s->data   = (float *)calloc((size_t)s->n_elem, sizeof(float));
    return idx;
}

static RtTensor *wrap_pool(int idx)
{
    RtTensor *rt = (RtTensor *)calloc(1, sizeof(RtTensor));
    rt->__rc__ = 1;
    rt->__sn___handle = (long long)idx;
    return rt;
}

static TPool *unwrap(RtTensor *rt) { return &g_pool[rt->__sn___handle]; }

/* ======================================================================
 * Graph recording mode — for training with ggml_opt
 *
 * In record mode, tensor ops create ggml_tensor nodes in a persistent
 * context instead of computing immediately. The resulting graph is then
 * used by ggml_opt for forward + backward + optimizer steps.
 *
 * Each pool handle maps to a ggml_tensor* in the record context.
 * Parameters (model weights) have their pool data uploaded to ggml.
 * ====================================================================== */

#define GRAPH_PARAM_CTX_SIZE   (32 * 1024 * 1024)   /* params + inputs (32MB) */
#define GRAPH_COMPUTE_CTX_SIZE (128 * 1024 * 1024)  /* intermediate ops (128MB) */

static bool                 g_record_mode  = false;
static struct ggml_context *g_param_ctx    = NULL;   /* static: params + inputs */
static struct ggml_context *g_compute_ctx  = NULL;   /* no_alloc: intermediate ops */
static struct ggml_context *g_record_ctx   = NULL;   /* points to g_compute_ctx for ops */
static struct ggml_tensor  *g_record_map[SN_TENSOR_MAX];

static struct ggml_tensor *rec_tensor(RtTensor *rt) {
    return g_record_map[rt->__sn___handle];
}

static RtTensor *rec_wrap(struct ggml_tensor *gt, int64_t ne0, int64_t ne1) {
    int idx = pool_alloc(ne0, ne1, 2);
    g_record_map[idx] = gt;
    return wrap_pool(idx);
}

/* ======================================================================
 * ggml backend — initialized lazily on first use
 * ====================================================================== */

static ggml_backend_t g_backend     = NULL;
static int            g_backend_gpu = 0;

static void ensure_backend(void)
{
    if (g_backend) return;

    /* Start with CPU — always available, no dynamic loading */
    g_backend = ggml_backend_cpu_init();

    /* Try to upgrade to a better backend (GPU) if available */
    ggml_backend_load_all();
    ggml_backend_t best = ggml_backend_init_best();
    if (best) {
        const char *name = ggml_backend_name(best);
        if (name && strcmp(name, "CPU") != 0) {
            /* Got a GPU backend — use it instead */
            ggml_backend_free(g_backend);
            g_backend = best;
            g_backend_gpu = 1;
        } else {
            /* init_best returned CPU — free the duplicate, keep ours */
            ggml_backend_free(best);
        }
    }
}

/* ======================================================================
 * ggml micro-graph helpers
 *
 * Each tensor op creates a small ggml context, builds a 1-op graph,
 * computes it on the backend, copies the result into the pool, and frees.
 * ====================================================================== */

/* Context size generous enough for any single-op graph */
#define GRAPH_CTX_SIZE (16 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096)

/* Pre-allocated zero'd buffer for micro-graph contexts.
 * Newer ggml versions assert tensor->buffer == NULL before allocation.
 * posix_memalign (ggml's default) doesn't zero memory, so stale
 * buffer pointers from prior ggml contexts can trigger the assertion.
 * Using a pre-zero'd buffer avoids this. */
static void *g_micro_ctx_buf = NULL;
static size_t g_micro_ctx_size = 0;

static struct ggml_context *micro_ctx_init(void) {
    size_t needed = GRAPH_CTX_SIZE;
    if (!g_micro_ctx_buf || g_micro_ctx_size < needed) {
        if (g_micro_ctx_buf) free(g_micro_ctx_buf);
        g_micro_ctx_size = needed;
        g_micro_ctx_buf = malloc(g_micro_ctx_size);
    }
    /* Zero the buffer to ensure all tensor fields (including buffer) start NULL */
    memset(g_micro_ctx_buf, 0, g_micro_ctx_size);
    struct ggml_init_params params = { g_micro_ctx_size, g_micro_ctx_buf, true };
    return ggml_init(params);
}

/* Input tensor tracking for upload before compute */
#define MAX_INPUTS 8
static struct ggml_tensor *g_inputs[MAX_INPUTS];
static const float        *g_input_data[MAX_INPUTS];
static int                 g_input_count = 0;

static void track_input(struct ggml_tensor *t, const float *host_data)
{
    if (g_input_count < MAX_INPUTS) {
        g_inputs[g_input_count]     = t;
        g_input_data[g_input_count] = host_data;
        g_input_count++;
    }
}

/* Run a graph with the global backend.
 * Returns the allocator — caller MUST call ggml_gallocr_free() after
 * reading results from the graph output tensors. */
static ggml_gallocr_t run_graph(struct ggml_context *ctx, struct ggml_cgraph *graph)
{
    (void)ctx;
    ensure_backend();

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(g_backend);
    ggml_gallocr_t alloc = ggml_gallocr_new(buft);
    ggml_gallocr_alloc_graph(alloc, graph);

    /* Upload tracked input data */
    for (int i = 0; i < g_input_count; i++) {
        ggml_backend_tensor_set(g_inputs[i], g_input_data[i], 0, ggml_nbytes(g_inputs[i]));
    }
    g_input_count = 0;

    ggml_backend_graph_compute(g_backend, graph);
    return alloc;
}

/* Forward declarations */
RtTensor *sn_tensor_zeros(long long rows, long long cols);
static void train_register_param(int pool_idx);

/* ======================================================================
 * Graph recording API
 * ====================================================================== */

static int g_pool_count_before_record = 0;

void sn_graph_begin(void) {
    ensure_backend();
    g_pool_count_before_record = g_pool_count;

    struct ggml_init_params p_params = { GRAPH_PARAM_CTX_SIZE, NULL, true };
    g_param_ctx = ggml_init(p_params);

    struct ggml_init_params c_params = { GRAPH_COMPUTE_CTX_SIZE, NULL, true };
    g_compute_ctx = ggml_init(c_params);

    g_record_ctx = g_compute_ctx;
    memset(g_record_map, 0, sizeof(g_record_map));
    g_record_mode = true;
}

void sn_graph_end(void) {
    if (g_compute_ctx) { ggml_free(g_compute_ctx); g_compute_ctx = NULL; }
    if (g_param_ctx)   { ggml_free(g_param_ctx);   g_param_ctx = NULL; }
    g_record_ctx = NULL;
    memset(g_record_map, 0, sizeof(g_record_map));
    g_record_mode = false;

    /* Reclaim pool slots allocated during recording */
    for (int i = g_pool_count_before_record; i < g_pool_count; i++) {
        if (g_pool[i].data) { free(g_pool[i].data); g_pool[i].data = NULL; }
    }
    g_pool_count = g_pool_count_before_record;
}

RtTensor *sn_graph_input(long long rows, long long cols) {
    if (!g_record_mode) return sn_tensor_zeros(rows, cols);
    /* Inputs go in the param context (statically allocated) */
    struct ggml_tensor *gt = ggml_new_tensor_2d(g_param_ctx, GGML_TYPE_F32, cols, rows);
    ggml_set_name(gt, "input");
    ggml_set_input(gt);
    return rec_wrap(gt, cols, rows);
}

RtTensor *sn_graph_param(RtTensor *rt) {
    if (!g_record_mode || !rt) return rt;
    TPool *s = unwrap(rt);
    int idx = (int)rt->__sn___handle;
    /* Parameters go in the param context (backend-allocated later) */
    struct ggml_tensor *gt = ggml_new_tensor_2d(g_param_ctx, GGML_TYPE_F32, s->ne[0], s->ne[1]);
    ggml_set_name(gt, "param");
    ggml_set_input(gt);
    ggml_set_param(gt);
    /* Data uploaded after ggml_backend_alloc_ctx_tensors in sn_graph_train */
    g_record_map[idx] = gt;
    /* Register for persistent AdamW state if training mode active */
    train_register_param(idx);
    return rt;
}

/* Custom optimizer params callback — returns user-configured lr/wd */
static struct ggml_opt_optimizer_params g_opt_params;

static struct ggml_opt_optimizer_params sn_get_opt_params(void *userdata) {
    (void)userdata;
    return g_opt_params;
}

double sn_graph_train(RtTensor *output_rt, RtTensor *input_rt,
                      SnArray *data_arr, SnArray *label_arr,
                      long long nsamples, long long nepochs,
                      long long nbatch, double val_split,
                      char *loss_type_str, char *optimizer_str,
                      double lr, double wd)
{
    if (!g_record_mode || !g_record_ctx) return -1.0;
    struct ggml_tensor *outputs = rec_tensor(output_rt);
    struct ggml_tensor *inputs  = rec_tensor(input_rt);
    if (!outputs || !inputs) return -1.0;
    ggml_set_output(outputs);

    int64_t ne_datapoint = inputs->ne[0];
    int64_t ne_label     = outputs->ne[0];

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_F32, GGML_TYPE_F32, ne_datapoint, ne_label, nsamples, 1);

    struct ggml_tensor *dd = ggml_opt_dataset_data(dataset);
    struct ggml_tensor *dl = ggml_opt_dataset_labels(dataset);
    long long dlen = sn_array_length(data_arr);
    long long llen = sn_array_length(label_arr);
    for (long long i = 0; i < dlen && i < ne_datapoint * nsamples; i++) {
        double *v = (double *)sn_array_get(data_arr, i);
        ((float *)dd->data)[i] = (float)(*v);
    }
    for (long long i = 0; i < llen && i < ne_label * nsamples; i++) {
        double *v = (double *)sn_array_get(label_arr, i);
        ((float *)dl->data)[i] = (float)(*v);
    }

    enum ggml_opt_loss_type loss_type = GGML_OPT_LOSS_TYPE_CROSS_ENTROPY;
    if (strcmp(loss_type_str, "mse") == 0) loss_type = GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR;

    enum ggml_opt_optimizer_type opt_type = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    if (strcmp(optimizer_str, "sgd") == 0) opt_type = GGML_OPT_OPTIMIZER_TYPE_SGD;

    /* Configure optimizer params from user-provided lr/wd */
    g_opt_params = ggml_opt_get_default_optimizer_params(NULL);
    g_opt_params.adamw.alpha = (float)lr;
    g_opt_params.adamw.wd    = (float)wd;
    g_opt_params.sgd.alpha   = (float)lr;
    g_opt_params.sgd.wd      = (float)wd;

    ggml_backend_t backends[] = { g_backend };
    /* Use a reasonable graph size upper bound instead of SN_TENSOR_MAX.
     * SN_TENSOR_MAX (65536) caused scheduler pre-allocation to exhaust ggml
     * backend memory after prior tensor operations fragmented the heap. */
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 1, 4096, false, false);

    /* Allocate backend buffers for all tensors in param context */
    ggml_backend_alloc_ctx_tensors(g_param_ctx, g_backend);

    /* Upload data from pool to backend buffers for all mapped param-ctx tensors */
    for (int i = 0; i < g_pool_count; i++) {
        struct ggml_tensor *gt = g_record_map[i];
        if (gt && gt->buffer && g_pool[i].data) {
            ggml_backend_tensor_set(gt, g_pool[i].data, 0,
                                    ggml_nbytes(gt) < (size_t)g_pool[i].n_elem * sizeof(float)
                                    ? ggml_nbytes(gt)
                                    : (size_t)g_pool[i].n_elem * sizeof(float));
        }
    }

    /* g_compute_ctx is no_alloc — ggml_opt manages its allocations.
     * inputs/outputs are in g_param_ctx (backend-allocated). */
    ggml_opt_fit(sched, g_compute_ctx, inputs, outputs,
                 dataset, loss_type, opt_type,
                 sn_get_opt_params,
                 nepochs, nbatch, (float)val_split, false);

    /* Read back trained parameters to pool.
     * After ggml_opt_fit, parameter data is in backend buffers.
     * Use ggml_backend_tensor_get if buffer exists, else memcpy from host. */
    for (int i = 0; i < g_pool_count; i++) {
        struct ggml_tensor *gt = g_record_map[i];
        if (gt && (gt->flags & GGML_TENSOR_FLAG_PARAM)) {
            TPool *s = &g_pool[i];
            if (gt->buffer) {
                ggml_backend_tensor_get(gt, s->data, 0, (size_t)s->n_elem * sizeof(float));
            } else if (gt->data) {
                memcpy(s->data, gt->data, (size_t)s->n_elem * sizeof(float));
            }
        }
    }

    ggml_backend_sched_free(sched);
    ggml_opt_dataset_free(dataset);
    return 0.0;
}

/* ======================================================================
 * Creation
 * ====================================================================== */

RtTensor *sn_tensor_zeros(long long rows, long long cols)
{
    int idx = pool_alloc(cols, rows, 2);
    return wrap_pool(idx);
}

RtTensor *sn_tensor_from_doubles(SnArray *data, long long rows, long long cols)
{
    int idx = pool_alloc(cols, rows, 2);
    TPool *s = &g_pool[idx];
    long long n = sn_array_length(data);
    for (long long i = 0; i < n && i < s->n_elem; i++) {
        double *p = (double *)sn_array_get(data, i);
        s->data[i] = (float)(*p);
    }
    return wrap_pool(idx);
}

SnArray *sn_tensor_to_doubles(RtTensor *rt)
{
    TPool *s = unwrap(rt);
    SnArray *arr = sn_array_new(sizeof(double), (int)s->n_elem);
    for (int64_t i = 0; i < s->n_elem; i++) {
        double v = (double)s->data[i];
        sn_array_push(arr, &v);
    }
    return arr;
}

SnArray *sn_tensor_shape(RtTensor *rt)
{
    TPool *s = unwrap(rt);
    SnArray *arr = sn_array_new(sizeof(long long), s->n_dims);
    for (int i = 0; i < s->n_dims; i++) {
        long long v = (long long)s->ne[i];
        sn_array_push(arr, &v);
    }
    return arr;
}

/* ======================================================================
 * Arithmetic — via ggml graphs
 * ====================================================================== */

RtTensor *sn_tensor_matmul(RtTensor *a, RtTensor *b)
{
    TPool *pa = unwrap(a);
    TPool *pb = unwrap(b);
    if (g_record_mode) {
        int64_t N = pb->ne[0], M = pa->ne[1];
        struct ggml_tensor *bt = ggml_cont(g_record_ctx, ggml_transpose(g_record_ctx, rec_tensor(b)));
        return rec_wrap(ggml_mul_mat(g_record_ctx, bt, rec_tensor(a)), N, M);
    }

    /* A is [M, K] (ne0=K, ne1=M), B is [K, N] (ne0=N, ne1=K)
     * ggml_mul_mat(x, y) computes y * x^T, needs ne0_x == ne0_y.
     * Strategy: transpose B to B^T [N, K] (ne0=K, ne1=N), then
     *   ggml_mul_mat(B^T, A) -> result [M, N] (ne0=N, ne1=M) */
    int64_t M = pa->ne[1];
    int64_t K = pa->ne[0];
    int64_t N = pb->ne[0];

    struct ggml_context *ctx = micro_ctx_init();

    struct ggml_tensor *ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor *tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, K);
    ggml_set_input(ta);
    ggml_set_input(tb);
    track_input(ta, pa->data);
    track_input(tb, pb->data);

    /* Transpose B so ne0 matches: B^T has ne0=K, ne1=N */
    struct ggml_tensor *bt = ggml_cont(ctx, ggml_transpose(ctx, tb));
    struct ggml_tensor *result = ggml_mul_mat(ctx, bt, ta);
    ggml_set_output(result);

    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_gallocr_t ga = run_graph(ctx, graph);

    /* Result: ne0=N, ne1=M -> [M rows, N cols] */
    int idx = pool_alloc(N, M, 2);
    ggml_backend_tensor_get(result, g_pool[idx].data, 0, (size_t)(M * N) * sizeof(float));

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return wrap_pool(idx);
}

RtTensor *sn_tensor_add(RtTensor *a, RtTensor *b)
{
    TPool *pa = unwrap(a);
    TPool *pb = unwrap(b);
    if (g_record_mode) return rec_wrap(ggml_add(g_record_ctx, rec_tensor(a), rec_tensor(b)), pa->ne[0], pa->ne[1]);

    struct ggml_context *ctx = micro_ctx_init();

    struct ggml_tensor *ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pa->ne[0], pa->ne[1]);
    struct ggml_tensor *tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pb->ne[0], pb->ne[1]);
    ggml_set_input(ta);
    ggml_set_input(tb);
    track_input(ta, pa->data);
    track_input(tb, pb->data);

    struct ggml_tensor *result = ggml_add(ctx, ta, tb);
    ggml_set_output(result);

    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_gallocr_t ga = run_graph(ctx, graph);

    int idx = pool_alloc(pa->ne[0], pa->ne[1], pa->n_dims);
    ggml_backend_tensor_get(result, g_pool[idx].data, 0, (size_t)g_pool[idx].n_elem * sizeof(float));

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return wrap_pool(idx);
}

RtTensor *sn_tensor_scale(RtTensor *t, double scalar)
{
    TPool *pt = unwrap(t);
    if (g_record_mode) return rec_wrap(ggml_scale(g_record_ctx, rec_tensor(t), (float)scalar), pt->ne[0], pt->ne[1]);

    struct ggml_context *ctx = micro_ctx_init();

    struct ggml_tensor *ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pt->ne[0], pt->ne[1]);
    ggml_set_input(ta);
    track_input(ta, pt->data);

    struct ggml_tensor *result = ggml_scale(ctx, ta, (float)scalar);
    ggml_set_output(result);

    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_gallocr_t ga = run_graph(ctx, graph);

    int idx = pool_alloc(pt->ne[0], pt->ne[1], pt->n_dims);
    ggml_backend_tensor_get(result, g_pool[idx].data, 0, (size_t)g_pool[idx].n_elem * sizeof(float));

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return wrap_pool(idx);
}

/* ======================================================================
 * Activations — via ggml graphs
 * ====================================================================== */

RtTensor *sn_tensor_relu(RtTensor *t)
{
    TPool *pt = unwrap(t);
    if (g_record_mode) return rec_wrap(ggml_relu(g_record_ctx, rec_tensor(t)), pt->ne[0], pt->ne[1]);

    struct ggml_context *ctx = micro_ctx_init();

    struct ggml_tensor *ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pt->ne[0], pt->ne[1]);
    ggml_set_input(ta);
    track_input(ta, pt->data);

    struct ggml_tensor *result = ggml_relu(ctx, ta);
    ggml_set_output(result);

    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_gallocr_t ga = run_graph(ctx, graph);

    int idx = pool_alloc(pt->ne[0], pt->ne[1], pt->n_dims);
    ggml_backend_tensor_get(result, g_pool[idx].data, 0, (size_t)g_pool[idx].n_elem * sizeof(float));

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return wrap_pool(idx);
}

RtTensor *sn_tensor_softmax(RtTensor *t, long long dim)
{
    TPool *pt = unwrap(t);
    (void)dim;
    if (g_record_mode) return rec_wrap(ggml_soft_max(g_record_ctx, rec_tensor(t)), pt->ne[0], pt->ne[1]);

    struct ggml_context *ctx = micro_ctx_init();

    struct ggml_tensor *ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pt->ne[0], pt->ne[1]);
    ggml_set_input(ta);
    track_input(ta, pt->data);

    struct ggml_tensor *result = ggml_soft_max(ctx, ta);
    ggml_set_output(result);

    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_gallocr_t ga = run_graph(ctx, graph);

    int idx = pool_alloc(pt->ne[0], pt->ne[1], pt->n_dims);
    ggml_backend_tensor_get(result, g_pool[idx].data, 0, (size_t)g_pool[idx].n_elem * sizeof(float));

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return wrap_pool(idx);
}

/* Dropout: implemented directly (random mask * scale) */
RtTensor *sn_tensor_dropout(RtTensor *t, double rate, int training)
{
    TPool *pt = unwrap(t);
    if (g_record_mode) { t->__rc__++; return t; } /* identity in graph mode — retain for caller ownership */
    int idx = pool_alloc(pt->ne[0], pt->ne[1], pt->n_dims);
    TPool *out = &g_pool[idx];

    if (!training || rate <= 0.0) {
        memcpy(out->data, pt->data, (size_t)pt->n_elem * sizeof(float));
        return wrap_pool(idx);
    }

    float scale = 1.0f / (1.0f - (float)rate);
    for (int64_t i = 0; i < pt->n_elem; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        out->data[i] = (r > (float)rate) ? pt->data[i] * scale : 0.0f;
    }
    return wrap_pool(idx);
}

/* ======================================================================
 * Normalization — implemented directly
 * ====================================================================== */

RtTensor *sn_tensor_batch_norm(RtTensor *t, RtTensor *weight, RtTensor *bias,
                               RtTensor *running_mean, RtTensor *running_var,
                               int training)
{
    if (g_record_mode) { t->__rc__++; return t; } /* identity in graph mode — retain for caller ownership */
    TPool *pt   = unwrap(t);
    TPool *pw   = unwrap(weight);
    TPool *pb   = unwrap(bias);
    TPool *pm   = unwrap(running_mean);
    TPool *pv   = unwrap(running_var);

    int64_t rows = pt->ne[1];
    int64_t cols = pt->ne[0];
    int idx = pool_alloc(cols, rows, pt->n_dims);
    TPool *out = &g_pool[idx];
    float eps = 1e-5f;

    if (!training) {
        /* Inference: use running stats */
        for (int64_t r = 0; r < rows; r++) {
            for (int64_t c = 0; c < cols; c++) {
                float x   = pt->data[r * cols + c];
                float m   = pm->data[c];
                float v   = pv->data[c];
                float w   = pw->data[c];
                float b   = pb->data[c];
                out->data[r * cols + c] = (x - m) / sqrtf(v + eps) * w + b;
            }
        }
    } else {
        /* Training: compute batch mean/var */
        float *mean = (float *)calloc((size_t)cols, sizeof(float));
        float *var  = (float *)calloc((size_t)cols, sizeof(float));

        for (int64_t r = 0; r < rows; r++)
            for (int64_t c = 0; c < cols; c++)
                mean[c] += pt->data[r * cols + c];
        for (int64_t c = 0; c < cols; c++)
            mean[c] /= (float)rows;

        for (int64_t r = 0; r < rows; r++)
            for (int64_t c = 0; c < cols; c++) {
                float d = pt->data[r * cols + c] - mean[c];
                var[c] += d * d;
            }
        for (int64_t c = 0; c < cols; c++)
            var[c] /= (float)rows;

        for (int64_t r = 0; r < rows; r++)
            for (int64_t c = 0; c < cols; c++) {
                float x = pt->data[r * cols + c];
                float w = pw->data[c];
                float b = pb->data[c];
                out->data[r * cols + c] = (x - mean[c]) / sqrtf(var[c] + eps) * w + b;
            }

        free(mean);
        free(var);
    }

    return wrap_pool(idx);
}

/* ======================================================================
 * Reduction & aggregation — implemented directly for GNN ops
 * ====================================================================== */

RtTensor *sn_tensor_mean_pool(RtTensor *node_embeddings, RtTensor *batch_index)
{
    TPool *px = unwrap(node_embeddings);
    TPool *pb = unwrap(batch_index);

    int64_t num_nodes = px->ne[1];
    int64_t feat_dim  = px->ne[0];

    if (g_record_mode) {
        /* Differentiable mean pool using a pooling matrix built from batch_index.
         *
         * When each node is its own graph (batch_index = [0,1,...,N-1]),
         * the pooling matrix is identity → mean pool is identity.
         * This is the standard training case (1 sample = 1 node = 1 graph).
         *
         * For the general case (multiple nodes per graph), build
         * P[num_graphs, num_nodes] and compute result = P × features
         * via ggml_mul_mat(features_T, P) → [feat_dim, num_graphs]. */

        /* Determine num_graphs from batch_index */
        int64_t num_graphs = 1;
        for (int64_t i = 0; i < pb->n_elem; i++) {
            int64_t b = (int64_t)pb->data[i];
            if (b + 1 > num_graphs) num_graphs = b + 1;
        }

        /* Identity check: num_graphs == num_nodes and batch[i] == i for all i.
         * Mean pool of 1-node graphs is identity — skip to preserve gradient flow
         * and avoid non-contiguous backward through transpose. */
        if (num_graphs == num_nodes) {
            int is_identity = 1;
            for (int64_t i = 0; i < num_nodes && i < pb->n_elem; i++) {
                if ((int64_t)pb->data[i] != i) { is_identity = 0; break; }
            }
            if (is_identity) {
                node_embeddings->__rc__++;
                return node_embeddings;
            }
        }

        /* General case: build pooling matrix on host */
        int pool_idx = pool_alloc(num_nodes, num_graphs, 2);
        TPool *pool_mat = &g_pool[pool_idx];
        memset(pool_mat->data, 0, (size_t)(num_graphs * num_nodes) * sizeof(float));

        float *count = (float *)calloc((size_t)num_graphs, sizeof(float));
        for (int64_t n = 0; n < num_nodes && n < pb->n_elem; n++)
            count[(int64_t)pb->data[n]] += 1.0f;
        for (int64_t n = 0; n < num_nodes && n < pb->n_elem; n++) {
            int64_t g = (int64_t)pb->data[n];
            float c = count[g] > 0.0f ? count[g] : 1.0f;
            pool_mat->data[g * num_nodes + n] = 1.0f / c;
        }
        free(count);

        /* Create ggml tensor for pooling matrix in param context */
        struct ggml_tensor *gpool = ggml_new_tensor_2d(g_param_ctx, GGML_TYPE_F32, num_nodes, num_graphs);
        ggml_set_name(gpool, "pool_mat");
        ggml_set_input(gpool);
        g_record_map[pool_idx] = gpool;

        /* Transpose features: [feat_dim, num_nodes] → [num_nodes, feat_dim] */
        struct ggml_tensor *feat_t = ggml_cont(g_record_ctx,
            ggml_transpose(g_record_ctx, rec_tensor(node_embeddings)));

        /* result = ggml_mul_mat(feat_t, gpool) → [feat_dim, num_graphs] */
        struct ggml_tensor *result = ggml_mul_mat(g_record_ctx, feat_t, gpool);
        return rec_wrap(result, feat_dim, num_graphs);
    }

    /* Find number of graphs (max batch index + 1) */
    int64_t num_graphs = 1;
    for (int64_t i = 0; i < pb->n_elem; i++) {
        int64_t b = (int64_t)pb->data[i];
        if (b + 1 > num_graphs) num_graphs = b + 1;
    }

    int idx = pool_alloc(feat_dim, num_graphs, 2);
    TPool *out = &g_pool[idx];

    float *count = (float *)calloc((size_t)num_graphs, sizeof(float));

    for (int64_t n = 0; n < num_nodes; n++) {
        int64_t b = (int64_t)pb->data[n];
        count[b] += 1.0f;
        for (int64_t f = 0; f < feat_dim; f++) {
            out->data[b * feat_dim + f] += px->data[n * feat_dim + f];
        }
    }

    for (int64_t g = 0; g < num_graphs; g++) {
        float c = count[g] > 0.0f ? count[g] : 1.0f;
        for (int64_t f = 0; f < feat_dim; f++) {
            out->data[g * feat_dim + f] /= c;
        }
    }

    free(count);
    return wrap_pool(idx);
}

long long sn_tensor_argmax(RtTensor *t, long long dim)
{
    TPool *pt = unwrap(t);
    (void)dim;

    /* Argmax across columns (dim 1) for the first row, or global argmax */
    float best_val = -1e30f;
    long long best_idx = 0;
    for (int64_t i = 0; i < pt->n_elem; i++) {
        if (pt->data[i] > best_val) {
            best_val = pt->data[i];
            best_idx = (long long)i;
        }
    }
    return best_idx;
}

RtTensor *sn_tensor_sparse_aggregate(RtTensor *features, RtTensor *edge_index,
                                     RtTensor *edge_weight, char *mode)
{
    TPool *px  = unwrap(features);
    TPool *pei = unwrap(edge_index);
    TPool *pew = unwrap(edge_weight);

    int64_t num_nodes = px->ne[1];
    int64_t feat_dim  = px->ne[0];

    /* In record mode, build dense adjacency matrix for differentiable aggregation */
    if (g_record_mode) {
        int64_t ne = (pei->ne[1] == 2) ? pei->ne[0] : pei->ne[1];

        /* Build adjacency matrix on host */
        int adj_idx = pool_alloc(num_nodes, num_nodes, 2);
        TPool *adj = &g_pool[adj_idx];
        memset(adj->data, 0, (size_t)(num_nodes * num_nodes) * sizeof(float));
        float *cnt = NULL;
        if (strcmp(mode, "mean") == 0 || strcmp(mode, "sum_normalized") == 0)
            cnt = (float *)calloc((size_t)num_nodes, sizeof(float));
        for (int64_t i = 0; i < ne; i++) {
            int64_t s, d;
            if (pei->ne[1] == 2) { s = (int64_t)pei->data[i]; d = (int64_t)pei->data[ne + i]; }
            else { s = (int64_t)pei->data[i*2]; d = (int64_t)pei->data[i*2+1]; }
            if (s < num_nodes && d < num_nodes) {
                adj->data[d * num_nodes + s] += pew->data[i];
                if (cnt) cnt[d] += 1.0f;
            }
        }
        if (cnt) {
            for (int64_t n = 0; n < num_nodes; n++) {
                float c = cnt[n] > 0 ? cnt[n] : 1.0f;
                for (int64_t s = 0; s < num_nodes; s++) adj->data[n*num_nodes+s] /= c;
            }
            free(cnt);
        }

        /* Identity check: if adjacency is identity matrix (self-loops only,
         * weight 1.0 each), aggregate is identity — skip to preserve gradient
         * flow and avoid non-contiguous backward through transpose. */
        int is_identity = (ne == num_nodes) ? 1 : 0;
        if (is_identity) {
            for (int64_t n = 0; n < num_nodes && is_identity; n++) {
                for (int64_t m = 0; m < num_nodes; m++) {
                    float expected = (n == m) ? 1.0f : 0.0f;
                    if (adj->data[n * num_nodes + m] != expected) { is_identity = 0; break; }
                }
            }
        }
        if (is_identity) {
            /* Adjacency is I — aggregate returns input unchanged */
            free(adj->data); adj->data = NULL;
            g_pool_count--; /* reclaim the adj slot */
            features->__rc__++;
            return features;
        }

        struct ggml_tensor *gadj = ggml_new_tensor_2d(g_param_ctx, GGML_TYPE_F32, num_nodes, num_nodes);
        ggml_set_name(gadj, "adj");
        ggml_set_input(gadj);
        /* Data uploaded after ggml_backend_alloc_ctx_tensors */
        g_record_map[adj_idx] = gadj;
        /* result = adj × features: ggml_mul_mat(a,b) = b @ a^T
         * Transpose features so ne[0] matches adj's ne[0] (num_nodes) */
        struct ggml_tensor *gfeat_t = ggml_cont(g_record_ctx, ggml_transpose(g_record_ctx, rec_tensor(features)));
        struct ggml_tensor *result = ggml_mul_mat(g_record_ctx, gfeat_t, gadj);
        return rec_wrap(result, feat_dim, num_nodes);
    }

    int64_t num_edges = pei->ne[0] > pei->ne[1] ? pei->ne[1] : pei->ne[0];

    /* edge_index: [2, num_edges] — created via fromDoubles(data, rows=2, cols=num_edges)
     * pool layout: ne[0]=cols=num_edges, ne[1]=rows=2
     * data[i] = src[i], data[ne[0]+i] = dst[i] */
    if (pei->ne[1] == 2) {
        /* Standard: [2 rows, num_edges cols] — ne[0]=num_edges, ne[1]=2 */
        num_edges = pei->ne[0];
    } else {
        /* Transposed: [num_edges rows, 2 cols] — ne[0]=2, ne[1]=num_edges */
        num_edges = pei->ne[1];
    }

    int idx = pool_alloc(feat_dim, num_nodes, 2);
    TPool *out = &g_pool[idx];

    for (int64_t i = 0; i < num_edges; i++) {
        int64_t s, d;
        if (pei->ne[1] == 2) {
            /* Standard: row 0 at offset 0, row 1 at offset num_edges */
            s = (int64_t)pei->data[i];
            d = (int64_t)pei->data[num_edges + i];
        } else {
            /* Transposed: each row is (src, dst) pair */
            s = (int64_t)pei->data[i * 2];
            d = (int64_t)pei->data[i * 2 + 1];
        }
        float w = pew->data[i];
        for (int64_t f = 0; f < feat_dim; f++) {
            out->data[d * feat_dim + f] += px->data[s * feat_dim + f] * w;
        }
    }

    /* Normalize if mean mode */
    if (strcmp(mode, "mean") == 0 || strcmp(mode, "sum_normalized") == 0) {
        float *count = (float *)calloc((size_t)num_nodes, sizeof(float));
        for (int64_t i = 0; i < num_edges; i++) {
            int64_t d;
            if (pei->ne[1] == 2) {
                d = (int64_t)pei->data[num_edges + i];
            } else {
                d = (int64_t)pei->data[i * 2 + 1];
            }
            count[d] += 1.0f;
        }
        for (int64_t n = 0; n < num_nodes; n++) {
            float c = count[n] > 0.0f ? count[n] : 1.0f;
            for (int64_t f = 0; f < feat_dim; f++) {
                out->data[n * feat_dim + f] /= c;
            }
        }
        free(count);
    }

    return wrap_pool(idx);
}

RtTensor *sn_tensor_attention_aggregate(RtTensor *features, RtTensor *edge_index,
                                        RtTensor *edge_weight, RtTensor *att_weight)
{
    TPool *px  = unwrap(features);
    TPool *pei = unwrap(edge_index);
    TPool *paw = unwrap(att_weight);

    int64_t num_nodes = px->ne[1];
    int64_t feat_dim  = px->ne[0];
    int64_t num_edges = pei->ne[0];

    if (g_record_mode) return sn_tensor_sparse_aggregate(features, edge_index, edge_weight, "sum");
    int64_t att_dim   = paw->n_elem;

    int idx = pool_alloc(feat_dim, num_nodes, 2);
    TPool *out = &g_pool[idx];

    /* Compute attention scores */
    float *scores = (float *)calloc((size_t)num_edges, sizeof(float));
    for (int64_t i = 0; i < num_edges; i++) {
        int64_t s = (int64_t)pei->data[i];
        int64_t d = (int64_t)pei->data[num_edges + i];
        float score = 0.0f;
        /* Dot product of concat(features[s], features[d]) with att_weight */
        for (int64_t f = 0; f < feat_dim && f < att_dim; f++) {
            score += px->data[s * feat_dim + f] * paw->data[f];
        }
        for (int64_t f = 0; f < feat_dim && (feat_dim + f) < att_dim; f++) {
            score += px->data[d * feat_dim + f] * paw->data[feat_dim + f];
        }
        /* LeakyReLU */
        scores[i] = score > 0.0f ? score : score * 0.2f;
    }

    /* Softmax per destination node */
    float *max_score = (float *)malloc((size_t)num_nodes * sizeof(float));
    float *sum_exp   = (float *)calloc((size_t)num_nodes, sizeof(float));
    for (int64_t n = 0; n < num_nodes; n++) max_score[n] = -1e30f;

    for (int64_t i = 0; i < num_edges; i++) {
        int64_t d = (int64_t)pei->data[num_edges + i];
        if (scores[i] > max_score[d]) max_score[d] = scores[i];
    }
    for (int64_t i = 0; i < num_edges; i++) {
        int64_t d = (int64_t)pei->data[num_edges + i];
        scores[i] = expf(scores[i] - max_score[d]);
        sum_exp[d] += scores[i];
    }

    /* Weighted aggregation */
    for (int64_t i = 0; i < num_edges; i++) {
        int64_t s = (int64_t)pei->data[i];
        int64_t d = (int64_t)pei->data[num_edges + i];
        float alpha = scores[i] / (sum_exp[d] + 1e-8f);
        for (int64_t f = 0; f < feat_dim; f++) {
            out->data[d * feat_dim + f] += px->data[s * feat_dim + f] * alpha;
        }
    }

    free(scores);
    free(max_score);
    free(sum_exp);
    return wrap_pool(idx);
}

/* ======================================================================
 * Loss
 * ====================================================================== */

RtTensor *sn_tensor_cross_entropy(RtTensor *probs, RtTensor *targets)
{
    TPool *pp = unwrap(probs);
    TPool *pt = unwrap(targets);

    int64_t rows = pp->ne[1];
    int64_t cols = pp->ne[0];

    int idx = pool_alloc(1, 1, 1);
    TPool *out = &g_pool[idx];

    float loss = 0.0f;
    for (int64_t r = 0; r < rows; r++) {
        int64_t target_class = (int64_t)pt->data[r];
        if (target_class >= 0 && target_class < cols) {
            float p = pp->data[r * cols + target_class];
            if (p < 1e-7f) p = 1e-7f;
            loss -= logf(p);
        }
    }
    out->data[0] = loss / (float)(rows > 0 ? rows : 1);

    return wrap_pool(idx);
}

/* ======================================================================
 * Initialization
 * ====================================================================== */

RtTensor *sn_tensor_init_kaiming(RtTensor *t)
{
    TPool *pt = unwrap(t);
    static int seeded = 0;
    if (!seeded) { srand((unsigned)time(NULL)); seeded = 1; }

    /* Kaiming uniform: U(-bound, bound) where bound = sqrt(6 / fan_in) */
    int64_t fan_in = pt->ne[0];
    float bound = sqrtf(6.0f / (float)(fan_in > 0 ? fan_in : 1));

    for (int64_t i = 0; i < pt->n_elem; i++) {
        float u = (float)rand() / (float)RAND_MAX;
        pt->data[i] = (2.0f * u - 1.0f) * bound;
    }

    return t;
}

/* ======================================================================
 * Device
 * ====================================================================== */

int sn_gpu_available(void)
{
    ensure_backend();
    return g_backend_gpu;
}

RtTensor *sn_tensor_to_device(RtTensor *t, char *device)
{
    (void)device;
    /* Backend handles device dispatch transparently.
     * Return the same tensor — data stays in the host pool. */
    return t;
}

/* ======================================================================
 * Persistence — simple binary format
 * ====================================================================== */

#define SN_TENSOR_MAGIC 0x534E544E /* "SNTN" */

void sn_model_save(SnArray *params, char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "model_save: cannot open %s\n", path); return; }

    uint32_t magic = SN_TENSOR_MAGIC;
    long long count = sn_array_length(params);
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&count, sizeof(count), 1, f);

    for (long long i = 0; i < count; i++) {
        RtTensor *rt = *(RtTensor **)sn_array_get(params, i);
        TPool *s = unwrap(rt);
        fwrite(&s->n_dims, sizeof(s->n_dims), 1, f);
        fwrite(s->ne, sizeof(int64_t), 4, f);
        fwrite(s->data, sizeof(float), (size_t)s->n_elem, f);
    }

    fclose(f);
}

SnArray *sn_model_load(char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "model_load: cannot open %s\n", path);
        return sn_array_new(sizeof(RtTensor *), 0);
    }

    uint32_t magic;
    long long count;
    fread(&magic, sizeof(magic), 1, f);
    if (magic != SN_TENSOR_MAGIC) {
        fprintf(stderr, "model_load: invalid file format\n");
        fclose(f);
        return sn_array_new(sizeof(RtTensor *), 0);
    }
    fread(&count, sizeof(count), 1, f);

    SnArray *arr = sn_array_new(sizeof(RtTensor *), (int)count);
    for (long long i = 0; i < count; i++) {
        int n_dims;
        int64_t ne[4];
        fread(&n_dims, sizeof(n_dims), 1, f);
        fread(ne, sizeof(int64_t), 4, f);

        int idx = pool_alloc(ne[0], ne[1], n_dims);
        g_pool[idx].ne[2] = ne[2];
        g_pool[idx].ne[3] = ne[3];
        fread(g_pool[idx].data, sizeof(float), (size_t)g_pool[idx].n_elem, f);

        RtTensor *rt = wrap_pool(idx);
        sn_array_push(arr, &rt);
    }

    fclose(f);
    return arr;
}

/* ======================================================================
 * Graph-aware training — manual forward+backward+AdamW per batch
 *
 * ggml_opt_fit cannot train variable-topology graphs because it requires
 * a 1:1 mapping between data rows and label rows.  With graph batching
 * the input has totalNodes rows but only numGraphs labels.
 *
 * Instead we use the lower-level ggml primitives:
 *   1. Record the forward pass (aggregate + meanPool + classifier → logits)
 *   2. Append per-sample weighted cross-entropy loss nodes
 *   3. ggml_set_loss + ggml_build_backward_expand → gradient computation
 *   4. ggml_opt_step_adamw per parameter → parameter updates (as graph nodes)
 *   5. Compute the full graph in one pass
 *   6. Read updated params + m/v back to the pool
 *
 * Persistent AdamW state (first/second moments) lives in host arrays
 * that survive sn_graph_begin / sn_graph_end cycles.
 * ====================================================================== */

#define TRAIN_MAX_PARAMS 256

typedef struct {
    float   *m;          /* first moment  */
    float   *v;          /* second moment */
    int64_t  n_elem;
    int      pool_idx;   /* which pool slot this param is */
} TrainParam;

static TrainParam g_tp[TRAIN_MAX_PARAMS];
static int        g_tp_count = 0;
static int        g_tp_step  = 0;
static float      g_tp_lr    = 0.001f;
static float      g_tp_wd    = 0.0001f;
static float      g_tp_beta1 = 0.9f;
static float      g_tp_beta2 = 0.999f;
static float      g_tp_eps   = 1e-8f;
static int        g_tp_active = 0;

void sn_train_init(double lr, double wd)
{
    /* Clean up any prior session */
    for (int i = 0; i < g_tp_count; i++) {
        if (g_tp[i].m) { free(g_tp[i].m); g_tp[i].m = NULL; }
        if (g_tp[i].v) { free(g_tp[i].v); g_tp[i].v = NULL; }
    }
    g_tp_count = 0;
    g_tp_step  = 0;
    g_tp_lr    = (float)lr;
    g_tp_wd    = (float)wd;
    g_tp_active = 1;
}

/* Register a pool slot as a trainable parameter (called from sn_graph_param
 * when training mode is active).  Allocates m/v on first encounter. */
static void train_register_param(int pool_idx)
{
    if (!g_tp_active) return;

    /* Already registered? */
    for (int i = 0; i < g_tp_count; i++) {
        if (g_tp[i].pool_idx == pool_idx) return;
    }
    if (g_tp_count >= TRAIN_MAX_PARAMS) {
        fprintf(stderr, "train: too many parameters (max %d)\n", TRAIN_MAX_PARAMS);
        return;
    }
    int64_t n = g_pool[pool_idx].n_elem;
    int idx = g_tp_count++;
    g_tp[idx].pool_idx = pool_idx;
    g_tp[idx].n_elem   = n;
    g_tp[idx].m        = (float *)calloc((size_t)n, sizeof(float));
    g_tp[idx].v        = (float *)calloc((size_t)n, sizeof(float));
}

/* Run one training step on the already-recorded forward graph.
 *
 * The Sindarin layer has already called:
 *   sn_graph_begin()
 *   sn_graph_param(w) for each weight
 *   input = sn_graph_input(totalNodes, featureDim)
 *   output = model.forward(graphTensors, true)   -- recorded with real topology
 *
 * This function:
 *   - Adds loss computation (per-sample weighted cross-entropy)
 *   - Marks loss with ggml_set_loss
 *   - Builds backward graph (ggml_build_backward_expand with NULL grad_accs)
 *   - Adds AdamW update nodes using ggml_opt_step_adamw
 *   - Allocates, uploads, computes, reads back
 *   - Returns batch loss
 */
double sn_train_step(RtTensor *logits_rt, RtTensor *input_rt,
                     SnArray *data_arr, SnArray *label_arr, SnArray *weight_arr,
                     long long num_graphs)
{
    if (!g_record_mode || !g_record_ctx) return -1.0;
    ensure_backend();

    struct ggml_tensor *logits = rec_tensor(logits_rt);
    struct ggml_tensor *inputs = rec_tensor(input_rt);
    if (!logits || !inputs) return -1.0;

    int64_t num_actions = logits->ne[0];
    struct ggml_context *ctx = g_record_ctx;

    /* --- Create label and weight tensors in param context --- */
    struct ggml_tensor *labels = ggml_new_tensor_2d(g_param_ctx, GGML_TYPE_F32,
                                                     num_actions, num_graphs);
    ggml_set_name(labels, "train_labels");
    ggml_set_input(labels);
    int lbl_pool = pool_alloc(num_actions, num_graphs, 2);
    g_record_map[lbl_pool] = labels;
    {
        long long llen = sn_array_length(label_arr);
        for (long long i = 0; i < llen && i < g_pool[lbl_pool].n_elem; i++) {
            double *v = (double *)sn_array_get(label_arr, i);
            g_pool[lbl_pool].data[i] = (float)(*v);
        }
    }

    struct ggml_tensor *weights = ggml_new_tensor_2d(g_param_ctx, GGML_TYPE_F32,
                                                      1, num_graphs);
    ggml_set_name(weights, "train_weights");
    ggml_set_input(weights);
    int wgt_pool = pool_alloc(1, num_graphs, 2);
    g_record_map[wgt_pool] = weights;
    {
        long long wlen = sn_array_length(weight_arr);
        for (long long i = 0; i < wlen && i < g_pool[wgt_pool].n_elem; i++) {
            double *v = (double *)sn_array_get(weight_arr, i);
            g_pool[wgt_pool].data[i] = (float)(*v);
        }
    }

    /* --- Build per-sample weighted cross-entropy loss ---
     *
     * ce[g] = -sum_a( label[g,a] * log(softmax(logits[g,a])) )
     * loss  = sum_g( ce[g] * weight[g] ) / num_graphs
     */
    struct ggml_tensor *sm     = ggml_soft_max(ctx, logits);               /* (A, G) */
    struct ggml_tensor *lp     = ggml_log(ctx, sm);                        /* (A, G) */
    struct ggml_tensor *prod   = ggml_mul(ctx, labels, lp);                /* (A, G) */
    struct ggml_tensor *ce_sum = ggml_sum_rows(ctx, prod);                 /* (1, G) */
    struct ggml_tensor *ce_neg = ggml_neg(ctx, ce_sum);                    /* (1, G) */
    struct ggml_tensor *wtd    = ggml_mul(ctx, ce_neg, weights);           /* (1, G) */
    struct ggml_tensor *total  = ggml_sum(ctx, wtd);                       /* scalar */
    struct ggml_tensor *loss   = ggml_scale(ctx, total,
                                            1.0f / (float)num_graphs);     /* scalar */

    ggml_set_output(loss);
    ggml_set_loss(loss);
    ggml_set_name(loss, "loss");

    /* --- Build forward graph --- */
    struct ggml_cgraph *gf = ggml_new_graph_custom(ctx, 16384, true);
    ggml_build_forward_expand(gf, loss);

    /* --- Collect param tensors --- */
    int n_params_in_graph = 0;
    struct ggml_tensor *param_tensors[TRAIN_MAX_PARAMS];
    int param_pool_indices[TRAIN_MAX_PARAMS];
    for (int i = 0; i < g_pool_count; i++) {
        struct ggml_tensor *gt = g_record_map[i];
        if (gt && (gt->flags & GGML_TENSOR_FLAG_PARAM)) {
            if (n_params_in_graph < TRAIN_MAX_PARAMS) {
                param_tensors[n_params_in_graph] = gt;
                param_pool_indices[n_params_in_graph] = i;
                n_params_in_graph++;
            }
        }
    }

    /* --- Build backward graph ---
     * Copy forward graph into gb (shares tensor pointers).
     * Pre-allocate grad_accs for PARAM tensors — ggml_build_backward_expand
     * only auto-creates them for LOSS tensors, not PARAMs.
     * The grad_accs array is indexed by forward graph node index. */
    struct ggml_cgraph *gb = ggml_new_graph_custom(ctx, 16384, true);
    ggml_graph_cpy(gf, gb);

    int n_fwd_nodes = ggml_graph_n_nodes(gb);
    struct ggml_tensor **grad_accs_arr = (struct ggml_tensor **)calloc(
        (size_t)n_fwd_nodes, sizeof(struct ggml_tensor *));

    for (int i = 0; i < n_fwd_nodes; i++) {
        struct ggml_tensor *node = ggml_graph_node(gb, i);
        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            /* Grad accumulators must be in param_ctx (backend-allocated)
             * because ggml_set_zero writes to ->data which requires allocation. */
            struct ggml_tensor *ga = ggml_dup_tensor(g_param_ctx, node);
            ggml_set_name(ga, "grad_acc");
            ggml_set_input(ga);
            grad_accs_arr[i] = ga;
        }
    }

    ggml_build_backward_expand(ctx, gb, grad_accs_arr);
    free(grad_accs_arr);

    /* --- Add AdamW update nodes for each parameter --- */
    g_tp_step++;

    float adamw_params_data[7];
    adamw_params_data[0] = g_tp_lr;
    adamw_params_data[1] = g_tp_beta1;
    adamw_params_data[2] = g_tp_beta2;
    adamw_params_data[3] = g_tp_eps;
    adamw_params_data[4] = g_tp_wd;
    adamw_params_data[5] = (float)g_tp_step;
    adamw_params_data[6] = 0.0f;

    struct ggml_tensor *adamw_params_t = ggml_new_tensor_1d(g_param_ctx, GGML_TYPE_F32, 7);
    ggml_set_name(adamw_params_t, "adamw_params");
    ggml_set_input(adamw_params_t);
    int adamw_pool = pool_alloc(7, 1, 1);
    g_record_map[adamw_pool] = adamw_params_t;
    memcpy(g_pool[adamw_pool].data, adamw_params_data, 7 * sizeof(float));

    struct ggml_tensor *m_tensors[TRAIN_MAX_PARAMS];
    struct ggml_tensor *v_tensors[TRAIN_MAX_PARAMS];
    int m_pools[TRAIN_MAX_PARAMS];
    int v_pools[TRAIN_MAX_PARAMS];

    for (int p = 0; p < n_params_in_graph; p++) {
        struct ggml_tensor *pt = param_tensors[p];
        int pidx = param_pool_indices[p];

        /* Find or register in training param state */
        int tp_idx = -1;
        for (int t = 0; t < g_tp_count; t++) {
            if (g_tp[t].pool_idx == pidx) { tp_idx = t; break; }
        }
        if (tp_idx < 0) {
            train_register_param(pidx);
            tp_idx = g_tp_count - 1;
        }

        /* Get gradient accumulator from the backward graph */
        struct ggml_tensor *grad = ggml_graph_get_grad_acc(gb, pt);
        if (!grad) {
            fprintf(stderr, "train_step: no gradient for param %d\n", p);
            continue;
        }

        /* Create m tensor (first moment) */
        struct ggml_tensor *mt = ggml_dup_tensor(g_param_ctx, pt);
        ggml_set_name(mt, "adam_m");
        ggml_set_input(mt);
        int mp = pool_alloc(pt->ne[0], pt->ne[1], 2);
        g_record_map[mp] = mt;
        memcpy(g_pool[mp].data, g_tp[tp_idx].m,
               (size_t)g_tp[tp_idx].n_elem * sizeof(float));
        m_tensors[p] = mt;
        m_pools[p] = mp;

        /* Create v tensor (second moment) */
        struct ggml_tensor *vt = ggml_dup_tensor(g_param_ctx, pt);
        ggml_set_name(vt, "adam_v");
        ggml_set_input(vt);
        int vp = pool_alloc(pt->ne[0], pt->ne[1], 2);
        g_record_map[vp] = vt;
        memcpy(g_pool[vp].data, g_tp[tp_idx].v,
               (size_t)g_tp[tp_idx].n_elem * sizeof(float));
        v_tensors[p] = vt;
        v_pools[p] = vp;

        /* Add AdamW step node — updates pt, mt, vt in-place */
        struct ggml_tensor *step = ggml_opt_step_adamw(
            ctx, pt, grad, mt, vt, adamw_params_t);
        ggml_build_forward_expand(gb, step);
    }

    /* --- Allocate backend buffers and upload data --- */
    ggml_backend_alloc_ctx_tensors(g_param_ctx, g_backend);

    /* Upload all mapped tensors from pool to backend */
    for (int i = 0; i < g_pool_count; i++) {
        struct ggml_tensor *gt = g_record_map[i];
        if (gt && gt->buffer && g_pool[i].data) {
            size_t pool_bytes = (size_t)g_pool[i].n_elem * sizeof(float);
            size_t gt_bytes   = ggml_nbytes(gt);
            size_t copy_bytes = pool_bytes < gt_bytes ? pool_bytes : gt_bytes;
            ggml_backend_tensor_set(gt, g_pool[i].data, 0, copy_bytes);
        }
    }

    /* --- Compute the full graph: forward → loss → backward → AdamW --- */
    ggml_backend_t backends[] = { g_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, NULL, 1, 16384, false, false);
    ggml_backend_sched_alloc_graph(sched, gb);

    /* Re-upload input tensors — sched alloc may have moved buffers */
    for (int i = 0; i < g_pool_count; i++) {
        struct ggml_tensor *gt = g_record_map[i];
        if (gt && gt->buffer && g_pool[i].data &&
            (gt->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_PARAM))) {
            size_t pool_bytes = (size_t)g_pool[i].n_elem * sizeof(float);
            size_t gt_bytes   = ggml_nbytes(gt);
            size_t copy_bytes = pool_bytes < gt_bytes ? pool_bytes : gt_bytes;
            ggml_backend_tensor_set(gt, g_pool[i].data, 0, copy_bytes);
        }
    }

    /* Reset gradient accumulators to 0; set loss gradient to 1 */
    ggml_graph_reset(gb);

    ggml_backend_sched_graph_compute(sched, gb);

    /* --- Read loss --- */
    float loss_val = 0.0f;
    ggml_backend_tensor_get(loss, &loss_val, 0, sizeof(float));

    /* --- Read back updated parameters to pool --- */
    for (int p = 0; p < n_params_in_graph; p++) {
        int pidx = param_pool_indices[p];
        struct ggml_tensor *pt = param_tensors[p];
        TPool *s = &g_pool[pidx];
        if (pt->buffer) {
            ggml_backend_tensor_get(pt, s->data, 0,
                                    (size_t)s->n_elem * sizeof(float));
        }
    }

    /* --- Read back updated m/v to persistent state --- */
    for (int p = 0; p < n_params_in_graph; p++) {
        int pidx = param_pool_indices[p];
        int tp_idx = -1;
        for (int t = 0; t < g_tp_count; t++) {
            if (g_tp[t].pool_idx == pidx) { tp_idx = t; break; }
        }
        if (tp_idx >= 0) {
            struct ggml_tensor *mt = m_tensors[p];
            struct ggml_tensor *vt = v_tensors[p];
            if (mt->buffer) {
                ggml_backend_tensor_get(mt, g_tp[tp_idx].m, 0,
                                        (size_t)g_tp[tp_idx].n_elem * sizeof(float));
            }
            if (vt->buffer) {
                ggml_backend_tensor_get(vt, g_tp[tp_idx].v, 0,
                                        (size_t)g_tp[tp_idx].n_elem * sizeof(float));
            }
        }
    }

    ggml_backend_sched_free(sched);

    return (double)loss_val;
}

void sn_train_finish(void)
{
    for (int i = 0; i < g_tp_count; i++) {
        if (g_tp[i].m) { free(g_tp[i].m); g_tp[i].m = NULL; }
        if (g_tp[i].v) { free(g_tp[i].v); g_tp[i].v = NULL; }
    }
    g_tp_count  = 0;
    g_tp_step   = 0;
    g_tp_active = 0;
}

/* ======================================================================
 * Lifecycle
 * ====================================================================== */

void sn_tensor_free(RtTensor *rt)
{
    TPool *s = unwrap(rt);
    if (s->data) {
        free(s->data);
        s->data = NULL;
        s->n_elem = 0;
    }
}
