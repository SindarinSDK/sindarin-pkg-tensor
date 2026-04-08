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
RtTensor *sn_tensor_from_doubles(SnArray *data, long long rows, long long cols);

/* ======================================================================
 * Graph recording API
 * ====================================================================== */

static int g_pool_count_before_record = 0;

/* State for sn_graph_train_epoch — persists across epoch calls inside one
 * sn_graph_begin/end cycle. Lazy-initialized on the first epoch call,
 * freed in sn_graph_end. */
static ggml_opt_context_t    g_opt_ctx              = NULL;
static ggml_backend_sched_t  g_opt_sched            = NULL;
static ggml_backend_buffer_t g_opt_param_buf        = NULL;
static struct ggml_tensor   *g_opt_loss_tensor      = NULL;
static struct ggml_tensor   *g_opt_features_tensor  = NULL;
static struct ggml_tensor   *g_opt_labels_tensor    = NULL;
static struct ggml_tensor   *g_opt_weights_tensor   = NULL;

/* Per-batch upload registry — tensors whose VALUES change per minibatch.
 *
 * Used for heterogeneous graph batching: when the caller's training set has
 * graphs with variable numNodes/numEdges, the dense adjacency and pooling
 * matrices baked into the static recorded graph cannot stay constant. Each
 * call to sn_tensor_sparse_aggregate / sn_tensor_mean_pool in record mode
 * registers its output ggml input tensor here, and sn_graph_train_epoch
 * walks the registry every batch to upload freshly assembled host buffers.
 *
 * See docs/issues/heterogeneous-graph-batching.md for the rationale. */
#define PB_KIND_ADJ  1
#define PB_KIND_POOL 2
#define PB_MODE_SUM            0
#define PB_MODE_SUM_NORMALIZED 1
#define PB_MODE_MEAN           2

#define MAX_PER_BATCH_TENSORS 32
static int g_pb_pool_idx[MAX_PER_BATCH_TENSORS]; /* pool slot of the registered ggml input tensor */
static int g_pb_kind[MAX_PER_BATCH_TENSORS];     /* PB_KIND_ADJ | PB_KIND_POOL */
static int g_pb_mode[MAX_PER_BATCH_TENSORS];     /* PB_MODE_* (ADJ only; POOL ignores) */
static int g_pb_count = 0;

static void track_per_batch(int pool_idx, int kind, int mode) {
    if (g_pb_count >= MAX_PER_BATCH_TENSORS) {
        fprintf(stderr, "per-batch tensor registry full (max %d)\n", MAX_PER_BATCH_TENSORS);
        abort();
    }
    g_pb_pool_idx[g_pb_count] = pool_idx;
    g_pb_kind[g_pb_count]     = kind;
    g_pb_mode[g_pb_count]     = mode;
    g_pb_count++;
}

static int parse_agg_mode(const char *mode) {
    if (mode && strcmp(mode, "sum_normalized") == 0) return PB_MODE_SUM_NORMALIZED;
    if (mode && strcmp(mode, "mean") == 0)           return PB_MODE_MEAN;
    return PB_MODE_SUM;
}

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

    /* Reset epoch-loop training state — fresh per begin/end cycle */
    g_opt_ctx              = NULL;
    g_opt_sched            = NULL;
    g_opt_param_buf        = NULL;
    g_opt_loss_tensor      = NULL;
    g_opt_features_tensor  = NULL;
    g_opt_labels_tensor    = NULL;
    g_opt_weights_tensor   = NULL;

    /* Reset per-batch upload registry — tensors get re-registered as the
     * record-mode forward pass walks the layers. */
    g_pb_count = 0;
}

void sn_graph_end(void) {
    /* Tear down epoch-loop training state if it was lazy-initialized */
    if (g_opt_ctx)       { ggml_opt_free(g_opt_ctx);                  g_opt_ctx       = NULL; }
    if (g_opt_sched)     { ggml_backend_sched_free(g_opt_sched);      g_opt_sched     = NULL; }
    if (g_opt_param_buf) { ggml_backend_buffer_free(g_opt_param_buf); g_opt_param_buf = NULL; }
    g_opt_loss_tensor     = NULL;
    g_opt_features_tensor = NULL;
    g_opt_labels_tensor   = NULL;
    g_opt_weights_tensor  = NULL;

    if (g_compute_ctx) { ggml_free(g_compute_ctx); g_compute_ctx = NULL; }
    if (g_param_ctx)   { ggml_free(g_param_ctx);   g_param_ctx = NULL; }
    g_record_ctx = NULL;
    memset(g_record_map, 0, sizeof(g_record_map));
    g_record_mode = false;
    g_pb_count = 0;

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

/* Like sn_graph_input, but pre-populates the host pool slot with caller data
 * so the same slot can be uploaded to the backend later via the standard
 * "iterate g_record_map and upload" loop. Used for non-PARAM inputs whose
 * values change between calls (labels, weights, batched node features). */
RtTensor *sn_graph_input_data(SnArray *data, long long rows, long long cols) {
    if (!g_record_mode) return sn_tensor_from_doubles(data, rows, cols);

    int idx = pool_alloc(cols, rows, 2);
    TPool *s = &g_pool[idx];
    long long n = sn_array_length(data);
    for (long long i = 0; i < n && i < s->n_elem; i++) {
        double *p = (double *)sn_array_get(data, i);
        s->data[i] = (float)(*p);
    }

    struct ggml_tensor *gt = ggml_new_tensor_2d(g_param_ctx, GGML_TYPE_F32, cols, rows);
    ggml_set_name(gt, "input_data");
    ggml_set_input(gt);
    g_record_map[idx] = gt;
    return wrap_pool(idx);
}

/* Execute the recorded forward graph that ends at `loss_rt` and read the
 * scalar value back. Used by tests to validate record-mode loss expressions
 * in isolation, without ggml_opt / backward / optimizer involvement. */
double sn_graph_compute_loss(RtTensor *loss_rt) {
    if (!g_record_mode || !g_record_ctx || !loss_rt) return 0.0;

    struct ggml_tensor *loss = rec_tensor(loss_rt);
    if (!loss) return 0.0;
    ggml_set_output(loss);

    ensure_backend();

    /* Allocate backend buffers for all tensors in the param context (inputs) */
    ggml_backend_buffer_t param_buf = ggml_backend_alloc_ctx_tensors(g_param_ctx, g_backend);

    /* Upload host data for every recorded input/param that has pool data */
    for (int i = 0; i < g_pool_count; i++) {
        struct ggml_tensor *gt = g_record_map[i];
        if (gt && gt->buffer && g_pool[i].data) {
            size_t want = (size_t)g_pool[i].n_elem * sizeof(float);
            size_t have = ggml_nbytes(gt);
            ggml_backend_tensor_set(gt, g_pool[i].data, 0, want < have ? want : have);
        }
    }

    /* Build the forward graph from the loss tensor's ancestors */
    struct ggml_cgraph *graph = ggml_new_graph_custom(g_compute_ctx, 4096, false);
    ggml_build_forward_expand(graph, loss);

    /* Allocate the compute graph via gallocr and compute */
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(g_backend);
    ggml_gallocr_t alloc = ggml_gallocr_new(buft);
    ggml_gallocr_alloc_graph(alloc, graph);
    ggml_backend_graph_compute(g_backend, graph);

    /* Read the scalar back */
    float loss_value = 0.0f;
    ggml_backend_tensor_get(loss, &loss_value, 0, sizeof(float));

    ggml_gallocr_free(alloc);
    if (param_buf) ggml_backend_buffer_free(param_buf);
    return (double)loss_value;
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
    return rt;
}

/* Custom optimizer params callback — returns user-configured lr/wd */
static struct ggml_opt_optimizer_params g_opt_params;

/* Cached training stats from last sn_graph_train call */
static double g_train_loss     = 0.0;
static double g_train_accuracy = 0.0;
static long long g_train_ndata = 0;

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
    ggml_backend_buffer_t param_buf = ggml_backend_alloc_ctx_tensors(g_param_ctx, g_backend);

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

    /* Use lower-level ggml_opt API instead of ggml_opt_fit so we can
     * capture the final training loss (ggml_opt_fit discards it). */
    int64_t nbatch_physical = inputs->ne[1];
    int64_t opt_period = nbatch / nbatch_physical;
    if (opt_period < 1) opt_period = 1;
    int64_t nbatches_logical = nsamples / nbatch;
    int64_t ibatch_split = (int64_t)((1.0f - (float)val_split) * nbatches_logical) * opt_period;
    int64_t idata_split = ibatch_split * nbatch_physical;

    int64_t epoch_counter = 1;
    struct ggml_opt_params opt_params_s = ggml_opt_default_params(sched, loss_type);
    opt_params_s.ctx_compute     = g_compute_ctx;
    opt_params_s.inputs          = inputs;
    opt_params_s.outputs         = outputs;
    opt_params_s.opt_period      = opt_period;
    opt_params_s.get_opt_pars    = sn_get_opt_params;
    opt_params_s.get_opt_pars_ud = &epoch_counter;
    opt_params_s.optimizer       = opt_type;
    ggml_opt_context_t opt_ctx = ggml_opt_init(opt_params_s);

    if (nbatch < nsamples) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1);
    }

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_val   = ggml_opt_result_init();

    for (; epoch_counter <= nepochs; ++epoch_counter) {
        if (nbatch < idata_split) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }
        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_val);
        fprintf(stderr, "ggml_opt_fit: epoch %04lld/%04lld:\n",
                (long long)epoch_counter, (long long)nepochs);
        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val,
                       idata_split,
                       ggml_opt_epoch_callback_progress_bar,
                       ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");
    }

    /* Extract final training stats and cache for accessor functions */
    ggml_opt_result_loss(result_train, &g_train_loss, NULL);
    ggml_opt_result_accuracy(result_train, &g_train_accuracy, NULL);
    {
        int64_t nd = 0;
        ggml_opt_result_ndata(result_train, &nd);
        g_train_ndata = (long long)nd;
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_val);

    /* Read back trained parameters to pool */
    for (int i = 0; i < g_pool_count; i++) {
        struct ggml_tensor *gt = g_record_map[i];
        if (gt && (gt->flags & GGML_TENSOR_FLAG_PARAM)) {
            TPool *s = &g_pool[i];
            float before = s->data[0];
            if (gt->buffer) {
                ggml_backend_tensor_get(gt, s->data, 0, (size_t)s->n_elem * sizeof(float));
            } else if (gt->data) {
                memcpy(s->data, gt->data, (size_t)s->n_elem * sizeof(float));
            }
            fprintf(stderr, "readback param pool=%d: before=%.6f after=%.6f buf=%p data=%p\n",
                    i, before, s->data[0], (void*)gt->buffer, (void*)gt->data);
        }
    }

    ggml_backend_sched_free(sched);
    ggml_opt_dataset_free(dataset);
    if (param_buf) ggml_backend_buffer_free(param_buf);
    return g_train_loss;
}

double sn_graph_train_loss(void)     { return g_train_loss; }
double sn_graph_train_accuracy(void) { return g_train_accuracy; }
long long sn_graph_train_ndata(void) { return g_train_ndata; }

/* ======================================================================
 * Per-epoch training driver
 *
 * This is the new training entry point. Unlike sn_graph_train (which
 * wraps ggml_opt_epoch with the dataset abstraction), this function:
 *
 *   - Uses ggml_opt with loss_type=GGML_OPT_LOSS_TYPE_SUM and a
 *     pre-built loss tensor as `outputs`. ggml_sum of a scalar is
 *     identity, so the entire loss expression flows through unchanged
 *     and backward propagates correctly.
 *   - Drives the per-batch loop manually via ggml_opt_alloc/eval, with
 *     three input uploads per batch (features, labels, weights).
 *   - Lazy-initializes the opt context on the first call within a
 *     sn_graph_begin/end cycle and reuses it across epochs. The state
 *     is freed in sn_graph_end.
 *
 * Caller is responsible for shuffling: pass a permutation of sample
 * indices in `batch_perm` (one entry per training sample). Per-epoch
 * shuffling means re-shuffling the permutation between calls.
 * ====================================================================== */
double sn_graph_train_epoch(
    RtTensor *loss_rt,
    RtTensor *features_rt,
    RtTensor *labels_rt,
    RtTensor *weights_rt,
    SnArray  *features_host,         /* total_samples * max_nodes * feature_dim doubles, zero-padded */
    SnArray  *labels_host,           /* total_samples * labels_per_sample doubles */
    SnArray  *weights_host,          /* total_samples doubles, one weight per sample */
    SnArray  *adj_host,              /* total_samples * max_nodes * max_nodes doubles, zero-padded */
    SnArray  *real_node_count_host,  /* total_samples doubles (per-sample real node count) */
    SnArray  *batch_perm,            /* total_samples doubles (integer-valued permutation) */
    long long feature_dim,
    long long labels_per_sample,
    long long n_per_batch,
    long long max_nodes_per_graph,
    char     *optimizer_str,
    double    lr,
    double    beta1,
    double    beta2,
    double    eps,
    double    wd)
{
    if (!g_record_mode || !g_record_ctx) return -1.0;

    struct ggml_tensor *loss     = rec_tensor(loss_rt);
    struct ggml_tensor *features = rec_tensor(features_rt);
    struct ggml_tensor *labels   = rec_tensor(labels_rt);
    struct ggml_tensor *weights  = rec_tensor(weights_rt);
    if (!loss || !features || !labels || !weights) return -1.0;

    /* Batched padded sizes — every batch lives in a tensor of these dims. */
    const long long nodes_per_batch    = n_per_batch * max_nodes_per_graph;
    const long long features_per_node  = feature_dim;
    const long long features_per_batch = nodes_per_batch * features_per_node;
    const long long adj_per_sample     = max_nodes_per_graph * max_nodes_per_graph;

    /* Update optimizer params (the callback reads from g_opt_params) */
    g_opt_params = ggml_opt_get_default_optimizer_params(NULL);
    g_opt_params.adamw.alpha = (float)lr;
    g_opt_params.adamw.beta1 = (float)beta1;
    g_opt_params.adamw.beta2 = (float)beta2;
    g_opt_params.adamw.eps   = (float)eps;
    g_opt_params.adamw.wd    = (float)wd;
    g_opt_params.sgd.alpha   = (float)lr;
    g_opt_params.sgd.wd      = (float)wd;

    /* Lazy initialize opt context on the first call.
     * Subsequent calls reuse the same opt context, sched, and param_buf. */
    if (!g_opt_ctx) {
        ggml_set_output(loss);

        enum ggml_opt_optimizer_type opt_type = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
        if (strcmp(optimizer_str, "sgd") == 0) opt_type = GGML_OPT_OPTIMIZER_TYPE_SGD;

        ggml_backend_t backends[] = { g_backend };
        g_opt_sched = ggml_backend_sched_new(backends, NULL, 1, 4096, false, false);

        /* Allocate backend buffers for params + recorded inputs in g_param_ctx */
        g_opt_param_buf = ggml_backend_alloc_ctx_tensors(g_param_ctx, g_backend);

        /* Upload host data for every recorded tensor that has pool data.
         * This populates params (from initKaiming) and the initial
         * features/labels/weights values (which the per-batch loop will
         * overwrite each batch). */
        for (int i = 0; i < g_pool_count; i++) {
            struct ggml_tensor *gt = g_record_map[i];
            if (gt && gt->buffer && g_pool[i].data) {
                size_t want = (size_t)g_pool[i].n_elem * sizeof(float);
                size_t have = ggml_nbytes(gt);
                ggml_backend_tensor_set(gt, g_pool[i].data, 0, want < have ? want : have);
            }
        }

        struct ggml_opt_params opt_params_s = ggml_opt_default_params(g_opt_sched, GGML_OPT_LOSS_TYPE_SUM);
        opt_params_s.ctx_compute     = g_compute_ctx;
        opt_params_s.inputs          = features;  /* required non-null for static graphs */
        opt_params_s.outputs         = loss;      /* OUR pre-built loss; sum-of-scalar = identity */
        opt_params_s.opt_period      = 1;
        opt_params_s.get_opt_pars    = sn_get_opt_params;
        opt_params_s.get_opt_pars_ud = NULL;
        opt_params_s.optimizer       = opt_type;

        g_opt_ctx              = ggml_opt_init(opt_params_s);
        /* Use ggml_opt_loss() — the loss tensor that ggml_opt_build creates
         * in ctx_static (with a real buffer). The original `loss` we built
         * lives in g_compute_ctx (no_alloc) and has no buffer to read from. */
        g_opt_loss_tensor      = ggml_opt_loss(g_opt_ctx);
        g_opt_features_tensor  = features;
        g_opt_labels_tensor    = labels;
        g_opt_weights_tensor   = weights;
    }

    /* Sanity check: caller must use the same tensors as the init call */
    if (features != g_opt_features_tensor ||
        labels   != g_opt_labels_tensor   ||
        weights  != g_opt_weights_tensor) {
        return -1.0;
    }

    long long total_samples = sn_array_length(batch_perm);
    long long n_batches     = total_samples / n_per_batch;
    if (n_batches == 0) return 0.0;

    /* Per-batch scratch buffers — sized from the (padded) batched shape.
     *
     *  features_buf : (n_per_batch * max_nodes) * feature_dim floats
     *  adj_buf      : (n_per_batch * max_nodes) * (n_per_batch * max_nodes) floats — block-diagonal
     *  pool_buf     : n_per_batch * (n_per_batch * max_nodes) floats — real-count weighted
     *  labels_buf   : n_per_batch * labels_per_sample floats
     *  weights_buf  : n_per_batch floats
     */
    size_t features_bytes = (size_t)features_per_batch * sizeof(float);
    size_t adj_bytes      = (size_t)nodes_per_batch * (size_t)nodes_per_batch * sizeof(float);
    size_t pool_bytes     = (size_t)n_per_batch * (size_t)nodes_per_batch * sizeof(float);
    size_t labels_bytes   = ggml_nbytes(labels);
    size_t weights_bytes  = ggml_nbytes(weights);

    float *features_buf = (float *)calloc((size_t)features_per_batch, sizeof(float));
    float *adj_buf      = (float *)calloc((size_t)nodes_per_batch * (size_t)nodes_per_batch, sizeof(float));
    float *pool_buf     = (float *)calloc((size_t)n_per_batch * (size_t)nodes_per_batch, sizeof(float));
    float *labels_buf   = (float *)malloc(labels_bytes);
    float *weights_buf  = (float *)malloc(weights_bytes);

    /* Sanity-check the recorded tensors actually have the expected padded shape.
     * If they don't, the caller built the template with the wrong dimensions
     * and the per-batch upload will overrun. Bail out instead of corrupting memory. */
    if ((long long)features->ne[1] != nodes_per_batch ||
        (long long)features->ne[0] != feature_dim) {
        fprintf(stderr,
                "sn_graph_train_epoch: features shape (%lld x %lld) != expected (%lld x %lld)\n",
                (long long)features->ne[0], (long long)features->ne[1],
                feature_dim, nodes_per_batch);
        free(features_buf); free(adj_buf); free(pool_buf);
        free(labels_buf); free(weights_buf);
        return -1.0;
    }

    double total_loss = 0.0;

    for (long long batch_idx = 0; batch_idx < n_batches; batch_idx++) {
        /* Zero the assembled buffers for this batch (block-diagonal layout
         * means most cells are padding zeros). */
        memset(features_buf, 0, features_bytes);
        memset(adj_buf,      0, adj_bytes);
        memset(pool_buf,     0, pool_bytes);

        /* For each sample slot in the batch, copy: padded features, the
         * sample's per-graph adjacency block at the right diagonal offset,
         * and the sample's pool row weighted by its REAL node count. */
        for (long long i = 0; i < n_per_batch; i++) {
            double *perm_val = (double *)sn_array_get(batch_perm, batch_idx * n_per_batch + i);
            long long sample_idx = (long long)(*perm_val);

            /* --- Padded features.
             * Source slice: features_host[sample_idx * max_nodes * feature_dim ..
             *                              (sample_idx+1) * max_nodes * feature_dim]
             * Destination slice: features_buf[(i * max_nodes) * feature_dim ..
             *                                  ((i+1) * max_nodes) * feature_dim]
             * Source is already zero-padded by Gnn.train(), so we just copy. */
            const long long src_feat_base = sample_idx * max_nodes_per_graph * feature_dim;
            const long long dst_feat_base = i * max_nodes_per_graph * feature_dim;
            for (long long j = 0; j < max_nodes_per_graph * feature_dim; j++) {
                double *v = (double *)sn_array_get(features_host, src_feat_base + j);
                features_buf[dst_feat_base + j] = (float)(*v);
            }

            /* --- Block-diagonal adjacency.
             * Each sample contributes a (max_nodes x max_nodes) block at
             * row offset (i * max_nodes), column offset (i * max_nodes).
             * The host adjacency for sample_idx is laid out row-major as a
             * (max_nodes x max_nodes) slice and is already zero beyond the
             * sample's real node count. */
            const long long src_adj_base = sample_idx * adj_per_sample;
            const long long row_off      = i * max_nodes_per_graph;
            for (long long r = 0; r < max_nodes_per_graph; r++) {
                for (long long c = 0; c < max_nodes_per_graph; c++) {
                    double *v = (double *)sn_array_get(adj_host, src_adj_base + r * max_nodes_per_graph + c);
                    adj_buf[(row_off + r) * nodes_per_batch + (row_off + c)] = (float)(*v);
                }
            }

            /* --- Pool matrix row for graph i.
             * pool_buf[i, i*max_nodes + k] = 1 / real_count[sample_idx]
             * for k in [0, real_count[sample_idx]); zero elsewhere.
             * Padded slots stay zero so they're excluded from the per-graph mean. */
            double *rc_v = (double *)sn_array_get(real_node_count_host, sample_idx);
            long long real_count = (long long)(*rc_v);
            if (real_count <= 0) real_count = 1; /* defensive — shouldn't happen */
            float inv = 1.0f / (float)real_count;
            for (long long k = 0; k < real_count; k++) {
                pool_buf[i * nodes_per_batch + (i * max_nodes_per_graph + k)] = inv;
            }

            /* --- Labels for this sample (labels_per_sample doubles) */
            for (long long j = 0; j < labels_per_sample; j++) {
                double *v = (double *)sn_array_get(labels_host, sample_idx * labels_per_sample + j);
                labels_buf[i * labels_per_sample + j] = (float)(*v);
            }

            /* --- One weight per sample */
            double *w = (double *)sn_array_get(weights_host, sample_idx);
            weights_buf[i] = (float)(*w);
        }

        /* Allocate the OPT graph for this batch (cached after first call) */
        ggml_opt_alloc(g_opt_ctx, /*backward =*/ true);

        /* Upload the batch's features/labels/weights to the backend buffers */
        ggml_backend_tensor_set(g_opt_features_tensor, features_buf, 0, features_bytes);
        ggml_backend_tensor_set(g_opt_labels_tensor,   labels_buf,   0, labels_bytes);
        ggml_backend_tensor_set(g_opt_weights_tensor,  weights_buf,  0, weights_bytes);

        /* DEBUG: dump min/max/anyNaN for each host-side buffer this batch */
        if (batch_idx == 0) {
            float fmin=1e30f, fmax=-1e30f; int fnan=0;
            for (long long k = 0; k < features_per_batch; k++) {
                float v = features_buf[k];
                if (v != v) fnan = 1;
                if (v < fmin) fmin = v;
                if (v > fmax) fmax = v;
            }
            float amin=1e30f, amax=-1e30f; int anan=0;
            for (long long k = 0; k < nodes_per_batch * nodes_per_batch; k++) {
                float v = adj_buf[k];
                if (v != v) anan = 1;
                if (v < amin) amin = v;
                if (v > amax) amax = v;
            }
            float pmin=1e30f, pmax=-1e30f; int pnan=0;
            for (long long k = 0; k < n_per_batch * nodes_per_batch; k++) {
                float v = pool_buf[k];
                if (v != v) pnan = 1;
                if (v < pmin) pmin = v;
                if (v > pmax) pmax = v;
            }
            float lmin=1e30f, lmax=-1e30f; int lnan=0;
            for (long long k = 0; k < n_per_batch * labels_per_sample; k++) {
                float v = labels_buf[k];
                if (v != v) lnan = 1;
                if (v < lmin) lmin = v;
                if (v > lmax) lmax = v;
            }
            float wmin=1e30f, wmax=-1e30f; int wnan=0;
            for (long long k = 0; k < n_per_batch; k++) {
                float v = weights_buf[k];
                if (v != v) wnan = 1;
                if (v < wmin) wmin = v;
                if (v > wmax) wmax = v;
            }
            fprintf(stderr,
                "[DBG] inputs: feat=[%.4g,%.4g]nan=%d adj=[%.4g,%.4g]nan=%d pool=[%.4g,%.4g]nan=%d lbl=[%.4g,%.4g]nan=%d w=[%.4g,%.4g]nan=%d\n",
                fmin, fmax, fnan, amin, amax, anan, pmin, pmax, pnan, lmin, lmax, lnan, wmin, wmax, wnan);
            fflush(stderr);

            /* Dump each PARAM tensor's host pool data — these are what the
             * loaded model contributed to this train() call. */
            for (int p = 0; p < g_pool_count; p++) {
                struct ggml_tensor *gt = g_record_map[p];
                if (!gt || !(gt->flags & GGML_TENSOR_FLAG_PARAM)) continue;
                TPool *s = &g_pool[p];
                if (!s->data || s->n_elem <= 0) continue;
                float pmin2=1e30f, pmax2=-1e30f; int pnan2=0;
                for (int64_t k = 0; k < s->n_elem; k++) {
                    float v = s->data[k];
                    if (v != v) pnan2 = 1;
                    if (v < pmin2) pmin2 = v;
                    if (v > pmax2) pmax2 = v;
                }
                fprintf(stderr,
                    "[DBG] param pool=%d name=%s shape=(%lld,%lld) range=[%.4g,%.4g] nan=%d\n",
                    p, gt->name, (long long)s->ne[0], (long long)s->ne[1], pmin2, pmax2, pnan2);
            }
            fflush(stderr);
        }

        /* Upload the batched adjacency / pool matrix to every tensor in the
         * per-batch registry. Within a single Gnn.train() call all ADJ
         * tensors share the same edgeIndex/edgeWeight + mode (because every
         * layer's `aggregate()` is called with the same args), so they all
         * receive the same buffer; same story for the (single) POOL tensor. */
        for (int r = 0; r < g_pb_count; r++) {
            int pool_idx = g_pb_pool_idx[r];
            struct ggml_tensor *gt = g_record_map[pool_idx];
            if (!gt || !gt->buffer) continue;

            if (g_pb_kind[r] == PB_KIND_ADJ) {
                size_t want = adj_bytes;
                size_t have = ggml_nbytes(gt);
                ggml_backend_tensor_set(gt, adj_buf, 0, want < have ? want : have);
            } else if (g_pb_kind[r] == PB_KIND_POOL) {
                size_t want = pool_bytes;
                size_t have = ggml_nbytes(gt);
                ggml_backend_tensor_set(gt, pool_buf, 0, want < have ? want : have);
            }
        }

        /* Run forward + backward + AdamW step (NULL result — we read loss directly) */
        ggml_opt_eval(g_opt_ctx, NULL);

        /* Read the scalar loss back */
        float batch_loss = 0.0f;
        ggml_backend_tensor_get(g_opt_loss_tensor, &batch_loss, 0, sizeof(float));
        total_loss += (double)batch_loss;

        /* DEBUG: dump batch loss + post-update param magnitudes on first batch */
        if (batch_idx == 0) {
            fprintf(stderr, "[DBG] batch 0 loss = %.6g (nan=%d)\n",
                    (double)batch_loss, batch_loss != batch_loss ? 1 : 0);
            for (int p = 0; p < g_pool_count; p++) {
                struct ggml_tensor *gt = g_record_map[p];
                if (!gt || !(gt->flags & GGML_TENSOR_FLAG_PARAM)) continue;
                TPool *s = &g_pool[p];
                if (!s->data || s->n_elem <= 0) continue;
                /* Read back from backend to see post-update values */
                if (gt->buffer) {
                    ggml_backend_tensor_get(gt, s->data, 0, (size_t)s->n_elem * sizeof(float));
                }
                float pmin3=1e30f, pmax3=-1e30f; int pnan3=0;
                for (int64_t k = 0; k < s->n_elem; k++) {
                    float v = s->data[k];
                    if (v != v) pnan3 = 1;
                    if (v < pmin3) pmin3 = v;
                    if (v > pmax3) pmax3 = v;
                }
                fprintf(stderr,
                    "[DBG] post-eval param pool=%d range=[%.4g,%.4g] nan=%d\n",
                    p, pmin3, pmax3, pnan3);
            }
            fflush(stderr);
        }
    }

    free(features_buf);
    free(adj_buf);
    free(pool_buf);
    free(labels_buf);
    free(weights_buf);

    /* Read back updated PARAM data into the pool slots so subsequent
     * inference (forward() outside record mode) sees the trained values. */
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

    return total_loss / (double)n_batches;
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

/* GNN-specific matmul: features × weight where weight is stored transposed
 * (ne[0]=inputDim matches features ne[0]=inputDim). No transpose needed in
 * the graph — gradients flow directly to the weight PARAM tensor.
 *
 * features: (inputDim, numNodes)   weight: (inputDim, outputDim)
 * result:   (outputDim, numNodes) */
RtTensor *sn_tensor_gnn_matmul(RtTensor *features, RtTensor *weight)
{
    TPool *pf = unwrap(features);
    TPool *pw = unwrap(weight);
    int64_t inputDim  = pf->ne[0];  /* contraction dim */
    int64_t numNodes  = pf->ne[1];
    int64_t outputDim = pw->ne[1];

    if (g_record_mode) {
        /* ggml_mul_mat(a, b) dots over ne[0]: both have ne[0]=inputDim.
         * result: (a.ne[1]=outputDim, b.ne[1]=numNodes) */
        struct ggml_tensor *result = ggml_mul_mat(g_record_ctx,
            rec_tensor(weight), rec_tensor(features));
        ggml_set_name(result, "gnn_matmul");
        return rec_wrap(result, outputDim, numNodes);
    }

    /* Non-record: ggml graph for single op */
    struct ggml_context *ctx = micro_ctx_init();
    struct ggml_tensor *tf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inputDim, numNodes);
    struct ggml_tensor *tw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inputDim, outputDim);
    ggml_set_input(tf);
    ggml_set_input(tw);
    track_input(tf, pf->data);
    track_input(tw, pw->data);

    struct ggml_tensor *result = ggml_mul_mat(ctx, tw, tf);
    ggml_set_output(result);

    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_gallocr_t ga = run_graph(ctx, graph);

    int idx = pool_alloc(outputDim, numNodes, 2);
    ggml_backend_tensor_get(result, g_pool[idx].data, 0, (size_t)(outputDim * numNodes) * sizeof(float));

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return wrap_pool(idx);
}

RtTensor *sn_tensor_matmul(RtTensor *a, RtTensor *b)
{
    TPool *pa = unwrap(a);
    TPool *pb = unwrap(b);
    if (g_record_mode) {
        int64_t K = pa->ne[0], M = pa->ne[1];
        int64_t N = pb->ne[0];
        /* Pool convention: A is (K, M), B is (N, K).
         * We want C = A @ B → (N, M).
         *
         * ggml_mul_mat(x, y) dots over ne0: result[i0,i1] = sum_k x[k,i0]*y[k,i1]
         * Need ne0_x == ne0_y == K (the contraction dim).
         * A already has ne0=K. B has ne0=N, so transpose B in the graph.
         * The RealOrko/ggml fork fixes repeat_back contiguous assertions
         * so ggml_transpose backward is safe. */
        /* Generic matmul: pre-transpose B on host for ggml_mul_mat.
         * For GNN weight matmuls, use sn_tensor_gnn_matmul instead
         * which stores weights in the correct layout for direct ggml_mul_mat. */
        int bt_pool = pool_alloc(K, N, 2);
        TPool *bt_s = &g_pool[bt_pool];
        for (int64_t r = 0; r < K; r++)
            for (int64_t c = 0; c < N; c++)
                bt_s->data[c * K + r] = pb->data[r * N + c];
        struct ggml_tensor *bt_gt = ggml_new_tensor_2d(g_param_ctx, GGML_TYPE_F32, K, N);
        ggml_set_name(bt_gt, "matmul_B_T");
        ggml_set_input(bt_gt);
        g_record_map[bt_pool] = bt_gt;
        struct ggml_tensor *result = ggml_mul_mat(g_record_ctx, bt_gt, rec_tensor(a));
        ggml_set_name(result, "matmul_result");
        return rec_wrap(result, N, M);
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
    if (g_record_mode) return t; /* identity in graph mode — borrow-inference handles rc at call site */
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
    if (g_record_mode) return t; /* identity — batchnorm removed from GNN architecture */
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
 * Layer normalization — normalizes across features per sample.
 * Works in both record mode (ggml_norm) and direct mode (C).
 * No batch statistics, no running state — identical in training/inference.
 * ====================================================================== */

RtTensor *sn_tensor_layer_norm(RtTensor *t, RtTensor *weight, RtTensor *bias)
{
    TPool *pt = unwrap(t);
    TPool *pw = unwrap(weight);
    TPool *pb = unwrap(bias);
    int64_t feat_dim  = pt->ne[0];
    int64_t num_nodes = pt->ne[1];
    float eps = 1e-5f;

    if (g_record_mode) {
        struct ggml_tensor *gt = rec_tensor(t);
        struct ggml_tensor *gw = rec_tensor(weight);
        struct ggml_tensor *gb = rec_tensor(bias);
        struct ggml_context *ctx = g_record_ctx;

        /* ggml_rms_norm: normalizes along ne[0] per row — has backward pass support */
        struct ggml_tensor *normed = ggml_rms_norm(ctx, gt, eps);
        /* scale and shift: output = weight * normed + bias */
        struct ggml_tensor *scaled = ggml_mul(ctx, normed, gw);
        struct ggml_tensor *output = ggml_add(ctx, scaled, gb);

        return rec_wrap(output, feat_dim, num_nodes);
    }

    /* Direct C implementation for inference */
    int idx = pool_alloc(feat_dim, num_nodes, 2);
    TPool *out = &g_pool[idx];

    for (int64_t r = 0; r < num_nodes; r++) {
        /* Compute RMS across features for this node */
        float sum_sq = 0.0f;
        for (int64_t c = 0; c < feat_dim; c++) {
            float v = pt->data[r * feat_dim + c];
            sum_sq += v * v;
        }
        float rms = sqrtf(sum_sq / (float)feat_dim + eps);

        /* Normalize by RMS, then scale and shift */
        for (int64_t c = 0; c < feat_dim; c++) {
            float x = pt->data[r * feat_dim + c] / rms;
            out->data[r * feat_dim + c] = x * pw->data[c] + pb->data[c];
        }
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
        /* Differentiable mean pool using a pooling matrix.
         *
         * Heterogeneous-batching note: the pooling matrix VALUES are no
         * longer baked into the static graph. We allocate the pool tensor
         * with the (batchSize x num_nodes) shape and register it with the
         * per-batch upload registry. sn_graph_train_epoch rebuilds it each
         * batch using the REAL per-sample node counts so padded slots get
         * weight 0 and the per-graph mean uses the right denominator. The
         * `batch_index` argument fixes the shape but its values matter only
         * for shape derivation. See docs/issues/heterogeneous-graph-batching.md.
         *
         * The previous identity short-circuit (skip pool entirely when each
         * node is its own graph) is GONE — same reasoning as in
         * sn_tensor_sparse_aggregate. Regime A consumers pay one identity
         * matmul; in exchange the topology contract is uniform. */

        /* Determine num_graphs from batch_index template (used only to
         * derive the static pool_mat row count). */
        int64_t num_graphs = 1;
        for (int64_t i = 0; i < pb->n_elem; i++) {
            int64_t b = (int64_t)pb->data[i];
            if (b + 1 > num_graphs) num_graphs = b + 1;
        }

        /* Allocate the pool tensor with zero data — first batch upload overwrites. */
        int pool_idx = pool_alloc(num_nodes, num_graphs, 2);
        TPool *pool_mat = &g_pool[pool_idx];
        memset(pool_mat->data, 0, (size_t)(num_graphs * num_nodes) * sizeof(float));

        struct ggml_tensor *gpool = ggml_new_tensor_2d(g_param_ctx, GGML_TYPE_F32, num_nodes, num_graphs);
        ggml_set_name(gpool, "pool_mat");
        ggml_set_input(gpool);
        g_record_map[pool_idx] = gpool;

        /* Register for per-batch upload. */
        track_per_batch(pool_idx, PB_KIND_POOL, 0);

        /* Transpose features INSIDE the graph (not on host) so per-batch
         * uploads of node_embeddings are reflected in the recorded graph.
         * Requires the patched ggml_repeat_back that handles non-contiguous
         * input tensors via internal ggml_cont (see vendor/ggml patch). */
        struct ggml_tensor *gfeat = rec_tensor(node_embeddings);
        struct ggml_tensor *feat_t = ggml_cont(g_record_ctx,
            ggml_transpose(g_record_ctx, gfeat));
        ggml_set_name(feat_t, "embed_T");

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

double sn_tensor_norm(RtTensor *t)
{
    TPool *pt = unwrap(t);
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < pt->n_elem; i++)
        sum_sq += pt->data[i] * pt->data[i];
    return (double)sqrtf(sum_sq);
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

    /* In record mode, build dense adjacency matrix for differentiable aggregation.
     *
     * Heterogeneous-batching note: the adjacency VALUES are no longer baked
     * into the static graph at record time. Instead we register the adjacency
     * tensor with the per-batch upload registry, and sn_graph_train_epoch
     * rebuilds + uploads the batched block-diagonal adjacency for each batch.
     * This is what lets training handle graphs with variable numNodes/numEdges
     * inside one train() call. The `edge_index` and `edge_weight` arguments
     * passed here describe the FIRST batch only — they fix the SHAPE of the
     * recorded graph but not the values. See docs/issues/heterogeneous-graph-batching.md.
     *
     * The previous "identity short-circuit" optimization (skip adj entirely
     * when adjacency is exactly I) is GONE: it depended on the template
     * adjacency being representative of every batch, which is false now.
     * Regime A consumers (every graph is a single self-looping node) pay
     * one identity matmul per layer for the simplification — small price. */
    if (g_record_mode) {
        /* Allocate the adjacency tensor in g_param_ctx with the batched
         * (num_nodes x num_nodes) shape. Data is left as zeros — the first
         * batch upload in sn_graph_train_epoch overwrites it. */
        int adj_idx = pool_alloc(num_nodes, num_nodes, 2);
        TPool *adj = &g_pool[adj_idx];
        memset(adj->data, 0, (size_t)(num_nodes * num_nodes) * sizeof(float));

        struct ggml_tensor *gadj = ggml_new_tensor_2d(g_param_ctx, GGML_TYPE_F32, num_nodes, num_nodes);
        ggml_set_name(gadj, "adj");
        ggml_set_input(gadj);
        g_record_map[adj_idx] = gadj;

        /* Register for per-batch upload. The mode determines how
         * sn_graph_train_epoch normalizes the assembled adjacency. */
        track_per_batch(adj_idx, PB_KIND_ADJ, parse_agg_mode(mode));

        /* Transpose features INSIDE the graph (not on host) so per-batch
         * uploads of `features` are reflected in the recorded graph.
         * Requires the patched ggml_repeat_back. */
        struct ggml_tensor *gfeat = rec_tensor(features);
        struct ggml_tensor *gfeat_t = ggml_cont(g_record_ctx,
            ggml_transpose(g_record_ctx, gfeat));
        ggml_set_name(gfeat_t, "feat_T");

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

/* Per-sample weighted cross-entropy loss.
 *
 * logits  : shape (numClasses, batchRows)  -- pre-softmax
 * labels  : shape (numClasses, batchRows)  -- one-hot encoded
 * weights : shape (1,          batchRows)  -- per-sample importance weights
 *
 * Math:
 *   per_sample_i = -sum_c (label[i,c] * log_softmax(logits[i,c]))
 *   loss         = sum_i (weight[i] * per_sample_i) / batchRows
 *
 * Convention: scale by 1/batchRows (not 1/sum(weights)). Weights are
 * relative importance, not effective sample counts. With weights all 1.0,
 * this exactly matches ggml_cross_entropy_loss.
 *
 * Works in both record mode (loss tensor wired into the recorded forward
 * graph for backward in train_step) and direct mode (one-shot eval, used
 * by tests/test_weighted_ce.sn).
 */
RtTensor *sn_tensor_weighted_cross_entropy(RtTensor *logits_rt,
                                           RtTensor *labels_rt,
                                           RtTensor *weights_rt)
{
    TPool *pl  = unwrap(logits_rt);
    TPool *plb = unwrap(labels_rt);
    TPool *pw  = unwrap(weights_rt);

    int64_t numClasses = pl->ne[0];
    int64_t batchRows  = pl->ne[1];
    float inv_n = -1.0f / (float)(batchRows > 0 ? batchRows : 1);

    if (g_record_mode) {
        struct ggml_context *ctx = g_record_ctx;
        struct ggml_tensor *gl  = rec_tensor(logits_rt);
        struct ggml_tensor *glb = rec_tensor(labels_rt);
        struct ggml_tensor *gw  = rec_tensor(weights_rt);

        struct ggml_tensor *softmax    = ggml_soft_max(ctx, gl);
        struct ggml_tensor *log_sm     = ggml_log(ctx, softmax);
        struct ggml_tensor *per_class  = ggml_mul(ctx, log_sm, glb);
        struct ggml_tensor *per_sample = ggml_sum_rows(ctx, per_class);
        struct ggml_tensor *weighted   = ggml_mul(ctx, per_sample, gw);
        struct ggml_tensor *summed     = ggml_sum(ctx, weighted);
        struct ggml_tensor *loss       = ggml_scale(ctx, summed, inv_n);
        ggml_set_name(loss, "weighted_ce_loss");
        return rec_wrap(loss, 1, 1);
    }

    /* Direct mode: build one-shot graph, execute, read scalar back */
    struct ggml_context *ctx = micro_ctx_init();

    struct ggml_tensor *tl  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, numClasses, batchRows);
    struct ggml_tensor *tlb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, numClasses, batchRows);
    struct ggml_tensor *tw  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1,          batchRows);
    ggml_set_input(tl);
    ggml_set_input(tlb);
    ggml_set_input(tw);
    track_input(tl,  pl->data);
    track_input(tlb, plb->data);
    track_input(tw,  pw->data);

    struct ggml_tensor *softmax    = ggml_soft_max(ctx, tl);
    struct ggml_tensor *log_sm     = ggml_log(ctx, softmax);
    struct ggml_tensor *per_class  = ggml_mul(ctx, log_sm, tlb);
    struct ggml_tensor *per_sample = ggml_sum_rows(ctx, per_class);
    struct ggml_tensor *weighted   = ggml_mul(ctx, per_sample, tw);
    struct ggml_tensor *summed     = ggml_sum(ctx, weighted);
    struct ggml_tensor *loss       = ggml_scale(ctx, summed, inv_n);
    ggml_set_output(loss);

    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, loss);
    ggml_gallocr_t ga = run_graph(ctx, graph);

    int idx = pool_alloc(1, 1, 1);
    ggml_backend_tensor_get(loss, g_pool[idx].data, 0, sizeof(float));

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return wrap_pool(idx);
}

/* ======================================================================
 * Initialization
 * ======================================================================
 *
 * Kaiming weight init uses a local xorshift64 PRNG rather than libc
 * rand(). Two reasons:
 *   1. libc rand() is a global: any other caller (including dropout)
 *      shares state, so init becomes order-dependent.
 *   2. The old code called srand(time(NULL)) once per process, which
 *      made model weights jitter across runs at 1-second resolution —
 *      untrained inference was silently non-reproducible even when the
 *      caller thought they had fixed the seed. See the seeded variant
 *      below and Gnn.createWithSeed in src/gnn.sn.
 */

static uint64_t xorshift64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* Fill `pt` with Kaiming-uniform values drawn from the given PRNG
 * state. U(-bound, bound) where bound = sqrt(6 / fan_in). */
static void kaiming_fill(TPool *pt, uint64_t *state)
{
    int64_t fan_in = pt->ne[0];
    float bound = sqrtf(6.0f / (float)(fan_in > 0 ? fan_in : 1));
    for (int64_t i = 0; i < pt->n_elem; i++) {
        /* Take the high 24 bits and map to [0, 1). */
        uint64_t r = xorshift64(state);
        float u = (float)(r >> 40) * (1.0f / 16777216.0f);
        pt->data[i] = (2.0f * u - 1.0f) * bound;
    }
}

/* Unseeded Kaiming init. Lazy-seeds a process-local xorshift state
 * from time(NULL) on first use, then advances that state across
 * subsequent calls. Same observable "different every run" behaviour
 * as the previous srand(time(NULL)) path — just without polluting
 * libc rand() and without the 1-second-resolution correlation gap.
 * Callers that need reproducibility should use the seeded variant. */
RtTensor *sn_tensor_init_kaiming(RtTensor *t)
{
    static uint64_t g_state = 0;
    if (g_state == 0) {
        g_state = (uint64_t)time(NULL);
        /* xorshift64 degenerates at state 0, so substitute the
         * golden-ratio constant if time() returned 0. */
        if (g_state == 0) g_state = 0x9E3779B97F4A7C15ULL;
    }
    kaiming_fill(unwrap(t), &g_state);
    return t;
}

/* Seeded Kaiming init. Produces bit-identical weights for the same
 * seed across process invocations and machine rebuilds. */
RtTensor *sn_tensor_init_kaiming_seeded(RtTensor *t, long long seed)
{
    uint64_t state = (uint64_t)seed;
    if (state == 0) state = 0x9E3779B97F4A7C15ULL;
    /* Warm-up rounds so small/adjacent seeds decorrelate before we
     * start sampling. xorshift has a long period but short-seed
     * trajectories look correlated for the first few outputs. */
    xorshift64(&state);
    xorshift64(&state);
    xorshift64(&state);
    kaiming_fill(unwrap(t), &state);
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

void sn_tensor_pool_reset(void)
{
    for (int i = 0; i < g_pool_count; i++) {
        if (g_pool[i].data) {
            free(g_pool[i].data);
            g_pool[i].data = NULL;
        }
    }
    g_pool_count = 0;
}

static int g_pool_checkpoint = 0;

long long sn_tensor_pool_checkpoint(void)
{
    g_pool_checkpoint = g_pool_count;
    return (long long)g_pool_checkpoint;
}

void sn_tensor_pool_restore(long long checkpoint)
{
    int cp = (int)checkpoint;
    if (cp < 0 || cp > g_pool_count) return;
    for (int i = cp; i < g_pool_count; i++) {
        if (g_pool[i].data) {
            free(g_pool[i].data);
            g_pool[i].data = NULL;
        }
    }
    g_pool_count = cp;
}
