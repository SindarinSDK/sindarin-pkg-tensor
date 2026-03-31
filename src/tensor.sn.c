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

    /* A is [M, K] (ne0=K, ne1=M), B is [K, N] (ne0=N, ne1=K)
     * ggml_mul_mat(x, y) computes y * x^T, needs ne0_x == ne0_y.
     * Strategy: transpose B to B^T [N, K] (ne0=K, ne1=N), then
     *   ggml_mul_mat(B^T, A) -> result [M, N] (ne0=N, ne1=M) */
    int64_t M = pa->ne[1];
    int64_t K = pa->ne[0];
    int64_t N = pb->ne[0];

    struct ggml_init_params params = { GRAPH_CTX_SIZE, NULL, true };
    struct ggml_context *ctx = ggml_init(params);

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

    struct ggml_init_params params = { GRAPH_CTX_SIZE, NULL, true };
    struct ggml_context *ctx = ggml_init(params);

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

    struct ggml_init_params params = { GRAPH_CTX_SIZE, NULL, true };
    struct ggml_context *ctx = ggml_init(params);

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

    struct ggml_init_params params = { GRAPH_CTX_SIZE, NULL, true };
    struct ggml_context *ctx = ggml_init(params);

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
    (void)dim; /* ggml_soft_max operates along ne[0] (columns) */

    struct ggml_init_params params = { GRAPH_CTX_SIZE, NULL, true };
    struct ggml_context *ctx = ggml_init(params);

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
    int64_t num_edges = pei->ne[0] > pei->ne[1] ? pei->ne[1] : pei->ne[0];

    /* edge_index: [2, num_edges], row 0 = src, row 1 = dst */
    if (pei->ne[1] == 2) {
        /* Stored as [num_edges, 2] — ne0=2, ne1=num_edges */
        num_edges = pei->ne[1];
    } else {
        /* Stored as [2, num_edges] — ne0=num_edges, ne1=2 */
        num_edges = pei->ne[0];
    }

    int idx = pool_alloc(feat_dim, num_nodes, 2);
    TPool *out = &g_pool[idx];

    for (int64_t i = 0; i < num_edges; i++) {
        int64_t s, d;
        if (pei->ne[1] == 2) {
            /* [2, num_edges]: row 0 at offset 0, row 1 at offset num_edges */
            s = (int64_t)pei->data[i];
            d = (int64_t)pei->data[num_edges + i];
        } else {
            s = (int64_t)pei->data[i];
            d = (int64_t)pei->data[num_edges + i];
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
                d = (int64_t)pei->data[num_edges + i];
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
