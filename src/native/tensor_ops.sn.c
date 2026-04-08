/* ==============================================================================
 * tensor_ops.sn.c — arithmetic, activations, normalization, reductions
 * ==============================================================================
 *
 * Direct-mode ops run through micro-graph helpers from tensor_backend.sn.c.
 * Record-mode paths append ggml nodes to g_compute_ctx (via rec_wrap from
 * tensor_record.sn.c) so they become part of the training graph.
 *
 * Ops in this file:
 *   Arithmetic  : sn_tensor_gnn_matmul, sn_tensor_matmul, sn_tensor_add,
 *                 sn_tensor_scale
 *   Activations : sn_tensor_relu, sn_tensor_softmax, sn_tensor_dropout
 *   Normalization: sn_tensor_batch_norm, sn_tensor_layer_norm
 *   Reductions  : sn_tensor_norm, sn_tensor_argmax
 * ============================================================================== */

#include "tensor_internal.h"

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

/* Layer normalization — normalizes across features per sample.
 * Works in both record mode (ggml_norm) and direct mode (C).
 * No batch statistics, no running state — identical in training/inference. */

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
 * Scalar reductions
 * ====================================================================== */

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
