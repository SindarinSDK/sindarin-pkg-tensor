/* ==============================================================================
 * tensor_train.sn.c — training metric callback + per-epoch training driver
 * ==============================================================================
 *
 * Contains:
 *   - Phase J / Phase M training metric callback (set/clear/emit + closure copy)
 *   - sn_graph_train_epoch_impl: shared per-epoch driver body used by both
 *     the weighted-CE and PPO wrappers
 *   - sn_graph_train_epoch:     weighted-CE public wrapper
 *   - sn_graph_train_epoch_ppo: PPO public wrapper
 *
 * Reads globals defined in tensor_record.sn.c (g_record_mode, g_record_ctx,
 * g_pool_count_before_record, g_opt_*, g_pb_*) and tensor_pool.sn.c (g_pool,
 * g_pool_count). Relies on rec_tensor from tensor_record.sn.c and
 * ensure_backend from tensor_backend.sn.c.
 * ============================================================================== */

#include "tensor_internal.h"

/* ============================================================================
 * Phase J / Phase M: training metric callback
 *
 * Consumers (skynet's MetricsClient) register a Sindarin lambda via
 * sn_graph_set_train_metric_callback. During Gnn.train() the package
 * emits (name, value, labels) tuples via sn_graph_emit_train_metric,
 * which invokes the registered closure if one is set.
 *
 * The `labels` arg is a Sindarin StringField[] — an SnArray of (key,
 * value) string pairs. It is passed as an opaque SnArray pointer
 * through the C→closure boundary; the callback's compiled Sindarin
 * body receives it back as StringField[] and can iterate.
 *
 * Closure ABI is copied from sindarin-pkg-threads/src/threads_internal.h:
 *   [0]              void *fn       — function pointer
 *   [sizeof(ptr)]    size_t size    — total closure bytes
 *   [ptr+size_t]     void *cleanup  — cleanup pointer (zero on copies)
 *   [...]            captured vars
 *
 * The callback is deep-copied on registration so it survives across
 * begin/end cycles and multiple train() invocations. Registering a new
 * callback frees the previously stored one. Clearing with
 * sn_graph_clear_train_metric_callback frees and nulls the slot.
 * ============================================================================ */

void *g_train_metric_cb = NULL;

void *sn_tensor_closure_copy(void *closure)
{
    size_t size = *(size_t *)((char *)closure + sizeof(void *));
    void *copy = malloc(size);
    if (!copy) { fprintf(stderr, "sn_tensor_closure_copy: oom\n"); exit(1); }
    memcpy(copy, closure, size);
    /* Zero the cleanup pointer so freeing the copy does not free
     * captured objects the caller still owns. */
    void **cleanup = (void **)((char *)copy + sizeof(void *) + sizeof(size_t));
    *cleanup = NULL;
    return copy;
}

void sn_graph_set_train_metric_callback(void *cb)
{
    if (g_train_metric_cb) { free(g_train_metric_cb); g_train_metric_cb = NULL; }
    if (cb) g_train_metric_cb = sn_tensor_closure_copy(cb);
}

void sn_graph_clear_train_metric_callback(void)
{
    if (g_train_metric_cb) { free(g_train_metric_cb); g_train_metric_cb = NULL; }
}

void sn_graph_emit_train_metric(char *name, double value, SnArray *labels)
{
    if (!g_train_metric_cb) return;
    typedef void (*CbFn)(void *closure, char *name, double value, SnArray *labels);
    CbFn fn = (CbFn)(*(void **)g_train_metric_cb);
    fn(g_train_metric_cb, name, value, labels);
}

/* ============================================================================
 * Phase 1.4: gradient clipping by global norm (post-hoc delta clipping)
 *
 * ggml's `ggml_opt_eval` runs forward + backward + optimizer step as a
 * single atomic call — there is no hook between backward and the
 * optimizer step where we could clip the gradients directly. Instead
 * we apply "delta clipping" around the call: snapshot every PARAM
 * tensor before eval, let ggml run its unmodified step, then compare
 * the post-eval values against the snapshot. If the global L2 of the
 * parameter delta exceeds `g_grad_clip_norm`, rewind the deltas to
 * length `g_grad_clip_norm` via `new = old + scale * (new - old)` and
 * upload the scaled values back to the backend.
 *
 * Mathematically this is equivalent to pre-step gradient clipping for
 * SGD (`delta = -lr * g` so bounding ||delta|| bounds ||g||). For
 * AdamW it's not strictly identical because the moments still see the
 * unscaled gradient, but it delivers the same stability property —
 * bounded step size per batch, preventing a single bad batch from
 * destroying the policy. This is the property that matters for the
 * "don't destroy a training run" Tier 0 goal.
 *
 * `g_grad_clip_norm` is a global because passing it through every
 * sn_graph_train_epoch_* variant's parameter list would add noise to
 * the ABI without enabling any feature. Trainer.run sets it before
 * the epoch loop via sn_graph_set_grad_clip_norm and resets it to
 * 0.0 afterwards.
 *
 * Value 0.0 means "disabled" (no snapshot, no overhead).
 * ============================================================================ */

double g_grad_clip_norm = 0.0;

void sn_graph_set_grad_clip_norm(double v)
{
    g_grad_clip_norm = v;
}

/* ======================================================================
 * Per-epoch training driver — shared implementation.
 *
 * The training entry point used by Gnn.train() (via sn_graph_train_epoch)
 * AND by Gnn.trainPpo() (via sn_graph_train_epoch_ppo). Key design points:
 *
 *   - Uses ggml_opt with loss_type=GGML_OPT_LOSS_TYPE_SUM and a
 *     pre-built loss tensor as `outputs`. ggml_sum of a scalar is
 *     identity, so the entire loss expression flows through unchanged
 *     and backward propagates correctly.
 *   - Drives the per-batch loop manually via ggml_opt_alloc/eval, with
 *     three OR four input uploads per batch:
 *         weighted CE: features, labels, weights
 *         PPO:         features, labels (= actionsOneHot), weights (= advantages),
 *                      oldLogProbs
 *   - Lazy-initializes the opt context on the first call within a
 *     sn_graph_begin/end cycle and reuses it across epochs. The state
 *     is freed in sn_graph_end.
 *
 * The PPO variant adds ONE extra per-batch host buffer (log π_old(a_i|s_i),
 * one scalar per sample) and ONE extra ggml input tensor upload. Everything
 * else — host buffer assembly, padded feature copy, block-diagonal adj,
 * pool matrix, attention mask, param readback — is identical between the
 * two loss kinds and lives here, shared.
 *
 * `old_log_probs_rt` and `old_log_probs_host` are both NULL for the
 * weighted-CE path and both non-NULL for the PPO path. The two must match
 * (either both NULL or both non-NULL); mixing is undefined. A caller that
 * switches loss kinds mid-begin/end cycle will trip the tensor-identity
 * sanity check below.
 *
 * Caller is responsible for shuffling: pass a permutation of sample
 * indices in `batch_perm` (one entry per training sample). Per-epoch
 * shuffling means re-shuffling the permutation between calls.
 * ====================================================================== */
static double sn_graph_train_epoch_impl(
    RtTensor *loss_rt,
    RtTensor *features_rt,
    RtTensor *labels_rt,
    RtTensor *weights_rt,
    RtTensor *old_log_probs_rt,     /* PPO only; NULL for weighted CE */
    RtTensor *value_targets_rt,     /* PPO only; NULL for weighted CE */
    RtTensor *old_values_rt,        /* PPO only; NULL for weighted CE */
    SnArray  *features_host,         /* total_samples * max_nodes * feature_dim doubles, zero-padded */
    SnArray  *labels_host,           /* total_samples * labels_per_sample doubles */
    SnArray  *weights_host,          /* total_samples doubles, one weight per sample */
    SnArray  *old_log_probs_host,    /* total_samples doubles, one per sample; PPO only; NULL for weighted CE */
    SnArray  *value_targets_host,    /* total_samples doubles; PPO only; NULL for weighted CE */
    SnArray  *old_values_host,       /* total_samples doubles; PPO only; NULL for weighted CE */
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

    /* PPO: resolve the optional 5th input tensor. */
    struct ggml_tensor *old_log_probs = NULL;
    if (old_log_probs_rt) {
        old_log_probs = rec_tensor(old_log_probs_rt);
        if (!old_log_probs) return -1.0;
    }

    /* Phase 4 PPO VF: resolve the two extra per-batch inputs. Both are
     * expected to be non-NULL whenever this is the PPO path (the
     * strategy passes (1,1) zero placeholders when VF is disabled) and
     * NULL whenever this is the weighted-CE path. */
    struct ggml_tensor *value_targets = NULL;
    struct ggml_tensor *old_values    = NULL;
    if (value_targets_rt) {
        value_targets = rec_tensor(value_targets_rt);
        if (!value_targets) return -1.0;
    }
    if (old_values_rt) {
        old_values = rec_tensor(old_values_rt);
        if (!old_values) return -1.0;
    }

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

        g_opt_ctx                  = ggml_opt_init(opt_params_s);
        /* Use ggml_opt_loss() — the loss tensor that ggml_opt_build creates
         * in ctx_static (with a real buffer). The original `loss` we built
         * lives in g_compute_ctx (no_alloc) and has no buffer to read from. */
        g_opt_loss_tensor            = ggml_opt_loss(g_opt_ctx);
        g_opt_features_tensor        = features;
        g_opt_labels_tensor          = labels;
        g_opt_weights_tensor         = weights;
        g_opt_old_log_probs_tensor   = old_log_probs; /* NULL in weighted-CE mode */
        g_opt_value_targets_tensor   = value_targets;  /* NULL in weighted-CE mode */
        g_opt_old_values_tensor      = old_values;     /* NULL in weighted-CE mode */

        /* Deferred optimizer state restore: if sn_opt_state_set_restore()
         * was called before this training cycle, load the saved m/v moments
         * and iter counter into the freshly-created g_opt_ctx. This gives
         * AdamW continuity across training rounds — the optimizer picks up
         * exactly where the previous round left off. */
        sn_opt_state_restore();
    }

    /* Sanity check: caller must use the same tensors as the init call.
     * Includes the optional PPO slots — switching loss kinds mid-cycle
     * (weighted CE → PPO or vice versa) is not supported and trips here. */
    if (features != g_opt_features_tensor ||
        labels   != g_opt_labels_tensor   ||
        weights  != g_opt_weights_tensor  ||
        old_log_probs != g_opt_old_log_probs_tensor ||
        value_targets != g_opt_value_targets_tensor ||
        old_values    != g_opt_old_values_tensor) {
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
    size_t features_bytes      = (size_t)features_per_batch * sizeof(float);
    size_t adj_bytes           = (size_t)nodes_per_batch * (size_t)nodes_per_batch * sizeof(float);
    size_t pool_bytes          = (size_t)n_per_batch * (size_t)nodes_per_batch * sizeof(float);
    size_t labels_bytes        = ggml_nbytes(labels);
    size_t weights_bytes       = ggml_nbytes(weights);
    /* PPO: old_log_probs is (1, n_per_batch); one scalar per sample. */
    size_t old_log_probs_bytes = old_log_probs ? ggml_nbytes(old_log_probs) : 0;
    /* Phase 4 PPO VF: value_targets and old_values are also (1, n_per_batch). */
    size_t value_targets_bytes = value_targets ? ggml_nbytes(value_targets) : 0;
    size_t old_values_bytes    = old_values    ? ggml_nbytes(old_values)    : 0;

    float *features_buf      = (float *)calloc((size_t)features_per_batch, sizeof(float));
    float *adj_buf           = (float *)calloc((size_t)nodes_per_batch * (size_t)nodes_per_batch, sizeof(float));
    float *att_mask_buf      = (float *)calloc((size_t)nodes_per_batch * (size_t)nodes_per_batch, sizeof(float));
    float *pool_buf          = (float *)calloc((size_t)n_per_batch * (size_t)nodes_per_batch, sizeof(float));
    float *labels_buf        = (float *)malloc(labels_bytes);
    float *weights_buf       = (float *)malloc(weights_bytes);
    float *old_log_probs_buf = old_log_probs ? (float *)malloc(old_log_probs_bytes) : NULL;
    float *value_targets_buf = value_targets ? (float *)malloc(value_targets_bytes) : NULL;
    float *old_values_buf    = old_values    ? (float *)malloc(old_values_bytes)    : NULL;

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
        free(att_mask_buf);
        if (old_log_probs_buf) free(old_log_probs_buf);
        if (value_targets_buf) free(value_targets_buf);
        if (old_values_buf)    free(old_values_buf);
        return -1.0;
    }

    double total_loss = 0.0;

    /* Phase 1.4: allocate per-param snapshot buffers up front if grad
     * clipping is enabled. These hold the "before ggml_opt_eval"
     * parameter state so we can diff against the post-step values
     * and clip the delta in place when it exceeds g_grad_clip_norm. */
    int clip_n_params = 0;
    float **clip_snapshots = NULL;
    float **clip_new_bufs  = NULL;
    int   *clip_param_pool_idx = NULL;
    if (g_grad_clip_norm > 0.0) {
        for (int i = 0; i < g_pool_count; i++) {
            struct ggml_tensor *gt = g_record_map[i];
            if (gt && (gt->flags & GGML_TENSOR_FLAG_PARAM)) clip_n_params++;
        }
        if (clip_n_params > 0) {
            clip_snapshots = (float **)malloc((size_t)clip_n_params * sizeof(float *));
            clip_new_bufs  = (float **)malloc((size_t)clip_n_params * sizeof(float *));
            clip_param_pool_idx = (int *)malloc((size_t)clip_n_params * sizeof(int));
            int j = 0;
            for (int i = 0; i < g_pool_count; i++) {
                struct ggml_tensor *gt = g_record_map[i];
                if (gt && (gt->flags & GGML_TENSOR_FLAG_PARAM)) {
                    size_t n_elem = (size_t)g_pool[i].n_elem;
                    clip_snapshots[j] = (float *)malloc(n_elem * sizeof(float));
                    clip_new_bufs[j]  = (float *)malloc(n_elem * sizeof(float));
                    clip_param_pool_idx[j] = i;
                    j++;
                }
            }
        }
    }

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

            /* --- PPO: one oldLogProb per sample */
            if (old_log_probs_buf) {
                double *olp = (double *)sn_array_get(old_log_probs_host, sample_idx);
                old_log_probs_buf[i] = (float)(*olp);
            }

            /* --- Phase 4 PPO VF: one valueTarget and one oldValue per sample.
             * Both buffers are non-NULL iff the caller is on the PPO path
             * (the weighted-CE wrapper passes NULL). Even when VF loss is
             * disabled (strategy passed (1,1) placeholder tensors), the
             * buffers are still uploaded so the static graph's sanity
             * check sees stable tensor identities across epochs. */
            if (value_targets_buf) {
                double *vt = (double *)sn_array_get(value_targets_host, sample_idx);
                value_targets_buf[i] = (float)(*vt);
            }
            if (old_values_buf) {
                double *ov = (double *)sn_array_get(old_values_host, sample_idx);
                old_values_buf[i] = (float)(*ov);
            }
        }

        /* Derive the Phase L attention additive mask from adj_buf:
         * 0.0f where adj_buf > 0 (edge exists), -10.0f elsewhere
         * (non-edge). Added to the logits before softmax. -10 is
         * chosen (instead of the naive -1e9) because -1e9 is
         * numerically extreme enough to make the softmax-backward's
         * gradient effectively zero at edge positions too (the
         * softmax output concentrates to ~0/1/num_edges per dest,
         * and the `y*(dy - dot(y,dy))` backward produces vanishing
         * values for the edge slots). -10 is still enough to
         * suppress non-edge attention (exp(-10) ≈ 4e-5) while
         * preserving gradient flow. */
        {
            const long long nn2 = nodes_per_batch * nodes_per_batch;
            for (long long k = 0; k < nn2; k++) {
                att_mask_buf[k] = (adj_buf[k] > 0.0f) ? 0.0f : -10.0f;
            }
        }

        /* Allocate the OPT graph for this batch (cached after first call) */
        ggml_opt_alloc(g_opt_ctx, /*backward =*/ true);

        /* Upload the batch's features/labels/weights to the backend buffers */
        ggml_backend_tensor_set(g_opt_features_tensor, features_buf, 0, features_bytes);
        ggml_backend_tensor_set(g_opt_labels_tensor,   labels_buf,   0, labels_bytes);
        ggml_backend_tensor_set(g_opt_weights_tensor,  weights_buf,  0, weights_bytes);

        /* PPO: upload the oldLogProbs slice for this batch. */
        if (g_opt_old_log_probs_tensor && old_log_probs_buf) {
            ggml_backend_tensor_set(g_opt_old_log_probs_tensor,
                                    old_log_probs_buf, 0, old_log_probs_bytes);
        }

        /* Phase 4 PPO VF: upload the valueTargets and oldValues slices
         * for this batch. When VF loss is disabled, the strategy has
         * allocated (1,1) placeholder tensors on the recorded graph and
         * their host buffers carry zeros — the backend still gets the
         * upload (a single float per tensor) so the sanity check
         * doesn't fire on the second batch. */
        if (g_opt_value_targets_tensor && value_targets_buf) {
            ggml_backend_tensor_set(g_opt_value_targets_tensor,
                                    value_targets_buf, 0, value_targets_bytes);
        }
        if (g_opt_old_values_tensor && old_values_buf) {
            ggml_backend_tensor_set(g_opt_old_values_tensor,
                                    old_values_buf, 0, old_values_bytes);
        }

        /* Upload the batched adjacency / pool matrix / attention mask
         * to every tensor in the per-batch registry. Within a single
         * Gnn.train() call all ADJ tensors share the same
         * edgeIndex/edgeWeight + mode (every layer's aggregate()
         * sees the same args), so they all receive the same buffer;
         * same story for the (single) POOL tensor. ATT_MASK tensors
         * (one per attention layer) all receive the same mask
         * buffer. */
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
            } else if (g_pb_kind[r] == PB_KIND_ATT_MASK) {
                size_t want = adj_bytes; /* same shape as adj_buf */
                size_t have = ggml_nbytes(gt);
                ggml_backend_tensor_set(gt, att_mask_buf, 0, want < have ? want : have);
            }
        }

        /* Phase 1.4: snapshot all PARAM tensors right before the step
         * so the delta-clip post-processor has a baseline to diff
         * against. Skipped when g_grad_clip_norm == 0.0. */
        if (clip_n_params > 0) {
            for (int k = 0; k < clip_n_params; k++) {
                int pi = clip_param_pool_idx[k];
                struct ggml_tensor *gt = g_record_map[pi];
                size_t bytes = (size_t)g_pool[pi].n_elem * sizeof(float);
                if (gt && gt->buffer) {
                    ggml_backend_tensor_get(gt, clip_snapshots[k], 0, bytes);
                } else if (gt && gt->data) {
                    memcpy(clip_snapshots[k], gt->data, bytes);
                }
            }
        }

        /* Run forward + backward + AdamW step (NULL result — we read loss directly) */
        ggml_opt_eval(g_opt_ctx, NULL);

        /* Read the scalar loss back */
        float batch_loss = 0.0f;
        ggml_backend_tensor_get(g_opt_loss_tensor, &batch_loss, 0, sizeof(float));
        total_loss += (double)batch_loss;

        /* Phase 1.4: post-hoc delta clipping. Read each PARAM's new
         * values, accumulate the global L2 of (new - snapshot), and if
         * it exceeds g_grad_clip_norm, scale the delta to length
         * g_grad_clip_norm and upload the clipped values back. The
         * new_bufs slab is reused across batches (allocated once
         * before the loop) to avoid per-batch malloc churn. */
        if (clip_n_params > 0) {
            double delta_sq = 0.0;
            for (int k = 0; k < clip_n_params; k++) {
                int pi = clip_param_pool_idx[k];
                struct ggml_tensor *gt = g_record_map[pi];
                size_t bytes = (size_t)g_pool[pi].n_elem * sizeof(float);
                if (gt && gt->buffer) {
                    ggml_backend_tensor_get(gt, clip_new_bufs[k], 0, bytes);
                } else if (gt && gt->data) {
                    memcpy(clip_new_bufs[k], gt->data, bytes);
                }
                float *old = clip_snapshots[k];
                float *nw  = clip_new_bufs[k];
                for (long long j = 0; j < g_pool[pi].n_elem; j++) {
                    double d = (double)nw[j] - (double)old[j];
                    delta_sq += d * d;
                }
            }
            double delta_norm = sqrt(delta_sq);
            if (delta_norm > g_grad_clip_norm) {
                double scale = g_grad_clip_norm / delta_norm;
                for (int k = 0; k < clip_n_params; k++) {
                    int pi = clip_param_pool_idx[k];
                    struct ggml_tensor *gt = g_record_map[pi];
                    float *old = clip_snapshots[k];
                    float *nw  = clip_new_bufs[k];
                    for (long long j = 0; j < g_pool[pi].n_elem; j++) {
                        double d = (double)nw[j] - (double)old[j];
                        nw[j] = (float)((double)old[j] + scale * d);
                    }
                    size_t bytes = (size_t)g_pool[pi].n_elem * sizeof(float);
                    if (gt && gt->buffer) {
                        ggml_backend_tensor_set(gt, nw, 0, bytes);
                    } else if (gt && gt->data) {
                        memcpy(gt->data, nw, bytes);
                    }
                }
            }
        }
    }

    /* Phase 1.4: free per-epoch grad-clip scratch buffers. */
    if (clip_n_params > 0) {
        for (int k = 0; k < clip_n_params; k++) {
            free(clip_snapshots[k]);
            free(clip_new_bufs[k]);
        }
        free(clip_snapshots);
        free(clip_new_bufs);
        free(clip_param_pool_idx);
    }

    free(features_buf);
    free(adj_buf);
    free(att_mask_buf);
    free(pool_buf);
    free(labels_buf);
    free(weights_buf);
    if (old_log_probs_buf) free(old_log_probs_buf);
    if (value_targets_buf) free(value_targets_buf);
    if (old_values_buf)    free(old_values_buf);

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

/* Weighted-CE training driver — thin wrapper over the shared impl. */
double sn_graph_train_epoch(
    RtTensor *loss_rt,
    RtTensor *features_rt,
    RtTensor *labels_rt,
    RtTensor *weights_rt,
    SnArray  *features_host,
    SnArray  *labels_host,
    SnArray  *weights_host,
    SnArray  *adj_host,
    SnArray  *real_node_count_host,
    SnArray  *batch_perm,
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
    return sn_graph_train_epoch_impl(
        loss_rt, features_rt, labels_rt, weights_rt,
        /*old_log_probs_rt=*/ NULL,
        /*value_targets_rt=*/ NULL,
        /*old_values_rt=*/    NULL,
        features_host, labels_host, weights_host,
        /*old_log_probs_host=*/ NULL,
        /*value_targets_host=*/ NULL,
        /*old_values_host=*/    NULL,
        adj_host, real_node_count_host, batch_perm,
        feature_dim, labels_per_sample, n_per_batch, max_nodes_per_graph,
        optimizer_str, lr, beta1, beta2, eps, wd);
}

/* PPO training driver — thin wrapper over the shared impl.
 *
 * Adds one extra per-batch host buffer (log π_old(a_i|s_i)) on top of the
 * weighted-CE contract. The caller (Gnn.trainPpo) assembles the host buffer
 * via a pre-train non-record forward pass on every sample. `labels_rt` holds
 * the one-hot action mask (same shape as supervised labels), `weights_rt`
 * holds the per-sample advantage A_i, and `old_log_probs_rt` holds log π_old.
 * The loss tensor is expected to be the output of
 * sn_tensor_ppo_clipped_loss(logits, oldLogProbs, actionsOneHot, advantages).
 */
double sn_graph_train_epoch_ppo(
    RtTensor *loss_rt,
    RtTensor *features_rt,
    RtTensor *labels_rt,
    RtTensor *weights_rt,
    RtTensor *old_log_probs_rt,
    RtTensor *value_targets_rt,
    RtTensor *old_values_rt,
    SnArray  *features_host,
    SnArray  *labels_host,
    SnArray  *weights_host,
    SnArray  *old_log_probs_host,
    SnArray  *value_targets_host,
    SnArray  *old_values_host,
    SnArray  *adj_host,
    SnArray  *real_node_count_host,
    SnArray  *batch_perm,
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
    return sn_graph_train_epoch_impl(
        loss_rt, features_rt, labels_rt, weights_rt, old_log_probs_rt,
        value_targets_rt, old_values_rt,
        features_host, labels_host, weights_host, old_log_probs_host,
        value_targets_host, old_values_host,
        adj_host, real_node_count_host, batch_perm,
        feature_dim, labels_per_sample, n_per_batch, max_nodes_per_graph,
        optimizer_str, lr, beta1, beta2, eps, wd);
}
