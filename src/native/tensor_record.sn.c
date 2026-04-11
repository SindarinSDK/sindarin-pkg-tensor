/* ==============================================================================
 * tensor_record.sn.c — graph recording mode state + begin/end/input/param API
 * ==============================================================================
 *
 * In record mode, tensor ops create ggml_tensor nodes in a persistent
 * context instead of computing immediately. The resulting graph is then
 * used by ggml_opt for forward + backward + optimizer steps.
 *
 * Each pool handle maps to a ggml_tensor* in the record context.
 * Parameters (model weights) have their pool data uploaded to ggml.
 *
 * This TU owns:
 *   - record-mode globals (g_record_mode, g_param_ctx, g_compute_ctx,
 *     g_record_ctx, g_record_map)
 *   - per-batch upload registry (g_pb_*)
 *   - opt state shared with tensor_train.sn.c (g_opt_ctx, g_opt_sched,
 *     g_opt_param_buf, g_opt_*_tensor, g_opt_params)
 *   - sn_graph_begin / sn_graph_end / sn_graph_input / sn_graph_input_data /
 *     sn_graph_param / sn_graph_compute_loss
 *   - the rec_tensor / rec_wrap / track_per_batch / parse_agg_mode helpers
 * ============================================================================== */

#include "tensor_internal.h"

/* ======================================================================
 * Record-mode globals
 * ====================================================================== */

bool                 g_record_mode  = false;
struct ggml_context *g_param_ctx    = NULL;   /* static: params + inputs */
struct ggml_context *g_compute_ctx  = NULL;   /* no_alloc: intermediate ops */
struct ggml_context *g_record_ctx   = NULL;   /* points to g_compute_ctx for ops */
struct ggml_tensor  *g_record_map[SN_TENSOR_MAX];

struct ggml_tensor *rec_tensor(RtTensor *rt) {
    return g_record_map[rt->__sn___handle];
}

RtTensor *rec_wrap(struct ggml_tensor *gt, int64_t ne0, int64_t ne1) {
    int idx = pool_alloc(ne0, ne1, 2);
    g_record_map[idx] = gt;
    return wrap_pool(idx);
}

/* ======================================================================
 * Opt state — lazy-initialized by tensor_train.sn.c, freed in sn_graph_end
 * ====================================================================== */

int g_pool_count_before_record = 0;

ggml_opt_context_t    g_opt_ctx              = NULL;
ggml_backend_sched_t  g_opt_sched            = NULL;
ggml_backend_buffer_t g_opt_param_buf        = NULL;
struct ggml_tensor   *g_opt_loss_tensor      = NULL;
struct ggml_tensor   *g_opt_features_tensor  = NULL;
struct ggml_tensor   *g_opt_labels_tensor    = NULL;
struct ggml_tensor   *g_opt_weights_tensor   = NULL;
/* PPO-only: the 5th per-batch input tensor holding
 * log π_old(a_i|s_i), one scalar per sample. NULL in weighted-CE mode;
 * set on the first call to sn_graph_train_epoch_ppo within a
 * begin/end cycle. See sn_tensor_ppo_clipped_loss in tensor_loss.sn.c and
 * docs/issues/ppo-clipped-objective.md. */
struct ggml_tensor   *g_opt_old_log_probs_tensor = NULL;
/* Phase 4: PPO VF-loss inputs — per-batch scalar-per-sample tensors
 * holding V_target (== advantages + V_old, from computeGae's returns)
 * and V_old (the frozen critic estimate snapshotted before training).
 * Both are non-NULL whenever sn_graph_train_epoch_ppo is the active
 * entry point, regardless of whether VF loss is actually enabled:
 * the strategy passes (1,1) placeholder tensors when valueCoeff=0
 * and the loss op gates the VF subgraph away, but the per-batch
 * upload registry still tracks them so re-entering the driver with
 * the same strategy doesn't trip the tensor-identity sanity check. */
struct ggml_tensor   *g_opt_value_targets_tensor = NULL;
struct ggml_tensor   *g_opt_old_values_tensor    = NULL;

struct ggml_opt_optimizer_params g_opt_params;

struct ggml_opt_optimizer_params sn_get_opt_params(void *userdata) {
    (void)userdata;
    return g_opt_params;
}

/* ======================================================================
 * Per-batch upload registry — tensors whose VALUES change per minibatch.
 *
 * Used for heterogeneous graph batching: when the caller's training set has
 * graphs with variable numNodes/numEdges, the dense adjacency and pooling
 * matrices baked into the static recorded graph cannot stay constant. Each
 * call to sn_tensor_sparse_aggregate / sn_tensor_mean_pool in record mode
 * registers its output ggml input tensor here, and sn_graph_train_epoch
 * walks the registry every batch to upload freshly assembled host buffers.
 *
 * See docs/issues/heterogeneous-graph-batching.md for the rationale.
 * ====================================================================== */

int g_pb_pool_idx[MAX_PER_BATCH_TENSORS];
int g_pb_kind[MAX_PER_BATCH_TENSORS];
int g_pb_mode[MAX_PER_BATCH_TENSORS];
int g_pb_count = 0;

void track_per_batch(int pool_idx, int kind, int mode) {
    if (g_pb_count >= MAX_PER_BATCH_TENSORS) {
        fprintf(stderr, "per-batch tensor registry full (max %d)\n", MAX_PER_BATCH_TENSORS);
        abort();
    }
    g_pb_pool_idx[g_pb_count] = pool_idx;
    g_pb_kind[g_pb_count]     = kind;
    g_pb_mode[g_pb_count]     = mode;
    g_pb_count++;
}

int parse_agg_mode(const char *mode) {
    if (mode && strcmp(mode, "sum_normalized") == 0) return PB_MODE_SUM_NORMALIZED;
    if (mode && strcmp(mode, "mean") == 0)           return PB_MODE_MEAN;
    return PB_MODE_SUM;
}

/* ======================================================================
 * Graph recording API — public entry points
 * ====================================================================== */

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
    g_opt_ctx                    = NULL;
    g_opt_sched                  = NULL;
    g_opt_param_buf              = NULL;
    g_opt_loss_tensor            = NULL;
    g_opt_features_tensor        = NULL;
    g_opt_labels_tensor          = NULL;
    g_opt_weights_tensor         = NULL;
    g_opt_old_log_probs_tensor   = NULL;
    g_opt_value_targets_tensor   = NULL;
    g_opt_old_values_tensor      = NULL;

    /* Reset per-batch upload registry — tensors get re-registered as the
     * record-mode forward pass walks the layers. */
    g_pb_count = 0;
}

/* Phase 2: pause/resume the record-mode dispatch flag around
 * direct-mode forward passes (e.g. the trainer's per-epoch drift
 * check). Toggles g_record_mode only — the compute / param contexts,
 * the per-batch upload registry, and the optimizer state all stay
 * allocated, so recording picks up exactly where it left off when
 * you call resume. Paired with sn_tensor_pool_checkpoint /
 * sn_tensor_pool_restore to reclaim any pool slots allocated while
 * paused.
 *
 * Intended use:
 *   long cp   = sn_tensor_pool_checkpoint();
 *   bool prev = sn_graph_pause_recording();
 *   // ... run inference forward passes on caller-owned graphs ...
 *   sn_graph_resume_recording(prev);
 *   sn_tensor_pool_restore(cp);
 *
 * Do NOT use this to jump in and out of an unrelated record session
 * — the pause only works for suspending ops within a single active
 * record session. */
bool sn_graph_pause_recording(void) {
    bool prev = g_record_mode;
    g_record_mode = false;
    return prev;
}

void sn_graph_resume_recording(bool prev) {
    g_record_mode = prev;
}

void sn_graph_end(void) {
    /* Tear down epoch-loop training state if it was lazy-initialized */
    if (g_opt_ctx)       { ggml_opt_free(g_opt_ctx);                  g_opt_ctx       = NULL; }
    if (g_opt_sched)     { ggml_backend_sched_free(g_opt_sched);      g_opt_sched     = NULL; }
    if (g_opt_param_buf) { ggml_backend_buffer_free(g_opt_param_buf); g_opt_param_buf = NULL; }
    g_opt_loss_tensor            = NULL;
    g_opt_features_tensor        = NULL;
    g_opt_labels_tensor          = NULL;
    g_opt_weights_tensor         = NULL;
    g_opt_old_log_probs_tensor   = NULL;
    g_opt_value_targets_tensor   = NULL;
    g_opt_old_values_tensor      = NULL;

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
    /* Data uploaded after ggml_backend_alloc_ctx_tensors in sn_graph_train_epoch */
    g_record_map[idx] = gt;
    return rt;
}
