/* ==============================================================================
 * tensor_internal.h — shared state and helpers for the split ggml bridge
 * ==============================================================================
 *
 * Each src/native/tensor_*.sn.c file is a separate translation unit. The
 * globals that used to be `static` in the monolithic tensor.sn.c now have
 * plain external linkage, declared here and defined in exactly one .sn.c
 * file. Helper prototypes follow the same pattern.
 *
 * The compiler's generated `sn_types.h` is force-included ahead of any .sn.c
 * file, so `__sn__Tensor`, `SnArray`, `sn_array_*` are already visible before
 * this header is #included.
 * ============================================================================== */

#ifndef SN_TENSOR_INTERNAL_H
#define SN_TENSOR_INTERNAL_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-opt.h>

/* RtTensor is __sn__Tensor, defined in the compiler-generated sn_types.h
 * (force-included). It has fields: int __rc__, long long __sn___handle. */
typedef __sn__Tensor RtTensor;

/* ======================================================================
 * Tensor pool (defined in tensor_pool.sn.c)
 * ====================================================================== */

#define SN_TENSOR_MAX 65536

typedef struct {
    float  *data;
    int64_t ne[4];    /* shape: ne[0]=cols, ne[1]=rows, ne[2], ne[3] */
    int     n_dims;
    int64_t n_elem;
} TPool;

extern TPool g_pool[SN_TENSOR_MAX];
extern int   g_pool_count;

int       pool_alloc(int64_t ne0, int64_t ne1, int n_dims);
RtTensor *wrap_pool(int idx);
TPool    *unwrap(RtTensor *rt);

/* ======================================================================
 * ggml backend + micro-graph helpers (defined in tensor_backend.sn.c)
 * ====================================================================== */

extern ggml_backend_t g_backend;
extern int            g_backend_gpu;

void ensure_backend(void);

/* Micro-graph helpers — for direct-mode (non-record) tensor ops that
 * build a single-op ggml graph, run it, read the result back, and free
 * the context. Each op grabs a fresh micro-context via micro_ctx_init(),
 * tracks its input tensors with track_input() so their host data is
 * uploaded by run_graph(), then calls run_graph() which returns an
 * allocator the caller must free after reading outputs. */
/* The largest direct-mode loss graph is sn_tensor_ppo_clipped_loss with the
 * Phase 4 VF extension: ~26 tensor nodes when VF clipping is active (policy
 * surrogate, entropy bonus, VF unclipped branch, and the clipped-max branch
 * with its two relu hinges). 48 tensors gives headroom for small op-count
 * growth without bumping the constant again. */
#define GRAPH_CTX_SIZE (48 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096)
#define MAX_INPUTS 8

struct ggml_context *micro_ctx_init(void);
void                 track_input(struct ggml_tensor *t, const float *host_data);
ggml_gallocr_t       run_graph(struct ggml_context *ctx, struct ggml_cgraph *graph);

/* ======================================================================
 * Graph recording mode (defined in tensor_record.sn.c)
 * ====================================================================== */

#define GRAPH_PARAM_CTX_SIZE   (32 * 1024 * 1024)   /* params + inputs (32MB) */
#define GRAPH_COMPUTE_CTX_SIZE (128 * 1024 * 1024)  /* intermediate ops (128MB) */

extern bool                 g_record_mode;
extern struct ggml_context *g_param_ctx;    /* static: params + inputs */
extern struct ggml_context *g_compute_ctx;  /* no_alloc: intermediate ops */
extern struct ggml_context *g_record_ctx;   /* points to g_compute_ctx for ops */
extern struct ggml_tensor  *g_record_map[SN_TENSOR_MAX];

struct ggml_tensor *rec_tensor(RtTensor *rt);
RtTensor           *rec_wrap(struct ggml_tensor *gt, int64_t ne0, int64_t ne1);

/* Per-batch upload registry — tensors whose VALUES change per minibatch.
 * See the comment block in tensor_record.sn.c for the full rationale. */
#define PB_KIND_ADJ       1
#define PB_KIND_POOL      2
#define PB_KIND_ATT_MASK  3  /* attention mask derived from adj_buf */

#define PB_MODE_SUM            0
#define PB_MODE_SUM_NORMALIZED 1
#define PB_MODE_MEAN           2

#define MAX_PER_BATCH_TENSORS 32

extern int g_pb_pool_idx[MAX_PER_BATCH_TENSORS];
extern int g_pb_kind[MAX_PER_BATCH_TENSORS];
extern int g_pb_mode[MAX_PER_BATCH_TENSORS];
extern int g_pb_count;

void track_per_batch(int pool_idx, int kind, int mode);
int  parse_agg_mode(const char *mode);

/* ======================================================================
 * Opt / training state (shared between tensor_record.sn.c's begin/end
 * and tensor_train.sn.c's per-epoch driver)
 * ====================================================================== */

extern int g_pool_count_before_record;

extern ggml_opt_context_t    g_opt_ctx;
extern ggml_backend_sched_t  g_opt_sched;
extern ggml_backend_buffer_t g_opt_param_buf;
extern struct ggml_tensor   *g_opt_loss_tensor;
extern struct ggml_tensor   *g_opt_features_tensor;
extern struct ggml_tensor   *g_opt_labels_tensor;
extern struct ggml_tensor   *g_opt_weights_tensor;
extern struct ggml_tensor   *g_opt_old_log_probs_tensor;
/* Phase 4: per-batch input tensors for the VF loss path. Both are
 * always non-NULL when sn_graph_train_epoch_ppo is the active entry
 * point (the strategy allocates (1,1) placeholders when valueCoeff=0)
 * and always NULL when sn_graph_train_epoch (weighted CE) is active.
 * The per-batch upload loop gates on `g_opt_value_targets_tensor !=
 * NULL` the same way the old-log-probs upload does. */
extern struct ggml_tensor   *g_opt_value_targets_tensor;
extern struct ggml_tensor   *g_opt_old_values_tensor;

extern struct ggml_opt_optimizer_params g_opt_params;
struct ggml_opt_optimizer_params sn_get_opt_params(void *userdata);

/* ======================================================================
 * Training metric callback (defined in tensor_train.sn.c)
 * ====================================================================== */

extern void *g_train_metric_cb;
void *sn_tensor_closure_copy(void *closure);

/* ======================================================================
 * Cross-TU public sn_* functions called from other .sn.c files.
 *
 * These must be forward-declared in the header — without a prototype,
 * C assumes implicit int-return, which silently truncates an RtTensor*
 * to 32 bits on LP64 and crashes at the first field access. (We hit
 * exactly this bug when splitting the original monolith.)
 *
 * Any time a new cross-TU call is added, add the prototype here too.
 * ====================================================================== */

RtTensor *sn_tensor_zeros(long long rows, long long cols);
RtTensor *sn_tensor_from_doubles(SnArray *data, long long rows, long long cols);
RtTensor *sn_graph_input_data(SnArray *data, long long rows, long long cols);

/* ======================================================================
 * Shared numerical constants
 * ====================================================================== */

/* eps floor used inside log(softmax+eps) in both weighted CE and PPO
 * clipped losses — bounds forward AND backward simultaneously so
 * REINFORCE-with-negative-weights doesn't NaN. ≈ exp(-20). See
 * docs/issues/ppo-clipped-objective.md and the comment on
 * sn_tensor_weighted_cross_entropy. */
#define LOG_SM_EPS (2.0e-9f)

#endif /* SN_TENSOR_INTERNAL_H */
