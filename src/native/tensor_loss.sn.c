/* ==============================================================================
 * tensor_loss.sn.c — loss ops: cross_entropy, weighted_cross_entropy,
 *                    ppo_clipped_loss
 * ==============================================================================
 *
 * All three ops work in both record mode (wired into a recorded forward
 * graph for backward in train_step) and direct mode (one-shot eval for
 * tests that validate the forward value). They share the LOG_SM_EPS floor
 * defined in tensor_internal.h.
 * ============================================================================== */

#include "tensor_internal.h"

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
 *   per_sample_i = -sum_c (label[i,c] * log(softmax(logits[i,c]) + eps))
 *   loss         = sum_i (weight[i] * per_sample_i) / batchRows
 *
 * Convention: scale by 1/batchRows (not 1/sum(weights)). Weights are
 * relative importance, not effective sample counts. With weights all 1.0,
 * this approximately matches ggml_cross_entropy_loss (offset by log(1+eps)
 * which is negligible for eps ~2e-9).
 *
 * Backward-safe softmax floor (the eps trick): we compute
 * `log(softmax(x) + eps)` instead of `log(softmax(x))` to bound BOTH the
 * forward and the backward. eps = LOG_SM_EPS (~2e-9, ≈ exp(-20)) gives:
 *   forward:  log(softmax + eps) ≥ log(eps) ≈ -20
 *   backward: d/dx log(softmax + eps) = 1/(softmax + eps) ≤ 1/eps ≈ 5e8
 *
 * Why this is necessary for REINFORCE policy gradient with negative
 * per-sample weights: the unbounded weighted CE drives log_softmax of
 * the chosen action toward -inf when weight < 0. Even if the FORWARD
 * loss is clamped (an earlier fix tried this with a relu-based clamp),
 * the BACKWARD through `ggml_log(softmax)` computes `1/softmax`, which
 * is +inf when softmax underflows to 0. That infinity propagates to NaN
 * through downstream ops, the AdamW step writes NaN params, and the
 * model dies on the very next forward pass. The eps trick caps both
 * sides of the autograd computation in one move.
 *
 * For consumers that need a mathematically clean unbounded objective
 * with negative weights, see docs/issues/ppo-clipped-objective.md for
 * the longer-term PPO-clipped surrogate.
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

        struct ggml_tensor *softmax = ggml_soft_max(ctx, gl);

        /* eps-shifted log: log(softmax + eps).
         * The eps constant is a 1-element scalar input tensor allocated
         * in g_param_ctx with its host data pre-loaded by
         * sn_graph_input_data. The lazy-init upload loop in
         * sn_graph_train_epoch copies it onto the backend buffer. We
         * cannot use ggml_new_f32 here because g_compute_ctx /
         * g_param_ctx are no_alloc — that helper aborts on no_alloc. */
        double eps_val[1] = { (double)LOG_SM_EPS };
        SnArray *eps_arr = sn_array_new(sizeof(double), 1);
        sn_array_push(eps_arr, &eps_val[0]);
        RtTensor *eps_rt = sn_graph_input_data(eps_arr, 1, 1);
        struct ggml_tensor *eps_t      = rec_tensor(eps_rt);
        struct ggml_tensor *softmax_safe = ggml_add1(ctx, softmax, eps_t);
        struct ggml_tensor *log_sm     = ggml_log(ctx, softmax_safe);
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

    struct ggml_tensor *softmax = ggml_soft_max(ctx, tl);

    /* Same eps-shift as the record path. The micro context is no_alloc,
     * so we feed eps in via the existing scalar-input + track_input
     * mechanism. The static float storage lives for the lifetime of
     * run_graph (which uploads tracked inputs and runs immediately). */
    static float eps_buf[1] = { LOG_SM_EPS };
    struct ggml_tensor *eps_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(eps_t);
    track_input(eps_t, eps_buf);
    struct ggml_tensor *softmax_safe = ggml_add1(ctx, softmax, eps_t);
    struct ggml_tensor *log_sm    = ggml_log(ctx, softmax_safe);
    struct ggml_tensor *per_class = ggml_mul(ctx, log_sm, tlb);
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

/* ===========================================================================
 * PPO-clipped surrogate loss (Schulman et al. 2017, arXiv:1707.06347)
 *
 *   logits        : (numClasses, batchRows) — current policy π_θ logits
 *   oldLogProbs   : (1,          batchRows) — log π_old(a_i|s_i), DETACHED
 *                                              from the recorded graph; the
 *                                              caller must have computed these
 *                                              with a pre-train forward pass.
 *   actionsOneHot : (numClasses, batchRows) — one-hot mask of chosen actions
 *   advantages    : (1,          batchRows) — A_i (already normalized by caller)
 *   epsilon       : clip range (Schulman default ≈ 0.2)
 *
 * Returns a scalar loss tensor (1,1), suitable either as the `outputs` of a
 * recorded training graph or as a one-shot direct-mode eval target.
 *
 * Math (both record and direct paths build the same expression):
 *
 *   log_pi       = log(softmax(logits) + LOG_SM_EPS)     # bounded log_softmax
 *   log_pi_a     = sum_classes(log_pi * actionsOneHot)   # one-hot gather
 *   ratio        = exp(log_pi_a - oldLogProbs)
 *   surrogate1   = ratio * advantages
 *   ratio_clip   = clamp(ratio, 1-eps, 1+eps)            # via two relu hinges
 *   surrogate2   = ratio_clip * advantages
 *   per_sample   = min(surrogate1, surrogate2)           # via 0.5*(a+b-|a-b|)
 *   loss         = -1/N * sum(per_sample)
 *
 * Design choices forced by the ggml fork:
 *
 * - The eps floor on log_softmax is fused into a single ggml_scale_bias
 *   (`1*softmax + LOG_SM_EPS`) instead of the input-scalar + ggml_add1
 *   pattern that sn_tensor_weighted_cross_entropy uses. No new input
 *   tensor is needed in g_param_ctx. Both formulations are mathematically
 *   equivalent; scale_bias keeps the param context small and the
 *   expression flat. The backward of GGML_OP_SCALE reads only the `s`
 *   parameter and ignores `b`, so the constant bias drops out of the
 *   gradient correctly (d/da (s*a + b) = s).
 *
 * - The clamp is built from two ggml_relu hinges instead of ggml_clamp
 *   because GGML_OP_CLAMP has no backward case in the fork (forward only
 *   at ggml.c GGML_OP_CLAMP). The two-hinge equivalence
 *     clamp(x, lo, hi) = x - relu(x - hi) + relu(lo - (x - relu(x - hi)))
 *   routes through GGML_UNARY_OP_RELU (backward via ggml_step),
 *   GGML_OP_SCALE (backward at ggml.c:6563), and GGML_OP_SUB/ADD —
 *   every node has a working backward.
 *
 * - The elementwise min is built from ggml_abs because the fork has no
 *   elementwise ggml_min/ggml_max. The identity
 *     min(a, b) = 0.5 * (a + b - |a - b|)
 *   is smooth except at a == b, where ggml_abs's backward uses
 *   GGML_UNARY_OP_SGN with sgn(0)=0 — that gives a clean zero
 *   subgradient at the crease, no smoothing required.
 *
 * - `actions` is accepted as a (numClasses, batchRows) one-hot mask rather
 *   than a (1, batchRows) integer-index tensor as the doc's prose API
 *   first suggests. In a static recorded graph the C op cannot convert
 *   dynamic integer indices into a one-hot at record time — the topology
 *   is baked once at sn_graph_begin. Accepting the one-hot directly lets
 *   callers reuse their existing labels buffer and matches the doc's own
 *   implementation sketch ("build a one-hot mask of the actions and
 *   compute sum_over_classes(log_pi * one_hot_actions)"). See
 *   docs/issues/ppo-clipped-objective.md for the full rationale.
 *
 * Why this op exists at all — and why it's SEPARATE from
 * sn_tensor_weighted_cross_entropy rather than replacing it: the weighted
 * CE loss is unbounded below when per-sample weights can be negative (the
 * REINFORCE-with-advantages regime), and even with the eps-floor safety net
 * it converges to a clamp-boundary local minimum rather than a true
 * policy-gradient update. The PPO clipped surrogate is bounded below by
 * construction (the `min` kills the unclipped branch once the policy has
 * moved enough in the favored direction), so multi-epoch training on the
 * same batch is mathematically sound. See
 * docs/issues/ppo-clipped-objective.md § "Why the current setup is wrong".
 * =========================================================================== */

/* Phase 4 — Value-function loss extension.
 *
 * The Phase 4.2 extension folds the PPO value-function term into this
 * same loss op rather than adding a second native entry point. The
 * rationale: the critic shares the upstream trunk with the actor
 * (same message-passing layers, same mean-pool embedding), so the
 * backward over the recorded graph wants a single scalar loss that
 * backprops through both heads. Splitting the policy and value loss
 * into two ops and adding their scalars at the Sindarin level would
 * force the trainer to stitch two backward passes over the same
 * recorded graph — ggml_opt only understands one loss tensor.
 *
 * The VF contribution is gated on value_coeff > 0.0: when the caller
 * passes valueCoeff=0 the VF sub-graph is not built and the policy-
 * only behaviour is bit-identical to pre-Phase-4. Callers that don't
 * want VF loss pass (1, 1) zero placeholder tensors for
 * value_estimates / value_targets / old_values; the placeholders
 * pass the unwrap() checks and the gate skips the rest.
 *
 * VF clipping follows the PPO2 form used by SB3 / CleanRL:
 *     V_clip = old_values + clamp(V_theta - old_values, -eps_v, +eps_v)
 *     L_VF_CLIP = max((V_theta - V_target)^2, (V_clip - V_target)^2)
 * i.e. the value loss uses the element-wise maximum of the unclipped
 * squared error and a squared error against a version of V_theta
 * rewound to within ±eps_v of the pre-update value. This is the
 * pessimistic form — if V has moved "in the direction the gradient
 * wants", the clipped path keeps the loss as large as the unclipped
 * one; if V has moved "past" the clip window, the clip freezes the
 * effective prediction and the squared error plateaus.
 *
 * NOTE on the VF clipping default: Engstrom et al. 2020 ("Implementation
 * Matters") found no benefit from VF clipping on their benchmarks, and
 * Andrychowicz et al. 2021 ("What Matters in On-Policy RL") reports
 * that it often *hurts* performance. SB3 defaults clip_range_vf to
 * None (off) for this reason; this package ships the same default —
 * PpoClipped.create passes vfClipRange=0.0 by default, which takes
 * the unclipped branch below. The code path is here for parity with
 * PPO2 / CleanRL and for consumers that want to reproduce those
 * defaults explicitly.
 */
RtTensor *sn_tensor_ppo_clipped_loss(RtTensor *logits_rt,
                                     RtTensor *old_log_probs_rt,
                                     RtTensor *actions_one_hot_rt,
                                     RtTensor *advantages_rt,
                                     RtTensor *value_estimates_rt,
                                     RtTensor *value_targets_rt,
                                     RtTensor *old_values_rt,
                                     double    epsilon,
                                     double    entropy_coeff,
                                     double    value_coeff,
                                     double    vf_clip_range)
{
    TPool *pl   = unwrap(logits_rt);
    TPool *polp = unwrap(old_log_probs_rt);
    TPool *poh  = unwrap(actions_one_hot_rt);
    TPool *padv = unwrap(advantages_rt);
    /* Value-head tensors are always non-NULL — the caller passes (1,1)
     * placeholders when valueCoeff==0.0. The gate below avoids reading
     * from them in the loss graph, so the placeholder data is never
     * observed. */
    TPool *pve  = unwrap(value_estimates_rt);
    TPool *pvt  = unwrap(value_targets_rt);
    TPool *pov  = unwrap(old_values_rt);
    (void)pve; (void)pvt; (void)pov;

    int64_t numClasses = pl->ne[0];
    int64_t batchRows  = pl->ne[1];
    const float inv_neg_n = -1.0f / (float)(batchRows > 0 ? batchRows : 1);
    const float inv_n     =  1.0f / (float)(batchRows > 0 ? batchRows : 1);
    const float eps_hi = 1.0f + (float)epsilon;
    const float eps_lo = 1.0f - (float)epsilon;
    const int   vf_enabled      = (value_coeff > 0.0);
    const int   vf_clip_enabled = (vf_clip_range > 0.0);
    const float vf_scale        = (float)(value_coeff * (double)inv_n);
    const float vf_clip_hi      = (float)  vf_clip_range;
    const float vf_clip_lo      = -(float) vf_clip_range;
    /* Entropy bonus: loss_total = loss_clipped - entropy_coeff * mean(H(p))
     * where H(p) = -Σ_classes p log p (per sample). To minimize loss_total
     * via gradient descent and INCREASE entropy, we add
     *   +entropy_coeff * mean(Σ p log p)   (= -entropy_coeff * mean(H))
     * to the existing clipped loss. mean is over batchRows samples; the
     * inner Σ is over numClasses, computed as ggml_sum over the full
     * (numClasses, batchRows) p_log_p tensor then scaled by 1/batchRows.
     * Set entropy_coeff = 0.0 to disable (matches pre-entropy behaviour).
     * Without this term PPO collapses to a deterministic policy on
     * datasets dominated by one class — see Phase 6 report 2026-04-08. */
    const float entropy_scale = (float)(entropy_coeff / (double)(batchRows > 0 ? batchRows : 1));

    if (g_record_mode) {
        struct ggml_context *ctx = g_record_ctx;
        struct ggml_tensor *gl   = rec_tensor(logits_rt);
        struct ggml_tensor *golp = rec_tensor(old_log_probs_rt);
        struct ggml_tensor *goh  = rec_tensor(actions_one_hot_rt);
        struct ggml_tensor *gadv = rec_tensor(advantages_rt);

        /* bounded log_softmax: log(softmax + LOG_SM_EPS) */
        struct ggml_tensor *softmax      = ggml_soft_max(ctx, gl);
        struct ggml_tensor *softmax_safe = ggml_scale_bias(ctx, softmax, 1.0f, LOG_SM_EPS);
        struct ggml_tensor *log_pi       = ggml_log(ctx, softmax_safe);

        /* log π(a_i|s_i) via one-hot gather: (1, batchRows) */
        struct ggml_tensor *masked       = ggml_mul(ctx, log_pi, goh);
        struct ggml_tensor *log_pi_a     = ggml_sum_rows(ctx, masked);

        /* ratio = exp(log π(a) - log π_old(a)) */
        struct ggml_tensor *log_ratio    = ggml_sub(ctx, log_pi_a, golp);
        struct ggml_tensor *ratio        = ggml_exp(ctx, log_ratio);

        /* surrogate1 = ratio * advantages */
        struct ggml_tensor *surrogate1   = ggml_mul(ctx, ratio, gadv);

        /* ratio_clip = clamp(ratio, 1-eps, 1+eps) via two relu hinges */
        struct ggml_tensor *r_minus_hi   = ggml_scale_bias(ctx, ratio,    1.0f, -eps_hi);
        struct ggml_tensor *over_hi      = ggml_relu(ctx, r_minus_hi);
        struct ggml_tensor *ratio_hi     = ggml_sub(ctx, ratio, over_hi);
        struct ggml_tensor *lo_minus_r   = ggml_scale_bias(ctx, ratio_hi, -1.0f, eps_lo);
        struct ggml_tensor *under_lo     = ggml_relu(ctx, lo_minus_r);
        struct ggml_tensor *ratio_clip   = ggml_add(ctx, ratio_hi, under_lo);

        /* surrogate2 = ratio_clip * advantages */
        struct ggml_tensor *surrogate2   = ggml_mul(ctx, ratio_clip, gadv);

        /* per_sample = min(surrogate1, surrogate2) = 0.5 * (a + b - |a - b|) */
        struct ggml_tensor *sum_s        = ggml_add(ctx, surrogate1, surrogate2);
        struct ggml_tensor *diff_s       = ggml_sub(ctx, surrogate1, surrogate2);
        struct ggml_tensor *abs_diff     = ggml_abs(ctx, diff_s);
        struct ggml_tensor *min_two_x    = ggml_sub(ctx, sum_s, abs_diff);
        struct ggml_tensor *per_sample   = ggml_scale(ctx, min_two_x, 0.5f);

        struct ggml_tensor *summed       = ggml_sum(ctx, per_sample);
        struct ggml_tensor *clipped_loss = ggml_scale(ctx, summed, inv_neg_n);

        /* Entropy bonus term — see comment near function top. softmax and
         * log_pi (= log(softmax + LOG_SM_EPS)) are already in scope. */
        struct ggml_tensor *p_log_p      = ggml_mul(ctx, softmax, log_pi);
        struct ggml_tensor *plp_sum      = ggml_sum(ctx, p_log_p);
        struct ggml_tensor *entropy_term = ggml_scale(ctx, plp_sum, entropy_scale);

        struct ggml_tensor *loss = ggml_add(ctx, clipped_loss, entropy_term);

        /* Phase 4 VF loss term (gated on value_coeff > 0).
         *
         *   diff       = value_estimates - value_targets
         *   L_unclip   = diff * diff                (elementwise)
         *   if vf_clip_range > 0:
         *     v_delta      = value_estimates - old_values
         *     v_delta_clip = clamp(v_delta, -eps_v, +eps_v) via two relu hinges
         *     V_clip       = old_values + v_delta_clip
         *     diff_clip    = V_clip - value_targets
         *     L_clipped    = diff_clip * diff_clip
         *     L_VF         = max(L_unclip, L_clipped) via abs-trick
         *   else:
         *     L_VF         = L_unclip
         *   loss_total    += value_coeff * mean(L_VF)
         *
         * The max of two tensors elementwise uses the same identity as
         * the policy min: max(a, b) = 0.5 * (a + b + |a - b|).
         */
        if (vf_enabled) {
            struct ggml_tensor *gve = rec_tensor(value_estimates_rt);
            struct ggml_tensor *gvt = rec_tensor(value_targets_rt);

            struct ggml_tensor *diff      = ggml_sub(ctx, gve, gvt);
            struct ggml_tensor *l_unclip  = ggml_mul(ctx, diff, diff);

            struct ggml_tensor *l_vf = l_unclip;
            if (vf_clip_enabled) {
                struct ggml_tensor *gov = rec_tensor(old_values_rt);

                struct ggml_tensor *v_delta       = ggml_sub(ctx, gve, gov);
                /* clamp(v_delta, vf_clip_lo, vf_clip_hi) via two relu hinges */
                struct ggml_tensor *vd_minus_hi   = ggml_scale_bias(ctx, v_delta,    1.0f, -vf_clip_hi);
                struct ggml_tensor *over_hi       = ggml_relu(ctx, vd_minus_hi);
                struct ggml_tensor *vd_hi         = ggml_sub(ctx, v_delta, over_hi);
                struct ggml_tensor *lo_minus_vd   = ggml_scale_bias(ctx, vd_hi,     -1.0f, vf_clip_lo);
                struct ggml_tensor *under_lo      = ggml_relu(ctx, lo_minus_vd);
                struct ggml_tensor *vd_clipped    = ggml_add(ctx, vd_hi, under_lo);

                struct ggml_tensor *v_clip        = ggml_add(ctx, gov, vd_clipped);
                struct ggml_tensor *diff_clip     = ggml_sub(ctx, v_clip, gvt);
                struct ggml_tensor *l_clipped     = ggml_mul(ctx, diff_clip, diff_clip);

                /* max(l_unclip, l_clipped) = 0.5 * (a + b + |a - b|) */
                struct ggml_tensor *sum_sq        = ggml_add(ctx, l_unclip, l_clipped);
                struct ggml_tensor *diff_sq       = ggml_sub(ctx, l_unclip, l_clipped);
                struct ggml_tensor *abs_diff_sq   = ggml_abs(ctx, diff_sq);
                struct ggml_tensor *max_two_x     = ggml_add(ctx, sum_sq, abs_diff_sq);
                l_vf                              = ggml_scale(ctx, max_two_x, 0.5f);
            }

            struct ggml_tensor *vf_sum   = ggml_sum(ctx, l_vf);
            struct ggml_tensor *vf_term  = ggml_scale(ctx, vf_sum, vf_scale);
            loss                         = ggml_add(ctx, loss, vf_term);
        }

        ggml_set_name(loss, "ppo_clipped_loss");
        return rec_wrap(loss, 1, 1);
    }

    /* Direct mode: one-shot eval for test_ppo_clipped_loss.sn and for
     * Phase 4.2 test_ppo_vf_loss_direct.sn. */
    struct ggml_context *ctx = micro_ctx_init();

    struct ggml_tensor *tl   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, numClasses, batchRows);
    struct ggml_tensor *tolp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1,          batchRows);
    struct ggml_tensor *toh  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, numClasses, batchRows);
    struct ggml_tensor *tadv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1,          batchRows);
    ggml_set_input(tl);
    ggml_set_input(tolp);
    ggml_set_input(toh);
    ggml_set_input(tadv);
    track_input(tl,   pl->data);
    track_input(tolp, polp->data);
    track_input(toh,  poh->data);
    track_input(tadv, padv->data);

    /* Phase 4 VF inputs — only constructed when enabled so the direct-
     * mode policy-only tests keep seeing the pre-Phase-4 graph shape. */
    struct ggml_tensor *tve = NULL;
    struct ggml_tensor *tvt = NULL;
    struct ggml_tensor *tov = NULL;
    if (vf_enabled) {
        tve = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, batchRows);
        tvt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, batchRows);
        ggml_set_input(tve);
        ggml_set_input(tvt);
        track_input(tve, pve->data);
        track_input(tvt, pvt->data);
        if (vf_clip_enabled) {
            tov = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, batchRows);
            ggml_set_input(tov);
            track_input(tov, pov->data);
        }
    }

    struct ggml_tensor *softmax      = ggml_soft_max(ctx, tl);
    struct ggml_tensor *softmax_safe = ggml_scale_bias(ctx, softmax, 1.0f, LOG_SM_EPS);
    struct ggml_tensor *log_pi       = ggml_log(ctx, softmax_safe);
    struct ggml_tensor *masked       = ggml_mul(ctx, log_pi, toh);
    struct ggml_tensor *log_pi_a     = ggml_sum_rows(ctx, masked);
    struct ggml_tensor *log_ratio    = ggml_sub(ctx, log_pi_a, tolp);
    struct ggml_tensor *ratio        = ggml_exp(ctx, log_ratio);
    struct ggml_tensor *surrogate1   = ggml_mul(ctx, ratio, tadv);

    struct ggml_tensor *r_minus_hi   = ggml_scale_bias(ctx, ratio,    1.0f, -eps_hi);
    struct ggml_tensor *over_hi      = ggml_relu(ctx, r_minus_hi);
    struct ggml_tensor *ratio_hi     = ggml_sub(ctx, ratio, over_hi);
    struct ggml_tensor *lo_minus_r   = ggml_scale_bias(ctx, ratio_hi, -1.0f, eps_lo);
    struct ggml_tensor *under_lo     = ggml_relu(ctx, lo_minus_r);
    struct ggml_tensor *ratio_clip   = ggml_add(ctx, ratio_hi, under_lo);

    struct ggml_tensor *surrogate2   = ggml_mul(ctx, ratio_clip, tadv);

    struct ggml_tensor *sum_s        = ggml_add(ctx, surrogate1, surrogate2);
    struct ggml_tensor *diff_s       = ggml_sub(ctx, surrogate1, surrogate2);
    struct ggml_tensor *abs_diff     = ggml_abs(ctx, diff_s);
    struct ggml_tensor *min_two_x    = ggml_sub(ctx, sum_s, abs_diff);
    struct ggml_tensor *per_sample   = ggml_scale(ctx, min_two_x, 0.5f);

    struct ggml_tensor *summed       = ggml_sum(ctx, per_sample);
    struct ggml_tensor *clipped_loss = ggml_scale(ctx, summed, inv_neg_n);

    /* Entropy bonus term — direct-mode mirror of the record-mode branch. */
    struct ggml_tensor *p_log_p      = ggml_mul(ctx, softmax, log_pi);
    struct ggml_tensor *plp_sum      = ggml_sum(ctx, p_log_p);
    struct ggml_tensor *entropy_term = ggml_scale(ctx, plp_sum, entropy_scale);

    struct ggml_tensor *loss = ggml_add(ctx, clipped_loss, entropy_term);

    /* Phase 4 VF loss term (direct-mode mirror of the record-mode branch
     * above). See comments on that branch for the math derivation. */
    if (vf_enabled) {
        struct ggml_tensor *diff      = ggml_sub(ctx, tve, tvt);
        struct ggml_tensor *l_unclip  = ggml_mul(ctx, diff, diff);

        struct ggml_tensor *l_vf = l_unclip;
        if (vf_clip_enabled) {
            struct ggml_tensor *v_delta       = ggml_sub(ctx, tve, tov);
            struct ggml_tensor *vd_minus_hi   = ggml_scale_bias(ctx, v_delta,    1.0f, -vf_clip_hi);
            struct ggml_tensor *over_hi       = ggml_relu(ctx, vd_minus_hi);
            struct ggml_tensor *vd_hi         = ggml_sub(ctx, v_delta, over_hi);
            struct ggml_tensor *lo_minus_vd   = ggml_scale_bias(ctx, vd_hi,     -1.0f, vf_clip_lo);
            struct ggml_tensor *under_lo      = ggml_relu(ctx, lo_minus_vd);
            struct ggml_tensor *vd_clipped    = ggml_add(ctx, vd_hi, under_lo);

            struct ggml_tensor *v_clip        = ggml_add(ctx, tov, vd_clipped);
            struct ggml_tensor *diff_clip     = ggml_sub(ctx, v_clip, tvt);
            struct ggml_tensor *l_clipped     = ggml_mul(ctx, diff_clip, diff_clip);

            struct ggml_tensor *sum_sq        = ggml_add(ctx, l_unclip, l_clipped);
            struct ggml_tensor *diff_sq       = ggml_sub(ctx, l_unclip, l_clipped);
            struct ggml_tensor *abs_diff_sq   = ggml_abs(ctx, diff_sq);
            struct ggml_tensor *max_two_x     = ggml_add(ctx, sum_sq, abs_diff_sq);
            l_vf                              = ggml_scale(ctx, max_two_x, 0.5f);
        }

        struct ggml_tensor *vf_sum  = ggml_sum(ctx, l_vf);
        struct ggml_tensor *vf_term = ggml_scale(ctx, vf_sum, vf_scale);
        loss                        = ggml_add(ctx, loss, vf_term);
    }

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
