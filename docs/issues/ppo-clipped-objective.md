# PPO-clipped surrogate objective

## Status: IMPLEMENTED

Landed as a full three-phase change:

- **Loss op** — `sn_tensor_ppo_clipped_loss(logits, oldLogProbs, actionsOneHot, advantages, epsilon)` in `src/tensor.sn.c` with both record and direct mode paths. Declared in `src/tensor.sn`.
- **Training path** — `Gnn.trainPpo(graphs, actions, advantages, optimizer, epochs, batchSize, epsilon, seed)` in `src/gnn.sn`. Computes `oldLogProbs` via a pre-train non-record forward pass, then delegates to the same private `Gnn.trainCore` helper that the weighted-CE `Gnn.train` uses — no duplicated code path. The per-epoch driver `sn_graph_train_epoch_ppo` is a thin wrapper over the existing `sn_graph_train_epoch_impl`; both loss kinds share the host-buffer assembly, block-diagonal adjacency, pool-matrix and attention-mask uploads, and the param read-back loop.
- **Tests** — `tests/test_ppo_clipped_loss.sn` (direct-mode math), `tests/test_ppo_clipped_loss_recorded.sn` (record-mode graph path), `tests/test_ppo_train.sn` (end-to-end training integration with mixed-sign advantages asserting boundedness, policy movement, and JSD-measurable distribution shift).

### Implementation notes that differ from the original proposal below

1. **Action shape resolution.** The API sketch below says `actions: Tensor (1, batchRows)` holding integer class indices; the implementation sketch then describes a one-hot gather via `ggml_mul + ggml_sum_rows`. In a static recorded graph the C op cannot convert dynamic integer indices into a one-hot at record time — the topology is baked once at `sn_graph_begin`. The implementation accepts `actionsOneHot: Tensor (numClasses, batchRows)` directly. This lets callers reuse their existing supervised labels buffer unchanged when opting into PPO, and matches the doc's own one-hot-gather implementation sketch.

2. **`ggml_scale_bias` instead of input-scalar + `ggml_add1`.** The sketch describes the `log_softmax` eps floor using the same scalar-input pattern as `sn_tensor_weighted_cross_entropy`. The implementation uses `ggml_scale_bias(softmax, 1.0, LOG_SM_EPS)` — which computes `1*softmax + LOG_SM_EPS` — because it needs no extra input tensor in `g_param_ctx`, collapses two ops into one, and has a verified backward (the backward of `GGML_OP_SCALE` reads only the `s` parameter from `op_params[0]`, so the bias drops out of the gradient correctly). The same shortcut is used for the two clip bounds (`ratio − (1+ε)` and `(1−ε) − ratio_hi`).

3. **Scoped to the policy term only, as proposed.** No value-function loss (`L^VF`), no entropy bonus (`S[π_θ]`). Consumers that need a critic head or exploration pressure layer those on top of the loss op themselves.

4. **`Gnn.trainPpo` vs extending `Gnn.train`.** The decision went to a separate method to avoid risking regressions in the well-tested supervised path. Existing `Gnn.train()` is unchanged to the caller; internally it now forwards to `Gnn.trainCore(..., "weighted_ce", {}, 0.0)` and `Gnn.trainPpo` forwards to `Gnn.trainCore(..., "ppo", oldLogProbsHost, epsilon)`. `trainCore` branches in exactly two places — the loss op construction and the `sn_graph_train_epoch_* ` call — and is otherwise identical for both loss kinds.

The rest of this document is the original design note, retained verbatim for historical context.

---

## Status (pre-implementation): PROPOSED — not implemented; tracked as the proper long-term fix

## TL;DR

`sn_tensor_weighted_cross_entropy` is currently the package's only loss
for policy-gradient training. With **negative per-sample weights**
(REINFORCE with mean-centred normalized advantages), it is unbounded
below: the optimizer can drive `log_softmax(chosen_action)` toward
`-∞`, eventually producing `NaN`. The package ships a numerical
safety net (a `relu`-based clamp at `log_softmax ≥ -20`, see
`sn_tensor_weighted_cross_entropy` in `src/tensor.sn.c`), but the
underlying objective is still mathematically wrong: it's "minimize the
unbounded weighted CE" rather than "take a bounded, trust-region-style
policy gradient step."

A proper long-term fix is to add a second loss op,
`sn_tensor_ppo_clipped_loss`, implementing the PPO-clipped surrogate
objective from Schulman et al. 2017
([arxiv:1707.06347](https://arxiv.org/abs/1707.06347)).
This document specifies what that op would look like, what it requires
from callers, and when it should land.

## Why the current setup is wrong

REINFORCE's policy-gradient identity is

> ∇ J(θ) = 𝔼[ A · ∇ log π_θ(a|s) ]

To express this as a loss that ggml can backprop through, the package
uses

> loss(θ) = −1/N · Σᵢ wᵢ · log π_θ(aᵢ|sᵢ)        (with wᵢ = Aᵢ)

For positive `wᵢ` and one-hot labels, the gradient of this loss
matches the policy-gradient identity (up to sign), and the loss is
bounded below at `0`. But for **negative `wᵢ`** the loss has terms

> contribᵢ = −1/N · wᵢ · log_softmax(correct_classᵢ)
>          = −1/N · (negative) · (≤ 0)
>          = ≤ 0

So a negative-advantage sample contributes a *negative* term to the
loss. The optimizer minimizes the loss → makes it more negative →
drives `log_softmax(correct_classᵢ)` toward `−∞`. There is no lower
bound. Once `log_softmax` underflows to `−∞`, the multiplication by
the zero entries of the one-hot label produces `NaN` (`−∞ · 0 = NaN`),
and the gradient poisons every parameter via the AdamW step.

This was first observed in skynet's crusher trainer with `epochs: 50`
on multi-node graphs. The smoking-gun loss progression was

```
−0.636, −1.134, −1.543, −1.942, −2.331, ..., −7.870, −nan, −nan, ...
```

— a clean monotonic divergence followed by overflow within ~18 epochs.

The math is right for *one* gradient step. It is wrong for many
gradient steps on the same loss because the loss is not a static
target — it represents an *operating point* of the policy, and
following its gradient blindly walks the policy off a cliff.

## What PPO does

PPO turns the unbounded REINFORCE objective into a **bounded surrogate**
that's safe to optimize for many epochs. Two ingredients:

1. **A frozen "old" policy snapshot.** Call its log-probabilities
   `log π_old(a|s)`. These are computed once at the start of training
   on the current batch and *not* differentiated through.
2. **A clipped probability ratio surrogate.** Define
   `r = π(a|s) / π_old(a|s) = exp(log π − log π_old)`. The PPO loss is
   then
   ```
   L = −1/N · Σᵢ min( rᵢ · Aᵢ ,  clip(rᵢ, 1−ε, 1+ε) · Aᵢ )
   ```
   with `ε ≈ 0.2`.

The `min` over the unclipped and clipped surrogates means the
optimizer cannot make the loss arbitrarily negative by pushing `r`
arbitrarily large or small: the loss is **bounded below**. The
gradient-vanishing is asymmetric, though — it only kicks in when `r`
has moved in the direction favored by the sign of `A`:

- `A > 0` and `r > 1+ε`: clipped branch wins, gradient is zero
  (further increasing `r` wouldn't help).
- `A < 0` and `r < 1−ε`: clipped branch wins, gradient is zero
  (further decreasing `r` wouldn't help).
- `A > 0` and `r < 1−ε`: unclipped branch wins, gradient is nonzero
  and pulls `r` back up toward 1.
- `A < 0` and `r > 1+ε`: unclipped branch wins, gradient is nonzero
  and pulls `r` back down toward 1.

So PPO-clipped is not a strict trust region: it zeroes gradient only
after the policy has already moved "enough" in the favored direction,
but still lets gradient flow to pull `r` back when an earlier epoch
drifted it the wrong way. This is a soft disincentive, not a hard
constraint, and production implementations layer a KL-divergence
early-stopping guard on top (e.g., abort the epoch if
`KL(π_old || π_θ) > 1.5 · target_kl`). Multi-epoch training on the
same batch is then safe in practice; PPO's typical regime is 3-10
epochs (Schulman et al. use K=3 for Atari and K=10 for MuJoCo;
stable-baselines3 defaults to K=10).

In this formulation:
- **Positive `Aᵢ` (good action):** the optimizer maximizes `r` toward
  `1+ε` and stops. Probability of the chosen action increases.
- **Negative `Aᵢ` (bad action):** the optimizer minimizes `r` toward
  `1−ε` and stops. Probability of the chosen action decreases. **The
  loss is bounded below** because the gradient for the negative branch
  vanishes once `r ≤ 1−ε`, regardless of how negative `A` is.

That's the property the current weighted-CE loss lacks.

### Scope: policy term only

The full PPO objective in Schulman et al. is

> L(θ) = L^CLIP(θ) − c₁ · L^VF(θ) + c₂ · S[π_θ](s)

where `L^VF` is a value-function MSE loss and `S[π_θ]` is an entropy
bonus (typically `c₁ = 1.0`, `c₂ = 0.01`). The op proposed here
implements **only the policy term `L^CLIP`**. A GNN policy doing
on-policy RL will usually want an entropy bonus for exploration and
(if it has a critic head) a value loss, but those are the caller's
responsibility to add to the training loss — the package just
exposes the clipped policy surrogate as a composable primitive.

## Proposed API

```sindarin
# In src/tensor.sn:
native fn sn_tensor_ppo_clipped_loss(
  logits: Tensor,         # (numClasses, batchRows) — current π_θ logits
  oldLogProbs: Tensor,    # (1,          batchRows) — log π_old(aᵢ|sᵢ), DETACHED
  actions: Tensor,        # (1,          batchRows) — chosen action indices
  advantages: Tensor,     # (1,          batchRows) — Aᵢ
  epsilon: double         # clip range, typically 0.2
): Tensor                 # scalar loss
```

Caller flow inside `Gnn.train()`:

```sindarin
# 1. Run a forward pass on the FROZEN params (no record mode) to get
#    π_old(aᵢ|sᵢ) for every sample, take log, store on host.
var oldLogProbsHost: double[] = ...  # one log-prob per sample

# 2. Enter record mode and build the PPO loss in the static graph.
sn_graph_begin()
... # register PARAMs, build forward pass
var loss: Tensor = sn_tensor_ppo_clipped_loss(
    logits, oldLogProbsInput, actionsInput, advantagesInput, 0.2)

# 3. Train as usual. The clipped surrogate is bounded so multi-epoch
#    training on the same batch is safe.
for epoch in epochs:
    sn_graph_train_epoch(...)
sn_graph_end()
```

## Implementation sketch (`src/tensor.sn.c`)

Record mode (the differentiable path) builds:

```
log_pi          = log_softmax(logits)                    # (numClasses, batchRows)
log_pi_a        = gather(log_pi, actions)                # (1, batchRows)  — log π(aᵢ|sᵢ)
log_ratio       = log_pi_a - oldLogProbs                 # (1, batchRows)
ratio           = exp(log_ratio)
surrogate1      = ratio * advantages
ratio_clipped   = clamp(ratio, 1-eps, 1+eps)
surrogate2      = ratio_clipped * advantages
per_sample      = min(surrogate1, surrogate2)
loss            = -1/N * sum(per_sample)
```

ggml ops needed (must all have working backward):
- `log_softmax`: build via `ggml_soft_max → ggml_log` with the same
  `relu`-based numerical floor as the current `weighted_cross_entropy`.
- `gather(log_pi, actions)` — **not** a direct `ggml_get_rows` call.
  `ggml_get_rows(a, b)` on an `a` of shape `(numClasses, batchRows)`
  selects along the outer (batch) dimension and copies the full
  `numClasses` inner dimension, which is the wrong axis: we want one
  scalar per batch row, indexed by the action class. The clean
  implementation is the same trick the existing
  `weighted_cross_entropy` uses: build a one-hot mask of the actions
  and compute `sum_over_classes(log_pi * one_hot_actions)`, which
  reduces to the per-sample `log π(aᵢ|sᵢ)` with a single
  `ggml_mul + ggml_sum_rows`. Both ops already have working backward
  in the fork.
- `ggml_sub`, `ggml_exp`, `ggml_mul`, `ggml_sum`, `ggml_scale` —
  standard, all have backward in the fork.
- `clamp(ratio, 1-eps, 1+eps)` — implement the same way as the
  log_softmax floor: two `relu` hinges, both differentiable. Avoid
  `ggml_clamp` directly because its backward isn't implemented in the
  current ggml fork (see `weighted_cross_entropy` for the workaround).
- `min(a, b)` element-wise — verified: the fork has no elementwise
  `ggml_min` / `ggml_max`, so use the `0.5 * (a + b - |a - b|)`
  identity with `ggml_abs`. `ggml_abs`'s backward is defined via
  `ggml_sgn` (see `src/ggml.c` `GGML_UNARY_OP_ABS` branch), and
  `sgn(0) = 0` gives a clean zero subgradient at the non-smooth point
  — no extra smoothing required.

Direct mode (the test path) is straightforward — same expression in a
one-shot allocating context.

## What callers must provide that they don't today

1. **`oldLogProbs`** — the chosen-action log-probability under the
   *frozen* parameters at the start of the training call. Today's
   `Gnn.train()` doesn't compute this. Adding it requires:
   - One extra forward pass per training call, in non-record mode,
     against the parameters as they entered the call.
   - Per-sample storage of `log π_old(aᵢ|sᵢ)` in a host buffer.
   - A new input tensor in the static graph for the upload.
2. **`actions`** — instead of one-hot labels. The current API uses
   one-hot labels passed to weighted CE. PPO needs the integer action
   index (or equivalent index lookup).
3. **A small refactor of `Gnn.train()`** to route REINFORCE-style
   training through the new loss when the caller opts in. The
   signature could grow a `lossKind: str` field on `Optimizer` or live
   as a separate `trainPpo()` method on `Gnn`. The straight
   cross-entropy path stays for supervised classification (skynet's
   `test_crusher_policy.sn`-style case).
4. **Sensible advantages.** PPO assumes the `advantages` tensor is
   already a reasonable per-sample estimate of `Aᵢ`. In production
   PPO this is computed via Generalized Advantage Estimation (GAE,
   λ ≈ 0.95) over a value-function critic, then per-minibatch
   normalized to mean-zero / unit-variance before being passed in.
   Skynet's current crusher trainer uses Monte-Carlo returns
   minus a baseline, which is a coarser estimate but compatible
   with the same op signature — the loss op doesn't care how the
   advantages were produced as long as they arrive normalized.

## Tradeoffs

| | Current weighted CE + clamp | PPO-clipped surrogate |
|---|---|---|
| Bounded loss with negative weights | ✓ (via `relu` clamp at `log_softmax ≥ -20`) | ✓ (intrinsic to the surrogate) |
| Mathematically clean policy gradient | ✗ (loss diverges past the clamp; converges to a clamp-boundary local min) | ✓ (well-defined surrogate per call; soft trust region via clip + optional KL early-stop) |
| Multi-epoch training on the same batch | ⚠️ Safe but doesn't improve much past 1-2 epochs (clamp neutralizes the gradient) | ✓ Safe and the standard practice (3-10 epochs) |
| Caller complexity | Existing API, no extra inputs | New input (`oldLogProbs`), forward pass before training, action indices |
| ggml ops needed | Already implemented | A few more ops (one-hot gather, clamp via relu, min via abs) — all verified available in the fork |
| Implementation cost | Done | Medium (loss op + caller plumbing in `Gnn.train`) |
| Skynet impact | Just works with existing trainer | Trainer needs to wire in `oldLogProbs` and switch loss kind |

## When to implement

**Trigger conditions:**

1. Skynet's REINFORCE training plateaus at a local minimum that's
   visibly worse than what a properly trust-region-bounded REINFORCE
   would reach. (The current safety net gets training to *not crash*
   but produces a sub-optimal policy.)
2. Or: any future consumer wants to do real on-policy RL with a GNN
   in this package and explicitly asks for PPO.
3. Or: the existing `relu`-clamp safety net starts failing on edge
   cases (e.g. extreme advantages, pathological mini-batches).

Until one of those happens, the existing safety net is acceptable.
Skynet should also reduce its `epochs:` config from 50 to 1-4 to match
on-policy RL convention regardless of which loss the package uses
(done in `skynet/config/crusher/model-engine.yaml`).

## Files that would change

- `src/tensor.sn.c` — new `sn_tensor_ppo_clipped_loss` (record + direct
  paths), helpers for the clipped surrogate building blocks.
- `src/tensor.sn` — native fn declaration.
- `src/gnn.sn` — `Gnn.train()` accepts an `oldLogProbs` host buffer
  (or computes it internally via a non-record forward pass over the
  input graphs); routes to the new loss when the optimizer kind asks
  for it.
- `tests/test_ppo_clipped_loss.sn` — new unit test: known-shape ratio
  + advantage cases, sanity-check that the loss is bounded as `r → ∞`
  and `r → 0`.
- `tests/test_ppo_train.sn` — new integration test: REINFORCE-style
  training with mixed-sign advantages, multi-epoch, asserting the
  model converges to a meaningfully different policy than the initial
  one.
- `docs/issues/ppo-clipped-objective.md` — this doc, updated to mark
  RESOLVED with implementation notes.

## References

- Schulman et al., "Proximal Policy Optimization Algorithms," 2017.
  https://arxiv.org/abs/1707.06347 — the original PPO-clipped paper.
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.),
  Chapter 13 — REINFORCE policy-gradient derivation.
- See also: `docs/issues/heterogeneous-graph-batching.md` for the
  resolved structural fix that surfaced this loss issue.
