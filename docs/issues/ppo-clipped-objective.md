# PPO-clipped surrogate objective

## Status: PROPOSED — not implemented; tracked as the proper long-term fix

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
arbitrarily large or small — once `r` leaves `[1−ε, 1+ε]`, the
clipped term takes over and gradient w.r.t. `θ` becomes zero. The
loss is bounded above (by the unclipped term), and the policy can
only move ε away from `π_old` per training call. Multi-epoch training
on the same batch is then safe; PPO's typical regime is 4-10 epochs.

In this formulation:
- **Positive `Aᵢ` (good action):** the optimizer maximizes `r` toward
  `1+ε` and stops. Probability of the chosen action increases.
- **Negative `Aᵢ` (bad action):** the optimizer minimizes `r` toward
  `1−ε` and stops. Probability of the chosen action decreases. **The
  loss is bounded below** because the gradient for the negative branch
  vanishes once `r ≤ 1−ε`, regardless of how negative `A` is.

That's the property the current weighted-CE loss lacks.

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
- `ggml_get_rows` for `gather(log_pi, actions)`. Available in ggml.
- `ggml_sub`, `ggml_exp`, `ggml_mul`, `ggml_sum`, `ggml_scale` —
  standard, all have backward.
- `clamp(ratio, 1-eps, 1+eps)` — implement the same way as the
  log_softmax floor: two `relu` hinges, both differentiable. Avoid
  `ggml_clamp` directly because its backward isn't implemented in the
  current ggml fork (see `weighted_cross_entropy` for the workaround).
- `min(a, b)` element-wise — verify ggml has it with backward; if not,
  implement as `0.5 * (a + b - |a - b|)` with `ggml_abs`.

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

## Tradeoffs

| | Current weighted CE + clamp | PPO-clipped surrogate |
|---|---|---|
| Bounded loss with negative weights | ✓ (via `relu` clamp at `log_softmax ≥ -20`) | ✓ (intrinsic to the surrogate) |
| Mathematically clean policy gradient | ✗ (loss diverges past the clamp; converges to a clamp-boundary local min) | ✓ (well-defined trust region per call) |
| Multi-epoch training on the same batch | ⚠️ Safe but doesn't improve much past 1-2 epochs (clamp neutralizes the gradient) | ✓ Safe and the standard practice (4-10 epochs) |
| Caller complexity | Existing API, no extra inputs | New input (`oldLogProbs`), forward pass before training, action indices |
| ggml ops needed | Already implemented | A few more ops (gather, clamp via relu, min) — likely all available |
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
