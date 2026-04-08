# Action Collapse Policy Gradient (ACPG)

## Status: RESEARCH NOTE — not implemented

Background reading for a future tensor-side feature. Captures the
academic state-of-the-art fix for deterministic policy collapse on
imbalanced action priors. The `sindarin-pkg-tensor` + `skynet`
stack currently uses a simpler fix (small-scale actor head init
+ literature-default PPO hyperparameters) that worked for the
crusher domain — see the "When to use ACPG" section for the
conditions under which we'd switch.

## TL;DR

The 2025 paper *"Imitate Optimal Policy: Prevail and Induce Action
Collapse in Policy Gradient"* (Ye & Liu, arXiv 2509.02737) observes
that any optimally-trained discrete-action policy network's final
layer always converges to the same geometric structure: a **simplex
equiangular tight frame (ETF)**. The ACPG algorithm short-circuits
the discovery by **initialising the action selection layer as a
fixed ETF and freezing it** — the upstream features are the only
thing learned. This provides a *structural* guarantee that the
policy cannot collapse to a deterministic distribution on imbalanced
data, because the action directions are hardcoded at maximum
pairwise angular separation and cannot drift toward each other.

Tested across REINFORCE, TRPO, PPO, and A3C with reported theoretical
and empirical improvements. The paper frames it as both an
optimisation trick (skip learning a geometry that training always
rediscovers) and a robustness mechanism (collapse becomes
mathematically impossible, not just unlikely).

## The problem ACPG solves

On strongly-imbalanced action priors (e.g. 86% of training samples
have `action = hold` because the environment's natural reward signal
makes hold the dominant choice), a policy network trained with
vanilla PPO will collapse: the softmax output saturates at
`P(dominant_action) = 1.0` to floating-point precision within a few
training rounds, and the entropy gradient at that simplex vertex is
exactly zero:

```
d/dx [softmax(x) * log(softmax(x))] = softmax * (1 - softmax) * (log(softmax) + 1)
```

At the vertex `softmax → 1`, the `(1 - softmax)` factor goes to zero
→ entropy regularisation cannot pull the policy back. Even with a
large `entropy_coeff`, the optimizer has no gradient to follow once
collapse is reached.

This is the failure mode `sindarin-pkg-tensor` + `skynet` hit in
Phase 6 verification (2026-04-08). We resolved it with:

1. `std=0.01` small-scale init for `classifierW2` per Andrychowicz
   et al. 2021 — centres the initial action distribution near
   uniform so the model *starts* in the simplex interior.
2. Literature-default PPO hyperparameters (`lr=0.00025`,
   `batch_size=64`, `entropy_coeff=0.01`, no `ε`-greedy) so
   per-round policy drift is bounded and doesn't overwhelm the
   entropy bonus's ability to keep the policy interior.

Both fixes make collapse *unlikely*. ACPG makes it *impossible*.

## What a simplex ETF is

An equiangular tight frame for `K` classes in a `D`-dimensional
embedding space (with `D ≥ K-1`) is a set of `K` unit vectors where:

- **Equiangular**: the angle between every pair of vectors is
  identical.
- **Tight frame**: the vectors span the embedding space with a
  specific normalisation property.
- **Simplex**: there are exactly `K` vectors, one per class, arranged
  at the vertices of a regular `(K-1)`-simplex.

For three actions (the `skynet` crusher case, `K=3`), a simplex ETF
is three unit vectors at 120° to each other — the maximum possible
angular separation for three vectors. For four actions, it's the
four vertices of a regular tetrahedron in 3D. In higher dimensions
it generalises similarly.

### Mathematical construction

The canonical construction, for `K` classes in `D ≥ K-1` dimensions,
builds a `(K × D)` matrix `M` whose rows are the ETF vertices:

```
M = sqrt(K / (K-1)) * (I_K - (1/K) * 1_K * 1_K^T) * P
```

where:
- `I_K` is the `K × K` identity matrix
- `1_K` is the `K × 1` all-ones vector
- `(1/K) * 1_K * 1_K^T` is a `K × K` matrix of `1/K` values (the
  "centering" projection)
- `P` is a random orthonormal `K × D` matrix (e.g. from the QR
  decomposition of a Gaussian `K × D` matrix) — this embeds the
  `K`-dimensional simplex into `D` dimensions at an arbitrary
  orientation.

The rows of `M` then satisfy:
- `||M_i|| = 1` for each row (unit length)
- `M_i · M_j = -1/(K-1)` for `i ≠ j` (equal pairwise inner product,
  which corresponds to an angle of `arccos(-1/(K-1))`)
- For `K=3`: inner product = `-0.5`, angle = `120°`
- For `K=4`: inner product = `-1/3`, angle ≈ `109.47°` (tetrahedral)
- For `K → ∞`: inner product → `0`, angle → `90°` (orthogonal limit)

Any state embedding `e ∈ R^D` produces logits `logits = M e`, which
measure how aligned the state is with each action direction. The
softmax over these logits is then the policy distribution.

## The ACPG algorithm

Vanilla PPO lets the action selection layer weights `W ∈ R^{K × D}`
be trained along with everything else. The weights drift during
training; on imbalanced data they drift toward a configuration where
one row dominates and the others are nearly zero, producing
`P(dominant) ≈ 1.0` collapse.

ACPG replaces this with:

1. **Construct `W` once at `Gnn.create` time** as a simplex ETF using
   the formula above. The random orientation `P` is drawn once from
   the seed.
2. **Freeze `W`** — mark it as non-trainable. Gradients flow through
   it (needed for backprop into the upstream features) but are
   never applied to update `W` itself.
3. **Train everything upstream**: the GNN layers, the first
   classifier layer `classifierW1`, and the `classifierB1` bias.
   `classifierB2` may also stay trainable or be zeroed — the paper
   tests both and finds minimal difference.
4. **Forward pass is identical to vanilla**: `logits = classifierW1(features) → ReLU → W * → + classifierB2 → softmax`.
5. **Loss is identical to vanilla PPO**: no change to the loss function or gradient computation.

The constraint is enforced purely at optimizer step time — the
parameter list passed to the optimizer excludes `W`. Everything else
about training stays the same.

## Why it prevents collapse (structurally)

With a trained `W`, collapse happens because the optimizer can drive
one row of `W` to large magnitude and the others to small, producing
`softmax` saturation at the large row's class. The gradient
supports this drift because the data rewards it.

With a frozen ETF `W`:

- The three (or `K`) row directions are **permanently at 120° (or
  the ETF angle) to each other**. They cannot drift toward each
  other because they cannot drift at all.
- The state embedding `e` can bias toward any action by moving
  closer to that action's direction — i.e. `e` can align with
  `M_hold` more than with `M_speed_up`. This increases
  `P(hold)`.
- But the softmax value `P(hold)` is bounded above by a function
  of the **embedding norm and the ETF angle**. For a fixed
  embedding norm, the maximum achievable `P(hold)` is strictly
  less than 1.0 because the other actions always receive some
  non-zero projection from the "centering" of the ETF.
- The only way to push `P(hold) → 1.0` is to let `||e|| → ∞`,
  which weight decay, normalisation layers, and the loss function
  collectively prevent.

The result is a **hard floor on policy entropy** that depends on
the maximum achievable embedding norm under training regularisation.
The floor is typically high enough that the entropy gradient stays
meaningfully non-zero throughout training, so PPO's optimizer can
always move the policy in response to data.

Put differently: ACPG converts "collapse is a saddle point the
optimizer falls into" into "collapse is unreachable by construction."

## Connection to Neural Collapse (classification-side prior art)

ACPG's insight isn't random — it extends the *Neural Collapse*
phenomenon observed in well-trained classification DNNs (Papyan,
Han, Donoho 2020, *"Prevalence of neural collapse during the terminal
phase of deep learning training"*).

Papyan et al. observed that when a classifier is trained to
convergence on a balanced dataset:

1. **Variability collapse**: last-layer features for each class
   collapse to their class mean (within-class variance → 0).
2. **Simplex ETF convergence**: the class means themselves converge
   to a simplex ETF geometry.
3. **Self-duality**: the classifier weight matrix also converges
   to the same ETF (up to scale and rotation).
4. **Nearest class-mean decision**: classification reduces to
   nearest-class-mean in the ETF geometry.

This is empirical — it happens at the end of training without being
designed for. The classification-side Neural Collapse community then
asked: why not start training *with* the ETF structure? The "Guiding
Neural Collapse" work (arXiv 2411.01248) and related lines explore
this for classification. ACPG is the policy-gradient version of
the same idea.

The philosophical message across both families is the same: **deep
learning problems often have a geometric endpoint that training
always rediscovers, and you can save time plus improve stability by
hardcoding it**.

## Comparison with what `sindarin-pkg-tensor` currently does

| Approach | Where it lives | What it guarantees | Cost |
|---|---|---|---|
| **Kaiming init (pre-Phase 6.3)** | Every weight, including `classifierW2` | Nothing — collapse was possible and happened | Cheap but broken for imbalanced priors |
| **Small-scale init (Phase 6.3, shipped)** | `classifierW2` only, `std=0.01` via `sn_tensor_init_small_scale_seeded` | Model *starts* near uniform; entropy bonus can push back on early drift | ~20 lines of C + binding |
| **Entropy bonus in loss (Phase 6.2, shipped)** | `sn_tensor_ppo_clipped_loss` | Adds `+coeff × mean(Σ p log p)` to loss, penalises peaked distributions *away from* the vertex | ~10 lines of ggml ops |
| **Literature PPO hyperparameters (Phase 6.4, shipped)** | skynet trainer.yaml | Per-round policy drift is small enough to not overshoot | Config-only |
| **ACPG (this doc)** | `Gnn.create` + `Gnn.parameters` + optimizer step path | Collapse is **structurally impossible** regardless of init, lr, batch size, or training duration | ~50 lines of C + architecture change |

The shipped stack (first three rows combined) keeps the policy in
the simplex interior via a combination of "start there" (small-scale
init), "push back if it drifts" (entropy bonus), and "don't overshoot"
(lr/batch tuning). It's the **soft-regularisation answer**.

ACPG is the **hard-constraint answer**: don't let the action directions
move at all.

## When we'd switch to ACPG

The current stack works for crusher (Phase 6.4 verification confirms
healthy `policy_entropy` of 0.67-0.96 across 27+ training rounds on
an 86%-hold prior). We would switch to ACPG only if:

1. **A new domain has even more extreme class imbalance** (e.g.
   98%/1%/1% or worse), and the literature defaults aren't enough
   to prevent collapse.
2. **Longer-horizon verification (hours or days)** reveals that
   `policy_entropy` drifts toward zero over time on a stable
   domain. The Phase 6.4 run was only ~5 minutes; a week-long
   production run might show slow drift the short run missed.
3. **Cross-domain generality matters**. ACPG's structural guarantee
   is domain-independent — there's no hyperparameter to tune per
   domain. If `skynet` eventually runs on 5+ domains
   simultaneously, the operations burden of per-domain lr/batch
   tuning might make ACPG's tune-free fix more attractive than
   per-domain hyperparameter sweeps.
4. **Speed of convergence becomes a blocker**. The paper reports
   ACPG trains faster than vanilla because the action-layer
   geometry is pre-solved — the model only has to learn the
   state features. On slow domains this could matter.

## When we'd NOT switch

- **Crusher is working.** The shipped fix is delivering healthy
  `policy_entropy`, real promotion deltas, and state-dependent
  action distributions. Replacing it with a more invasive change
  would be churn without observable benefit.
- **Debuggability matters.** With a learned `classifierW2`, you
  can read the weight matrix to see "what does the policy think
  each action looks like in feature space". With a fixed ETF,
  `W` is meaningless — it's a random orthogonal projection.
  Interpreting policies requires tracing through the state
  embeddings instead.
- **Close-to-vanilla-PPO compatibility.** The shipped stack is a
  nearly vanilla PPO with one additional init call and standard
  hyperparameters. External PPO literature (benchmarks, bug
  reports, tuning guides, newly-published tricks) transfers
  cleanly. ACPG is a non-standard extension that would diverge
  the codebase from the mainline PPO ecosystem.
- **Action spaces that aren't semantically symmetric.** ACPG
  assumes the `K` actions should be maximally separated in
  embedding space, which is only correct if the actions are
  symmetric under the problem structure. A domain where one
  action is "do nothing" and others are "intervene with
  increasing magnitude" may have an asymmetric optimal
  geometry that ETF discards.

## Implementation sketch (if we ever add it)

A first-cut ACPG implementation for `sindarin-pkg-tensor` would
need the following pieces:

### C-side (`src/tensor.sn.c`)

1. **`sn_tensor_init_simplex_etf(t: Tensor, seed: long)`** — new
   init function that fills a `(K, D)` tensor with the simplex ETF
   construction described above. Pure CPU numpy-like code, no ggml
   autograd required (this is an init, not a graph op). Roughly
   30 lines of C (QR decomposition of a Gaussian `K × D` matrix
   via Gram-Schmidt, then the ETF formula).

2. **No change to `sn_tensor_ppo_clipped_loss`** — loss function
   stays identical. The freeze is a *parameter list* concern, not
   a *loss* concern.

### Sindarin-side (`src/tensor.sn` + `src/gnn.sn`)

3. **`fn Tensor.initSimplexEtfSeeded(seed: long): void`** — method
   binding to the new C init.

4. **New `Gnn` field: `classifierW2Frozen: bool`** — indicates
   whether `classifierW2` should be excluded from the parameter
   list. Default `false` for backwards compatibility.

5. **`fn Gnn.createEtfWithSeed(config: GnnConfig, seed: long): Gnn`**
   — new static constructor that builds a Gnn with
   `classifierW2` initialised as a simplex ETF and
   `classifierW2Frozen = true`. Mirrors the existing
   `Gnn.createWithSeed` but swaps the init call.

6. **Modify `Gnn.parameters`** to conditionally exclude
   `classifierW2` based on the frozen flag. One `if` branch.

7. **Modify `Gnn.save` / `Gnn.load`** to persist the frozen flag
   alongside the weights so round-tripping preserves the
   ACPG architecture. Currently the parameter list order is
   position-dependent, so we'd need to either (a) serialize the
   flag as a new header field, or (b) always save all params
   including the ETF (then re-mark frozen on load).

### skynet-side

8. **New yaml key: `model.use_etf: false`** in trainer.yaml,
   model-engine.yaml, backtest.yaml — per-domain switch.

9. **`trainer/main.sn`**: when `use_etf` is set, call
   `Gnn.createEtfWithSeed` instead of `Gnn.createWithSeed`.
   Same for `model/main.sn` and `backtest/main.sn`.

### Tests

10. **`tests/test_simplex_etf_init.sn`** — verifies the ETF
    geometry: unit-norm rows, equal pairwise inner products
    matching `-1/(K-1)`, orthogonality of centered rows.

11. **`tests/test_gnn_etf_training.sn`** — end-to-end PPO training
    with ACPG on the crusher-like imbalanced dataset from
    `test_trainer_offline_crusher_ppo.sn`, asserting that
    `policy_entropy` never crosses a collapse floor (say, 0.3)
    across a training run designed to hammer the 86%-hold prior.

Total scope estimate: ~150-200 lines across tensor + skynet,
with most of the risk in the save/load path.

## Related research directions

If we go down this path, adjacent work worth reviewing first:

- **KL regularization to a fixed uniform prior** — mathematically
  equivalent to an entropy bonus (KL to uniform = -H + const), so
  same saddle-point problem. Covered in the earlier literature
  research (see session notes dated 2026-04-08).
- **Probability floor / label smoothing** — clamp `softmax > p_min`
  before computing loss. Conceptually similar to ACPG (both prevent
  reaching the vertex) but applied at the loss function rather
  than the parameter geometry. Simpler but not in standard
  literature.
- **Target KL early stopping** — `stable-baselines3`'s opt-in
  `target_kl` parameter aborts the epoch loop when cumulative KL
  exceeds a threshold, which limits per-round drift. Complementary
  to ACPG, not a replacement.
- **Reward weighting for class imbalance** — Lin et al. 2019,
  Yang et al. 2022 show that scaling rewards inversely by class
  frequency can offset imbalanced priors in RL classification.
  Domain-specific (requires known class distribution in advance).
- **`ReLU + LayerNorm` as an implicit ETF inducer** — a line of
  classification work shows that normalization layers nudge
  features toward ETF geometry naturally, which may partially
  explain why our GNN doesn't collapse *further* than it does.

## References

1. **Ye, T. & Liu, X. (2025).** *Imitate Optimal Policy: Prevail
   and Induce Action Collapse in Policy Gradient.* arXiv
   2509.02737. [Link](https://arxiv.org/abs/2509.02737) — the
   ACPG paper. Core reference for everything in this document.
   Also at [OpenReview ve0q46za2O](https://openreview.net/forum?id=ve0q46za2O).

2. **Papyan, V., Han, X. & Donoho, D. L. (2020).** *Prevalence of
   neural collapse during the terminal phase of deep learning
   training.* PNAS 117 (40): 24652-24663.
   [Link](https://www.pnas.org/doi/pdf/10.1073/pnas.2015509117) —
   the foundational Neural Collapse paper that inspired ACPG.
   Establishes the simplex ETF phenomenon in classification and
   names the four "NC" properties (variability collapse, ETF
   geometry, self-duality, nearest-class-mean).

3. **Yaras, C., Cai, T., Zhu, Z. & Vidal, R. (2024).** *Guiding
   Neural Collapse: Optimising Towards the Nearest Simplex
   Equiangular Tight Frame.* arXiv 2411.01248.
   [Link](https://arxiv.org/abs/2411.01248) — classification-side
   work on using ETF constraints *during* training rather than
   discovering them at convergence. Direct methodological
   ancestor of ACPG.

4. **Zhu, Z. et al. (2021).** *A Geometric Analysis of Neural
   Collapse with Unconstrained Features.* NeurIPS.
   [Link](https://arxiv.org/abs/2105.02375) — theoretical analysis
   showing the ETF geometry is the unique global minimizer of the
   cross-entropy loss under the "unconstrained features" model.
   Relevant to understanding *why* the ETF is always the
   end-state, not just empirically.

5. **Xie, Y. et al. (2023).** *ETF Transformer for Imbalanced
   Semantic Segmentation.* (PMC PMC11548193). [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC11548193/)
   — applies fixed ETF classifier heads to the imbalanced
   segmentation problem. Evidence that the fixed-ETF approach
   generalises beyond vanilla classification to high-imbalance
   settings. Relevant to the skynet domain-imbalance problem.

6. **Yang, Y. et al. (2022).** *Inducing Neural Collapse in
   Imbalanced Learning: Do We Really Need a Learnable Classifier
   at the End of Deep Neural Network?* ICLR. OpenReview A6EmxI3_Xc.
   [Link](https://openreview.net/forum?id=A6EmxI3_Xc) — directly
   addresses the question "can we just skip training the final
   layer?" in a classification context. Answer: yes, and it helps
   on imbalanced data. This is the strongest piece of prior art
   for ACPG's claim that fixing the final layer as an ETF works
   even when the data is imbalanced.

7. **Andrychowicz, M. et al. (2021).** *What Matters in On-Policy
   Reinforcement Learning? A Large-Scale Empirical Study.* ICLR.
   [Link](https://arxiv.org/abs/2006.05990) — the empirical study
   that established `std=0.01` small-scale policy output init as
   the standard practice. Complements ACPG (same problem,
   different axis of attack). Our current shipped fix is based
   on this paper.

8. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov,
   O. (2017).** *Proximal Policy Optimization Algorithms.* arXiv
   1707.06347. [Link](https://arxiv.org/abs/1707.06347) — the
   original PPO paper. Sets the default hyperparameters
   (`lr=3e-4`, `batch=64`, `epochs=10`, `clip=0.2`) that Phase
   6.4 adopted. ACPG is a modification to the PPO training loop;
   start here to understand what the algorithm was originally.

9. **Ahmed, Z., Le Roux, N., Norouzi, M. & Schuurmans, D. (2019).**
   *Understanding the Impact of Entropy on Policy Optimization.*
   ICML. [Link](https://proceedings.mlr.press/v97/ahmed19a/ahmed19a.pdf) —
   the saddle-point geometry of entropy regularisation in
   policy optimisation. Explains *why* entropy bonus alone cannot
   escape the vertex and motivates structural approaches like
   ACPG.

10. **Huang, S. et al. (2022).** *The 37 Implementation Details
    of Proximal Policy Optimization.* ICLR Blog Track.
    [Link](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
    — the practitioner reference for all the "small details"
    that make PPO actually work, including the `std=0.01` actor
    head init. ACPG can be viewed as one more entry on this
    list, if a relatively invasive one.

## Session history

- **2026-04-08 Phase 6 verification**: first encountered policy
  collapse on crusher; initial literature research identified
  ACPG as the academic state-of-the-art.
- **2026-04-08 Phase 6.4 resolution**: shipped the soft-regularisation
  fix (small-scale init + literature hyperparameters) which
  resolved collapse on crusher. ACPG deferred as "the stronger
  fix we'd switch to if the soft fix failed".
- **This document**: written for future reference so the design
  decision is recoverable and ACPG can be picked up without
  re-running the literature search if the failure mode re-emerges
  on a new domain.
