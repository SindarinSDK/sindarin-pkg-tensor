# PPO + GNN for AC Optimal Power Flow (López-Cardona et al. 2022)

## Status: RESEARCH NOTE — reference architecture, not directly applicable

Background reading for understanding how a published PPO + GNN
stack targets a real-world graph-structured optimisation problem.
The paper is directly relevant to `sindarin-pkg-tensor` because it
implements essentially the same algorithmic stack (PPO policy
gradient + graph neural network with message passing) that
`Gnn.trainPpo` supports, but in a different domain (electrical
power grids instead of skynet's industrial process control).

This note captures what the paper does, how the architecture
compares to what `Gnn` currently offers, and which hyperparameter
and design choices are worth cross-referencing when tuning PPO
+ GNN setups in the future.

## TL;DR

López-Cardona, Bernárdez, Barlet-Ros, Cabellos-Aparicio (2022)
propose combining PPO with a Graph Neural Network to solve the AC
Optimal Power Flow problem — deciding which generators to dispatch
in an electrical grid to minimise cost subject to physical
constraints. The policy is parameterised by an actor-GNN that
reads the grid topology as a graph (nodes = buses, edges =
transmission lines) and outputs a distribution over generator
dispatch actions, trained with standard PPO. The approach
generalises across unseen grid topologies and achieves up to 30%
cost reduction versus the DC-approximation baseline on IEEE 30-bus
benchmarks.

The relevance for `sindarin-pkg-tensor`: this is a working,
published instance of the same PPO-on-GNN pattern the `Gnn.trainPpo`
API is designed for, with a different architecture
(message-passing with concat readout) and very different
hyperparameters (`lr=0.003`, `batch=25`, `epochs=3`) than either
the skynet crusher defaults or the Phase 6.4 literature defaults.
Useful as a real-world sanity check for "how should a PPO + GNN
stack look in production" and as a comparison point for
architecture choices.

## The problem the paper addresses

**AC Optimal Power Flow (ACOPF)** is the problem of dispatching
electrical generators in a grid to meet demand at minimum cost,
subject to the physical constraints of the AC power flow equations
(voltage magnitudes, phase angles, line capacities, generator
limits). It's non-convex and notoriously hard.

Grid operators solve ACOPF every few minutes in real-world
operation. Traditional numerical methods are slow and don't
gracefully handle topology changes (e.g. lines going offline).
Most practical systems fall back to **DC Optimal Power Flow**
(DCOPF), a linearised approximation that's faster but can be
5-15% suboptimal.

The paper's target: a learned dispatcher that's fast enough for
real-time operation, handles topology changes without retraining,
and closes the gap to the true AC optimum.

## Why this matters for `sindarin-pkg-tensor`

Four direct overlaps with the `Gnn.trainPpo` use case:

1. **Same algorithmic stack.** PPO-clipped surrogate objective +
   GNN-parameterised actor. `sindarin-pkg-tensor` already ships
   both (`sn_tensor_ppo_clipped_loss`, `Gnn` class, `trainPpo`
   method). A user who wants to build an ACOPF-style dispatcher
   on top of this package has all the primitives already.

2. **Graph-structured state.** Both domains naturally represent
   state as a graph. Crusher/skynet builds a temporal/similarity/
   causal graph from process telemetry; ACOPF has the grid
   topology baked in. Both train on the same `Gnn.forward`
   signature.

3. **Discrete action space over graph elements.** Both pick
   actions per node (skynet picks which action to take, ACOPF
   picks which generator to dispatch). Same softmax-over-nodes
   output pattern.

4. **Real-world deployment constraints.** Both need to run with
   reasonable throughput, generalise across state variations,
   and be safe enough to deploy in production-like environments.

The architectures and hyperparameters differ in instructive ways —
see the comparison table below.

## Architecture

### Node and edge features

The grid is a graph where:

- **Nodes = electrical buses**. Node feature vector:
  `X_n^AC = [V_n, θ_n, P_n, Q_n]` — 4-dim, with voltage magnitude
  `V`, phase angle `θ`, real power `P`, reactive power `Q`.

- **Edges = transmission lines**. Edge feature vector:
  `e_{n,m}^ACLine = [R_{n,m}, X_{n,m}]` — 2-dim, with electrical
  resistance `R` and reactance `X`.

Tiny compared to skynet's 24-dim node features (12 continuous
process signals + 12 digital flags) — but this reflects that the
power flow equations are physical laws with a known closed-form
state representation, whereas industrial process control needs
empirically-discovered features.

### Message Passing Neural Network (MPNN)

The model is a standard 3-phase MPNN:

1. **Message function** — each node processes its neighbours'
   current hidden states through a 2-layer MLP, producing a
   per-edge message vector.
2. **Aggregation** — incoming messages at each node are combined
   via **concatenation of min, max, and mean** over the neighbour
   set. This is unusual: most GNNs use just `mean` or `sum` or
   `max`; the concat gives the downstream update function access
   to multiple statistics of the neighbour distribution.
3. **Update** — the node's new hidden state is computed by a
   3-layer MLP that takes the old state + the aggregated message.

**`k = 4` message passing iterations**, propagating information up
to 4 hops through the graph.

**Node hidden state = 16 dimensions**.

Compare to `sindarin-pkg-tensor`'s `Gnn`:

| Detail | López-Cardona 2022 | `sindarin-pkg-tensor` |
|---|---|---|
| Message passing iterations | 4 | `numLayers` (typically 3 in skynet) |
| Hidden dim | 16 | `hiddenDim` (typically 32 in skynet) |
| Aggregation | `concat(min, max, mean)` | `sum`, `sum_normalized`, or `mean` (arch-dependent) |
| Edge features | resistance, reactance | edge weights (scalar) |
| Update function | 3-layer MLP | linear + ReLU (GCN/SAGE) or attention (GAT) |
| Message function | 2-layer MLP | implicit (aggregation does the work) |

### Actor-GNN (policy head)

- **Input**: individual node representations (16-dim) for each
  **generator node** only (not all buses, just the ones that can
  be dispatched).
- **Readout**: a 3-layer MLP applied *independently to each
  generator's representation*.
- **Output**: `N` probability values, one per generator, forming
  a categorical distribution over "which generator to dispatch
  next".
- **Action selection**: sample a generator index from this
  distribution, then increment that generator's dispatched power
  by one discrete "portion".
- **Portions per generator**: 50. So each generator's full
  range from `P_min` to `P_max` is divided into 50 steps, and
  the actor's choice is which generator gets bumped up by one
  step on each timestep.

This is a **per-node softmax with action-masking** — only
generator nodes are candidates. Non-generator buses exist in
the graph but don't contribute to the action distribution. This
is a useful pattern that `sindarin-pkg-tensor` doesn't currently
have a first-class API for, but could be synthesised by masking
the logits before softmax.

### Critic-GNN (value head)

- **Input**: a **global graph aggregate** — concatenation of
  `sum`, `min`, and `max` of all node hidden states → `3 × 16 = 48`
  dimensions. The critic sees the whole graph at once, not
  per-node.
- **Readout**: a 3-layer MLP with a single scalar output.
- **Output**: `V(s)` — the state-value function estimate.

The actor operates at node level (to pick a specific generator),
the critic operates at graph level (to score the overall state).
Both share the same upstream message-passing backbone.

Compare to `sindarin-pkg-tensor`: the current `Gnn.forward`
returns a single `GnnOutput` with both `logits` (per-graph,
after mean-pool) and `embedding`. There's no separate critic
head — value function support is an application-level concern.
The López-Cardona architecture is a **true actor-critic**, with
two distinct output heads sharing the message-passing body.

### Reward function

```
r(t) = {
  MinMaxScaler(cost_t) - MinMaxScaler(cost_{t-1})  [normal case]
  cte₁ = -1                                         [generator at P_max]
  cte₂ = -2                                         [infeasible solution]
}
```

The primary reward is the **change in normalised cost** between
consecutive timesteps, so positive rewards correspond to cost
improvements. MinMax scaling normalises across episodes with
different starting costs.

Hard-capped constants for invalid actions (generator maxed out →
`-1`, grid solver fails to converge → `-2`). This gives the
policy clear "don't do that" signals without letting a single
bad action dominate the gradient.

Compare to skynet's `safe_throughput` reward in
`config/crusher/reward-engine.yaml`:

```yaml
reward_function: safe_throughput
safe_throughput:
  throughput_key: weight_cvr04
  throughput_scale: 100.0
  safety_key: amps_cru01
  safety_limit: 170.0
  penalty: -5.0
```

Both use the same "continuous reward signal + hard penalty for
bounds violations" pattern. Different scale (skynet `±5` vs
ACOPF `±2`) and different normalisation strategy (skynet uses
fixed divide-by-100, ACOPF uses per-episode MinMax), but the
design philosophy matches.

## Training hyperparameters

The paper reports these values:

| Parameter | López-Cardona 2022 | skynet Phase 6.4 | SB3/CleanRL standard |
|---|---|---|---|
| `learning_rate` | **`0.003`** | `0.00025` | `0.0003` (MuJoCo) / `0.00025` (Atari) |
| `minibatch_size` | **`25`** | `64` | `64-256` |
| `epochs_per_update` | **`3`** | `10` | `10` |
| `optimizer` | ADAM | ADAM (via `Optimizer.adamw`) | ADAM |
| `episodes` | 500 | N/A (continuous loop) | domain-specific |
| `horizon` / `n_steps` | 125 | 10 cycles (train_interval) | 2048 (SB3 MuJoCo) |
| `message_iterations` | 4 | `numLayers` = 3 | N/A (MLP) |
| `hidden_dim` (GNN) | 16 | 32 | N/A |
| Generator action portions | 50 | N/A (3 discrete actions) | N/A |

**`lr = 0.003`** is an order of magnitude higher than our
literature-tuned Phase 6.4 value of `0.00025`, and 10x higher
than the SB3 MuJoCo default. At `batch=25` and `epochs=3`, their
cumulative per-update policy drift is:

```
drift ≈ lr × epochs × (rollout_size / batch) × gradient_scale
     = 0.003 × 3 × (125 / 25) × ~1
     = 0.045
```

Versus skynet Phase 6.4:

```
drift ≈ 0.00025 × 10 × (200 / 64) × ~1
     ≈ 0.0078
```

López-Cardona 2022 is doing ~6x more policy drift per update
than skynet Phase 6.4. **And they report it works.** This could
mean:

1. ACOPF has a less imbalanced action prior than crusher, so
   collapse risk is lower and aggressive updates are fine.
2. Their reward signal (cost-delta, near-zero most of the time,
   heavily regularised by the MinMax scaler) produces smaller
   effective gradients than skynet's reward-weighted advantages.
3. The 125-step horizon is short enough that training fully
   resets between updates.
4. Or it works despite being aggressive, and they just didn't
   hit the collapse failure mode we did on crusher.

Hard to say without reproducing. But it's a good reminder that
**PPO hyperparameters are domain-dependent** — the Phase 6.4
"literature defaults" aren't universal, they're one point in a
multi-dimensional tuning space that includes the reward
distribution, action prior, and rollout size.

## Empirical results

The paper reports on the IEEE 30-bus benchmark (a standard
academic grid with 30 buses, 41 transmission lines, 6 generators):

- **Load variation tests** (±10% load changes): improvement ratio
  up to `1.30×` better than DCOPF — i.e. the learned dispatcher
  gets ~30% closer to the true AC optimum than the DC
  approximation does.
- **Load removal tests** (drop up to 50% of loads): comparable
  or superior performance, full convergence.
- **Edge removal tests** (drop transmission lines): ratios
  ranging `0.52×` to `2.83×`. The wide range reflects that some
  topology changes are much easier to handle than others — the
  learned policy handles some perfectly and struggles with
  others.
- **Cost deviation from true ACOPF optimum**: typically
  `0.4%-1.1%` — i.e. the learned dispatcher is within 1% of the
  provably-optimal solution on most test cases.

**Convergence**: full convergence on the topology tests; 100%
convergence on load variations up to about 10%.

**Generalisation**: the key contribution — the trained policy
handles topology changes (added or removed lines) without
retraining, because the GNN naturally processes whatever graph
it's given. Prior DRL-only approaches required retraining for
each topology.

## Comparison with skynet's PPO + GNN stack

| Aspect | López-Cardona 2022 | skynet + `Gnn` |
|---|---|---|
| **Domain** | AC Optimal Power Flow (ACOPF) | Industrial process control (crusher) |
| **Graph source** | Physical grid topology (fixed, small) | Dynamically constructed from telemetry (varies) |
| **Node count** | 30 (IEEE 30-bus) | varies, typically 5-50 per graph |
| **Node features** | 4 physical quantities | 24 sensor values |
| **Action space** | Pick a generator to dispatch (6 generators × 50 portions) | 3 abstract actions (speed_up, speed_down, hold) |
| **Action head** | Per-node softmax over generators only | Graph-level softmax after mean-pool |
| **Critic head** | Separate GNN head with global readout | None (not implemented) |
| **GNN architecture** | MPNN with concat(min, max, mean) | GCN / GAT / GraphSAGE |
| **Loss** | PPO-clipped | PPO-clipped (`sn_tensor_ppo_clipped_loss`) |
| **Message iterations** | 4 | 3 |
| **Hidden dim** | 16 | 32 |
| **Learning rate** | 0.003 | 0.00025 |
| **Training data** | 500 episodes × 125 steps = 62,500 transitions | accumulated on-the-fly, ~200 sample cap per class |
| **Training cadence** | Batch PPO at end of each rollout | Continuous, every N orchestrator cycles |
| **Actor init** | Not specified (presumably Kaiming defaults) | `std=0.01` small-scale (Phase 6.3) |
| **Advantage normalisation** | Not specified | Per-batch mean-centred (trainer.sn) |

The biggest architectural gap is **the critic head**.
López-Cardona's actor-critic setup uses the critic's `V(s)`
estimate as a variance-reduction baseline for the PPO advantage
calculation — the standard textbook PPO recipe. `sindarin-pkg-tensor`
currently doesn't ship a critic: the skynet trainer computes
advantages as "reward minus batch mean" (a crude 1-step Monte
Carlo baseline), which is higher-variance than the critic
approach.

## What we could adopt

Not all of this is directly portable, but several ideas are
worth cataloguing for future work:

### 1. Separate critic head

**Why**: standard PPO uses actor-critic with a learned value
function. The critic's `V(s)` is used in the advantage
computation `A(s, a) = r(s, a) + γ V(s') - V(s)` (GAE), which is
dramatically lower-variance than skynet's "reward minus batch
mean" baseline.

**Where**: `Gnn.forward` could return a second scalar
`valueEstimate` in `GnnOutput`, backed by a parallel classifier
head on top of the mean-pooled graph embedding.

**Cost**: ~30 lines in `Gnn.forward`, additional parameter
storage, new loss term `c1 × MSE(V_pred, R_target)` in the PPO
loss function, new tensor.sn.c function for the VF loss.
Non-trivial but well-scoped.

**Impact**: lower-variance gradients → faster convergence and
potentially higher stability. Paper also enables GAE which is
the standard advantage formula.

### 2. Concat(min, max, mean) aggregation

**Why**: standard GNN aggregators (sum, mean, max) lose
information. Concatenating all three gives the update function
access to multiple statistics of the neighbour set at once.
López-Cardona 2022 uses this and reports strong results.

**Where**: would be a new arch variant in `GnnLayer.create`
alongside `"gcn"`, `"gat"`, `"sage"` — call it `"mpnn"` or
`"concat"`.

**Cost**: ~50 lines in gnn.sn + corresponding ggml ops for the
per-neighbour aggregation.

**Impact**: potentially better feature extraction on graphs
where neighbourhood heterogeneity matters (variable degree
nodes, clusters with different statistics). Hard to estimate
without empirical comparison.

### 3. Action masking for per-node policies

**Why**: some domains (ACOPF: only generators can be dispatched;
skynet: some actions may be invalid given state) need to mask
out illegal actions before softmax. López-Cardona's actor-GNN
implicitly does this by only reading generator node
representations.

**Where**: `Gnn.forward` could accept an optional
`actionMask: bool[]` that zeroes out logits at masked positions
before softmax.

**Cost**: ~20 lines + an extra input tensor in
`sn_tensor_ppo_clipped_loss`.

**Impact**: opens up per-node-action problems where the current
graph-level softmax doesn't fit.

### 4. MinMax reward normalisation

**Why**: skynet currently clips rewards to `[-5, 5]` via fixed
bounds. López-Cardona normalises per-episode via MinMax scaling,
which handles domain-shift (a new simulator producing rewards in
a different range) without retuning.

**Where**: `src/reward/reward_engine.sn`.

**Cost**: moderate — needs an episode buffer to track min/max
within the current episode, then retroactive rescaling at
episode end. Trickier in skynet's continuous-loop setting where
"episodes" don't have clean boundaries.

**Impact**: better cross-domain portability at the cost of some
complexity.

## What we wouldn't adopt

Not every architectural choice in the paper is worth copying:

- **`lr = 0.003`** — much too aggressive for our regime based
  on Phase 6 evidence. Their stable-at-0.003 result doesn't
  generalise to crusher's imbalanced priors.
- **`hidden_dim = 16`** — our feature space is 24-dim, so 16-dim
  hidden would be a bottleneck. Crusher's 32-dim hidden is
  already close to minimal.
- **Per-generator softmax** — we have 3 abstract actions, not
  per-generator dispatch. The action shape is different.
- **Per-episode MinMax reward scaling** — skynet doesn't have
  clean episode boundaries (the learning loop is continuous),
  so per-episode normalisation doesn't directly map.

## When to re-read this doc

- If `skynet` adds a new domain where the action space is
  **per-element of the graph** (e.g. "dispatch this worker",
  "route to this queue") — the actor-GNN readout pattern is
  directly applicable.
- If advantage variance becomes a concern — the critic head
  adoption is the next step.
- If a new domain needs **generalisation across graph
  topologies** (unseen graph shapes at inference time) — this
  paper proves PPO + GNN can do it and offers a template.
- If tuning PPO-on-GNN hyperparameters for a new domain — their
  `lr=0.003, batch=25, epochs=3` is a useful "aggressive but
  works" point in the tuning space, worth trying before falling
  back to the more conservative literature defaults.

## References

1. **López-Cardona, Á., Bernárdez, G., Barlet-Ros, P. &
   Cabellos-Aparicio, A. (2022).** *Proximal Policy Optimization
   with Graph Neural Networks for Optimal Power Flow.* arXiv
   2212.12470. [Link](https://arxiv.org/abs/2212.12470) —
   the paper this note summarises. v3 revision at
   [arXiv:2212.12470v3](https://arxiv.org/html/2212.12470v3).

2. **Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O.
   & Dahl, G. E. (2017).** *Neural Message Passing for
   Quantum Chemistry.* ICML.
   [Link](https://arxiv.org/abs/1704.01212) — the foundational
   Message Passing Neural Network (MPNN) paper that
   López-Cardona 2022's GNN architecture follows. Defines the
   three-step message/aggregate/update framework.

3. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A. &
   Klimov, O. (2017).** *Proximal Policy Optimization
   Algorithms.* arXiv 1707.06347.
   [Link](https://arxiv.org/abs/1707.06347) — the PPO paper.
   López-Cardona 2022 uses PPO-clipped with no algorithmic
   modifications.

4. **Schulman, J., Moritz, P., Levine, S., Jordan, M. &
   Abbeel, P. (2016).** *High-Dimensional Continuous Control
   Using Generalized Advantage Estimation.* ICLR.
   [Link](https://arxiv.org/abs/1506.02438) — Generalized
   Advantage Estimation (GAE), the variance-reduced advantage
   formula that needs a critic. Relevant to adopting the
   critic head proposed above.

5. **IEEE 30-bus test case** — the standard academic benchmark
   López-Cardona 2022 evaluates on. Widely used in the power
   systems and ML-for-grid literature. The case is distributed
   as part of MATPOWER and similar packages.

6. **Andrychowicz, M. et al. (2021).** *What Matters in
   On-Policy Reinforcement Learning? A Large-Scale Empirical
   Study.* ICLR. [Link](https://arxiv.org/abs/2006.05990) —
   our Phase 6.4 hyperparameter defaults come from this paper.
   Useful cross-reference when comparing the López-Cardona
   hyperparameters to the "literature consensus" values
   skynet uses.

7. **Huang, S. et al. (2022).** *The 37 Implementation Details
   of Proximal Policy Optimization.* ICLR Blog Track.
   [Link](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
   — practitioner reference for PPO. Complements this paper
   for anyone building a PPO + GNN stack from scratch.

8. **Related work in ML-for-grid**: Donnot, B., Guyon, I.,
   Schoenauer, M., Marot, A. & Panciatici, P. (2018). *Fast
   Power System Security Analysis with Guided Dropout.* and
   subsequent papers from the Learning to Run a Power Network
   (L2RPN) challenge series. López-Cardona 2022 builds on
   this body of work.

## Session history

- **2026-04-08**: read the paper after the ACPG research note
  was written. Relevant for understanding how published PPO +
  GNN stacks look in production, as a reference for future
  architecture decisions in `sindarin-pkg-tensor` and skynet.
- Informs potential future work on: critic heads, action
  masking, alternative GNN aggregators, and cross-topology
  generalisation.
