# The Golden Path: Fix `sindarin-pkg-tensor` So Skynet Can Learn

## Status: RESOLVED — all phases complete, 19/19 tests green, release tag pending

## Resolution summary

Every item in the Part 7 acceptance checklist below is green. The full
test suite (19 binaries, ~1600+ assertions) passes reliably. The work
landed in phases labeled A–K; each phase left the suite green before
the next started, and every behavioural claim has a regression test
backing it.

| Phase | Shipped | Regression guard |
|---|---|---|
| A — Sage fix + Bug B3 regression guard | `src/gnn.sn::sageForward` now uses `gnnMatmul`; covers gat/gcn/sage | `tests/test_weight_update_proven.sn` (126 assertions) |
| B — Weighted loss behavioural proof | Unchanged code; behavioural proof the weighted CE steers the gradient | `tests/test_weighted_loss.sn` (4 assertions) |
| C — `TrainResult` Phase C diagnostics | Extended struct with 9 new fields (`lossCurve`, `gradNormCurve`, per-param L2/max-abs-delta, weight/input stats, real accuracy); populated in `Gnn.train()` | `tests/test_train_result_diagnostics.sn` (85 assertions) |
| D — Diagnostic accessors | **Skipped** — subsumed by Phase C's in-struct diagnostics. See §4.3 note. | n/a |
| E — RL reward-weighted policy proof | Unchanged code; contrast proof that weighted and unweighted runs learn opposite policies on the same data | `tests/test_reward_weighted_policy.sn` (8 assertions) |
| F — `predictBatch` + `distributionDivergence` | Added `Gnn.predictBatch()` and the free `distributionDivergence(a, b, kind)` (`l1`, `l2`, `kl`, `js`) | `tests/test_predict_batch_and_divergence.sn` (37 assertions) |
| G — Batch-composition invariance (Test 6 reformulation) | Unchanged code; the invariant from GNN literature (`forward(G) == forward(batch([..G..]))[i]`) and its permutation twin | `tests/test_batch_composition_invariance.sn` (63 assertions) |
| H — Strict crusher e2e | Strict replacement for the prints-only predecessor; JSD thresholds at `ln(2)`, determinism rerun at 1e-6 | `tests/test_crusher_policy_e2e.sn` (78 assertions) |
| I — Strict save/load roundtrip under train | Strict replacement for the 0.01-tolerance predecessor; 1e-6 tolerance + full save→load→continue-train vs never-saved reference | `tests/test_save_load_roundtrip_under_train.sn` (701 assertions) |
| J — Metric callback API | `sn_graph_set_train_metric_callback` / `_clear_` / `_emit_` — consumer registers `fn(str, double): void`, package emits per-epoch and post-train | `tests/test_train_metric_callback.sn` (68 assertions) |
| K — Cleanup + release | Deleted legacy `sn_graph_train` (~145 LOC dead C code), stale accessors, obsolete exploratory test, stale comments. Docs updated. | full suite stays 19/19 after deletion |

### Bugs found beyond the original audit

1. **GraphSAGE was completely broken, not just gradient-flow degraded** (Phase A). The old `sn_tensor_matmul` host pre-transpose path misinterprets the GNN weight layout `(ne0=inputDim, ne1=outputDim)`, producing the wrong output shape so `.add(bias)` crashes in `ggml_add`. No prior test exercised sage. Fix is a one-liner in `src/gnn.sn::sageForward`. `docs/issues/weight-update-failure.md` was updated to record the follow-up.

2. **Compiler refcount double-free on `arr2d[i] = local` reassignment for refcounted `T[][]`** (Phase C). Surfaced during Phase C's rolling parameter snapshot. Reported as `docs/issues/compiler-problems.31.md` issue #50 with a 12-line pure-Sindarin minimal repro. Fixed upstream during this work. Workaround was removed after the fix; Phase C's `gradNormCurve` now uses the intended L2-of-param-delta computation and the 12-line repro is kept as a regression guard at `tests/exploratory/repro_array_reassign.sn`.

3. **Test 6 was vestigial as written** (Phase G). The doc's original "train/predict shape contract" test was written under the homogeneous-batch assumption that predated heterogeneous batching; its literal form became vacuous. Reformulated as the canonical GNN batch-composition invariance test the literature (PyG / DGL / OGB) actually checks. See the new §Part 3 Test 6 block for the rationale.

### Deviations from the doc

- **§4.2 metric callback signature** ships as `fn(str, double): void` not `fn(str, double, StringField[]): void`. Rationale in §4.2 "Implementation deviation (Phase J)". Callers pack label values into the metric name string; a future `_v2` entry point can add structured labels without breaking this surface.
- **§4.3 diagnostic accessors** not shipped separately. All the data they would expose is already on `TrainResult`, so the accessors would be duplicate state. Recorded as Phase D "SKIPPED" in the progress log.

### Skynet unblock

Every item in the consumer-side `skynet/docs/issues/golden-path.md` Part 6 inheritance table that depends on this package is now ready to land. Skynet can:
- Pass `advantages` directly to `Gnn.train()` (Phase B + E).
- Delete its dead `batchGraphsToTensors()` helper and call `Gnn.train(graphs, ...)` directly (Phase G + the pre-existing heterogeneous batching).
- Register a `MetricsClient`-forwarding callback via `sn_graph_set_train_metric_callback` (Phase J).
- Use `Gnn.predictBatch()` + `distributionDivergence(a, b, "js")` for the canonical-pair probe (Phase F).
- Read `TrainResult.paramNormBefore/After/MaxAbsDelta` for per-train weight trajectory (Phase C).
- Rely on the batch-composition invariance proof (Phase G) for the "train and predict see the same inputs" invariant at the package boundary.

The skynet-side work to consume all of this is tracked in `skynet/docs/issues/golden-path.md` Part 4 step 1.

## Progress log

| Step | Status | Notes |
|---|---|---|
| **Step 0** — establish baseline | DONE | All existing tests passed against the broken self-loop template — confirmed every existing test was single-node and tripped the identity short-circuit, so the multi-node code path was effectively dead. Documented in commit `b4d9b59`. |
| **Step 1** — failing baseline `tests/test_real_graph_topology.sn` | DONE | Two graphs with identical features but different edge structure (chain vs self-loops). Originally failed with `loss=ln(2)` because the trainer collapsed both samples into identical inputs via the self-loop template. Committed in `b4d9b59`. |
| **Step 2a** — `batchGraphs(graphs[]): GraphTensors` helper + 89-assertion unit test | DONE | Pure-Sindarin graph batching with offset edge indices and per-row batchIndex. Committed in `b4d9b59`. |
| **Step 2b** — `sn_tensor_weighted_cross_entropy` op (record + direct mode) + 5-assertion unit test | DONE | Bypasses ggml-opt's hardcoded loss_type enum via the `loss_type=SUM` injection point discovered in `vendor/ggml/src/ggml-opt.cpp:396-401`. Committed in `b4d9b59`. |
| **Step 2c.1** — record-mode test for the loss op + `sn_graph_input_data` + `sn_graph_compute_loss` helpers | DONE | Validates the record-mode branch produces identical results to direct-mode within 1e-4. 4/4 PASS. |
| **Step 2c.2** — `sn_graph_train_epoch` C entry point + by-hand model smoke test | DONE | Static-graph ggml_opt with `loss_type=SUM` and our pre-built loss as `outputs`. Lazy-init in first epoch call, persists across epochs in `sn_graph_begin/end` cycle. Loss drops 0.598 → 0.017 in 30 epochs on a 2x2 by-hand classifier. |
| **Step 2c.3** — `Gnn.train()` rewrite + migrate all 5 existing tests + ggml patch | DONE | Gnn.train rewritten to take `graphs[]` instead of flat features. New signature: `train(graphs[], labels[], weights[], optimizer, epochs, batchSize, valSplit, seed)`. All 5 tests migrated. **10/10 tests pass.** Multi-node `test_real_graph_topology` no longer crashes (ggml patches at `RealOrko/ggml@sn-pkg-tensor`) and now converges (loss `0.034`, JSD `0.51`). The convergence fix was in `src/gnn.sn`: replaced the per-epoch within-batch shuffle (which scrambled the static-topology slot binding) with an across-batch shuffle, plus added a runtime precondition assertion. See `docs/issues/ggml-issue.md` "Final status" and the new `docs/issues/heterogeneous-graph-batching.md` for the architectural precondition this surfaced. |

## Outstanding work

1. ~~Fix the multi-node convergence bug.~~ **DONE** in Step 2c.3 — root
   cause was the per-epoch within-batch shuffle in `Gnn.train()`
   scrambling the static-topology slot binding, not a ggml or weighted-CE
   bug. Fix was replacing it with an across-batch shuffle. See
   `docs/issues/ggml-issue.md` "Final status".
2. **Bug B3 attWeight sub-bug** (called out in Step 0): GAT attention weights
   never receive gradients because `sn_tensor_attention_aggregate` falls back
   to `sparse_aggregate` in record mode and never references `attWeight`.
   Separate from B1; tracked but not yet fixed.
3. **Steps 2c.4 onwards** — `TrainResult` diagnostics, `predictBatch` /
   `distributionDivergence`, metric callback API, release tagging.
4. ~~**Heterogeneous-graph batching**~~ **DONE.** `Gnn.train()` now
   handles graphs with variable `numNodes` and variable `numEdges`
   inside a single multi-batch call via pad-to-max + per-batch upload
   of features, dense adjacency, and pool matrix. The `numNodes` and
   `numEdges` precondition assertions are gone; the only remaining
   shape constraint is uniform `featureDim`. See
   `docs/issues/heterogeneous-graph-batching.md` for the implementation
   write-up. Regression guard:
   `tests/test_heterogeneous_graph_batching.sn`.

## Skynet unblock status

**Skynet crusher is fully unblocked.** Earlier revisions of this
section claimed crusher was "single-node graphs by construction" — that
was wrong. The crusher pipeline builds graphs on demand from the
orchestrator's per-cycle observation poll (`pollObservations(...,
50)`), so persisted graphs span a wide distribution of node counts (3
to dozens) and edge counts. A live DB sample after 2 minutes of
runtime showed graphs with `(nodes, edges)` of `(4, 6)`, `(9, 36)`,
`(3, 3)`, `(8, 28)`, … — squarely in Regime C of the
heterogeneous-graph-batching doc.

Skynet hit the multi-batch precondition assertion the moment its trainer
started training on real graphs. The pad-to-max + per-batch upload work
in `Gnn.train()` resolves this. The skynet trainer's old
`usableSamples` divisibility workaround is gone (`skynet/src/trainer/trainer.sn`).
`test_crusher_policy.sn` and `test_heterogeneous_graph_batching.sn`
both pass.

## Why this document exists

`skynet` is stuck. It cannot learn a state-conditional policy on the crusher domain, no matter what reward function, simulator parameter, or hyperparameter is changed. The full skynet-side audit is in `skynet/docs/issues/golden-path.md`. The summary that matters here:

> **Three of the four root-cause bugs in skynet's learning loop live in this package, not in skynet.**
>
> - `Gnn.train()` builds a fake "self-loop template graph" instead of training on the caller's actual graph topology. Message passing across real edges never happens during training.
> - `sn_graph_train` accepts no per-sample weights. Reward-weighted loss is impossible from the API surface, so REINFORCE / policy gradient cannot be expressed. Skynet computes advantages and silently drops them on the floor because there is nowhere to pass them.
> - Weights never update during training. Only biases learn. Already documented in `weight-update-failure.md`. Caused by the host-side weight pre-transpose in `sn_tensor_matmul`.
>
> Skynet cannot work around any of these. They have to be fixed here.

This document defines the contract changes the tensor package must make, the integration tests that prove the fixes are real, and the structured metrics the package must surface so that the consuming application (skynet) can build observability on top without having to instrument the C side itself.

**The shape of the work is: fix the API, prove it with tests that match the production scenario, then expose the diagnostics that let the production scenario be observed continuously.** If we get this right, the skynet trainer becomes a thin shim and most of the skynet golden-path Tier 0 work disappears.

---

## Part 1 — The four bugs, restated against this package

### Bug B1 — `Gnn.train()` discards caller graph topology

`src/gnn.sn:223-280`. The current implementation builds:

```sindarin
# Build template graph: batchSize nodes, each a self-loop with weight 1.0.
# Each node maps to its own graph via batchIndex so meanPool is identity.
for var i: int = 0; i < batchSize; i++ =>
  edgeSrc.push(i.toDouble())
  edgeDst.push(i.toDouble())
  edgeWts.push(1.0)
  batchIdx.push(i.toDouble())
```

The caller passes a flat `data: double[]` of `nsamples * featureDim` and the method ignores graph structure entirely. Every "graph" in the template batch is a single isolated self-loop node, so message passing across actual edges is a no-op throughout training. The very same `Gnn` object then runs `forward(GraphTensors, false)` at inference time on **real** topology (real `edgeIndex`, real `batchIndex`, multi-node graphs). The model is trained on a different distribution from the one it serves on. This is the root reason for the predict/train asymmetry the user observed in skynet.

### Bug B2 — `sn_graph_train` has no weights argument

`src/tensor.sn.c:280-420`. Signature:

```c
double sn_graph_train(RtTensor *output_rt, RtTensor *input_rt,
                      SnArray *data_arr, SnArray *label_arr,
                      long long nsamples, long long nepochs,
                      long long nbatch, double val_split,
                      char *loss_type_str, char *optimizer_str,
                      double lr, double wd)
```

There is no per-sample weight tensor. The body uses `GGML_OPT_LOSS_TYPE_CROSS_ENTROPY` directly via `ggml_opt_epoch`, which is unweighted. There is no way for a caller to multiply the loss of a sample by its reward / advantage / importance weight. Policy gradient (the entire point of an RL training loop) cannot be expressed against this API.

`docs/issues/weight-update-failure.md` notes that `sn_train_step` was added (`tensor.sn.c` debug prints in the issue refer to it) but it is **not exposed in `tensor.sn`**, so the skynet side cannot reach it. Either the API is incomplete or it was wired up and then unwired — either way, what skynet imports today does not support weights.

### Bug B3 — Weights never receive gradient updates

`src/tensor.sn.c` `sn_tensor_matmul` host-side pre-transposes the weight matrix to work around a `repeat_back` non-contiguous stride assertion. This creates a **disconnected copy** in a new pool slot. The optimizer trains the copy. Readback reads the original. Net result: only biases learn. This is the existing issue in `weight-update-failure.md`. The proposed fix `sn_tensor_gnn_matmul` is partway in (`tensor.sn:60`, `tensor.sn:129`), but per the issue it is blocked on a gallocr segfault during compute and the bug is still in effect for skynet today.

### Bug B4 — `Gnn.train()` does not return enough information for the caller to know what happened

The method returns:

```sindarin
struct TrainResult =>
  loss: double
  accuracy: double
  epochs: int
```

These are top-line scalars. The caller cannot tell:

- Which parameters changed and by how much (needed to detect Bug B3 from skynet without tailing stderr).
- What the per-batch loss curve was (needed to detect divergence, plateauing, NaN spikes).
- Whether the gradient norm was non-zero (needed to detect "gradients were computed but failed to flow").
- What the input-distribution statistics were (needed to detect garbage-in conditions like Bug A from the skynet doc).
- Whether the weights tensor it received was actually used or silently discarded.

The only diagnostic that does exist is the host-side `fprintf(stderr, "readback param pool=%d ...")` at `tensor.sn.c:411`. This is the right idea but in the wrong place — it cannot be queried, aggregated, or alerted on.

---

## Part 2 — The contract changes this package needs to make

### 2.1 — `sn_graph_train` must accept per-sample weights

**Required signature change** (C side):

```c
double sn_graph_train(RtTensor *output_rt, RtTensor *input_rt,
                      SnArray *data_arr, SnArray *label_arr,
                      SnArray *weight_arr,    /* NEW: nsamples doubles, may be NULL */
                      long long nsamples, long long nepochs,
                      long long nbatch, double val_split,
                      char *loss_type_str, char *optimizer_str,
                      double lr, double wd)
```

**Semantics:**
- `weight_arr` length is exactly `nsamples`, one weight per training sample.
- If `weight_arr` is NULL or all-ones, behaviour must be identical to today's unweighted CE.
- The C implementation multiplies the per-sample CE loss by the weight before reduction. The natural ggml expression is `softmax → log → mul(labels) → sum_rows → neg → mul(weights) → sum → scale_by(1/n)`. (`docs/issues/weight-update-failure.md` already describes this pattern in §"Reward-Weighted Loss".)
- The optimizer step still uses the standard AdamW path; only the loss computation changes.

**Sindarin-side surface change** (`src/tensor.sn`):

```sindarin
native fn sn_graph_train(output: Tensor, input: Tensor,
                         data: double[], labels: double[], weights: double[],
                         nsamples: int, epochs: int, batchSize: int, valSplit: double,
                         lossType: str, optimizer: str, lr: double, wd: double): double
```

**Sindarin-side caller change** (`src/gnn.sn`):

```sindarin
fn train(data: double[], labels: double[], weights: double[],
         numSamples: int, batchSize: int, featureDim: int,
         optimizer: Optimizer, epochs: int, valSplit: double): TrainResult
```

`weights` is required by signature so callers cannot accidentally drop it. To express "no weighting", the caller passes an array of 1.0s — explicit, not implicit.

### 2.2 — `Gnn.train()` must train on real graphs

The fake template-graph builder in `gnn.sn:223-262` must be deleted. The method must accept either a `GraphTensors` instance or, more usefully, a list of graphs that it batches internally. The two reasonable shapes:

**Shape A (graph-aware, batch-internal):**

```sindarin
fn train(graphs: GraphTensors[], labels: double[], weights: double[],
         optimizer: Optimizer, epochs: int, batchSize: int, valSplit: double): TrainResult
```

The method internally batches groups of `batchSize` graphs into a single `GraphTensors` (concatenate node features, offset edge indices, build batchIndex), runs `self.forward(batched, true)` inside `sn_graph_begin/end`, and calls the new weighted `sn_graph_train`. This is what `skynet/src/trainer/trainer.sn:90-149`'s currently-dead `batchGraphsToTensors()` already does — that logic should move into the tensor package so every consumer gets it for free.

**Shape B (caller batches, library trains):**

```sindarin
fn train(batched: GraphTensors, labels: double[], weights: double[],
         optimizer: Optimizer, epochs: int, valSplit: double): TrainResult
```

Caller is responsible for batching multiple graphs into one `GraphTensors`. The library only handles the train step.

Shape A is preferable because it pushes a correctness-critical primitive (graph batching with edge index offsetting) into the package where it gets tested once, instead of duplicated in every consumer. Shape B is acceptable as a lower-level escape hatch but should not be the only API.

**Either way: no more `template self-loop graph`. The caller's edges, node features, and batchIndex are what train sees.**

### 2.3 — Weight updates must actually happen (Bug B3 / `weight-update-failure.md`)

This is already documented and partway in flight. The acceptance criterion shifts though. Today the criterion is "the diagnostic test in `tests/test_train_diagnostics.sn` shows weight norm changing". That is necessary but not sufficient. The new criterion is:

- The crusher policy integration test (Part 3 below) passes with predictions that *change after training* and that *correctly classify the canonical samples*. The package's existing `tests/test_train_save_infer.sn` already has the structure for this and currently includes assertion `if preDelta1 < 0.01 => println("FAIL: overloaded predictions unchanged after training")`. It must pass.
- Per-layer weight L2 norms must change between consecutive `train()` calls by at least a configurable threshold (default `1e-5`). Surfaced as a structured metric (Part 4 below).

If the gallocr segfault from `weight-update-failure.md` is the blocker, the alternatives listed there (single-context approach, manual backward) are now in scope. Whatever it takes to get a real gradient onto the weight tensors.

### 2.4 — `TrainResult` must carry per-train diagnostics

Extended struct:

```sindarin
struct TrainResult =>
  loss: double                    # final training loss
  accuracy: double                # final training accuracy
  epochs: int                     # epochs actually executed
  lossCurve: double[]             # per-epoch loss
  gradNormCurve: double[]         # per-epoch L2 of accumulated gradient
  paramNormBefore: double[]       # per-parameter L2 norm at train() entry
  paramNormAfter: double[]        # per-parameter L2 norm at train() exit
  paramMaxAbsDelta: double[]      # per-parameter max abs change
  weightSumIn: double             # sum of caller-supplied weights (sanity check Bug C/B2)
  weightVarianceIn: double        # variance of caller weights (zero ⇒ unweighted)
  inputMean: double               # mean of input feature tensor across all batches
  inputStd: double                # std  of input feature tensor across all batches
```

`paramNormBefore` / `paramNormAfter` / `paramMaxAbsDelta` are arrays in the same order as `model.parameters()`. The caller can index them by layer.

These fields are not optional. The struct is the only thing the package can hand back across the FFI boundary, and it is the only place we can guarantee that the diagnostic data exists *for the exact training call that the consumer cares about*. This is what makes Bug B3 visible to skynet without skynet having to tail a container's stderr.

Cost: a few hundred bytes per `train()` call. Worth it.

### 2.5 — Optional, but valuable: `Gnn.predictBatch()` for canonical-pair probes

The skynet golden-path doc Tier 1 #7 calls for a "canonical-pair JSD probe" to be run periodically against the live champion. That probe needs to predict on two fixed graphs and compute the JSD between the resulting distributions. This pattern is generic, so put it in the package:

```sindarin
fn predictBatch(graphs: GraphTensors[]): GnnOutput[]
```

and a free function:

```sindarin
fn distributionDivergence(a: double[], b: double[], kind: str): double
```

where `kind` is `l1`, `l2`, `kl`, or `js`. This means the consumer's "is the model learning a state-conditional policy?" KPI becomes a one-liner against the package, not a hand-rolled implementation in the model engine.

---

## Part 3 — Integration tests: the contract that proves the fixes are real

**Principle: every change above must have a test that fails if the change regresses. No item is marked DONE without one.** This is the rule whose absence let `skynet/docs/design-corrections.md` claim §2 and §3 were RESOLVED while the actual code disagreed.

The integration tests below are designed so that **passing all of them means skynet's crusher domain will learn a policy automatically.** They are deliberately framed against the same scenarios skynet runs in production.

### Test 1 — `tests/test_weighted_loss.sn` (Bug B2)

**What it tests:** `sn_graph_train` and `Gnn.train()` actually use the weights array.

**Setup:** Two identical graphs with opposite labels.
- Graph A: features `[1, 0, 0]`, label `[1, 0, 0]` (action 0).
- Graph B: features `[0, 1, 0]`, label `[0, 0, 1]` (action 2).

**Two training runs:**
- Run 1: weights `[10.0, 0.0]`. Train 200 epochs.
- Run 2: weights `[0.0, 10.0]`. Train 200 epochs.

**Assertions:**
- After Run 1, `forward(Graph A)` predicts action 0 with `prob > 0.7`, `forward(Graph B)` predicts action 0 with `prob > 0.5` (because Graph B's loss was zero-weighted, so the model never learned action 2).
- After Run 2, `forward(Graph B)` predicts action 2 with `prob > 0.7`, and Graph A is *not* required to predict 0 (its training was zero-weighted).
- `result.weightSumIn` matches the expected sum.
- `result.weightVarianceIn > 0` in both runs.

**Why this test:** If Bug B2 regresses (weights silently ignored or default-1.0'd), both runs produce identical models — both samples got equal effective weight — and at least one of these assertions fails.

### Test 2 — `tests/test_real_graph_topology.sn` (Bug B1)

**What it tests:** `Gnn.train()` learns from the actual edge structure, not from a self-loop template.

**Setup:** Two graphs with **identical node features but different topology**.
- Graph A: 3 nodes with features `[[1,0,0],[0,1,0],[0,0,1]]`, edges form a chain `0→1→2`.
- Graph B: same 3 nodes with features `[[1,0,0],[0,1,0],[0,0,1]]`, edges form an isolated set (no edges).

For a GAT or sum-aggregate GNN, the message passing on Graph A will produce a different embedding (specifically, node 2 sees a contribution from nodes 0 and 1 via the chain) than on Graph B (each node sees only itself). After mean-pooling and the classifier head, the two graphs should produce different logits.

**Train** on a balanced labelled set: Graph A → action 0, Graph B → action 1, weights all 1.0, 200 epochs.

**Assertions:**
- After training, `forward(Graph A)` predicts action 0 with `prob > 0.7`.
- After training, `forward(Graph B)` predicts action 1 with `prob > 0.7`.
- The Jensen–Shannon divergence between `forward(A).probs` and `forward(B).probs` is `> 0.3`.

**Why this test:** If `Gnn.train()` reverts to building a self-loop template graph, the trained model has never seen the difference between A and B, and the JSD will be near zero. Test fails. This is the test that will not let Bug B1 silently come back.

### Test 3 — `tests/test_weight_update_proven.sn` (Bug B3)

**What it tests:** All trainable parameters — weights, not just biases — receive non-zero updates.

**Setup:** A small `Gnn` with `inputDim=4, hiddenDim=8, numActions=3, numLayers=2, arch="gat"`. 20 random training samples, balanced labels, weights all 1.0.

**Procedure:**
- Capture `model.parameters()` L2 norms before train.
- `train(... epochs=50)`.
- Read the returned `TrainResult.paramNormAfter` and `TrainResult.paramMaxAbsDelta`.
- Independently capture `model.parameters()` L2 norms after train and compare to the returned values (cross-check that the result struct matches reality).

**Assertions:**
- For **every** parameter (weights AND biases), `paramMaxAbsDelta[i] > 1e-5`.
- For **every weight matrix** (the matrices, specifically — index them by knowing that `parameters()` returns weights and biases interleaved), the L2 norm changes by at least `1e-4` between before and after.
- The values returned in `TrainResult` match those measured externally.

**Why this test:** This is the direct continuous test for Bug B3. It will fail today. It must pass before this golden path is closed.

### Test 4 — `tests/test_crusher_policy_e2e.sn` (the existence proof)

**What it tests:** The actual crusher domain learns a state-conditional policy from end to end. **This test is the contract between this package and skynet. If it passes, skynet will be unblocked.**

**Setup:** This test already exists in skeletal form as `tests/test_crusher_policy.sn`. It needs to be brought up against the new API and made deterministic. The existing test trains 200 epochs on a small balanced set and reports `correct/total`. That format is fine; the new requirements are stricter.

- 24 features (matching the skynet `feature_dim: 24`).
- 3 actions: `speed_up=0, speed_down=1, hold=2`.
- 5 samples per class, normalized to `[0, 1]` (the existing test already does this).
- Architecture: `gat`, `hiddenDim=32`, `numLayers=3`, `dropoutRate=0.0` (matches `skynet/config/crusher/model-engine.yaml`).
- Optimizer: AdamW, `lr=0.01`, `wd=0.0`.
- 200 epochs, `batchSize=5`.
- Weights: all 1.0 for this test (this is supervised classification, not RL — the RL story is Test 5).

**Assertions:**
- All 5 SPEED_UP samples predict `argmax == 0`.
- All 5 SPEED_DOWN samples predict `argmax == 1`.
- All 5 HOLD samples predict `argmax == 2`.
- The Jensen–Shannon divergence between SPEED_UP[0] and SPEED_DOWN[0] predictions is `> 0.4`.
- The Jensen–Shannon divergence between SPEED_DOWN[0] and HOLD[0] predictions is `> 0.2`.
- Predictions are deterministic: re-running the test with the same seed produces identical probability values to within `1e-6`.

**Why this test:** This is the *exact* failure mode the skynet user is trying to escape. If this test passes, skynet's claim "the model learns when tested in isolation" becomes "the model learns in production" because the production code path now goes through the same `Gnn.train()` and `Gnn.forward()` that this test uses.

### Test 5 — `tests/test_reward_weighted_policy.sn` (the RL existence proof)

**What it tests:** Policy-gradient style training with reward weights produces a reward-aware policy.

**Setup:** A 2-state, 2-action toy problem.
- State A (features `[1, 0]`): the "good" action is 0; reward `+5` if the agent picks 0, `-5` if it picks 1.
- State B (features `[0, 1]`): the "good" action is 1; reward `+5` if the agent picks 1, `-5` if it picks 0.

Generate a **deliberately imbalanced** training set: 80% of samples come from picking the *bad* action in each state (so the empirical action frequency is the wrong policy), and the labels record the actually-taken action. The weights array contains the per-sample reward (or its z-scored advantage).

**Procedure:**
- Train 300 epochs with the new weighted API.
- `forward(State A)` and `forward(State B)`.

**Assertions:**
- `forward(State A).probs[0] > 0.7`.
- `forward(State B).probs[1] > 0.7`.
- Without the weighted-loss API (i.e. if you re-run the test passing weights `[1.0, 1.0, ...]`), the unweighted training learns the *bad* policy because the bad action dominates the dataset. So this test must demonstrate that the weighted run does the *opposite* of what the unweighted run does on the same data.

**Why this test:** This is the smallest closed-form proof that `weights` actually steers the gradient. It is also the closest possible analogue to what skynet's REINFORCE pipeline needs. If this test passes, the skynet trainer can pass `advantages` to `Gnn.train()` and expect the gradient to follow the rewards.

### Test 6 — `tests/test_batch_composition_invariance.sn` (reformulated Phase G)

**Original form (superseded):** "train/predict shape contract" — build K `GraphTensors` with uniform N, M, F; call `Gnn.train(graphs, ...)`; call `Gnn.predictBatch(graphs)`; assert shape equality.

**Why that form was superseded:** it was written under the homogeneous-batch assumption that predated the heterogeneous-batching work in `docs/issues/heterogeneous-graph-batching.md`. Once `Gnn.train()` and `Gnn.forward()` both take `GraphTensors` by type, the "same shape at the train/predict boundary" property is enforced by the function signatures and is not a behavioural invariant that can be violated by a bug. The literal test is vacuous.

**Reformulation (what the test actually catches):** the canonical GNN batch-correctness invariant from the literature (PyG, DGL, OGB):

> A graph's prediction must be independent of which other graphs it is batched with. Formally: `forward(G_i) == forward(batchGraphs([G_1..G_K]))[i]` up to float32 precision for every `i`.

**What the reformulated test catches:**
- `batchIndex` / `mean_pool` leakage (nodes from graph B contributing to graph A's pooled embedding).
- Edge index offset bugs (edges in graph B pointing at row indices that land in graph A's slot).
- Padding contamination (zero-padded rows' post-activation bias affecting other graphs' aggregation).
- Normalization mode mismatch (e.g. `sum_normalized` computed per-batch instead of per-graph).
- Permutation dependence (reordering `[G1, G2, G3]` → `[G3, G1, G2]` changing predictions).
- **The Bug B1 spirit**, post-train: if train's internal pad-to-max forward saw something different from what predict sees for the same graph, Part C of the test would fail because the trained model's batched and solo predictions would diverge.

**Structure:**
- **Part A** — Three heterogeneous graphs (3, 5, 2 nodes). Compute solo `forward()` probs for each. Compute `forward(batchGraphs([...]))` once and slice per-graph rows. Assert row-by-row equivalence to `1e-5`.
- **Part B** — Rebuild the merged graph with the inputs permuted to `[G3, G1, G2]`. Assert the per-graph rows match the original solo probs at their new slots.
- **Part C** — Briefly train the model on the three graphs (exercising the pad-to-max / per-batch-upload path). Re-run Part A on the trained model. Assert the invariance still holds.

Sanity checks precede every assertion block: each solo prediction is a probability distribution, distinct graphs produce distinct predictions, each batched row sums to 1.

**Historical pointer:** the original "shape contract" framing is preserved in the audit trail as part of this file. It was a legitimate concern in the homogeneous-batch era; heterogeneous batching reframed the right invariant.

### Test 7 — `tests/test_save_load_roundtrip_under_train.sn` (Bug D + state corruption)

**What it tests:** Save → load → train → save → load → forward produces identical predictions to a freshly-trained never-saved model on the same data.

**Procedure:** Mirrors `tests/test_train_save_infer.sn` but with the new API and stricter assertions.

**Assertions:** Predictions must match within `1e-6` between the save/load path and the no-save path. The set of trainable parameters in the loaded model must equal the set of trainable parameters in the saved model.

**Why this test:** Skynet's champion/challenger ratchet depends entirely on save/load preserving model state. If save/load silently drops a layer (or a bias, or the classifier head), every promotion is a regression and the user will never know.

---

## Part 4 — Structured metrics this package must expose

**Principle: the diagnostic data must be queryable by the consumer, not buried in `fprintf(stderr, ...)`.**

The consumer (skynet) has a metrics store, a Postgres schema, and a dashboard. It does not have access to container stderr in production. This package therefore needs to make its diagnostics available *through the API surface*, not through process logs.

There are two layers:

### 4.1 — Per-call diagnostics, returned in `TrainResult` and `GnnOutput`

Already enumerated in §2.4 for `TrainResult`. For `GnnOutput`, no change is needed today — the existing `probs`, `logits`, `embedding` is sufficient as long as the consumer reads `probs.toDoubles()` and computes its own input-fingerprint statistics. Optional follow-up: add a `forwardWithDiagnostics()` variant that returns input mean/std/hash too, so the consumer doesn't have to recompute.

### 4.2 — Optional named metric callback

For consumers that want continuous, train-by-train metric streaming (skynet does), provide a simple callback registration:

```sindarin
fn setTrainMetricCallback(callback: fn(name: str, value: double, labels: StringField[]): void): void
```

The C side calls back into the Sindarin callback once per epoch with `(name, value, labels)` tuples like:
- `("train_loss", lossValue, [{"epoch", "5"}])`
- `("grad_norm_l2", gradNorm, [{"epoch", "5"}])`
- `("param_norm_before", normBefore, [{"layer", "0"}, {"kind", "weight"}])`
- `("param_norm_after", normAfter, [{"layer", "0"}, {"kind", "weight"}])`
- `("weight_sum_in", sumWeights, [])`

The consumer registers a callback that forwards into its existing `MetricsClient` (skynet's `metrics_client.sn`). The package itself is unaware of how the metrics are stored; it just emits.

**Implementation deviation (Phase J):** the shipped signature is the two-argument form

```sindarin
native fn sn_graph_set_train_metric_callback(cb: fn(str, double): void): void
native fn sn_graph_clear_train_metric_callback(): void
```

without the `labels: StringField[]` parameter. Rationale:

1. `StringField` is not in the SDK yet and the cost of building it for one consumer is outsized versus the value.
2. Callers can pack the label values into the metric name string (`"param_norm_before/layer_0/weight"` in place of the structured `[{"layer","0"},{"kind","weight"}]`). This is what skynet's `metrics_client.sn` already does for its own metric registry.
3. The two-arg form is forward-compatible: a future `_v2` entry point with `StringField[]` can be added alongside without breaking this surface.

The metric **set** shipped in Phase J matches the doc list above minus the labels: `train_loss`, `grad_norm_l2`, `weight_sum_in`, `weight_variance_in`, `input_mean`, `input_std`, `accuracy`, `param_norm_before/{i}`, `param_norm_after/{i}`, `param_max_abs_delta/{i}`. Per-param metrics use `{i}` in the name because `{i}` indexes `Gnn.parameters()` and is the contract that lets the caller correlate emissions back to `TrainResult.paramNormBefore[i]` etc.

Behaviour: the callback is deep-copied on registration (so it survives across multiple `train()` calls), invoked synchronously from Sindarin-side `Gnn.train()` at epoch boundaries and once more at the end of the train call for the scalar and per-parameter fields, and cleanly detached via `sn_graph_clear_train_metric_callback()`.

Regression guard: `tests/test_train_metric_callback.sn`.

**Why this matters:** with this in place, skynet's Tier 2 #8 (per-train weight L2 trajectory) and Tier 2 #9 (per-batch input/label/weight summary) are automatic. The skynet-side trainer only has to register a callback and the dashboard panels light up.

### 4.3 — Diagnostic accessors

Add three free functions in `tensor.sn`:

```sindarin
native fn sn_graph_last_train_loss_curve(): double[]
native fn sn_graph_last_train_grad_curve(): double[]
native fn sn_graph_last_train_input_stats(): double[]   # [mean, std, min, max]
```

Cheap to implement (just expose the data the C side already collects in `g_train_loss` etc.) and cheap for callers to use. Skynet's Tier 2 #9 becomes a single call after each train.

---

## Part 5 — Order of operations inside this package

1. **Land Test 3 (`test_weight_update_proven.sn`) against the current code first.** It will fail. We need the failing baseline so we can prove the fix worked.
2. **Fix Bug B3 (weight updates).** This is the prerequisite for everything else. The existing `weight-update-failure.md` work plan applies here. Acceptance: Test 3 passes deterministically.
3. **Land the new `sn_graph_train` signature with `weights`.** Wire it through `tensor.sn` and `gnn.sn`. Default to all-ones internally if a NULL is passed for safety, but make the Sindarin signature non-optional so callers cannot accidentally drop it. Acceptance: Test 1 (`test_weighted_loss.sn`) passes.
4. **Replace the self-loop template builder in `Gnn.train()`.** Use Shape A from §2.2: accept a list of `GraphTensors` and batch internally. Move the (currently dead) `batchGraphsToTensors()` logic from `skynet/src/trainer/trainer.sn:90-149` into the package, where it gets a single test instead of being copy-pasted across consumers. Acceptance: Test 2 (`test_real_graph_topology.sn`) passes.
5. **Extend `TrainResult` to carry per-train diagnostics.** §2.4. No new logic — just plumb the data the C side already has into the returned struct. Acceptance: Test 3 reads them and they match externally measured values.
6. **Land Test 4 (`test_crusher_policy_e2e.sn`).** This is the production-shape proof. If 1-5 are right, it passes. If it doesn't, debug *here*, in the 200-line test, not in the 9-service skynet cluster.
7. **Land Test 5 (`test_reward_weighted_policy.sn`).** Proves policy gradient works.
8. **Land Tests 6 and 7.** Regression guards.
9. **Add the metric callback API (§4.2) and diagnostic accessors (§4.3).** These do not block the core fix — they unblock the consumer's observability story.
10. **Tag a release.** Bump `sindarin-pkg-tensor` version. Skynet pulls it in via `sn --install` (per `skynet/CLAUDE.md` `!!! PACKAGE CHANGES !!!`).

The whole sequence is mechanically verifiable: each step has an integration test that ratchets the package one step closer to the production scenario, and step 10 cannot happen until 1-9 are green.

---

## Part 6 — How this maps onto skynet (the inheritance contract)

Once this package is at the state defined in Parts 2-4, the skynet golden-path doc reduces to a much smaller scope. Specifically:

| Skynet golden-path Tier 0 item | Status after this package is fixed |
|---|---|
| #1 Make `Gnn.train()` accept and apply per-sample weights | **DONE in package.** Skynet just passes `advantages`. |
| #2 Train on real graphs | **DONE in package.** Skynet deletes `trainer.sn:266-281` mean-pool path and calls the new `Gnn.train(graphs, ...)`. Five lines. |
| #3 Decide on rebalancing | Stays in skynet, but with weighted loss working it likely just becomes "remove the SQL `PARTITION BY` and let weights do the work". |
| #4 Land tensor weight-update fix | **DONE in package.** |

| Skynet Tier 1 item | Status after this package is fixed |
|---|---|
| #5 Log full prediction distribution | Trivially available via `forward().probs.toDoubles()`. Skynet just logs it. |
| #6 Per-cycle input fingerprint | Optional `forwardWithDiagnostics()` (§4.1) makes this a one-liner. |
| #7 Continuous canonical-pair JSD probe | Use `Gnn.predictBatch()` (§2.5) and `distributionDivergence()` (§2.5). One function call. |

| Skynet Tier 2 item | Status after this package is fixed |
|---|---|
| #8 Per-train parameter trajectory | Returned in `TrainResult.paramNormBefore/After/MaxAbsDelta` (§2.4). No work. |
| #9 Per-batch input/label/weight summary | Returned in `TrainResult.weightSumIn/weightVarianceIn/inputMean/inputStd` (§2.4) and via the metric callback (§4.2). |
| #10 Train/predict shape contract assertion | Already enforced by the package (Test 6). Skynet inherits it for free. |

| Skynet Tier 3 item | Status after this package is fixed |
|---|---|
| #11 Backtest confusion matrix | Stays in skynet. Easy now that `forward()` is reliable. |
| #12 Backtest determinism check | Stays in skynet. Test 4's determinism assertion proves the package is deterministic for a given seed; skynet just has to assert the same. |
| #13 Promotion evidence bundle | Stays in skynet. The pieces it needs (paramNorm, gradCurve, input stats) are now available from `TrainResult`. |

| Skynet Tier 4 item | Status after this package is fixed |
|---|---|
| #14 Wrap "model can learn policy" integration test as a service | The test is *Test 4* in this doc. Skynet's job is to call into it via FFI or replay it as a probe — minimal new logic. |
| #15 Synthetic eval cycle | Trivial with `Gnn.predictBatch()`. |

| Skynet Tier 5 item | Status after this package is fixed |
|---|---|
| #16-#20 Dashboard panels | Pure skynet UI work, fed by the metrics emitted via §4.2 callback. |

**Net result: of the 20 items in skynet's golden-path plan, 11 of them (#1, #2, #4, #5, #6, #7, #8, #9, #10, plus partial credit for #14, #15) become trivial-to-zero work in skynet once this package is fixed.** Almost everything skynet's user has been struggling with comes down to the four bugs in this document.

That is what the user meant by "If we get the tensor package right, there is a fair bit in skynet that should follow naturally."

---

## Part 7 — Acceptance checklist

This document is closed when:

- [x] Test 1 `tests/test_weighted_loss.sn` passes. (Phase B)
- [x] Test 2 `tests/test_real_graph_topology.sn` passes. (pre-existing; now part of the 19/19 suite)
- [x] Test 3 `tests/test_weight_update_proven.sn` passes across gat/gcn/sage. (Phase A)
- [x] Test 4 `tests/test_crusher_policy_e2e.sn` passes deterministically; determinism rerun matches to 1e-6. (Phase H)
- [x] Test 5 `tests/test_reward_weighted_policy.sn` passes, including the contrast check that the unweighted run learns the wrong policy. (Phase E)
- [x] Test 6 `tests/test_batch_composition_invariance.sn` passes. (Phase G — literal Test 6 superseded; see §Part 3 Test 6 for rationale)
- [x] Test 7 `tests/test_save_load_roundtrip_under_train.sn` passes with 1e-6 tolerance, param-set equality, and continue-train vs never-saved reference identity. (Phase I)
- [x] `Gnn.train()` accepts a `weights: double[]` parameter and forwards it to the training path (`sn_graph_train_epoch` — the legacy `sn_graph_train` was deleted in Phase K).
- [x] `Gnn.train()` no longer constructs a self-loop template graph; it trains on caller-supplied topology. (Pre-existing; Bug B1 was fixed before this round, regression-guarded by Test 2.)
- [x] `TrainResult` exposes `paramNormBefore`, `paramNormAfter`, `paramMaxAbsDelta`, `weightSumIn`, `weightVarianceIn`, `inputMean`, `inputStd`, `lossCurve`, `gradNormCurve`. (Phase C)
- [x] `Gnn.predictBatch()` and `distributionDivergence()` exist and have unit tests. (Phase F)
- [x] A metric callback API exists and the package emits at least the metric set defined in §4.2. (Phase J — ships as two-arg signature, see §4.2 "Implementation deviation")
- [x] `weight-update-failure.md` is closed and references this doc. (Phase A follow-up note added)
- [ ] `skynet/docs/issues/golden-path.md` Part 4 step 1 ("Fix the tensor package first") is marked DONE. (Blocked on consumer-side work)
- [ ] A new release of `sindarin-pkg-tensor` is tagged and pushed. (Phase K deliberate stop point — pending user approval for `git tag` / `git push`)

## Files referenced (post-resolution)

### Package sources
- `src/gnn.sn::Gnn.train()` — weights-aware, heterogeneous-batch, Phase C diagnostics, Phase J metric emissions
- `src/gnn.sn::Gnn.predictBatch` — Phase F addition
- `src/gnn.sn::sageForward` — Phase A fix (now uses `gnnMatmul`)
- `src/tensor.sn::TrainResult` — Phase C extended struct with per-train diagnostics
- `src/tensor.sn::batchGraphs` — pre-existing; used by Phase G's invariance test
- `src/tensor.sn::distributionDivergence` — Phase F addition (`l1`/`l2`/`kl`/`js`)
- `src/tensor.sn::sn_graph_set_train_metric_callback` / `_clear_` / `_emit_` — Phase J callback surface
- `src/tensor.sn.c::sn_graph_train_epoch` — the training entry point (legacy `sn_graph_train` was deleted in Phase K)
- `src/tensor.sn.c::sn_tensor_gnn_matmul` — correct weight layout path used by every GNN layer forward + classifier head (Phase A fix ensured sage uses it too)

### Tests (all passing, 19/19)
- `tests/test_weighted_loss.sn` — Phase B (Test 1)
- `tests/test_real_graph_topology.sn` — pre-existing (Test 2)
- `tests/test_weight_update_proven.sn` — Phase A (Test 3, 126 assertions)
- `tests/test_crusher_policy_e2e.sn` — Phase H (Test 4, strict replacement)
- `tests/test_reward_weighted_policy.sn` — Phase E (Test 5)
- `tests/test_batch_composition_invariance.sn` — Phase G (Test 6 reformulated)
- `tests/test_save_load_roundtrip_under_train.sn` — Phase I (Test 7, strict replacement)
- `tests/test_train_result_diagnostics.sn` — Phase C cross-check
- `tests/test_predict_batch_and_divergence.sn` — Phase F unit test
- `tests/test_train_metric_callback.sn` — Phase J unit test
- `tests/exploratory/repro_array_reassign.sn` — regression guard for compiler issue #50

### Docs
- `docs/issues/weight-update-failure.md` — Bug B3, with the Phase A sage follow-up
- `docs/issues/heterogeneous-graph-batching.md` — the pad-to-max / per-batch upload architecture
- `docs/issues/compiler-problems.31.md` issue #50 — double-free bug found during Phase C, now fixed upstream
- `skynet/docs/issues/golden-path.md` — the consumer-side audit this doc unblocks
