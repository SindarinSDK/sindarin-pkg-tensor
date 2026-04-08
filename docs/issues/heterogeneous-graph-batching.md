# Heterogeneous-graph batching

## Status: RESOLVED — pad-to-max + per-batch upload, landed in `Gnn.train()`

## TL;DR

`Gnn.train()` now handles graphs with **variable `numNodes` and variable
`numEdges`** within a single multi-batch `train()` call. The only shape
precondition is that every graph share the same `featureDim`. Internally,
each sample is padded to `maxNodes = max(graphs[i].numNodes)`, the dense
adjacency and the pooling matrix are rebuilt per batch on the host side
and uploaded to the recorded ggml graph alongside features/labels/weights,
and the per-graph mean pool denominator uses the **real** node count so
padded slots do not dilute the embedding.

| Regime | Definition | Status |
|---|---|---|
| **A** | Every graph is single-node (1 node, self-loop only) | Works (Regime A consumers pay one identity matmul per layer for the simplification — small price) |
| **B** | Multi-node graphs, identical topology across the training set | Works |
| **C** | Multi-node graphs with *different* `numNodes`/`numEdges` per graph, multi-batch training | **Works.** This is the change documented here. |

The historical `numNodes` and `numEdges` precondition assertions in
`Gnn.train()` are gone. The only remaining shape check is on `featureDim`.

## Where the constraint used to come from

`Gnn.train()` builds a **static recorded forward graph** inside
`sn_graph_begin/end`. Before this change, that static graph baked in:

- node count and feature dimension
- adjacency matrix entries (from a single representative batch)
- `batchIndex` and the derived pooling matrix
- every shape that the recorded ggml ops were allocated against

Per-batch upload (`sn_graph_train_epoch` in `src/tensor.sn.c`) only
refreshed the *float values* in the input tensors (`nodeFeatures`,
`labels`, `weights`). It never modified the topology. So every batch had
to occupy exactly the shape the first batch was recorded with.

## How the limitation surfaced

The original bug was tracked in `docs/issues/ggml-issue.md`:
`tests/test_real_graph_topology.sn` failed to converge even after the
ggml backward-pass patches landed. The root cause turned out to be the
per-epoch within-batch shuffle in `Gnn.train()` permuting which graph's
features landed in which slot of the static topology, while the slot
topology and label position stayed fixed. The result was a contradictory
training task that locked the optimizer onto the maximum-entropy fixed
point.

The first fix was to replace the within-batch shuffle with an
across-batch shuffle (a no-op for single-batch training). The deeper
realization was that *the architecture only worked correctly if all
graphs in the training set were interchangeable across batch slots*,
which became the multi-batch precondition documented in earlier
revisions of this file.

The skynet consumer then drove the next step: skynet's
on-demand `GRAPH_BUILD` path produces graphs with variable node and edge
counts (3–9 nodes and 3–36 edges per the live DB), which immediately
tripped the multi-batch precondition. That made full Option 3 the
load-bearing fix.

## The implementation: pad-to-max + per-batch upload

### Caller side (`src/gnn.sn` `Gnn.train()`)

For each `train()` call:

1. Compute `maxNodes = max(graphs[i].numNodes)` across the whole training
   set.
2. Build three flat host buffers, indexed by sample:
   - `featuresHost`: `totalSamples * maxNodes * featureDim` doubles, with
     each sample's real features in the first `numNodes[i] * featureDim`
     slots and zeros in the padding tail.
   - `adjHost`: `totalSamples * maxNodes * maxNodes` doubles, each
     sample's slice holding its dense adjacency with the layer's
     normalization mode (`sum_normalized` for GCN, `mean` for SAGE,
     `sum` for GAT-during-record) **already applied at host build time**,
     and zeros in the padded rows/columns.
   - `realNodeCountHost`: `totalSamples` doubles (one per sample).
3. Build a **padded template batched graph** from the first
   `batchSize` graphs to fix the recorded forward graph's shape:
   - features: `(batchSize * maxNodes, featureDim)`
   - edgeIndex: a single self-loop per padded slot — values are
     overwritten per batch, only the *shape* matters at record time.
   - batchIndex: `[0, 0, …, 0, 1, …, 1, …, batchSize-1, …]` so
     `mean_pool` derives `num_graphs == batchSize` at record time.
4. Pass `featuresHost`, `adjHost`, `realNodeCountHost`, and
   `maxNodesPerGraph` into the new `sn_graph_train_epoch` entry point
   each epoch.

### C side (`src/tensor.sn.c`)

A new **per-batch tensor registry** tracks every adjacency tensor
allocated by `sn_tensor_sparse_aggregate` in record mode (one per
message-passing layer) and the single pool tensor allocated by
`sn_tensor_mean_pool` in record mode. The registry is reset in
`sn_graph_begin` and `sn_graph_end`.

`sn_graph_train_epoch` was extended with three new parameters
(`adjHost`, `realNodeCountHost`, `maxNodesPerGraph`) and a per-batch
assembly loop:

- For each of the `batchSize` sample slots in the batch:
  - copy the sample's padded feature slice into the batched
    `(batchSize * maxNodes, featureDim)` features buffer at the right
    block offset
  - copy the sample's `(maxNodes × maxNodes)` adjacency slice into the
    block-diagonal `(batchSize * maxNodes, batchSize * maxNodes)` adjacency
    buffer at the right diagonal block
  - write the sample's pool row: `pool[s, s*maxNodes + k] = 1/realCount[s]`
    for `k ∈ [0, realCount[s])`, zeros elsewhere
  - copy labels and the per-sample weight as before
- Upload the batched features/labels/weights to their tensors as before
- Upload the batched adjacency to **every** registered `ADJ` tensor (all
  layers share the same edges within a `train()` call so the data is the
  same)
- Upload the batched pool matrix to the registered `POOL` tensor

The previous "identity short-circuit" optimizations in
`sn_tensor_sparse_aggregate` and `sn_tensor_mean_pool` (skip the op
entirely when the adjacency is `I` / when each node is its own graph)
are gone. They depended on the template adjacency being representative
of every batch, which is no longer guaranteed. Regime A consumers (every
graph is a single self-looping node) pay one identity matmul per layer
in exchange for a uniform topology contract.

### Why no explicit node mask is needed

Padding nodes get zero features, zero adjacency rows/cols, and zero
pool-matrix columns. Tracing each architecture:

- **GCN / GAT-in-record**: layer = `matmul(x, W) + bias → aggregate →
  relu → (residual)`. After `aggregate`, padding rows are zero because
  `adj[pad, :] = 0`. The residual `+ x` keeps them zero (x was
  zero-padded). Padding rows stay zero through the entire forward pass.
- **SAGE**: layer = `aggregate(x) → matmul(agg, W) + bias → relu →
  (residual)`. Padding rows become `bias` after the `matmul + bias`
  step and the residual leaks `relu(bias_prev)` from the previous
  layer. **But** because `pool_mat[:, pad_cols] = 0`, the pooled
  embedding is independent of padding activations and the gradient
  backflow to padding activations is exactly zero. So even with nonzero
  padding internally, the loss and gradient-to-weights are correct.

## Remaining gaps

### True sparse GAT attention during training

`sn_tensor_attention_aggregate` in record mode currently falls back to
plain `"sum"` aggregation (`src/tensor.sn.c:1403`). GAT in training
therefore behaves identically to a sum-aggregation GNN — there is no
per-edge attention softmax, and the `attWeight` parameters are
unused on the gradient path. This was true before this change and
remains true after.

A future fix would replace the fallback with a real attention path —
either dense (compute attention scores between every pair of padded
nodes, mask out non-edges) or sparse (use ggml scatter ops if/when they
land). Out of scope for this work.

### Variable `featureDim` across graphs

Still unsupported and likely never needed: feature dimension is set by
the consumer's feature key registry and is constant across a training
run by construction.

### Per-call cost of `maxNodes`

`maxNodes` is computed per `train()` call. If the training set has one
extreme outlier (e.g. 999 nodes when the median is 5), the static
recorded graph and per-batch buffers scale to `(batchSize * 999)²` for
the adjacency tensor, which can dominate memory. Consumers with very
heavy-tailed node-count distributions should consider dropping the tail
before passing into `train()`. No automatic mitigation today.

## Files referenced

- `src/gnn.sn` — `Gnn.train()` definition and the host-side padded
  buffer construction
- `src/tensor.sn` — updated `sn_graph_train_epoch` native signature
- `src/tensor.sn.c` — per-batch tensor registry, modified
  `sn_tensor_sparse_aggregate` and `sn_tensor_mean_pool` record paths,
  rewritten `sn_graph_train_epoch`
- `tests/test_heterogeneous_graph_batching.sn` — Regime C regression
  guard (variable `numNodes` and `numEdges`, multi-batch)
- `tests/test_real_graph_topology.sn` — original Regime B convergence
  test (must continue to pass)
- `docs/issues/ggml-issue.md` — the convergence-bug investigation that
  surfaced the limitation
- `docs/issues/golden-path.md` — the parent plan
