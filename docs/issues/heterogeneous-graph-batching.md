# Heterogeneous-graph batching: architectural limitation and resolution options

## Status: KNOWN LIMITATION — deferred until a real-graph consumer arrives

## TL;DR

`Gnn.train()` requires every graph in a single `train()` call to share
the same `numNodes` and `featureDim`, plus the same `numEdges` when the
training set spans more than one batch. A runtime assertion in
`src/gnn.sn` enforces this. The package therefore supports three
operating regimes today:

| Regime | Definition | Status |
|---|---|---|
| **A** | Every graph is single-node (1 node, self-loop only) | Works fully — used by skynet crusher |
| **B** | Multi-node graphs, but every graph in the training set has identical topology (same `numNodes`, `numEdges`, edges) | Works |
| **C** | Multi-node graphs with *different* topology per graph | Only works when `totalSamples == batchSize` (one batch). Multi-batch is impossible without an architectural change |

This document records *why* the limitation exists, *who* it affects,
and *what* the resolution options look like when a consumer needs more.

## Where the constraint comes from

`Gnn.train()` builds a **static recorded forward graph** inside
`sn_graph_begin/end`. The static graph bakes in:

- node count and feature dimension
- adjacency matrix entries (edges + weights)
- `batchIndex` and the derived pooling matrix
- every shape that the recorded ggml ops were allocated against

Per-batch upload (`sn_graph_train_epoch` in `src/tensor.sn.c`) only
refreshes the *float values* in the input tensors (`nodeFeatures`,
`labels`, `weights`). It never modifies the topology. So every batch
must occupy exactly the shape the first batch was recorded with.

This is the right design for the cases the package was built for
(skynet crusher: every state observation is a 1-node graph with a
self-loop, so the topology is uniform by construction). It is the
*wrong* design for general GNN workloads where every input graph has
its own structure.

## How the limitation surfaced

The bug that exposed this was tracked in `docs/issues/ggml-issue.md`:
`tests/test_real_graph_topology.sn` failed to converge even after the
ggml backward-pass patches landed. The root cause turned out to be the
per-epoch within-batch shuffle in `Gnn.train()` permuting which graph's
features landed in which slot of the static topology, while the slot
topology and label position stayed fixed. The result was a
contradictory training task that locked the optimizer onto the
maximum-entropy fixed point.

The fix was to replace the within-batch shuffle with an across-batch
shuffle (which is a no-op for single-batch training). But the deeper
realization is that *the architecture only works correctly if all
graphs in the training set are interchangeable across batch slots*,
which is what this document captures.

## Who is affected

**Not affected (Regime A):**
- skynet crusher and any other RL-style training where each "graph" is
  a single state observation modelled as a 1-node graph
- any toy classification test where graphs are constructed from a fixed
  template

**Conditionally affected (Regime B):**
- Trainers where every input graph is constructed from the same scaffold
  (same node count, same edges, only features vary). Examples:
  per-frame sensor networks, fixed-topology meshes, fixed-arity scene
  graphs. These work today as long as the caller doesn't try to mix
  shapes within a single `train()` call.

**Hard-limited (Regime C):**
- Real GNN workloads on heterogeneous graphs: molecules (different
  atom counts), social graphs (different friend counts), citation
  networks, knowledge graphs.
- RL on variable-arity environments: multi-agent settings where the
  number of agents changes between episodes, environments with
  spawnable/destroyable entities.
- Curriculum learning that adds/removes nodes mid-training.

For Regime C consumers, the only workaround today is `batchSize == 1`
(one graph per batch), which defeats the purpose of mini-batching and
significantly slows training.

## Resolution options

| Option | Description | Cost | When to use |
|---|---|---|---|
| **1. Do nothing** | Document the precondition, assert at runtime, accept the limitation | Done (this doc + assertion in `gnn.sn`) | Until a real consumer needs more |
| **2. Re-record per batch** | Tear down `sn_graph_begin/end` and rebuild the static graph between every batch | Defeats the caching that motivated the design. Significantly slower. Loses optimizer state across rebuilds | Only as a quick escape hatch |
| **3. Pad-to-max + masking** | Pad every graph in a batch to the max node/edge count in the batch. Add a mask tensor that zeros padded contributions in attention/aggregation. Static graph holds the max shape | Standard playbook (PyTorch Geometric, DGL). Requires masking logic in `sn_tensor_sparse_aggregate`, `sn_tensor_mean_pool`, and the GAT attention path. Medium implementation cost | The right long-term answer |
| **4. Per-epoch-group recompile** | Bucket graphs by shape, run training in passes — one bucket at a time, rebuilding the static graph between buckets but reusing within. AdamW state must be persisted across rebuilds | Compromise between (2) and (3). Avoids per-batch rebuild cost but still loses some caching. Useful for curriculum schedules | Only if curriculum learning is the driver |

**Recommendation**: Option 1 stays in place until a concrete consumer
needs more. When that happens, jump to Option 3 (pad-to-max + masks).
Skip Option 2; it's a footgun. Skip Option 4 unless curriculum
learning is on the roadmap.

## Notes for future work on Option 3

If we end up implementing pad-to-max + masking, the touch points are:

- `src/gnn.sn` `Gnn.train()` — drop the precondition assertion;
  compute `maxNodes`, `maxEdges` per batch; pad host buffers; allocate
  a `nodeMask` and `edgeMask` tensor of the appropriate shape
- `src/tensor.sn.c` `sn_tensor_sparse_aggregate` (record mode) — accept
  a mask tensor and zero contributions where the mask is 0
- `src/tensor.sn.c` `sn_tensor_mean_pool` (record mode) — divide by the
  *unmasked* count per graph, not the static count
- `src/tensor.sn.c` `sn_tensor_attention_aggregate` — apply the mask
  before the softmax over neighbours
- `src/tensor.sn.c` `sn_graph_train_epoch` — upload the mask buffer
  alongside the feature/label/weight buffers each batch

The static graph would then hold tensors sized for the *worst-case*
shape across all training batches. Per-batch uploads would refresh
both the data and the mask. Existing single-shape consumers (Regimes A
and B) would pay only the cost of an always-1 mask, which is small.

This is non-trivial work but well-trodden: every production GNN
framework does it.

## Files referenced

- `src/gnn.sn` — `Gnn.train()` definition + precondition assertion
- `src/tensor.sn.c` — `sn_graph_begin/end`, `sn_graph_train_epoch`,
  `sn_tensor_sparse_aggregate`, `sn_tensor_mean_pool`,
  `sn_tensor_attention_aggregate`
- `docs/issues/ggml-issue.md` — the convergence-bug investigation that
  surfaced this limitation
- `docs/issues/golden-path.md` — the parent plan that this constraint
  is now part of
- `tests/test_real_graph_topology.sn` — the test that exposed the
  shuffle bug and proves the fix
