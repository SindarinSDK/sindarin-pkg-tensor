# Weight Update Failure in GNN Training

## Status: RESOLVED

GNN weight matrices now update during training. The fix shipped across
two fronts: a new weight-layout-aware matmul (`sn_tensor_gnn_matmul`)
and backward-pass contiguity patches in the ggml fork. See
`docs/issues/ggml-issue.md` for the ggml side and the commits listed
under "Resolution" below.

## Original symptom

GNN weight matrices were never updated during training. Only bias
tensors received gradient updates. The model appeared to train (loss
decreased) but weights were static — predictions were determined
entirely by random initialization.

## Root cause

The `sn_tensor_matmul` record-mode path pre-transposed weight matrices
on the host to work around a ggml `repeat_back` non-contiguous stride
assertion. This created a disconnected copy in a new pool slot. The
ggml optimizer trained the copy, but results were never written back
to the original weight tensor.

```
sn_graph_param(weight)  → registers original tensor with PARAM flag
matmul(x, weight)       → creates HOST-TRANSPOSED COPY in new pool slot
ggml_opt_fit / sn_train_step → trains the copy
readback                → reads original (unchanged)
```

Biases were unaffected because `ggml_add` consumes the bias tensor
directly with no intermediate copy.

## Resolution

1. **`sn_tensor_gnn_matmul`** (commit `422cf5f`, exposed as
   `tensor.gnnMatmul()` in `src/tensor.sn`). Stores weights transposed
   at construction time (`ne[0]=inputDim`), so `ggml_mul_mat` consumes
   the weight PARAM tensor directly without any host-side transpose or
   intermediate pool slot. All GNN layers in `src/gnn.sn` use
   `gnnMatmul` (GCN, GraphSAGE, GAT forward passes and the classifier
   head).

2. **ggml backward-pass contiguity patches** (commits `6cdcdc9` →
   `de8613f`). Fix `ggml_repeat_back` and the TRANSPOSE/PERMUTE
   backward expansions to produce contiguous gradients. Without these
   patches the backward pass crashed on multi-node topologies; with
   them, gradients flow correctly through
   `cont(transpose(features)) → mul_mat → add bias → relu`. Consumed
   via the `vcpkg-overlay/ggml` port pinned to
   `RealOrko/ggml@master`. See `docs/issues/ggml-issue.md`.

3. **`Gnn.train()` rewrite** (commit `edbef76`). Replaced the
   self-loop template graph with real-topology training via
   `sn_graph_train_epoch`. This is the integration point that proves
   gradients flow end-to-end: `test_train_save_infer.sn` and
   `test_crusher_policy.sn` both assert predictions change after
   training, and `test_real_graph_topology.sn` converges on a
   multi-node topology that previously crashed.

## Verification

- `ggml-issue.md` "Final status" records a total L2 norm delta of
  ≈1.63 across all 10 parameters after training — every weight and
  bias moves significantly.
- `tests/test_real_graph_topology.sn` converges (loss ≈0.034, JSD ≈0.51)
  on two graphs with identical features but different edge structure —
  impossible if weights were frozen.
- `tests/test_crusher_policy.sn` passes 15/15 with predictions that
  change from their pre-train values.

## Historical notes (preserved for context)

The original investigation considered two alternatives to `gnnMatmul`:

- **Single-context approach** — use one ggml context for params + compute
  to avoid cross-context allocation issues. Not needed; the
  `gnnMatmul` + ggml-patch approach kept the param/compute context
  split working.
- **Manual backward pass** — skip `ggml_build_backward_expand` and
  implement gradients by hand for the small set of ops used. Not
  needed; the upstream backward path works once
  `repeat_back`/transpose/permute produce contiguous grads.

Both alternatives remain viable escape hatches if the ggml patches
ever need to be reverted.
