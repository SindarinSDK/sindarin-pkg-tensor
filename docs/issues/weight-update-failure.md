# Weight Update Failure in GNN Training

## Status: OPEN — Root cause found, fix in progress

## Summary

GNN weight matrices have never been updated during training. Only bias tensors receive gradient updates. The model appeared to train (loss decreased) but weights were static — predictions were determined entirely by random initialization.

## Root Cause

The `sn_tensor_matmul` record-mode path pre-transposes weight matrices on the host to work around a ggml `repeat_back` non-contiguous stride assertion. This creates a disconnected copy in a new pool slot. The ggml optimizer trains the copy, but results are never written back to the original weight tensor.

```
sn_graph_param(weight)  → registers original tensor with PARAM flag
matmul(x, weight)       → creates HOST-TRANSPOSED COPY in new pool slot
ggml_opt_fit / sn_train_step → trains the copy
readback                → reads original (unchanged)
```

Biases work because `ggml_add` uses the bias tensor directly — no copy.

## Evidence

Diagnostic test (`tests/test_train_diagnostics.sn`) shows:
- Weight norms identical before/after 1000 epochs of training
- Bias at pool 6 changes (0.000 → -0.009)
- All weight pools unchanged (delta = 0.000000000)

## Fix Approach: `gnnMatmul`

Added `sn_tensor_gnn_matmul` which stores weights transposed (ne[0]=inputDim) so `ggml_mul_mat` works without any transpose in the graph. Weight PARAM tensors are directly in the computation graph — gradients should flow to them.

Changes made:
- `tensor.sn`: Added `gnnMatmul()` method and `sn_tensor_gnn_matmul` native fn
- `tensor.sn.c`: Implemented `sn_tensor_gnn_matmul` for both record and direct modes
- `gnn.sn`: Weight creation swapped to `Tensor.zeros(outputDim, inputDim)`, all matmuls changed to `gnnMatmul`

## Remaining Blocker: Backend Compute

With `gnnMatmul`, the weight PARAM tensors are directly in the graph. Gradient accumulators are allocated. Loss gradient is seeded to 1.0. But:

1. **`ggml_backend_sched_graph_compute`**: Gradients are zero after compute. The sched may be calling `ggml_graph_reset` internally or splitting the graph in a way that breaks gradient flow.

2. **`ggml_backend_graph_compute` (via gallocr)**: Segfaults during compute. The combined forward+backward+AdamW graph (~100+ nodes) crashes when allocated purely through gallocr.

## Files Changed

- `sindarin-pkg-tensor/src/tensor.sn` — added gnnMatmul, norm
- `sindarin-pkg-tensor/src/tensor.sn.c` — sn_tensor_gnn_matmul, sn_tensor_norm, sn_tensor_layer_norm, debug prints in sn_train_step
- `sindarin-pkg-tensor/src/gnn.sn` — weight storage swapped, gnnMatmul calls, residual connections
- `sindarin-pkg-tensor/tests/test_train_diagnostics.sn` — per-epoch weight norm + gradient tracing
- `sindarin-pkg-tensor/tests/test_layernorm_overfit.sn` — 2-sample overfit test
- `sindarin-pkg-tensor/tests/test_opt_fit.sn` — ggml_opt_fit weight update test

## Next Steps

1. **Debug the gallocr segfault**: The crash happens in `ggml_backend_graph_compute` after `ggml_gallocr_alloc_graph`. Need to determine if it's a graph size issue, allocation overlap, or ggml bug.

2. **Alternative: single-context approach**: Instead of separate param_ctx and compute_ctx, use ONE ggml context for everything. This avoids cross-context allocation issues.

3. **Alternative: manual backward**: Skip `ggml_build_backward_expand` entirely. Implement manual gradient computation for the small set of ops used (mul_mat, add, relu, softmax, cross-entropy). This gives full control over gradient flow.

## Skynet Changes (not yet committed)

- `skynet/src/orchestrator/orchestrator.sn` — abs-loss champion ratchet
- `skynet/src/executor/executor.sn` — skip guardrails in paper mode
- `skynet/src/reward/reward_engine.sn` — multi_alignment reward function
- `skynet/config/crusher/reward-engine.yaml` — multi-alignment config
- `skynet/config/crusher/model-engine.yaml` — epoch/patience config
