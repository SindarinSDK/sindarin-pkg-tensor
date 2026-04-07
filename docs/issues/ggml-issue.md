# ggml backward-pass contiguity assertions block multi-node GNN training

## Status: RESOLVED — patched in fork (RealOrko/ggml@sn-pkg-tensor)

The original ggml-side crashes are fixed by the two patches documented
below. The "remaining convergence" issue noted earlier was tracked down
to a separate Sindarin-side bug in the training driver (per-epoch
within-batch shuffle scrambling the static-topology slot binding) and
fixed in `src/gnn.sn`. See "Final status" at the end of this document
and `docs/issues/heterogeneous-graph-batching.md` for the architectural
precondition that the fix surfaces.

## Background

While implementing the per-batch training driver for `Gnn.train()` (Step 2c
of `docs/issues/golden-path.md`), the multi-node graph topology test
(`tests/test_real_graph_topology.sn`) hit a crash inside ggml's backward
pass that the existing fix in upstream RealOrko/ggml does not cover.

The path that triggers it is the in-graph transpose used by
`sn_tensor_sparse_aggregate` and `sn_tensor_mean_pool` in record mode:

```c
struct ggml_tensor *gfeat_t = ggml_cont(g_record_ctx,
    ggml_transpose(g_record_ctx, gfeat));
struct ggml_tensor *result = ggml_mul_mat(g_record_ctx, gfeat_t, gadj);
```

This is necessary so that per-batch uploads to `gfeat` (the features input)
are reflected in the recorded forward graph. The previous behaviour was a
host-side pre-transpose that snapshots the features at record time,
breaking per-batch upload (see Step 2c.3 findings in
`docs/issues/golden-path.md`). The in-graph transpose is the correct fix —
but it lights up two stale assumptions in ggml's backward pass.

## Symptom 1 — `ggml_compute_forward_repeat_back` assertion

```
vendor/ggml/src/ggml-cpu/ops.cpp:1838: GGML_ASSERT(nb00 == sizeof(float)) failed
#3  ggml_compute_forward_repeat_back
#4  ggml_graph_compute_thread
#5  GOMP_parallel
#6  ggml_graph_compute
#7  ggml_backend_cpu_graph_compute
...
#10 ggml_opt_eval
```

`ggml_compute_forward_repeat_back_f32` (at `vendor/ggml/src/ggml-cpu/ops.cpp:1816-1872`)
asserts that its src0 is contiguous along dim 0 (`nb[0] == sizeof(float)`).
The TODO comment at line 1836 acknowledges the limitation:

```c
// TODO: support for transposed / permuted tensors
GGML_ASSERT(nb0  == sizeof(float));
GGML_ASSERT(nb00 == sizeof(float));
```

The upstream fix in commit `04cd197e` (`fix: ensure contiguous input to
repeat_back in backward pass`) wraps grad tensors in `ggml_cont` at the
four backward call sites that build `repeat_back`:

- `ggml.c:6367-6376` (ADD backward, src1 broadcast)
- `ggml.c:6412-6420` (MUL backward, src1 broadcast)
- `ggml.c:6475-6479` (REPEAT backward)
- `ggml.c:6520-6533` (MUL_MAT backward, when src0 differs in shape)

This is incomplete: any **other** call site that constructs a `repeat_back`
op (including future ones added by package authors or other ggml backward
expansions) will re-trip the same assertion. The bias broadcast in our
multi-layer GNN goes through a chain that doesn't always match the
patched call sites.

## Symptom 2 — `binary-ops` non-contiguous src0 assertion

After patching the repeat_back path, training proceeds further but hits a
different assertion in the same family:

```
vendor/ggml/src/ggml-cpu/binary-ops.cpp:59: GGML_ASSERT(nb00 == sizeof(src0_t)) failed
```

The assertion is in `apply_binary_op` (used by ADD, SUB, MUL, DIV):

```cpp
GGML_ASSERT( nb0 == sizeof(dst_t));
GGML_ASSERT(nb00 == sizeof(src0_t));
```

It fires because the **TRANSPOSE** and **PERMUTE** backward expansions
store their result without wrapping it in `ggml_cont`:

```c
case GGML_OP_TRANSPOSE: {
    if (src0_needs_grads) {
        ggml_add_or_set(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
    }
} break;
```

`ggml_transpose` returns a non-contiguous view (it just swaps `ne[0]`/`ne[1]`
and `nb[0]`/`nb[1]`). When the stored grad is later consumed as the **src0**
of another binary op (e.g. accumulating gradients via `ggml_add_or_set` →
`ggml_add_impl`), the contiguity assertion fires.

`ggml_permute` has the same issue.

## Patches (RealOrko/ggml@sn-pkg-tensor commit `4ebc6531`)

Two minimal additions to `vendor/ggml/src/ggml.c`:

### Patch 1 — `ggml_repeat_back` self-guards its src0

```c
struct ggml_tensor * ggml_repeat_back(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    // The CPU forward op asserts nb[0] == sizeof(type) — see TODO in
    // ggml_compute_forward_repeat_back_f32. Wrap non-contiguous src in
    // ggml_cont so the invariant holds regardless of caller. Stronger
    // than the per-call-site cont wraps in ggml_compute_backward.
    if (a->nb[0] != ggml_type_size(a->type)) {
        a = ggml_cont(ctx, a);
    }

    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);
    result->op     = GGML_OP_REPEAT_BACK;
    result->src[0] = a;
    return result;
}
```

This guarantees the forward-op invariant at the construction site. Future
callers don't need to know about it.

### Patch 2 — TRANSPOSE / PERMUTE backward stores contiguous grads

```c
case GGML_OP_PERMUTE: {
    if (src0_needs_grads) {
        // ...
        // Wrap in cont — ggml_permute produces a non-contiguous view
        // and downstream binary ops assert nb00 == sizeof(type).
        ggml_add_or_set(ctx, cgraph, isrc0,
            ggml_cont(ctx, ggml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3])));
    }
} break;
case GGML_OP_TRANSPOSE: {
    if (src0_needs_grads) {
        // Wrap in cont — ggml_transpose produces a non-contiguous view
        // and downstream binary ops assert nb00 == sizeof(type).
        ggml_add_or_set(ctx, cgraph, isrc0,
            ggml_cont(ctx, ggml_transpose(ctx, grad)));
    }
} break;
```

Stored gradients are always contiguous, so any downstream consumer
satisfies the binary-op contiguity assertion.

## Acceptance

After these two patches:

- `ggml_compute_forward_repeat_back` assertion: **gone**.
- `ggml-cpu/binary-ops.cpp` non-contiguous src0 assertion: **gone**.
- `tests/test_real_graph_topology.sn` no longer crashes — runs to
  completion through 200 epochs of training.
- All 9 single-node tests in the package still pass — no regressions.
- **Param updates verified**: a debug print of L2 norms before/after
  training shows every weight and bias moves significantly (total norm
  delta ≈ 1.63 across 10 params). Gradients flow correctly through the
  whole `cont(transpose(features)) → mul_mat → add bias → relu` chain.

## Final status — convergence bug was NOT in ggml

After the patches, `test_real_graph_topology.sn` no longer crashed but
also did not converge: loss stayed at `ln(2) ≈ 0.693`, post-train probs
collapsed to `[0.5, 0.5]`. The "remaining issue" section originally
listed three suspects (sign error in weighted-CE, wrong cont/transpose
values, GAT capacity collapse). All three were wrong.

The actual root cause was in `src/gnn.sn:334`, the per-epoch
`rng.shuffleDouble(perm)` in `Gnn.train()`. The shuffle permuted which
graph's *features* landed in which *slot* of the static batched
topology, but the topology itself is fixed at record time. When two
graphs share node-feature distributions but differ in edges (which is
exactly what `test_real_graph_topology.sn` was designed to expose),
shuffling alternated which embedding got paired with which label across
epochs. The model was being asked to predict opposite labels for
identical inputs every other epoch, locking the optimizer onto the
maximum-entropy `[0.5, 0.5]` fixed point — exactly the observed symptom.

**Verification**: temporarily commenting out the shuffle line and
re-running the test produced loss `0.0328`, graphA prob[0] `0.733`,
graphB prob[1] `0.99994`, JSD `0.367`. All assertions passed.

**Permanent fix**: replaced the within-batch shuffle with an
across-batch shuffle that permutes batch presentation order across
epochs while keeping the within-batch slot-to-graph binding stable. For
the topology test (1 batch), this collapses to a no-op. For
multi-batch training, it randomizes batch order without scrambling the
slot binding. See the new docstring on `Gnn.train` in `src/gnn.sn` and
the precondition assertion that fires when caller-supplied graphs
violate the static-topology requirement.

**Architectural precondition surfaced by this work**: the static-
batched-topology design fundamentally requires every graph in a single
`train()` call to share `numNodes` and `featureDim`, and additionally
share `numEdges` when `len(graphs) > batchSize`. This was always
implicitly true; the convergence failure made it explicit. See
`docs/issues/heterogeneous-graph-batching.md` for the long-term plan to
support variable-shape graph batches.

The ggml patches in this document are correct, complete, and necessary.
They are not the cause of any remaining package issue. This document is
closed.

## How the patch is consumed

The patches are committed to `RealOrko/ggml` on branch `sn-pkg-tensor`
at SHA `4ebc65314b17901daea2655e1525caa0efbce625`. The
`sindarin-pkg-tensor` repo consumes them via a vcpkg overlay port:

```
vcpkg-overlay/ggml/portfile.cmake     # vcpkg_from_github(REPO RealOrko/ggml, REF 4ebc6531...)
vcpkg-overlay/ggml/vcpkg.json         # port metadata
vcpkg-configuration.json              # overlay-ports = ["./vcpkg-overlay"]
```

`.github/workflows/ci.yml` and `.github/workflows/release.yml` were
updated to bootstrap vcpkg, run `vcpkg install` (which picks up the
overlay automatically via `vcpkg-configuration.json`), and copy the
installed static libs into `libs/<platform>/lib`. The previous inline
`git clone https://github.com/ggml-org/ggml.git && cmake ...` block was
removed.

To bump the patch: rebase / commit on `RealOrko/ggml@sn-pkg-tensor`,
push, recompute the SHA512 of the new tarball, and update
`vcpkg-overlay/ggml/portfile.cmake` (REF + SHA512).

## Upstreaming

The patches are minimal and self-contained. They could be PR'd against
upstream `ggml-org/ggml` directly — both are bug fixes with
straightforward justification (the existing TODO acknowledges the
non-contiguous-input limitation; the fix is to either remove the
limitation or guarantee the precondition at construction time).

For now, they live in the RealOrko fork to keep the change isolated and
to avoid blocking on upstream review.
