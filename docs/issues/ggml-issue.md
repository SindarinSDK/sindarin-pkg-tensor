# ggml backward-pass contiguity assertions block multi-node GNN training

## Status: PATCHED in fork (RealOrko/ggml@sn-pkg-tensor), partial fix

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

## Acceptance and known remaining issue

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

**However**: the model still doesn't converge for the multi-node case.
Loss stays at ~`ln(2) = 0.6931` (the random-classification baseline) even
after 200 epochs, and post-train predictions for both graphs converge to
roughly `[0.5, 0.5]`. This is a *separate* bug, not in ggml's contiguity
invariants. The most likely candidates:

1. **Sign error somewhere in the weighted-CE loss expression** (record
   mode branch of `sn_tensor_weighted_cross_entropy` in
   `src/tensor.sn.c`). The optimizer is making predictions slightly
   *worse* than random over time, which is the classic symptom of
   maximizing instead of minimizing.

2. **`ggml_cont` of a non-contiguous transposed view computes the wrong
   values.** The forward result has the right shape but maybe reads from
   the underlying buffer ignoring strides. Needs a hand-computed
   verification on a tiny example.

3. **The loss landscape for this specific test is degenerate.** The two
   training graphs have identical node features and only differ in edge
   structure; if the GAT attention falls back to sum-aggregate (which it
   does — see `sn_tensor_attention_aggregate` record-mode fallback at
   `tensor.sn.c:1133`, plus the unmoving `attWeight` params noted in
   Step 0 of `golden-path.md`), the model may not have enough capacity
   to discriminate them in 200 epochs.

The first cause is the most likely and the most actionable. It needs a
focused investigation in the next session.

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
