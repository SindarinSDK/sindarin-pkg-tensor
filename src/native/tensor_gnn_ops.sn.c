/* ==============================================================================
 * tensor_gnn_ops.sn.c — GNN aggregation ops (mean_pool, sparse_aggregate,
 *                       attention_aggregate)
 * ==============================================================================
 *
 * These are the only ops that register themselves with the per-batch upload
 * registry (via track_per_batch) — adjacency, pool matrix, and attention
 * mask tensors all have values that change per minibatch under the
 * heterogeneous batching contract. See docs/issues/heterogeneous-graph-batching.md.
 * ============================================================================== */

#include "tensor_internal.h"

RtTensor *sn_tensor_mean_pool(RtTensor *node_embeddings, RtTensor *batch_index)
{
    TPool *px = unwrap(node_embeddings);
    TPool *pb = unwrap(batch_index);

    int64_t num_nodes = px->ne[1];
    int64_t feat_dim  = px->ne[0];

    if (g_record_mode) {
        /* Differentiable mean pool using a pooling matrix.
         *
         * Heterogeneous-batching note: the pooling matrix VALUES are no
         * longer baked into the static graph. We allocate the pool tensor
         * with the (batchSize x num_nodes) shape and register it with the
         * per-batch upload registry. sn_graph_train_epoch rebuilds it each
         * batch using the REAL per-sample node counts so padded slots get
         * weight 0 and the per-graph mean uses the right denominator. The
         * `batch_index` argument fixes the shape but its values matter only
         * for shape derivation. See docs/issues/heterogeneous-graph-batching.md.
         *
         * The previous identity short-circuit (skip pool entirely when each
         * node is its own graph) is GONE — same reasoning as in
         * sn_tensor_sparse_aggregate. Regime A consumers pay one identity
         * matmul; in exchange the topology contract is uniform. */

        /* Determine num_graphs from batch_index template (used only to
         * derive the static pool_mat row count). */
        int64_t num_graphs = 1;
        for (int64_t i = 0; i < pb->n_elem; i++) {
            int64_t b = (int64_t)pb->data[i];
            if (b + 1 > num_graphs) num_graphs = b + 1;
        }

        /* Allocate the pool tensor with zero data — first batch upload overwrites. */
        int pool_idx = pool_alloc(num_nodes, num_graphs, 2);
        TPool *pool_mat = &g_pool[pool_idx];
        memset(pool_mat->data, 0, (size_t)(num_graphs * num_nodes) * sizeof(float));

        struct ggml_tensor *gpool = ggml_new_tensor_2d(g_param_ctx, GGML_TYPE_F32, num_nodes, num_graphs);
        ggml_set_name(gpool, "pool_mat");
        ggml_set_input(gpool);
        g_record_map[pool_idx] = gpool;

        /* Register for per-batch upload. */
        track_per_batch(pool_idx, PB_KIND_POOL, 0);

        /* Transpose features INSIDE the graph (not on host) so per-batch
         * uploads of node_embeddings are reflected in the recorded graph.
         * Requires the patched ggml_repeat_back that handles non-contiguous
         * input tensors via internal ggml_cont (see vendor/ggml patch). */
        struct ggml_tensor *gfeat = rec_tensor(node_embeddings);
        struct ggml_tensor *feat_t = ggml_cont(g_record_ctx,
            ggml_transpose(g_record_ctx, gfeat));
        ggml_set_name(feat_t, "embed_T");

        /* result = ggml_mul_mat(feat_t, gpool) → [feat_dim, num_graphs] */
        struct ggml_tensor *result = ggml_mul_mat(g_record_ctx, feat_t, gpool);
        return rec_wrap(result, feat_dim, num_graphs);
    }

    /* Find number of graphs (max batch index + 1) */
    int64_t num_graphs = 1;
    for (int64_t i = 0; i < pb->n_elem; i++) {
        int64_t b = (int64_t)pb->data[i];
        if (b + 1 > num_graphs) num_graphs = b + 1;
    }

    int idx = pool_alloc(feat_dim, num_graphs, 2);
    TPool *out = &g_pool[idx];

    float *count = (float *)calloc((size_t)num_graphs, sizeof(float));

    for (int64_t n = 0; n < num_nodes; n++) {
        int64_t b = (int64_t)pb->data[n];
        count[b] += 1.0f;
        for (int64_t f = 0; f < feat_dim; f++) {
            out->data[b * feat_dim + f] += px->data[n * feat_dim + f];
        }
    }

    for (int64_t g = 0; g < num_graphs; g++) {
        float c = count[g] > 0.0f ? count[g] : 1.0f;
        for (int64_t f = 0; f < feat_dim; f++) {
            out->data[g * feat_dim + f] /= c;
        }
    }

    free(count);
    return wrap_pool(idx);
}

RtTensor *sn_tensor_sparse_aggregate(RtTensor *features, RtTensor *edge_index,
                                     RtTensor *edge_weight, char *mode)
{
    TPool *px  = unwrap(features);
    TPool *pei = unwrap(edge_index);
    TPool *pew = unwrap(edge_weight);

    int64_t num_nodes = px->ne[1];
    int64_t feat_dim  = px->ne[0];

    /* In record mode, build dense adjacency matrix for differentiable aggregation.
     *
     * Heterogeneous-batching note: the adjacency VALUES are no longer baked
     * into the static graph at record time. Instead we register the adjacency
     * tensor with the per-batch upload registry, and sn_graph_train_epoch
     * rebuilds + uploads the batched block-diagonal adjacency for each batch.
     * This is what lets training handle graphs with variable numNodes/numEdges
     * inside one train() call. The `edge_index` and `edge_weight` arguments
     * passed here describe the FIRST batch only — they fix the SHAPE of the
     * recorded graph but not the values. See docs/issues/heterogeneous-graph-batching.md.
     *
     * The previous "identity short-circuit" optimization (skip adj entirely
     * when adjacency is exactly I) is GONE: it depended on the template
     * adjacency being representative of every batch, which is false now.
     * Regime A consumers (every graph is a single self-looping node) pay
     * one identity matmul per layer for the simplification — small price. */
    if (g_record_mode) {
        /* Allocate the adjacency tensor in g_param_ctx with the batched
         * (num_nodes x num_nodes) shape. Data is left as zeros — the first
         * batch upload in sn_graph_train_epoch overwrites it. */
        int adj_idx = pool_alloc(num_nodes, num_nodes, 2);
        TPool *adj = &g_pool[adj_idx];
        memset(adj->data, 0, (size_t)(num_nodes * num_nodes) * sizeof(float));

        struct ggml_tensor *gadj = ggml_new_tensor_2d(g_param_ctx, GGML_TYPE_F32, num_nodes, num_nodes);
        ggml_set_name(gadj, "adj");
        ggml_set_input(gadj);
        g_record_map[adj_idx] = gadj;

        /* Register for per-batch upload. The mode determines how
         * sn_graph_train_epoch normalizes the assembled adjacency. */
        track_per_batch(adj_idx, PB_KIND_ADJ, parse_agg_mode(mode));

        /* Transpose features INSIDE the graph (not on host) so per-batch
         * uploads of `features` are reflected in the recorded graph.
         * Requires the patched ggml_repeat_back. */
        struct ggml_tensor *gfeat = rec_tensor(features);
        struct ggml_tensor *gfeat_t = ggml_cont(g_record_ctx,
            ggml_transpose(g_record_ctx, gfeat));
        ggml_set_name(gfeat_t, "feat_T");

        struct ggml_tensor *result = ggml_mul_mat(g_record_ctx, gfeat_t, gadj);
        return rec_wrap(result, feat_dim, num_nodes);
    }

    int64_t num_edges = pei->ne[0] > pei->ne[1] ? pei->ne[1] : pei->ne[0];

    /* edge_index: [2, num_edges] — created via fromDoubles(data, rows=2, cols=num_edges)
     * pool layout: ne[0]=cols=num_edges, ne[1]=rows=2
     * data[i] = src[i], data[ne[0]+i] = dst[i] */
    if (pei->ne[1] == 2) {
        /* Standard: [2 rows, num_edges cols] — ne[0]=num_edges, ne[1]=2 */
        num_edges = pei->ne[0];
    } else {
        /* Transposed: [num_edges rows, 2 cols] — ne[0]=2, ne[1]=num_edges */
        num_edges = pei->ne[1];
    }

    int idx = pool_alloc(feat_dim, num_nodes, 2);
    TPool *out = &g_pool[idx];

    for (int64_t i = 0; i < num_edges; i++) {
        int64_t s, d;
        if (pei->ne[1] == 2) {
            /* Standard: row 0 at offset 0, row 1 at offset num_edges */
            s = (int64_t)pei->data[i];
            d = (int64_t)pei->data[num_edges + i];
        } else {
            /* Transposed: each row is (src, dst) pair */
            s = (int64_t)pei->data[i * 2];
            d = (int64_t)pei->data[i * 2 + 1];
        }
        float w = pew->data[i];
        for (int64_t f = 0; f < feat_dim; f++) {
            out->data[d * feat_dim + f] += px->data[s * feat_dim + f] * w;
        }
    }

    /* Normalize if mean mode */
    if (strcmp(mode, "mean") == 0 || strcmp(mode, "sum_normalized") == 0) {
        float *count = (float *)calloc((size_t)num_nodes, sizeof(float));
        for (int64_t i = 0; i < num_edges; i++) {
            int64_t d;
            if (pei->ne[1] == 2) {
                d = (int64_t)pei->data[num_edges + i];
            } else {
                d = (int64_t)pei->data[i * 2 + 1];
            }
            count[d] += 1.0f;
        }
        for (int64_t n = 0; n < num_nodes; n++) {
            float c = count[n] > 0.0f ? count[n] : 1.0f;
            for (int64_t f = 0; f < feat_dim; f++) {
                out->data[n * feat_dim + f] /= c;
            }
        }
        free(count);
    }

    return wrap_pool(idx);
}

RtTensor *sn_tensor_attention_aggregate(RtTensor *features, RtTensor *edge_index,
                                        RtTensor *edge_weight, RtTensor *att_src, RtTensor *att_dst)
{
    TPool *px  = unwrap(features);
    TPool *pei = unwrap(edge_index);
    TPool *pas = unwrap(att_src);
    TPool *pad = unwrap(att_dst);

    int64_t num_nodes = px->ne[1];
    int64_t feat_dim  = px->ne[0];
    int64_t num_edges = pei->ne[0];

    if (g_record_mode) {
        /* Phase L: attention as a differentiable ggml subgraph.
         *
         * The previous record-mode implementation fell back to
         * sparse_aggregate with "sum", which (a) silently degraded
         * GAT training to sum aggregation and (b) never referenced
         * the attention parameters, so they received no gradients
         * and stayed at their init value forever. At inference time
         * the full attention code then ran with zero attention
         * weights, producing a degenerate uniform-attention pass
         * that looked like mean aggregation. Net effect: `arch:
         * "gat"` was sum-train / mean-infer with no learned edge
         * weighting. Closed out as part of
         * docs/issues/golden-path.md Phase L.
         *
         * Attention weights are stored as TWO separate parameter
         * tensors (att_src and att_dst, each shape (feat_dim, 1))
         * rather than one (feat_dim*2, 1) tensor split by view.
         * ggml's VIEW op backward path does accumulate into the
         * source via ggml_acc_or_set, but empirically (verified by
         * test_weight_update_proven.sn Phase L) gradients from the
         * forward chain `view → reshape_2d → mul_mat → …` did not
         * reach the PARAM tensor behind the view — the updated
         * values after training exactly equaled the init values,
         * max-abs-delta = 0. Splitting the parameter sidesteps this
         * entirely: each half is a standalone ggml PARAM tensor
         * used directly in mul_mat, and gradients flow through the
         * standard backward path.
         *
         * Implementation sketch (all ggml ops with known backward):
         *
         *   1. src_scores[s] = sum_f features[f, s] * att_src[f]
         *      via ggml_mul_mat(features, att_src).
         *   2. dst_scores[d] = sum_f features[f, d] * att_dst[f] same.
         *   3. Broadcast src_scores → h_src_rows[s, d] = src_scores[s]
         *      via ggml_repeat onto a (nn, nn) template (the mask).
         *   4. Reshape + broadcast dst_scores → h_dst_cols[s, d] =
         *      dst_scores[d] similarly.
         *   5. scores = h_src_rows + h_dst_cols.
         *   6. scores_leaky = relu(scores) - 0.2 * relu(-scores)
         *      (LEAKY_RELU has no backward in this fork).
         *   7. Per-destination softmax with a per-batch-uploaded
         *      attention mask: mask[d, s] = 0 for edges, -1e9 for
         *      non-edges. ggml_soft_max_ext with max_bias=0 adds the
         *      mask to the logits before softmax, so non-edge
         *      positions collapse to ~0 probability.
         *   8. out[f, d] = sum_s features[f, s] * alpha[s, d] via
         *      ggml_mul_mat(features_T, alpha), same contraction
         *      sparse_aggregate uses.
         *
         * The attention mask tensor is registered as PB_KIND_ATT_MASK
         * and uploaded per batch from att_mask_buf, which is derived
         * from adj_buf inside sn_graph_train_epoch (non-zero adj cell
         * → edge exists → mask cell = 0; else -1e9).
         */
        int64_t nn = num_nodes;
        int64_t fd = feat_dim;

        /* Per-batch uploaded edge indicator mask: 1.0 at edges, 0.0
         * elsewhere. Applied multiplicatively (not additively) so
         * non-edge positions zero out the attention weights after
         * softmax, followed by per-destination renormalization. */
        int mask_idx = pool_alloc(nn, nn, 2);
        TPool *mask_pool = &g_pool[mask_idx];
        memset(mask_pool->data, 0, (size_t)(nn * nn) * sizeof(float));
        struct ggml_tensor *gmask = ggml_new_tensor_2d(g_param_ctx, GGML_TYPE_F32, nn, nn);
        ggml_set_name(gmask, "att_edge_mask");
        ggml_set_input(gmask);
        g_record_map[mask_idx] = gmask;
        track_per_batch(mask_idx, PB_KIND_ATT_MASK, 0);

        /* Dedicated (nn, nn) zero tensor used ONLY as the shape
         * template for broadcast-add when building the score matrix.
         * Separate from gmask so gmask is referenced only once in the
         * forward graph. */
        int zero_idx = pool_alloc(nn, nn, 2);
        TPool *zero_pool = &g_pool[zero_idx];
        memset(zero_pool->data, 0, (size_t)(nn * nn) * sizeof(float));
        struct ggml_tensor *gzero = ggml_new_tensor_2d(g_param_ctx, GGML_TYPE_F32, nn, nn);
        ggml_set_name(gzero, "att_zero_base");
        ggml_set_input(gzero);
        g_record_map[zero_idx] = gzero;

        /* Reference the existing recorded tensors. Both att_src and
         * att_dst are registered as PARAM tensors via sn_graph_param
         * by Gnn.train(), so rec_tensor returns their ggml handles. */
        struct ggml_tensor *gfeat    = rec_tensor(features); /* ne[0]=fd, ne[1]=nn */
        struct ggml_tensor *gatt_src = rec_tensor(att_src);  /* ne[0]=fd, ne[1]=1 */
        struct ggml_tensor *gatt_dst = rec_tensor(att_dst);  /* ne[0]=fd, ne[1]=1 */

        /* Full Phase L attention subgraph.
         *
         * The attention mask is baked into the logits as a MODEST
         * additive bias (-10 for non-edges, 0 for edges) rather than
         * the naive -1e9. Bisected finding: values of -1e9 (or
         * anything large enough to hard-zero the softmax output at
         * non-edge positions) numerically kill the gradient of the
         * softmax backward pass even at edge positions — the
         * `y*(dy - dot(y,dy))` formula produces effectively-zero
         * outputs once the softmax concentrates to near-0/near-1.
         * -10 is still enough to suppress non-edge attention
         * (exp(-10) ≈ 4.5e-5 per position) while preserving a
         * meaningful gradient path back to att_src / att_dst.
         *
         * Other hard-learned constraints from the bisect:
         *  - ggml_soft_max_ext(a, mask, ...) with a per-batch-uploaded
         *    mask breaks gradient flow upstream; fold the mask into
         *    the logits manually via ggml_add and use plain
         *    ggml_soft_max instead. Mathematically identical when
         *    max_bias=0 and scale=1, because soft_max_ext adds the
         *    mask to the scaled logits (ops.cpp::soft_max_f32:5301).
         *  - The (nn, nn) score matrix is built via broadcast-add
         *    against a dedicated zero tensor (gzero), not the mask
         *    itself — referencing gmask more than once in the graph
         *    empirically correlated with gradient loss.
         */
        struct ggml_tensor *src_scores = ggml_mul_mat(g_record_ctx, gfeat, gatt_src); /* (nn,1) */
        struct ggml_tensor *dst_scores = ggml_mul_mat(g_record_ctx, gfeat, gatt_dst); /* (nn,1) */

        /* Build the (nn, nn) score matrix S[s,d] = src_scores[s] + dst_scores[d]
         * via broadcast-add. gzero is the dedicated zero-valued shape
         * template; gmask is kept pristine for its single use in the
         * softmax mask-fold below. */
        struct ggml_tensor *score_plus_src = ggml_add(g_record_ctx, gzero, src_scores);
        struct ggml_tensor *dst_row = ggml_reshape_2d(g_record_ctx, dst_scores, 1, nn);
        struct ggml_tensor *scores_t = ggml_add(g_record_ctx, score_plus_src, dst_row);

        /* LeakyReLU (manual composition — LEAKY_RELU has no backward
         * pass in this fork). */
        struct ggml_tensor *pos_part = ggml_relu(g_record_ctx, scores_t);
        struct ggml_tensor *neg_in   = ggml_neg(g_record_ctx, scores_t);
        struct ggml_tensor *neg_relu = ggml_relu(g_record_ctx, neg_in);
        struct ggml_tensor *neg_part = ggml_scale(g_record_ctx, neg_relu, 0.2f);
        struct ggml_tensor *scores_leaky = ggml_sub(g_record_ctx, pos_part, neg_part);

        /* Fold the mask into the logits manually. gmask has 0 for
         * edges, -1e9 for non-edges. Adding it to scores_leaky forces
         * non-edge positions to a very negative logit so the softmax
         * output at those positions is ~0 (correctly zeroing out
         * non-edge attention). */
        struct ggml_tensor *scores_masked = ggml_add(g_record_ctx, scores_leaky, gmask);

        /* Per-destination softmax over ne[0] (sources). Plain
         * ggml_soft_max (not soft_max_ext with mask) to preserve
         * gradient flow through the scores_masked chain. */
        struct ggml_tensor *alpha = ggml_soft_max(g_record_ctx, scores_masked);

        /* Aggregate: out[f, d] = sum_s features[f, s] * alpha[s, d]
         * Same contraction pattern sparse_aggregate uses at its tail. */
        struct ggml_tensor *gfeat_t = ggml_cont(g_record_ctx,
            ggml_transpose(g_record_ctx, gfeat));
        ggml_set_name(gfeat_t, "att_feat_T");
        struct ggml_tensor *result = ggml_mul_mat(g_record_ctx, gfeat_t, alpha);
        ggml_set_name(result, "att_result");
        return rec_wrap(result, fd, nn);
    }

    int idx = pool_alloc(feat_dim, num_nodes, 2);
    TPool *out = &g_pool[idx];

    /* Compute attention scores using split att_src/att_dst params.
     * Each param is a flat (feat_dim,) vector in memory. */
    float *scores = (float *)calloc((size_t)num_edges, sizeof(float));
    for (int64_t i = 0; i < num_edges; i++) {
        int64_t s = (int64_t)pei->data[i];
        int64_t d = (int64_t)pei->data[num_edges + i];
        float score = 0.0f;
        for (int64_t f = 0; f < feat_dim; f++) {
            score += px->data[s * feat_dim + f] * pas->data[f];
            score += px->data[d * feat_dim + f] * pad->data[f];
        }
        /* LeakyReLU */
        scores[i] = score > 0.0f ? score : score * 0.2f;
    }

    /* Softmax per destination node */
    float *max_score = (float *)malloc((size_t)num_nodes * sizeof(float));
    float *sum_exp   = (float *)calloc((size_t)num_nodes, sizeof(float));
    for (int64_t n = 0; n < num_nodes; n++) max_score[n] = -1e30f;

    for (int64_t i = 0; i < num_edges; i++) {
        int64_t d = (int64_t)pei->data[num_edges + i];
        if (scores[i] > max_score[d]) max_score[d] = scores[i];
    }
    for (int64_t i = 0; i < num_edges; i++) {
        int64_t d = (int64_t)pei->data[num_edges + i];
        scores[i] = expf(scores[i] - max_score[d]);
        sum_exp[d] += scores[i];
    }

    /* Weighted aggregation */
    for (int64_t i = 0; i < num_edges; i++) {
        int64_t s = (int64_t)pei->data[i];
        int64_t d = (int64_t)pei->data[num_edges + i];
        float alpha = scores[i] / (sum_exp[d] + 1e-8f);
        for (int64_t f = 0; f < feat_dim; f++) {
            out->data[d * feat_dim + f] += px->data[s * feat_dim + f] * alpha;
        }
    }

    free(scores);
    free(max_score);
    free(sum_exp);
    return wrap_pool(idx);
}

/* Note: pew is unused in record mode (edge weights are folded into the adj
 * upload inside sn_graph_train_epoch). Suppress the "unused variable" warning
 * here rather than at every call site. */
