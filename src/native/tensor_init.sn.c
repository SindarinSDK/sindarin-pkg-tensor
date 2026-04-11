/* ==============================================================================
 * tensor_init.sn.c — weight initialization (Kaiming + small-scale)
 * ==============================================================================
 *
 * Uses a local xorshift64 PRNG rather than libc rand(). Two reasons:
 *   1. libc rand() is a global: any other caller (including dropout)
 *      shares state, so init becomes order-dependent.
 *   2. The old code called srand(time(NULL)) once per process, which
 *      made model weights jitter across runs at 1-second resolution —
 *      untrained inference was silently non-reproducible even when the
 *      caller thought they had fixed the seed. Seeded variants below
 *      + Gnn.createWithSeed in src/gnn.sn close that gap.
 * ============================================================================== */

#include "tensor_internal.h"

static uint64_t xorshift64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* Fill `pt` with Kaiming-uniform values drawn from the given PRNG
 * state. U(-bound, bound) where bound = sqrt(6 / fan_in). */
static void kaiming_fill(TPool *pt, uint64_t *state)
{
    int64_t fan_in = pt->ne[0];
    float bound = sqrtf(6.0f / (float)(fan_in > 0 ? fan_in : 1));
    for (int64_t i = 0; i < pt->n_elem; i++) {
        /* Take the high 24 bits and map to [0, 1). */
        uint64_t r = xorshift64(state);
        float u = (float)(r >> 40) * (1.0f / 16777216.0f);
        pt->data[i] = (2.0f * u - 1.0f) * bound;
    }
}

/* Unseeded Kaiming init. Lazy-seeds a process-local xorshift state
 * from time(NULL) on first use, then advances that state across
 * subsequent calls. Same observable "different every run" behaviour
 * as the previous srand(time(NULL)) path — just without polluting
 * libc rand() and without the 1-second-resolution correlation gap.
 * Callers that need reproducibility should use the seeded variant. */
RtTensor *sn_tensor_init_kaiming(RtTensor *t)
{
    static uint64_t g_state = 0;
    if (g_state == 0) {
        g_state = (uint64_t)time(NULL);
        /* xorshift64 degenerates at state 0, so substitute the
         * golden-ratio constant if time() returned 0. */
        if (g_state == 0) g_state = 0x9E3779B97F4A7C15ULL;
    }
    kaiming_fill(unwrap(t), &g_state);
    return t;
}

/* Seeded Kaiming init. Produces bit-identical weights for the same
 * seed across process invocations and machine rebuilds. */
RtTensor *sn_tensor_init_kaiming_seeded(RtTensor *t, long long seed)
{
    uint64_t state = (uint64_t)seed;
    if (state == 0) state = 0x9E3779B97F4A7C15ULL;
    /* Warm-up rounds so small/adjacent seeds decorrelate before we
     * start sampling. xorshift has a long period but short-seed
     * trajectories look correlated for the first few outputs. */
    xorshift64(&state);
    xorshift64(&state);
    xorshift64(&state);
    kaiming_fill(unwrap(t), &state);
    return t;
}

/* Fill `pt` with small-scale uniform values U(-bound, bound) where
 * bound = std * sqrt(3) so that the resulting empirical standard
 * deviation matches the requested `std`. Used for the policy output
 * layer in PPO actors — Andrychowicz et al. 2021 ("What Matters in
 * On-Policy Reinforcement Learning") established std=0.01 as the
 * standard for centring the action distribution near uniform at init,
 * which prevents the deterministic-policy collapse on imbalanced
 * action priors observed in skynet Phase 6/6.1/6.2 verification. */
static void small_scale_fill(TPool *pt, double std, uint64_t *state)
{
    float bound = (float)(std * 1.7320508075688772);  /* std * sqrt(3) */
    for (int64_t i = 0; i < pt->n_elem; i++) {
        uint64_t r = xorshift64(state);
        float u = (float)(r >> 40) * (1.0f / 16777216.0f);
        pt->data[i] = (2.0f * u - 1.0f) * bound;
    }
}

/* Unseeded small-scale init. Same lazy-seed strategy as
 * sn_tensor_init_kaiming. */
RtTensor *sn_tensor_init_small_scale(RtTensor *t, double std)
{
    static uint64_t g_state = 0;
    if (g_state == 0) {
        g_state = (uint64_t)time(NULL);
        if (g_state == 0) g_state = 0x9E3779B97F4A7C15ULL;
    }
    small_scale_fill(unwrap(t), std, &g_state);
    return t;
}

/* Seeded small-scale init. Produces bit-identical weights for the same
 * seed across runs. Same warm-up rounds as the Kaiming variant. */
RtTensor *sn_tensor_init_small_scale_seeded(RtTensor *t, double std, long long seed)
{
    uint64_t state = (uint64_t)seed;
    if (state == 0) state = 0x9E3779B97F4A7C15ULL;
    xorshift64(&state);
    xorshift64(&state);
    xorshift64(&state);
    small_scale_fill(unwrap(t), std, &state);
    return t;
}

/* Phase 3: orthogonal init (Saxe et al. 2014, "Exact solutions to the
 * nonlinear dynamics of learning in deep linear neural networks").
 * Produces a weight matrix whose rows are orthonormal, scaled by
 * `gain`. For policy/value heads in PPO the standard setup is:
 *   - policy head: gain = 0.01  (small-scale, centres π near uniform)
 *   - value head:  gain = 1.0
 *   - hidden weights: gain = sqrt(2) for ReLU, 1.0 for tanh
 *
 * Algorithm: modified Gram-Schmidt on rows.
 *   1. Fill the tensor with iid N(0, 1) samples (Box-Muller from xorshift)
 *   2. For each row r:
 *        - Subtract the projection onto every previously-orthonormalized row
 *        - Normalize to unit length and multiply by `gain`
 *
 * Tensor shape convention: the caller creates a tensor via
 * Tensor.zeros(rows, cols) which stores ne[0] = cols, ne[1] = rows;
 * the row-major data layout is `data[r * cols + c]`. Gram-Schmidt on
 * rows produces up to min(rows, cols) orthogonal vectors — for the
 * `rows > cols` degenerate case the last `rows - cols` rows will be
 * numerically near-zero after orthogonalization. In practice every
 * real use case (policy/value head, classifier head, GCN layer) has
 * `rows <= cols`, and we deliberately do not try to handle the
 * pathological case transparently here. */
static float sample_gaussian(uint64_t *state)
{
    /* Box-Muller: two uniforms in (0, 1) -> one standard normal */
    uint64_t r1 = xorshift64(state);
    uint64_t r2 = xorshift64(state);
    float u1 = (float)(r1 >> 40) * (1.0f / 16777216.0f);
    float u2 = (float)(r2 >> 40) * (1.0f / 16777216.0f);
    /* Clamp u1 away from 0 so logf is finite. */
    if (u1 < 1e-7f) u1 = 1e-7f;
    float mag = sqrtf(-2.0f * logf(u1));
    return mag * cosf(2.0f * 3.14159265358979323846f * u2);
}

static void orthogonal_fill(TPool *pt, double gain, uint64_t *state)
{
    int64_t cols = pt->ne[0];
    int64_t rows = pt->ne[1];

    /* Step 1: fill with iid N(0, 1) Gaussian samples. */
    for (int64_t i = 0; i < pt->n_elem; i++) {
        pt->data[i] = sample_gaussian(state);
    }

    /* Step 2: modified Gram-Schmidt on rows to produce a UNIT-norm
     * orthonormal basis. Working at unit norm throughout means the
     * projection coefficient is simply dot(row_r, row_p) — no gain
     * factors to track. We apply `gain` in a single final pass
     * after the orthonormalization is complete. */
    for (int64_t r = 0; r < rows; r++) {
        float *row_r = pt->data + (size_t)(r * cols);

        /* Subtract projections onto previously orthonormalized
         * (unit-norm) rows. With row_p unit-norm, proj_p(row_r) is
         * exactly (row_r · row_p) * row_p. */
        for (int64_t p = 0; p < r; p++) {
            float *row_p = pt->data + (size_t)(p * cols);
            float dot = 0.0f;
            for (int64_t c = 0; c < cols; c++) {
                dot += row_r[c] * row_p[c];
            }
            for (int64_t c = 0; c < cols; c++) {
                row_r[c] -= dot * row_p[c];
            }
        }

        /* Normalize row_r to unit length. */
        float norm_sq = 0.0f;
        for (int64_t c = 0; c < cols; c++) {
            norm_sq += row_r[c] * row_r[c];
        }
        float norm = sqrtf(norm_sq);
        if (norm < 1e-8f) {
            /* Degenerate row (rows > cols region, or catastrophic
             * numerical cancellation). Zero it out and move on —
             * subsequent rows orthogonalize against this zero
             * vector, which is a no-op. */
            for (int64_t c = 0; c < cols; c++) row_r[c] = 0.0f;
            continue;
        }
        float inv = 1.0f / norm;
        for (int64_t c = 0; c < cols; c++) {
            row_r[c] *= inv;
        }
    }

    /* Step 3: scale the whole matrix by `gain`. This preserves
     * orthogonality (scaling every row by the same constant doesn't
     * change dot products' sign or their zero-ness) and sets each
     * non-degenerate row's L2 norm to exactly `gain`. */
    float g = (float)gain;
    for (int64_t i = 0; i < pt->n_elem; i++) {
        pt->data[i] *= g;
    }
}

RtTensor *sn_tensor_init_orthogonal_seeded(RtTensor *t, double gain, long long seed)
{
    uint64_t state = (uint64_t)seed;
    if (state == 0) state = 0x9E3779B97F4A7C15ULL;
    /* Warm-up rounds so small/adjacent seeds decorrelate. Same
     * convention as the Kaiming and small-scale seeded variants. */
    xorshift64(&state);
    xorshift64(&state);
    xorshift64(&state);
    orthogonal_fill(unwrap(t), gain, &state);
    return t;
}
