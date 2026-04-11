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
 * Produces a (semi-)orthogonal weight matrix scaled by `gain`. For
 * policy/value heads and hidden layers in PPO the standard setup is:
 *   - policy head:    gain = 0.01 (small-scale, centres π near uniform)
 *   - value head:     gain = 1.0
 *   - ReLU hidden:    gain = sqrt(2)  (Andrychowicz 2021, SB3/CleanRL)
 *   - tanh hidden:    gain = 1.0
 *
 * Rectangular semantics (matches torch.nn.init.orthogonal_):
 *   - rows <= cols: rows are orthonormal, i.e. W Wᵀ = gain² · I_rows.
 *   - rows  > cols: columns are orthonormal, i.e. Wᵀ W = gain² · I_cols.
 *     (the "semi-orthogonal" case Saxe et al. 2014 define for
 *     expansion layers — e.g. skynet's first GNN layer, whose shape
 *     is hidden_dim × input_dim with hidden_dim > input_dim)
 *
 * Algorithm: modified Gram-Schmidt on the shorter axis.
 *   1. Fill n_elem = rows*cols slots with iid N(0, 1) samples
 *      (Box-Muller from xorshift).
 *   2. Run modified Gram-Schmidt on rows, treating the matrix as
 *      (inner_rows = min(rows, cols), inner_cols = max(rows, cols)).
 *      This always orthogonalizes the shorter of the two axes, so no
 *      row collapses from the rows > cols dimensionality mismatch.
 *   3. Scale everything by `gain` in a single final pass.
 *
 * For rows > cols the Gram-Schmidt runs on a scratch buffer laid out
 * as (cols, rows), and the result is written transposed back into
 * `pt->data` so the caller's (rows, cols) view sees orthonormal
 * columns. The scratch buffer is freed before the gain scale pass,
 * which then runs in-place over `pt->data` exactly like the
 * rows <= cols path.
 *
 * Tensor shape convention: the caller creates a tensor via
 * Tensor.zeros(rows, cols) which stores ne[0] = cols, ne[1] = rows;
 * the row-major data layout is `data[r * cols + c]`. */
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

/* In-place modified Gram-Schmidt on the rows of a row-major
 * (inner_rows, inner_cols) buffer. Requires inner_rows <= inner_cols
 * — the caller is responsible for arranging this via the scratch
 * transpose path when necessary. Produces UNIT-norm orthonormal rows;
 * the gain scale is applied by the caller in a single final pass. */
static void gram_schmidt_rows(float *buf, int64_t inner_rows, int64_t inner_cols)
{
    for (int64_t r = 0; r < inner_rows; r++) {
        float *row_r = buf + (size_t)(r * inner_cols);

        /* Subtract projections onto previously orthonormalized
         * (unit-norm) rows. With row_p unit-norm, proj_p(row_r) is
         * exactly (row_r · row_p) * row_p. */
        for (int64_t p = 0; p < r; p++) {
            float *row_p = buf + (size_t)(p * inner_cols);
            float dot = 0.0f;
            for (int64_t c = 0; c < inner_cols; c++) {
                dot += row_r[c] * row_p[c];
            }
            for (int64_t c = 0; c < inner_cols; c++) {
                row_r[c] -= dot * row_p[c];
            }
        }

        /* Normalize row_r to unit length. */
        float norm_sq = 0.0f;
        for (int64_t c = 0; c < inner_cols; c++) {
            norm_sq += row_r[c] * row_r[c];
        }
        float norm = sqrtf(norm_sq);
        if (norm < 1e-8f) {
            /* Catastrophic numerical cancellation: two random
             * Gaussian draws collided onto the same direction. The
             * caller's shape precondition (inner_rows <= inner_cols)
             * rules out the dimensionality-exhaustion case that the
             * old code path had to handle, so this branch now only
             * fires on genuine numerical pathology. Zero it out and
             * move on — subsequent rows orthogonalize against this
             * zero vector, which is a no-op. */
            for (int64_t c = 0; c < inner_cols; c++) row_r[c] = 0.0f;
            continue;
        }
        float inv = 1.0f / norm;
        for (int64_t c = 0; c < inner_cols; c++) {
            row_r[c] *= inv;
        }
    }
}

static void orthogonal_fill(TPool *pt, double gain, uint64_t *state)
{
    int64_t cols = pt->ne[0];
    int64_t rows = pt->ne[1];
    float g = (float)gain;

    if (rows <= cols) {
        /* Step 1: fill pt->data with iid N(0, 1) Gaussian samples. */
        for (int64_t i = 0; i < pt->n_elem; i++) {
            pt->data[i] = sample_gaussian(state);
        }

        /* Step 2: Gram-Schmidt on rows; rows <= cols guarantees the
         * shorter-axis precondition holds. */
        gram_schmidt_rows(pt->data, rows, cols);

        /* Step 3: gain scale in place. */
        for (int64_t i = 0; i < pt->n_elem; i++) {
            pt->data[i] *= g;
        }
        return;
    }

    /* rows > cols: semi-orthogonal case. Build a (cols, rows) scratch
     * matrix, orthonormalize its rows (cols rows, each in R^rows —
     * feasible since cols < rows), then transpose back into pt->data
     * so the caller's (rows, cols) view sees orthonormal columns. */
    size_t n_elem = (size_t)rows * (size_t)cols;
    float *scratch = (float *)malloc(n_elem * sizeof(float));
    if (scratch == NULL) {
        /* Allocation failure is terminal: the tensor would otherwise
         * be left in a partially-initialized state. Zero it out so
         * the caller at least gets a well-defined (if useless)
         * tensor instead of uninitialized memory. */
        for (int64_t i = 0; i < pt->n_elem; i++) {
            pt->data[i] = 0.0f;
        }
        return;
    }

    /* Step 1: fill the scratch buffer with iid N(0, 1) Gaussian
     * samples. Consumes exactly n_elem samples, matching the
     * rows <= cols path's RNG consumption for the same element count. */
    for (size_t i = 0; i < n_elem; i++) {
        scratch[i] = sample_gaussian(state);
    }

    /* Step 2: Gram-Schmidt on the scratch treated as (cols, rows).
     * inner_rows = cols, inner_cols = rows, and cols < rows so the
     * precondition holds. */
    gram_schmidt_rows(scratch, cols, rows);

    /* Step 3: transpose scratch[c * rows + r] -> pt->data[r * cols + c].
     * After this, pt->data's cols columns (each a vector in R^rows)
     * are orthonormal. */
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < cols; c++) {
            pt->data[r * cols + c] = scratch[c * rows + r];
        }
    }

    free(scratch);

    /* Step 4: gain scale in place. Scaling every entry by the same
     * constant preserves Wᵀ W = I up to gain² factor, so the result
     * has column L2 norm = gain. */
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
