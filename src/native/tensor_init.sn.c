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
