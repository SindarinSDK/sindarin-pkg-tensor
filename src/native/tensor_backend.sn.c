/* ==============================================================================
 * tensor_backend.sn.c — ggml backend init + micro-graph helpers + device ops
 * ==============================================================================
 * Owns:
 *   - the global ggml backend handle and lazy-init path
 *   - the micro-graph scratch context (pre-zero'd, reused across ops)
 *   - the input-tracking machinery for direct-mode ops
 *   - sn_gpu_available / sn_tensor_to_device public entry points
 *
 * Every direct-mode tensor op (in tensor_ops.sn.c, tensor_gnn_ops.sn.c,
 * tensor_loss.sn.c) goes through micro_ctx_init / track_input / run_graph
 * from here.
 * ============================================================================== */

#include "tensor_internal.h"

/* ======================================================================
 * ggml backend — initialized lazily on first use
 * ====================================================================== */

ggml_backend_t g_backend     = NULL;
int            g_backend_gpu = 0;

void ensure_backend(void)
{
    if (g_backend) return;

    /* Start with CPU — always available, no dynamic loading */
    g_backend = ggml_backend_cpu_init();

    /* Try to upgrade to a better backend (GPU) if available */
    ggml_backend_load_all();
    ggml_backend_t best = ggml_backend_init_best();
    if (best) {
        const char *name = ggml_backend_name(best);
        if (name && strcmp(name, "CPU") != 0) {
            /* Got a GPU backend — use it instead */
            ggml_backend_free(g_backend);
            g_backend = best;
            g_backend_gpu = 1;
        } else {
            /* init_best returned CPU — free the duplicate, keep ours */
            ggml_backend_free(best);
        }
    }
}

/* ======================================================================
 * ggml micro-graph helpers
 *
 * Each tensor op creates a small ggml context, builds a 1-op graph,
 * computes it on the backend, copies the result into the pool, and frees.
 * ====================================================================== */

/* Pre-allocated zero'd buffer for micro-graph contexts.
 * Newer ggml versions assert tensor->buffer == NULL before allocation.
 * posix_memalign (ggml's default) doesn't zero memory, so stale
 * buffer pointers from prior ggml contexts can trigger the assertion.
 * Using a pre-zero'd buffer avoids this. */
static void *g_micro_ctx_buf = NULL;
static size_t g_micro_ctx_size = 0;

struct ggml_context *micro_ctx_init(void) {
    size_t needed = GRAPH_CTX_SIZE;
    if (!g_micro_ctx_buf || g_micro_ctx_size < needed) {
        if (g_micro_ctx_buf) free(g_micro_ctx_buf);
        g_micro_ctx_size = needed;
        g_micro_ctx_buf = malloc(g_micro_ctx_size);
    }
    /* Zero the buffer to ensure all tensor fields (including buffer) start NULL */
    memset(g_micro_ctx_buf, 0, g_micro_ctx_size);
    struct ggml_init_params params = { g_micro_ctx_size, g_micro_ctx_buf, true };
    return ggml_init(params);
}

/* Input tensor tracking for upload before compute */
static struct ggml_tensor *g_inputs[MAX_INPUTS];
static const float        *g_input_data[MAX_INPUTS];
static int                 g_input_count = 0;

void track_input(struct ggml_tensor *t, const float *host_data)
{
    if (g_input_count < MAX_INPUTS) {
        g_inputs[g_input_count]     = t;
        g_input_data[g_input_count] = host_data;
        g_input_count++;
    }
}

/* Run a graph with the global backend.
 * Returns the allocator — caller MUST call ggml_gallocr_free() after
 * reading results from the graph output tensors. */
ggml_gallocr_t run_graph(struct ggml_context *ctx, struct ggml_cgraph *graph)
{
    (void)ctx;
    ensure_backend();

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(g_backend);
    ggml_gallocr_t alloc = ggml_gallocr_new(buft);
    ggml_gallocr_alloc_graph(alloc, graph);

    /* Upload tracked input data */
    for (int i = 0; i < g_input_count; i++) {
        ggml_backend_tensor_set(g_inputs[i], g_input_data[i], 0, ggml_nbytes(g_inputs[i]));
    }
    g_input_count = 0;

    ggml_backend_graph_compute(g_backend, graph);
    return alloc;
}

/* ======================================================================
 * Device
 * ====================================================================== */

int sn_gpu_available(void)
{
    ensure_backend();
    return g_backend_gpu;
}

RtTensor *sn_tensor_to_device(RtTensor *t, char *device)
{
    (void)device;
    /* Backend handles device dispatch transparently.
     * Return the same tensor — data stays in the host pool. */
    return t;
}
