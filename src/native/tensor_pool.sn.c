/* ==============================================================================
 * tensor_pool.sn.c — tensor pool + creation ops + lifecycle
 * ==============================================================================
 * Owns the handle-to-host-float-array map. Every tensor handle in the
 * package is an index into g_pool. Creation ops allocate a pool slot and
 * return a wrapped RtTensor. Lifecycle ops (free, reset, checkpoint,
 * restore) operate directly on the pool.
 * ============================================================================== */

#include "tensor_internal.h"

/* ----- Pool storage (definitions for extern decls in the header) ----- */

TPool g_pool[SN_TENSOR_MAX];
int   g_pool_count = 0;

/* Allocate a new pool slot and return its index */
int pool_alloc(int64_t ne0, int64_t ne1, int n_dims)
{
    if (g_pool_count >= SN_TENSOR_MAX) {
        fprintf(stderr, "tensor pool exhausted\n");
        abort();
    }
    int idx = g_pool_count++;
    TPool *s = &g_pool[idx];
    s->ne[0] = ne0;
    s->ne[1] = ne1;
    s->ne[2] = 1;
    s->ne[3] = 1;
    s->n_dims = n_dims;
    s->n_elem = ne0 * ne1;
    s->data   = (float *)calloc((size_t)s->n_elem, sizeof(float));
    return idx;
}

RtTensor *wrap_pool(int idx)
{
    RtTensor *rt = (RtTensor *)calloc(1, sizeof(RtTensor));
    rt->__rc__ = 1;
    rt->__sn___handle = (long long)idx;
    return rt;
}

TPool *unwrap(RtTensor *rt) { return &g_pool[rt->__sn___handle]; }

/* ======================================================================
 * Creation
 * ====================================================================== */

RtTensor *sn_tensor_zeros(long long rows, long long cols)
{
    int idx = pool_alloc(cols, rows, 2);
    return wrap_pool(idx);
}

RtTensor *sn_tensor_from_doubles(SnArray *data, long long rows, long long cols)
{
    int idx = pool_alloc(cols, rows, 2);
    TPool *s = &g_pool[idx];
    long long n = sn_array_length(data);
    for (long long i = 0; i < n && i < s->n_elem; i++) {
        double *p = (double *)sn_array_get(data, i);
        s->data[i] = (float)(*p);
    }
    return wrap_pool(idx);
}

SnArray *sn_tensor_to_doubles(RtTensor *rt)
{
    TPool *s = unwrap(rt);
    SnArray *arr = sn_array_new(sizeof(double), (int)s->n_elem);
    for (int64_t i = 0; i < s->n_elem; i++) {
        double v = (double)s->data[i];
        sn_array_push(arr, &v);
    }
    return arr;
}

SnArray *sn_tensor_shape(RtTensor *rt)
{
    TPool *s = unwrap(rt);
    SnArray *arr = sn_array_new(sizeof(long long), s->n_dims);
    for (int i = 0; i < s->n_dims; i++) {
        long long v = (long long)s->ne[i];
        sn_array_push(arr, &v);
    }
    return arr;
}

/* ======================================================================
 * Lifecycle
 * ====================================================================== */

void sn_tensor_free(RtTensor *rt)
{
    TPool *s = unwrap(rt);
    if (s->data) {
        free(s->data);
        s->data = NULL;
        s->n_elem = 0;
    }
}

void sn_tensor_pool_reset(void)
{
    for (int i = 0; i < g_pool_count; i++) {
        if (g_pool[i].data) {
            free(g_pool[i].data);
            g_pool[i].data = NULL;
        }
    }
    g_pool_count = 0;
}

static int g_pool_checkpoint = 0;

long long sn_tensor_pool_checkpoint(void)
{
    g_pool_checkpoint = g_pool_count;
    return (long long)g_pool_checkpoint;
}

void sn_tensor_pool_restore(long long checkpoint)
{
    int cp = (int)checkpoint;
    if (cp < 0 || cp > g_pool_count) return;
    for (int i = cp; i < g_pool_count; i++) {
        if (g_pool[i].data) {
            free(g_pool[i].data);
            g_pool[i].data = NULL;
        }
    }
    g_pool_count = cp;
}
