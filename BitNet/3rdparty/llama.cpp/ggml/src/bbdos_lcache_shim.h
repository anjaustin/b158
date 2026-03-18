/*
 * BBDOS L-Cache Shim Header
 */

#ifndef BBDOS_LCACHE_SHIM_H
#define BBDOS_LCACHE_SHIM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BBDOS_Context BBDOS_Context;

BBDOS_Context* bbdos_create(int cache_size);
void bbdos_destroy(BBDOS_Context* ctx);
uint64_t bbdos_capability(void);
const char* bbdos_version(void);

int bbdos_cache_attention(
    BBDOS_Context* ctx,
    const float* q, const float* k, const float* v,
    int seq_len, int head_dim,
    float* output
);

int bbdos_cache_ffn(
    BBDOS_Context* ctx,
    const float* input,
    const uint8_t* weights,
    const float* scales,
    int input_size, int hidden_size, int output_size,
    float* output
);

void bbdos_stats(BBDOS_Context* ctx, int* hits, int* misses, int* evictions);

void bbdos_mtfp_pack(const float* input, int8_t* output, int count);
void bbdos_mtfp_unpack(const int8_t* input, float* output, int count);

#ifdef __cplusplus
}
#endif

#endif
