/*
 * ggml-lcache.h - L-Cache Integration for llama.cpp
 * 
 * Provides caching for attention and FFN layer outputs to accelerate
 * repeated inference patterns (e.g., multi-turn conversations).
 */

#ifndef GGML_LCACHE_H
#define GGML_LCACHE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_LCACHE_MAX_SLOTS 512
#define GGML_LCACHE_MAX_NAME 64

typedef struct {
    char op_name[GGML_LCACHE_MAX_NAME];
    uint64_t input_hash;
    uint8_t* output_data; // Packed ternary data (2-bits per element)
    size_t output_size;   // Size in elements
    int64_t last_access;
    int hit_count;
} ggml_lcache_entry_t;

typedef struct {
    ggml_lcache_entry_t slots[GGML_LCACHE_MAX_SLOTS];
    int n_slots;
    int cache_hits;
    int cache_misses;
    int evictions;
    bool enabled;
    // Timing stats
    uint64_t lookup_time_ns;
    uint64_t store_time_ns;
    int lookup_count;
    int store_count;

    float * unpack_buffer; // Internal buffer for unpacking on hit
    size_t unpack_buffer_size;
} ggml_lcache_t;

// L-Cache API
ggml_lcache_t* ggml_lcache_init(int max_slots);
void ggml_lcache_destroy(ggml_lcache_t* cache);
void ggml_lcache_enable(ggml_lcache_t* cache, bool enable);
void ggml_lcache_clear(ggml_lcache_t* cache);

// Compute-side caching (check before compute, store after)
bool ggml_lcache_lookup(ggml_lcache_t* cache, const char* op_name, uint64_t input_hash, float** output);
void ggml_lcache_store(ggml_lcache_t* cache, const char* op_name, uint64_t input_hash, float* output, size_t size);

// Simple hash for data content
uint64_t ggml_lcache_hash_tensor(const void* data, size_t size);

// Stats
void ggml_lcache_get_stats(ggml_lcache_t* cache, int* hits, int* misses, int* evictions);

// Timing stats
typedef struct {
    uint64_t lookup_time_ns;
    uint64_t store_time_ns;
    int lookup_count;
    int store_count;
} ggml_lcache_stats_t;

void ggml_lcache_get_timing(ggml_lcache_stats_t* stats);

// Accessor functions (for external initialization)
void ggml_lcache_set(ggml_lcache_t* cache);
ggml_lcache_t* ggml_lcache_get(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_LCACHE_H
