/*
 * ggml-lcache.c - L-Cache Implementation for llama.cpp
 */

#include "ggml-lcache.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <stdatomic.h>
#include <math.h>

// MTFP16 packing (8 trits per element, 2 bits per trit = 16 bits per element)
static void ggml_lcache_mtfp_pack(const float * src, uint8_t * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = src[i];
        uint8_t trits[8] = {0};
        
        // Simple 8-trit quantization logic
        int sign = (v < 0) ? 1 : 0;
        float abs_v = fabsf(v);
        int exp = 0;
        if (abs_v >= 1.0f) {
            while (abs_v >= 3.0f && exp < 40) { abs_v /= 3.0f; exp++; }
        } else if (abs_v > 0) {
            while (abs_v < 1.0f && exp > -40) { abs_v *= 3.0f; exp--; }
        }
        int mantissa = (int)(abs_v * 9.0f);

        // Pack into 16 bits (2 bytes)
        trits[0] = sign;
        trits[1] = (exp + 40) % 3;
        trits[2] = ((exp + 40) / 3) % 3;
        trits[3] = ((exp + 40) / 9) % 3;
        trits[4] = ((exp + 40) / 27) % 3;
        trits[5] = mantissa % 3;
        trits[6] = (mantissa / 3) % 3;
        trits[7] = (mantissa / 9) % 3;

        uint16_t packed = 0;
        for (int t = 0; t < 8; t++) packed |= (uint16_t)(trits[t] << (t * 2));
        memcpy(dst + i * 2, &packed, 2);
    }
}

static void ggml_lcache_mtfp_unpack(const uint8_t * src, float * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint16_t packed;
        memcpy(&packed, src + i * 2, 2);
        
        int sign = (packed >> 0) & 0x03;
        int exp = ((packed >> 2) & 0x03) + 
                  ((packed >> 4) & 0x03) * 3 + 
                  ((packed >> 6) & 0x03) * 9 + 
                  ((packed >> 8) & 0x03) * 27 - 40;
        int mantissa = ((packed >> 10) & 0x03) + 
                       ((packed >> 12) & 0x03) * 3 + 
                       ((packed >> 14) & 0x03) * 9;

        float res = (float)mantissa / 9.0f * powf(3.0f, (float)exp);
        dst[i] = (sign == 1) ? -res : res;
    }
}

// FNV-1a hash
static uint64_t hash_bytes(const uint8_t* data, size_t size) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < size; i++) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

ggml_lcache_t* ggml_lcache_init(int max_slots) {
    if (max_slots > GGML_LCACHE_MAX_SLOTS) max_slots = GGML_LCACHE_MAX_SLOTS;
    if (max_slots < 1) max_slots = 64;
    
    ggml_lcache_t* cache = (ggml_lcache_t*)calloc(1, sizeof(ggml_lcache_t));
    cache->n_slots = max_slots;
    cache->enabled = true;
    
    cache->unpack_buffer_size = 1024 * 1024; // 1M elements default
    cache->unpack_buffer = (float*)malloc(cache->unpack_buffer_size * sizeof(float));

    return cache;
}

void ggml_lcache_destroy(ggml_lcache_t* cache) {
    if (!cache) return;
    
    for (int i = 0; i < cache->n_slots; i++) {
        if (cache->slots[i].output_data) {
            free(cache->slots[i].output_data);
        }
    }
    
    if (cache->unpack_buffer) free(cache->unpack_buffer);
    free(cache);
}

void ggml_lcache_enable(ggml_lcache_t* cache, bool enable) {
    if (cache) cache->enabled = enable;
}

void ggml_lcache_clear(ggml_lcache_t* cache) {
    if (!cache) return;
    
    for (int i = 0; i < cache->n_slots; i++) {
        if (cache->slots[i].output_data) {
            free(cache->slots[i].output_data);
            cache->slots[i].output_data = NULL;
        }
        cache->slots[i].op_name[0] = '\0';
    }
    
    cache->cache_hits = 0;
    cache->cache_misses = 0;
    cache->evictions = 0;
}

bool ggml_lcache_lookup(ggml_lcache_t* cache, const char* op_name, uint64_t input_hash, float** output) {
    if (!cache || !cache->enabled || !output) return false;
    
    uint64_t start = clock();
    
    for (int i = 0; i < cache->n_slots; i++) {
        if (cache->slots[i].op_name[0] &&
            strcmp(cache->slots[i].op_name, op_name) == 0 &&
            cache->slots[i].input_hash == input_hash) {
            
            cache->slots[i].last_access = clock();
            cache->slots[i].hit_count++;
            cache->cache_hits++;
            
            // Ensure unpack buffer is large enough
            if (cache->slots[i].output_size > cache->unpack_buffer_size) {
                cache->unpack_buffer_size = cache->slots[i].output_size;
                cache->unpack_buffer = (float*)realloc(cache->unpack_buffer, cache->unpack_buffer_size * sizeof(float));
            }
            
            ggml_lcache_mtfp_unpack(cache->slots[i].output_data, cache->unpack_buffer, cache->slots[i].output_size);
            *output = cache->unpack_buffer;
            
            cache->lookup_time_ns += (clock() - start) * (1000000000ULL / CLOCKS_PER_SEC);
            cache->lookup_count++;
            return true;
        }
    }
    
    cache->cache_misses++;
    cache->lookup_time_ns += (clock() - start) * (1000000000ULL / CLOCKS_PER_SEC);
    cache->lookup_count++;
    return false;
}

void ggml_lcache_store(ggml_lcache_t* cache, const char* op_name, uint64_t input_hash, float* output, size_t n_bytes) {
    if (!cache || !cache->enabled || !output) return;
    
    uint64_t start = clock();
    size_t n_elements = n_bytes / sizeof(float);
    size_t packed_size = n_elements * 2; // 16 bits per element (8 trits)

    // Check if already cached
    for (int i = 0; i < cache->n_slots; i++) {
        if (cache->slots[i].op_name[0] &&
            strcmp(cache->slots[i].op_name, op_name) == 0 &&
            cache->slots[i].input_hash == input_hash) {
            
            // Update existing entry
            if (cache->slots[i].output_data && cache->slots[i].output_size != n_elements) {
                free(cache->slots[i].output_data);
                cache->slots[i].output_data = (uint8_t*)malloc(packed_size);
            }
            ggml_lcache_mtfp_pack(output, cache->slots[i].output_data, n_elements);
            cache->slots[i].output_size = n_elements;
            cache->slots[i].last_access = clock();
            
            cache->store_time_ns += (clock() - start) * (1000000000ULL / CLOCKS_PER_SEC);
            cache->store_count++;
            return;
        }
    }
    
    // Find empty slot
    for (int i = 0; i < cache->n_slots; i++) {
        if (cache->slots[i].op_name[0] == 0) {
            strncpy(cache->slots[i].op_name, op_name, GGML_LCACHE_MAX_NAME - 1);
            cache->slots[i].input_hash = input_hash;
            cache->slots[i].output_data = (uint8_t*)malloc(packed_size);
            ggml_lcache_mtfp_pack(output, cache->slots[i].output_data, n_elements);
            cache->slots[i].output_size = n_elements;
            cache->slots[i].last_access = clock();
            cache->slots[i].hit_count = 0;
            return;
        }
    }
    
    // LRU eviction
    int lru_idx = 0;
    int64_t oldest = INT64_MAX;
    for (int i = 0; i < cache->n_slots; i++) {
        if (cache->slots[i].last_access < oldest) {
            oldest = cache->slots[i].last_access;
            lru_idx = i;
        }
    }
    
    free(cache->slots[lru_idx].output_data);
    strncpy(cache->slots[lru_idx].op_name, op_name, GGML_LCACHE_MAX_NAME - 1);
    cache->slots[lru_idx].input_hash = input_hash;
    cache->slots[lru_idx].output_data = (uint8_t*)malloc(packed_size);
    ggml_lcache_mtfp_pack(output, cache->slots[lru_idx].output_data, n_elements);
    cache->slots[lru_idx].output_size = n_elements;
    cache->slots[lru_idx].last_access = clock();
    cache->slots[lru_idx].hit_count = 0;
    cache->evictions++;
    
    cache->store_time_ns += (clock() - start) * (1000000000ULL / CLOCKS_PER_SEC);
    cache->store_count++;
}

uint64_t ggml_lcache_hash_tensor(const void* data, size_t size) {
    if (!data || size == 0) return 0;
    return hash_bytes((const uint8_t*)data, size);
}

void ggml_lcache_get_stats(ggml_lcache_t* cache, int* hits, int* misses, int* evictions) {
    if (!cache) return;
    if (hits) *hits = cache->cache_hits;
    if (misses) *misses = cache->cache_misses;
    if (evictions) *evictions = cache->evictions;
}

void ggml_lcache_get_timing(ggml_lcache_stats_t* stats) {
    ggml_lcache_t* cache = ggml_lcache_get();
    if (!cache || !stats) return;
    stats->lookup_time_ns = cache->lookup_time_ns;
    stats->store_time_ns = cache->store_time_ns;
    stats->lookup_count = cache->lookup_count;
    stats->store_count = cache->store_count;
}

// Debug function to print cache stats
void ggml_lcache_dump_stats(void) {
    ggml_lcache_t* cache = ggml_lcache_get();
    if (!cache) return;
    
    fprintf(stderr, "=== L-Cache Stats ===\n");
    fprintf(stderr, "Hits: %d, Misses: %d, Evictions: %d\n", 
            cache->cache_hits, cache->cache_misses, cache->evictions);
    fprintf(stderr, "Lookups: %d (%.2f us avg)\n", 
            cache->lookup_count, 
            cache->lookup_count > 0 ? (double)cache->lookup_time_ns / cache->lookup_count / 1000.0 : 0);
    fprintf(stderr, "Stores: %d (%.2f us avg)\n", 
            cache->store_count,
            cache->store_count > 0 ? (double)cache->store_time_ns / cache->store_count / 1000.0 : 0);
}
