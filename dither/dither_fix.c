#include "dither_avx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define VOCAB 10000
#define SAMPLES 1000
#define BAYER_SIZE 8

static float bayer_inv[BAYER_SIZE][BAYER_SIZE] = {
    { 0.0f/64.0f, 32.0f/64.0f,  8.0f/64.0f, 40.0f/64.0f,
      2.0f/64.0f, 34.0f/64.0f, 10.0f/64.0f, 42.0f/64.0f},
    {48.0f/64.0f, 16.0f/64.0f, 56.0f/64.0f, 24.0f/64.0f,
     50.0f/64.0f, 18.0f/64.0f, 58.0f/64.0f, 26.0f/64.0f},
    {12.0f/64.0f, 44.0f/64.0f,  4.0f/64.0f, 36.0f/64.0f,
     14.0f/64.0f, 46.0f/64.0f,  6.0f/64.0f, 38.0f/64.0f},
    {60.0f/64.0f, 28.0f/64.0f, 52.0f/64.0f, 20.0f/64.0f,
     62.0f/64.0f, 30.0f/64.0f, 54.0f/64.0f, 22.0f/64.0f},
    { 3.0f/64.0f, 35.0f/64.0f, 11.0f/64.0f, 43.0f/64.0f,
      1.0f/64.0f, 33.0f/64.0f,  9.0f/64.0f, 41.0f/64.0f},
    {51.0f/64.0f, 19.0f/64.0f, 59.0f/64.0f, 27.0f/64.0f,
     49.0f/64.0f, 17.0f/64.0f, 57.0f/64.0f, 25.0f/64.0f},
    {15.0f/64.0f, 47.0f/64.0f,  7.0f/64.0f, 39.0f/64.0f,
     13.0f/64.0f, 45.0f/64.0f,  5.0f/64.0f, 37.0f/64.0f},
    {63.0f/64.0f, 31.0f/64.0f, 55.0f/64.0f, 23.0f/64.0f,
     61.0f/64.0f, 29.0f/64.0f, 53.0f/64.0f, 21.0f/64.0f}
};

void dither_token_only(float* logits, int vocab_size, float intensity) {
    if (intensity <= 0.0f) return;
    
    float max_logit = -1e30f, min_logit = 1e30f;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
        if (logits[i] < min_logit) min_logit = logits[i];
    }
    float range = max_logit - min_logit;
    if (range < 1e-6f) range = 1.0f;
    
    for (int tid = 0; tid < vocab_size; tid++) {
        int y = tid % BAYER_SIZE;
        int x = 0;
        float threshold = bayer_inv[y][x];
        float dither = intensity * (threshold - 0.5f) * 2.0f * range;
        logits[tid] += dither;
    }
}

void dither_strong_bias(float* logits, int vocab_size, float intensity) {
    if (intensity <= 0.0f) return;
    
    float max_logit = -1e30f, min_logit = 1e30f;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
        if (logits[i] < min_logit) min_logit = logits[i];
    }
    float range = max_logit - min_logit;
    if (range < 1e-6f) range = 1.0f;
    
    for (int tid = 0; tid < vocab_size; tid++) {
        float dither = intensity * (1.0f - (float)tid / vocab_size) * range;
        logits[tid] += dither;
    }
}

void dither_sigmoid_bias(float* logits, int vocab_size, float intensity) {
    if (intensity <= 0.0f) return;
    
    float max_logit = -1e30f, min_logit = 1e30f;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
        if (logits[i] < min_logit) min_logit = logits[i];
    }
    float range = max_logit - min_logit;
    if (range < 1e-6f) range = 1.0f;
    
    for (int tid = 0; tid < vocab_size; tid++) {
        float sigmoid = 1.0f / (1.0f + expf(-((float)tid - vocab_size/2.0f) / (vocab_size/4.0f)));
        float dither = intensity * (0.5f - sigmoid) * 2.0f * range;
        logits[tid] += dither;
    }
}

void dither_hash(float* logits, int vocab_size, int position, float intensity) {
    if (intensity <= 0.0f) return;
    
    float max_logit = -1e30f, min_logit = 1e30f;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
        if (logits[i] < min_logit) min_logit = logits[i];
    }
    float range = max_logit - min_logit;
    if (range < 1e-6f) range = 1.0f;
    
    for (int tid = 0; tid < vocab_size; tid++) {
        int hash = (position * 123457 + tid * 193759) % 256;
        float threshold = (hash / 256.0f) - 0.5f;
        float dither = intensity * threshold * 2.0f * range;
        logits[tid] += dither;
    }
}

void dither_token_weighted(float* logits, int vocab_size, float intensity) {
    if (intensity <= 0.0f) return;
    
    float max_logit = -1e30f, min_logit = 1e30f;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
        if (logits[i] < min_logit) min_logit = logits[i];
    }
    float range = max_logit - min_logit;
    if (range < 1e-6f) range = 1.0f;
    
    for (int tid = 0; tid < vocab_size; tid++) {
        float weight = (float)(tid % 17) / 17.0f - 0.5f;
        float dither = intensity * weight * 2.0f * range;
        logits[tid] += dither;
    }
}

static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

int sample_simple(float* logits, int n, uint64_t rng) {
    float max_l = -1e30f;
    for (int i = 0; i < n; i++) if (logits[i] > max_l) max_l = logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { logits[i] = expf(logits[i] - max_l); sum += logits[i]; }
    for (int i = 0; i < n; i++) logits[i] /= sum;
    float r = (splitmix64(&rng) & 0xFFFFFFFF) / (float)0xFFFFFFFF;
    float c = 0;
    for (int i = 0; i < n; i++) { c += logits[i]; if (c >= r) return i; }
    return n-1;
}

typedef struct { int unique; double entropy; } Metrics;

void dither_apply_with_pos(float* logits, int vocab_size, int position, float intensity) {
    dither_apply_avx(logits, vocab_size, position, intensity);
}

Metrics run_test(int method, float param, int varying_pos) {
    Metrics m = {0, 0.0};
    int* counts = calloc(VOCAB, sizeof(int));
    if (!counts) return m;
    
    uint64_t rng = 12345;
    for (int s = 0; s < SAMPLES; s++) {
        float logits[VOCAB];
        for (int i = 0; i < VOCAB; i++) logits[i] = (i < 10) ? 5.0f : 0.0f;
        
        int pos = varying_pos ? s : 42;
        
        switch(method) {
            case 0: break; // no dither
            case 1: dither_apply_avx(logits, VOCAB, pos, param); break;
            case 2: dither_token_only(logits, VOCAB, param); break;
            case 3: dither_hash(logits, VOCAB, pos, param); break;
            case 4: dither_token_weighted(logits, VOCAB, param); break;
            case 5: dither_strong_bias(logits, VOCAB, param); break;
            case 6: dither_sigmoid_bias(logits, VOCAB, param); break;
        }
        
        int tok = sample_simple(logits, VOCAB, rng + s);
        counts[tok]++;
    }
    
    for (int i = 0; i < VOCAB; i++) if (counts[i] > 0) {
        m.unique++;
        double p = (double)counts[i] / SAMPLES;
        m.entropy -= p * log2(p);
    }
    free(counts);
    return m;
}

int main(void) {
    printf("=== DITHER FIX TESTING ===\n\n");
    
    Metrics baseline = run_test(0, 0.0f, 0);
    printf("Baseline (no dither):     unique=%d, entropy=%.3f\n", baseline.unique, baseline.entropy);
    
    printf("\n--- ORIGINAL (position-dependent) ---\n");
    Metrics orig_fixed = run_test(1, 0.0809f, 0);
    Metrics orig_vary = run_test(1, 0.0809f, 1);
    printf("Fixed position:  unique=%d, entropy=%.3f (delta %d)\n", 
           orig_fixed.unique, orig_fixed.entropy, orig_fixed.unique - baseline.unique);
    printf("Varying pos:    unique=%d, entropy=%.3f (delta %d)\n", 
           orig_vary.unique, orig_vary.entropy, orig_vary.unique - baseline.unique);
    
    printf("\n--- FIX 1: Token-only (position ignored) ---\n");
    Metrics fix1 = run_test(2, 0.0809f, 1);
    printf("Token-only:     unique=%d, entropy=%.3f (delta %d)\n", 
           fix1.unique, fix1.entropy, fix1.unique - baseline.unique);
    
    printf("\n--- FIX 2: Hash-based (deterministic) ---\n");
    Metrics fix3 = run_test(3, 0.0809f, 1);
    printf("Hash-based:     unique=%d, entropy=%.3f (delta %d)\n", 
           fix3.unique, fix3.entropy, fix3.unique - baseline.unique);
    
    printf("\n--- FIX 3: Token-weighted ---\n");
    Metrics fix4 = run_test(4, 0.0809f, 1);
    printf("Token-weighted: unique=%d, entropy=%.3f (delta %d)\n", 
           fix4.unique, fix4.entropy, fix4.unique - baseline.unique);
    
    printf("\n--- FIX 4: Strong linear bias (lower tokens = higher bonus) ---\n");
    Metrics fix5 = run_test(5, 0.05f, 1);
    printf("Strong bias:    unique=%d, entropy=%.3f (delta %d)\n", 
           fix5.unique, fix5.entropy, fix5.unique - baseline.unique);
    
    printf("\n--- FIX 5: Sigmoid bias (favors extremes) ---\n");
    Metrics fix6 = run_test(6, 0.1f, 1);
    printf("Sigmoid bias:   unique=%d, entropy=%.3f (delta %d)\n", 
           fix6.unique, fix6.entropy, fix6.unique - baseline.unique);
    
    printf("\n=== RESULTS SUMMARY ===\n");
    printf("%-20s %-10s %-10s\n", "Method", "Unique", "vs Baseline");
    printf("%-20s %-10s %-10s\n", "------", "-------", "----------");
    printf("%-20s %-10d %-10d\n", "Baseline", baseline.unique, 0);
    printf("%-20s %-10d %-10d\n", "Original (fixed)", orig_fixed.unique, orig_fixed.unique - baseline.unique);
    printf("%-20s %-10d %-10d\n", "Original (vary)", orig_vary.unique, orig_vary.unique - baseline.unique);
    printf("%-20s %-10d %-10d\n", "Fix1 (token)", fix1.unique, fix1.unique - baseline.unique);
    printf("%-20s %-10d %-10d\n", "Fix4 (strong)", fix5.unique, fix5.unique - baseline.unique);
    printf("%-20s %-10d %-10d\n", "Fix5 (sigmoid)", fix6.unique, fix6.unique - baseline.unique);
    
    return 0;
}
