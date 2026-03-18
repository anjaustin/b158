#include "dither_avx.h"
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static float bayer_matrix[BAYER_SIZE][BAYER_SIZE] = {
    { 0.0f/64.0f - 0.5f, 32.0f/64.0f - 0.5f,  8.0f/64.0f - 0.5f, 40.0f/64.0f - 0.5f,
      2.0f/64.0f - 0.5f, 34.0f/64.0f - 0.5f, 10.0f/64.0f - 0.5f, 42.0f/64.0f - 0.5f},
    {48.0f/64.0f - 0.5f, 16.0f/64.0f - 0.5f, 56.0f/64.0f - 0.5f, 24.0f/64.0f - 0.5f,
     50.0f/64.0f - 0.5f, 18.0f/64.0f - 0.5f, 58.0f/64.0f - 0.5f, 26.0f/64.0f - 0.5f},
    {12.0f/64.0f - 0.5f, 44.0f/64.0f - 0.5f,  4.0f/64.0f - 0.5f, 36.0f/64.0f - 0.5f,
     14.0f/64.0f - 0.5f, 46.0f/64.0f - 0.5f,  6.0f/64.0f - 0.5f, 38.0f/64.0f - 0.5f},
    {60.0f/64.0f - 0.5f, 28.0f/64.0f - 0.5f, 52.0f/64.0f - 0.5f, 20.0f/64.0f - 0.5f,
     62.0f/64.0f - 0.5f, 30.0f/64.0f - 0.5f, 54.0f/64.0f - 0.5f, 22.0f/64.0f - 0.5f},
    { 3.0f/64.0f - 0.5f, 35.0f/64.0f - 0.5f, 11.0f/64.0f - 0.5f, 43.0f/64.0f - 0.5f,
      1.0f/64.0f - 0.5f, 33.0f/64.0f - 0.5f,  9.0f/64.0f - 0.5f, 41.0f/64.0f - 0.5f},
    {51.0f/64.0f - 0.5f, 19.0f/64.0f - 0.5f, 59.0f/64.0f - 0.5f, 27.0f/64.0f - 0.5f,
     49.0f/64.0f - 0.5f, 17.0f/64.0f - 0.5f, 57.0f/64.0f - 0.5f, 25.0f/64.0f - 0.5f},
    {15.0f/64.0f - 0.5f, 47.0f/64.0f - 0.5f,  7.0f/64.0f - 0.5f, 39.0f/64.0f - 0.5f,
     13.0f/64.0f - 0.5f, 45.0f/64.0f - 0.5f,  5.0f/64.0f - 0.5f, 37.0f/64.0f - 0.5f},
    {63.0f/64.0f - 0.5f, 31.0f/64.0f - 0.5f, 55.0f/64.0f - 0.5f, 23.0f/64.0f - 0.5f,
     61.0f/64.0f - 0.5f, 29.0f/64.0f - 0.5f, 53.0f/64.0f - 0.5f, 21.0f/64.0f - 0.5f}
};

void dither_init(void) {
    printf("[DITHER] Bayer matrix initialized (8x8)\n");
}

void dither_apply_avx(float* logits, int vocab_size, int position, float intensity) {
    if (intensity <= 0.0f) return;
    
    float max_logit = -INFINITY;
    float min_logit = INFINITY;
    
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
        if (logits[i] < min_logit) min_logit = logits[i];
    }
    
    float logit_range = max_logit - min_logit;
    if (logit_range < 1e-6f) logit_range = 1.0f;
    
    int x = position % BAYER_SIZE;
    
    for (int token_id = 0; token_id < vocab_size; token_id++) {
        int y = token_id % BAYER_SIZE;
        float threshold = bayer_matrix[y][x];
        float dither = intensity * threshold * 2.0f * logit_range;
        
        float new_val = logits[token_id] + dither;
        if (new_val > 1e30f) new_val = 1e30f;
        if (new_val < -1e30f) new_val = -1e30f;
        logits[token_id] = new_val;
    }
}

void softmax_avx(float* logits, int vocab_size) {
    float max_logit = -INFINITY;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    int vec_width = 8;
    float sum = 0.0f;
    
    for (int i = 0; i < vocab_size; i += vec_width) {
        int remaining = vocab_size - i;
        int process = (remaining >= vec_width) ? vec_width : remaining;
        
        if (process == vec_width) {
            __m256 logits_vec = _mm256_loadu_ps(&logits[i]);
            __m256 max_vec = _mm256_set1_ps(max_logit);
            __m256 diff = _mm256_sub_ps(logits_vec, max_vec);
            
            float vals[8];
            _mm256_storeu_ps(vals, diff);
            for (int j = 0; j < 8; j++) {
                vals[j] = expf(vals[j]);
            }
            __m256 exp_vec = _mm256_loadu_ps(vals);
            _mm256_storeu_ps(&logits[i], exp_vec);
            
            for (int j = 0; j < 8; j++) {
                sum += vals[j];
            }
        } else {
            for (int j = 0; j < process; j++) {
                float val = expf(logits[i + j] - max_logit);
                logits[i + j] = val;
                sum += val;
            }
        }
    }
    
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= sum;
    }
}

static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

int sample_token(float* logits, int vocab_size, float temperature, int top_k, float top_p, uint64_t rng_state) {
    if (temperature != 1.0f) {
        float inv_temp = 1.0f / temperature;
        for (int i = 0; i < vocab_size; i++) {
            logits[i] *= inv_temp;
        }
    }
    
    if (top_k > 0 && top_k < vocab_size) {
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] < logits[top_k - 1]) {
                logits[i] = -1e30f;
            }
        }
    }
    
    softmax_avx(logits, vocab_size);
    
    if (top_p > 0.0f && top_p < 1.0f) {
        float cumsum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += logits[i];
            if (cumsum > top_p) {
                for (int j = i + 1; j < vocab_size; j++) {
                    logits[j] = 0.0f;
                }
                break;
            }
        }
        
        softmax_avx(logits, vocab_size);
    }
    
    float r = (splitmix64(&rng_state) & 0xFFFFFFFF) / (float)0xFFFFFFFF;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (cumsum >= r) {
            return i;
        }
    }
    
    return vocab_size - 1;
}

void benchmark_dither(int vocab_size, int num_positions, int iterations) {
    printf("\n=== AVX Dither Benchmark ===\n");
    printf("Vocab: %d, Positions: %d, Iterations: %d\n", vocab_size, num_positions, iterations);
    
    float* logits = (float*)aligned_alloc(32, vocab_size * sizeof(float));
    if (!logits) {
        fprintf(stderr, "Failed to allocate memory\n");
        return;
    }
    
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = ((float)rand() / RAND_MAX) * 10.0f;
    }
    
    clock_t start = clock();
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int pos = 0; pos < num_positions; pos++) {
            float* test_logits = (float*)malloc(vocab_size * sizeof(float));
            memcpy(test_logits, logits, vocab_size * sizeof(float));
            
            dither_apply_avx(test_logits, vocab_size, pos, 0.0809f);
            
            free(test_logits);
        }
    }
    
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Time: %.3f seconds\n", elapsed);
    printf("Rate: %.0f dithers/sec\n", (iterations * num_positions) / elapsed);
    
    free(logits);
}

bool test_deterministic(void) {
    printf("\n=== Test: Deterministic at Zero ===\n");
    
    float logits1[100] = {0};
    float logits2[100] = {0};
    
    for (int i = 0; i < 100; i++) {
        logits1[i] = ((float)rand() / RAND_MAX) * 10.0f;
        logits2[i] = logits1[i];
    }
    
    dither_apply_avx(logits1, 100, 42, 0.0f);
    dither_apply_avx(logits2, 100, 42, 0.0f);
    
    bool identical = true;
    for (int i = 0; i < 100; i++) {
        if (fabsf(logits1[i] - logits2[i]) > 1e-6f) {
            identical = false;
            break;
        }
    }
    
    printf("Result: %s\n", identical ? "PASS" : "FAIL");
    return identical;
}

bool test_entropy_reduction(void) {
    printf("\n=== Test: Entropy Reduction ===\n");
    
    int num_samples = 2000;
    int vocab_size = 1000;
    
    int* baseline_counts = calloc(vocab_size, sizeof(int));
    int* dithered_counts = calloc(vocab_size, sizeof(int));
    
    uint64_t rng_state = 12345;
    
    for (int i = 0; i < num_samples; i++) {
        float logits[vocab_size];
        for (int j = 0; j < vocab_size; j++) {
            logits[j] = 4.0f + ((float)rand() / RAND_MAX) * 2.0f;
        }
        
        int token = sample_token(logits, vocab_size, 0.7f, 42, 0.963f, rng_state + i);
        baseline_counts[token]++;
    }
    
    float intensity = 0.0809f;
    for (int i = 0; i < num_samples; i++) {
        float logits[vocab_size];
        for (int j = 0; j < vocab_size; j++) {
            logits[j] = 4.0f + ((float)rand() / RAND_MAX) * 2.0f;
        }
        
        dither_apply_avx(logits, vocab_size, i, intensity);
        int token = sample_token(logits, vocab_size, 0.7f, 42, 0.963f, rng_state + i);
        dithered_counts[token]++;
    }
    
    double baseline_entropy = 0.0;
    double dithered_entropy = 0.0;
    
    for (int i = 0; i < vocab_size; i++) {
        if (baseline_counts[i] > 0) {
            double p = (double)baseline_counts[i] / num_samples;
            baseline_entropy -= p * log2(p);
        }
        if (dithered_counts[i] > 0) {
            double p = (double)dithered_counts[i] / num_samples;
            dithered_entropy -= p * log2(p);
        }
    }
    
    int baseline_unique = 0;
    int dithered_unique = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (baseline_counts[i] > 0) baseline_unique++;
        if (dithered_counts[i] > 0) dithered_unique++;
    }
    
    printf("Baseline: unique=%d, entropy=%.3f\n", baseline_unique, baseline_entropy);
    printf("Dithered: unique=%d, entropy=%.3f\n", dithered_unique, dithered_entropy);
    printf("Delta: unique=%d, entropy=%+.3f\n", dithered_unique - baseline_unique, dithered_entropy - baseline_entropy);
    
    bool focus = (dithered_unique < baseline_unique) || (dithered_entropy < baseline_entropy);
    printf("Result: %s\n", focus ? "PASS (focusing)" : "FAIL (no focus)");
    
    free(baseline_counts);
    free(dithered_counts);
    
    return focus;
}
