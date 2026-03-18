#include "dither_avx.h"
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VOCAB_SIZE 32000
#define SAMPLES 500

typedef struct {
    float intensity;
    int unique_tokens;
    double entropy;
    float avg_top10_rank;
} Result;

float close_race_logits[VOCAB_SIZE];

static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

int sample_direct(float* logits, int vocab_size, float temperature, uint64_t rng_state) {
    if (temperature != 1.0f) {
        float inv_temp = 1.0f / temperature;
        for (int i = 0; i < vocab_size; i++) logits[i] *= inv_temp;
    }
    
    float max_logit = -1e30f;
    for (int i = 0; i < vocab_size; i++) if (logits[i] > max_logit) max_logit = logits[i];
    
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_logit);
        sum += logits[i];
    }
    for (int i = 0; i < vocab_size; i++) logits[i] /= sum;
    
    float r = (splitmix64(&rng_state) & 0xFFFFFFFF) / (float)0xFFFFFFFF;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (cumsum >= r) return i;
    }
    return vocab_size - 1;
}

void setup_close_race(void) {
    srand(42);
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (i < 10) {
            close_race_logits[i] = 5.0f + ((float)rand() / RAND_MAX) * 0.1f;
        } else {
            close_race_logits[i] = ((float)rand() / RAND_MAX) * 0.5f;
        }
    }
}

Result test_intensity(float intensity) {
    Result r = {intensity, 0, 0.0, 0.0};
    
    int* counts = calloc(VOCAB_SIZE, sizeof(int));
    if (!counts) return r;
    
    uint64_t rng = 12345;
    
    int top1_tokens[10] = {0};
    
    for (int s = 0; s < SAMPLES; s++) {
        float logits[VOCAB_SIZE];
        memcpy(logits, close_race_logits, VOCAB_SIZE * sizeof(float));
        
        if (intensity > 0) {
            dither_apply_avx(logits, VOCAB_SIZE, s, intensity);
        }
        
        int tok = sample_direct(logits, VOCAB_SIZE, 0.528f, rng + s);
        counts[tok]++;
        
        for (int i = 0; i < 10; i++) {
            if (tok == i) top1_tokens[i]++;
        }
    }
    
    int unique = 0;
    double entropy = 0.0;
    double top10_sum = 0.0;
    int top10_count = 0;
    
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (counts[i] > 0) {
            unique++;
            double p = (double)counts[i] / SAMPLES;
            entropy -= p * log2(p);
            if (i < 10) {
                top10_sum += counts[i];
                top10_count++;
            }
        }
    }
    
    r.unique_tokens = unique;
    r.entropy = entropy;
    r.avg_top10_rank = top10_count > 0 ? (double)top10_sum / SAMPLES : 0.0;
    
    printf("    Top tokens selected: ");
    for (int i = 0; i < 10; i++) printf("%d ", top1_tokens[i]);
    printf("\n");
    
    free(counts);
    return r;
}

int main(void) {
    printf("=== SMALL VOCAB TEST (matching research) ===\n\n");
    printf("Testing with small vocab to see if effect scales up\n\n");
    
    dither_init();
    setup_close_race();
    
    float intensities[] = {0.0, 0.0162, 0.0349, 0.0809, 0.1618};
    int n = sizeof(intensities) / sizeof(intensities[0]);
    
    printf("%-12s %-10s %-10s %-15s\n", "Intensity", "Unique", "Entropy", "Top-10 Share");
    printf("%-12s %-10s %-10s %-15s\n", "----------", "-------", "------", "-----------");
    
    Result baseline = test_intensity(0.0);
    printf("%-12.4f %-10d %-10.3f %-15.2f%%\n", 0.0f, baseline.unique_tokens, baseline.entropy, baseline.avg_top10_rank * 100);
    
    for (int i = 1; i < n; i++) {
        Result r = test_intensity(intensities[i]);
        double entropy_delta = r.entropy - baseline.entropy;
        int unique_delta = r.unique_tokens - baseline.unique_tokens;
        printf("%-12.4f %-10d %-10.3f %+d %-13.2f%%\n", 
               intensities[i], r.unique_tokens, r.entropy, unique_delta, r.avg_top10_rank * 100);
    }
    
    printf("\n=== INTERPRETATION ===\n\n");
    printf("Original research on close_race distribution:\n");
    printf("  Baseline unique: 87\n");
    printf("  Dithered unique: 33  (intensity=0.0809)\n");
    printf("  Entropy delta: -2.263 bits\n\n");
    
    printf("If our results show similar reduction:\n");
    printf("  - Mechanism is valid\n");
    printf("  - Effect is DISTRIBUTION-DEPENDENT (not universal)\n\n");
    
    printf("If our results show NO reduction:\n");
    printf("  - Implementation bug\n");
    printf("  - Or original research was on different model/sampling\n");
    
    return 0;
}
