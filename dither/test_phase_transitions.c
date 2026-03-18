#include "dither_avx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define NUM_INTENSITIES 12
#define SAMPLES_PER_INTENSITY 500
#define VOCAB_SIZE 1000

static float intensities[NUM_INTENSITIES] = {
    0.0f,    // baseline
    0.0073f, // alpha (lower bound)
    0.037f,  // between A and B
    0.05f,   // near A→B boundary
    0.074f,  // upper bound of A
    0.096f,  // lower bound of B
    0.12f,   // middle of B
    0.143f,  // upper bound of B
    0.144f,  // lower bound of C (claimed transition)
    0.155f,  // middle of C
    0.1618f, // phi/10 (upper bound)
    0.20f    // above range (test extrapolation)
};

typedef struct {
    float intensity;
    int unique_tokens;
    double entropy;
    int top_token_freq;
    float avg_rank;
} IntensityResult;

static float simulate_logit_mode(int token_id, int mode) {
    float base = 0.0f;
    
    if (mode == 0) {
        base = 5.0f - (token_id * 0.01f);
    }
    else if (mode == 1) {
        if (token_id < 10) base = 8.0f;
        else if (token_id < 20) base = 5.0f;
        else base = 1.0f;
    }
    else if (mode == 2) {
        base = 4.0f + sinf(token_id * 0.1f) * 2.0f;
    }
    
    return base + ((float)rand() / RAND_MAX) * 0.5f;
}

IntensityResult run_intensity_test(float intensity, int mode) {
    IntensityResult result = {0};
    result.intensity = intensity;
    
    int* token_counts = calloc(VOCAB_SIZE, sizeof(int));
    if (!token_counts) return result;
    
    uint64_t rng_state = 12345;
    
    for (int sample = 0; sample < SAMPLES_PER_INTENSITY; sample++) {
        float logits[VOCAB_SIZE];
        
        for (int i = 0; i < VOCAB_SIZE; i++) {
            logits[i] = simulate_logit_mode(i, mode);
        }
        
        if (intensity > 0.0f) {
            dither_apply_avx(logits, VOCAB_SIZE, sample, intensity);
        }
        
        int token = sample_token(logits, VOCAB_SIZE, 0.7f, 42, 0.963f, rng_state + sample);
        token_counts[token]++;
    }
    
    result.unique_tokens = 0;
    result.top_token_freq = 0;
    double entropy = 0.0;
    double rank_sum = 0.0;
    
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (token_counts[i] > 0) {
            result.unique_tokens++;
            if (token_counts[i] > result.top_token_freq) {
                result.top_token_freq = token_counts[i];
            }
            double p = (double)token_counts[i] / SAMPLES_PER_INTENSITY;
            entropy -= p * log2(p);
            rank_sum += (double)token_counts[i] * (i + 1);
        }
    }
    
    result.entropy = entropy;
    result.avg_rank = rank_sum / SAMPLES_PER_INTENSITY;
    
    free(token_counts);
    return result;
}

void analyze_phase_transitions(void) {
    printf("\n=== PHASE TRANSITION VERIFICATION ===\n\n");
    
    printf("Testing with simulated multimodal logit distributions\n");
    printf("Each mode has different selection bias:\n");
    printf("  Mode 0: Gradient (prefers low token IDs)\n");
    printf("  Mode 1: Cluster (prefers tokens 0-20)\n");
    printf("  Mode 2: Oscillating (sinusoidal preference)\n\n");
    
    for (int mode = 0; mode < 3; mode++) {
        printf("--- Mode %d ---\n", mode);
        printf("%-12s %-12s %-10s %-12s %-12s\n", "Intensity", "Unique", "Entropy", "Top Freq", "Avg Rank");
        printf("%-12s %-12s %-10s %-12s %-12s\n", "----------", "-------", "------", "---------", "--------");
        
        IntensityResult prev = {0};
        int transitions = 0;
        
        for (int i = 0; i < NUM_INTENSITIES; i++) {
            IntensityResult r = run_intensity_test(intensities[i], mode);
            
            printf("%-12.4f %-12d %-10.3f %-12d %-12.2f\n",
                   r.intensity, r.unique_tokens, r.entropy, r.top_token_freq, r.avg_rank);
            
            if (i > 0) {
                double entropy_delta = r.entropy - prev.entropy;
                double rank_delta = r.avg_rank - prev.avg_rank;
                
                if (fabs(entropy_delta) > 0.1 || fabs(rank_delta) > 50.0) {
                    transitions++;
                    printf("  >>> TRANSITION at %.4f: entropy Δ%.3f, rank Δ%.2f\n",
                           r.intensity, entropy_delta, rank_delta);
                }
            }
            
            prev = r;
        }
        
        printf("Transitions detected: %d\n\n", transitions);
    }
}

void test_boundary_precisions(void) {
    printf("\n=== BOUNDARY PRECISION TEST ===\n\n");
    
    float boundaries[] = {0.074f, 0.143f, 0.144f, 0.1618f};
    float test_intensities[8];
    
    for (int b = 0; b < 4; b++) {
        printf("Testing boundary %.4f:\n", boundaries[b]);
        
        test_intensities[0] = boundaries[b] - 0.010f;
        test_intensities[1] = boundaries[b] - 0.005f;
        test_intensities[2] = boundaries[b] - 0.001f;
        test_intensities[3] = boundaries[b];
        test_intensities[4] = boundaries[b] + 0.001f;
        test_intensities[5] = boundaries[b] + 0.005f;
        test_intensities[6] = boundaries[b] + 0.010f;
        
        for (int i = 0; i < 7; i++) {
            IntensityResult r = run_intensity_test(test_intensities[i], 1);
            printf("  %.4f: unique=%d, entropy=%.3f, avg_rank=%.2f\n",
                   test_intensities[i], r.unique_tokens, r.entropy, r.avg_rank);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    printf("=== Phase Transition Verification Test ===\n");
    printf("Testing whether intensity values produce discrete phase transitions\n");
    printf("in token selection behavior.\n\n");
    
    dither_init();
    
    analyze_phase_transitions();
    test_boundary_precisions();
    
    printf("\n=== INTERPRETATION ===\n\n");
    printf("If phase transitions are REAL:\n");
    printf("  - There should be discrete jumps in entropy/unique tokens at boundaries\n");
    printf("  - Boundaries: ~0.074 (A→B), ~0.144 (B→C), ~0.1618 (C→D)\n\n");
    printf("If phase transitions are NOT real:\n");
    printf("  - Changes will be gradual across intensity spectrum\n");
    printf("  - No discrete jumps at specific intensity values\n\n");
    
    return 0;
}
