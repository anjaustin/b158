#include "dither_avx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VOCAB 1000
#define SAMPLES 10000

static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

int sample_simple(float* logits, int n, uint64_t rng) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        logits[i] = expf(logits[i]);
        sum += logits[i];
    }
    for (int i = 0; i < n; i++) logits[i] /= sum;
    float r = (splitmix64(&rng) & 0xFFFFFFFF) / (float)0xFFFFFFFF;
    float c = 0;
    for (int i = 0; i < n; i++) {
        c += logits[i];
        if (c >= r) return i;
    }
    return n-1;
}

int main(void) {
    printf("=== FIXED POSITION TEST ===\n\n");
    printf("Testing with SAME position for all samples\n");
    printf("This tests if dither creates consistent bias\n\n");
    
    dither_init();
    
    int counts_no_dither[VOCAB] = {0};
    int counts_fixed_pos[VOCAB] = {0};
    
    printf("Test 1: No dither (baseline)\n");
    for (int s = 0; s < SAMPLES; s++) {
        float logits[VOCAB];
        for (int i = 0; i < VOCAB; i++) logits[i] = (i < 10) ? 5.0f : 0.0f;
        int t = sample_simple(logits, VOCAB, 12345 + s);
        counts_no_dither[t]++;
    }
    
    int unique = 0;
    for (int i = 0; i < VOCAB; i++) if (counts_no_dither[i] > 0) unique++;
    printf("  Unique tokens: %d\n", unique);
    
    printf("Test 2: Dither at FIXED position (42)\n");
    for (int s = 0; s < SAMPLES; s++) {
        float logits[VOCAB];
        for (int i = 0; i < VOCAB; i++) logits[i] = (i < 10) ? 5.0f : 0.0f;
        dither_apply_avx(logits, VOCAB, 42, 0.0809f);
        int t = sample_simple(logits, VOCAB, 12345 + s);
        counts_fixed_pos[t]++;
    }
    
    unique = 0;
    for (int i = 0; i < VOCAB; i++) if (counts_fixed_pos[i] > 0) unique++;
    printf("  Unique tokens: %d\n", unique);
    
    printf("\n  Top 10 token distribution:\n  ");
    for (int i = 0; i < 10; i++) {
        printf("%d:%d ", i, counts_fixed_pos[i]);
    }
    printf("\n");
    
    printf("\nTest 3: Dither at RANDOM positions\n");
    int counts_random[VOCAB] = {0};
    for (int s = 0; s < SAMPLES; s++) {
        float logits[VOCAB];
        for (int i = 0; i < VOCAB; i++) logits[i] = (i < 10) ? 5.0f : 0.0f;
        dither_apply_avx(logits, VOCAB, s, 0.0809f);
        int t = sample_simple(logits, VOCAB, 12345 + s);
        counts_random[t]++;
    }
    
    unique = 0;
    for (int i = 0; i < VOCAB; i++) if (counts_random[i] > 0) unique++;
    printf("  Unique tokens: %d\n", unique);
    printf("  Top 10: ");
    for (int i = 0; i < 10; i++) {
        printf("%d:%d ", i, counts_random[i]);
    }
    printf("\n");
    
    printf("\n=== CONCLUSION ===\n");
    printf("If FIXED position has fewer unique than RANDOM:\n");
    printf("  -> Dither creates consistent bias (focusing effect)\n");
    printf("If they're the same:\n");
    printf("  -> Position-averaging washes out the effect\n");
    
    return 0;
}
