#include "dither_avx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VOCAB 5000
#define SEQ_LEN 20
#define SAMPLES 100

static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

int sample_topk(float* logits, int n, int top_k, uint64_t rng) {
    if (top_k <= 0 || top_k >= n) top_k = n;
    
    float max_l = -1e30f;
    for (int i = 0; i < n; i++) if (logits[i] > max_l) max_l = logits[i];
    
    float threshold = -1e30f;
    if (top_k < n) {
        float max1 = -1e30f, max2 = -1e30f, max3 = -1e30f;
        for (int i = 0; i < n; i++) {
            if (logits[i] > max1) { max3 = max2; max2 = max1; max1 = logits[i]; }
            else if (logits[i] > max2) { max3 = max2; max2 = logits[i]; }
            else if (logits[i] > max3) { max3 = logits[i]; }
        }
        threshold = max3;
    }
    
    float sum = 0;
    for (int i = 0; i < n; i++) {
        if (logits[i] < threshold && top_k < n) logits[i] = -1e30f;
        else {
            logits[i] = expf(logits[i] - max_l);
            sum += logits[i];
        }
    }
    for (int i = 0; i < n; i++) logits[i] /= sum;
    
    float r = (splitmix64(&rng) & 0xFFFFFFFF) / (float)0xFFFFFFFF;
    float c = 0;
    for (int i = 0; i < n; i++) { c += logits[i]; if (c >= r) return i; }
    return n-1;
}

void dither_seq_fixed(float* logits, int vocab_size, float intensity) {
    if (intensity <= 0.0f) return;
    float max_l = -1e30f, min_l = 1e30f;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_l) max_l = logits[i];
        if (logits[i] < min_l) min_l = logits[i];
    }
    float range = max_l - min_l;
    if (range < 1e-6f) range = 1.0f;
    for (int tid = 0; tid < vocab_size; tid++) {
        float bias = intensity * (1.0f - (float)tid / vocab_size) * range;
        logits[tid] += bias;
    }
}

typedef struct { int seq_len; int unique; int* tokens; } Sequence;

Sequence generate_seq(int method, float intensity, uint64_t rng) {
    Sequence seq = {0, 0, NULL};
    seq.tokens = malloc(SEQ_LEN * sizeof(int));
    
    float base_logits[VOCAB];
    for (int i = 0; i < VOCAB; i++) base_logits[i] = (i < 20) ? 5.0f : 0.0f;
    
    for (int pos = 0; pos < SEQ_LEN; pos++) {
        float logits[VOCAB];
        memcpy(logits, base_logits, VOCAB * sizeof(float));
        
        if (method == 1) dither_apply_avx(logits, VOCAB, pos, intensity);
        if (method == 2) dither_seq_fixed(logits, VOCAB, intensity);
        
        seq.tokens[pos] = sample_topk(logits, VOCAB, 40, rng + pos);
    }
    seq.seq_len = SEQ_LEN;
    
    int unique_map[VOCAB] = {0};
    for (int i = 0; i < SEQ_LEN; i++) unique_map[seq.tokens[i]] = 1;
    for (int i = 0; i < VOCAB; i++) if (unique_map[i]) seq.unique++;
    
    return seq;
}

int main(void) {
    printf("=== END-TO-END TEST: Token-by-Token Generation ===\n\n");
    printf("Simulating %d tokens x %d sequences\n", SEQ_LEN, SAMPLES);
    printf("Each sequence: top-20 tokens at 5.0, rest at 0.0\n\n");
    
    dither_init();
    
    printf("Test 1: No dither (baseline)\n");
    int total_unique = 0;
    for (int s = 0; s < SAMPLES; s++) {
        Sequence seq = generate_seq(0, 0.0f, 12345 + s);
        total_unique += seq.unique;
        free(seq.tokens);
    }
    printf("  Avg unique tokens per seq: %.1f\n\n", (float)total_unique / SAMPLES);
    
    printf("Test 2: Original dither (position-varying)\n");
    total_unique = 0;
    for (int s = 0; s < SAMPLES; s++) {
        Sequence seq = generate_seq(1, 0.08f, 12345 + s);
        total_unique += seq.unique;
        free(seq.tokens);
    }
    printf("  Avg unique tokens per seq: %.1f\n\n", (float)total_unique / SAMPLES);
    
    printf("Test 3: Fixed bias dither (position-independent)\n");
    total_unique = 0;
    for (int s = 0; s < SAMPLES; s++) {
        Sequence seq = generate_seq(2, 0.05f, 12345 + s);
        total_unique += seq.unique;
        free(seq.tokens);
    }
    printf("  Avg unique tokens per seq: %.1f\n\n", (float)total_unique / SAMPLES);
    
    printf("=== CONCLUSION ===\n");
    printf("If Test 3 shows lower unique than Test 1:\n");
    printf("  -> Position-independent dither works in generation\n");
    printf("  -> We can now control output characteristics\n");
    
    return 0;
}
