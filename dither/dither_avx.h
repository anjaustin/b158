#ifndef DITHER_AVX_H
#define DITHER_AVX_H

#include <stdint.h>
#include <stdbool.h>

#define VOCAB_SIZE 32000
#define BAYER_SIZE 8

typedef struct {
    float* logits;
    int vocab_size;
} LogitsContext;

void dither_init(void);

void dither_apply_avx(float* logits, int vocab_size, int position, float intensity);

void softmax_avx(float* logits, int vocab_size);

int sample_token(float* logits, int vocab_size, float temperature, int top_k, float top_p, uint64_t rng_state);
int sample_token_notopk(float* logits, int vocab_size, float temperature, float top_p, uint64_t rng_state);

void benchmark_dither(int vocab_size, int num_positions, int iterations);

bool test_deterministic(void);

bool test_entropy_reduction(void);

#endif
