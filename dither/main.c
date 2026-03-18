#include "dither_avx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define ALPHA (1.0f / 137.0f)
#define PHI ((1.0f + sqrtf(5.0f)) / 2.0f)

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --benchmark       Run performance benchmark\n");
    printf("  --test-det        Test deterministic at zero\n");
    printf("  --test-ent        Test entropy reduction\n");
    printf("  --test-all        Run all tests\n");
    printf("  --vocab N         Vocabulary size (default: 32000)\n");
    printf("  --intensity F    Dithering intensity (default: 0.0809)\n");
    printf("  --help            Show this help\n");
}

int main(int argc, char** argv) {
    int vocab_size = 32000;
    float intensity = PHI / 20.0f;
    bool benchmark = false;
    bool test_det = false;
    bool test_ent = false;
    bool test_all = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0) {
            benchmark = true;
        } else if (strcmp(argv[i], "--test-det") == 0) {
            test_det = true;
        } else if (strcmp(argv[i], "--test-ent") == 0) {
            test_ent = true;
        } else if (strcmp(argv[i], "--test-all") == 0) {
            test_all = true;
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--intensity") == 0 && i + 1 < argc) {
            intensity = atof(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    
    printf("=== BitNet Dither (AVX2) ===\n");
    printf("Alpha (1/137): %.5f\n", ALPHA);
    printf("Phi/10: %.5f\n", PHI / 10.0f);
    printf("Default intensity: %.4f\n", intensity);
    
    dither_init();
    
    if (benchmark) {
        benchmark_dither(vocab_size, 100, 1000);
    }
    
    if (test_det || test_all) {
        test_deterministic();
    }
    
    if (test_ent || test_all) {
        test_entropy_reduction();
    }
    
    if (!benchmark && !test_det && !test_ent && !test_all) {
        printf("\nNo tests selected. Use --help for options.\n");
    }
    
    return 0;
}
