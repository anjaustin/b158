#include "dither_avx.h"
#include <stdio.h>

int main(void) {
    dither_init();
    
    float logits[16] = {0};
    
    printf("Setting up close race:\n");
    for (int i = 0; i < 16; i++) {
        if (i < 5) {
            logits[i] = 5.0f + (i * 0.02f);
        } else {
            logits[i] = 0.0f;
        }
        printf("  logits[%d] = %.4f\n", i, logits[i]);
    }
    
    printf("\nBefore dither (position=0, intensity=0.0809):\n");
    dither_apply_avx(logits, 16, 0, 0.0809f);
    for (int i = 0; i < 16; i++) {
        printf("  logits[%d] = %.4f\n", i, logits[i]);
    }
    
    printf("\n\n=== Different positions ===\n");
    float logits2[16];
    for (int pos = 0; pos < 8; pos++) {
        for (int i = 0; i < 16; i++) {
            logits2[i] = (i < 5) ? 5.0f + (i * 0.02f) : 0.0f;
        }
        dither_apply_avx(logits2, 16, pos, 0.0809f);
        printf("Position %d: top token = ", pos);
        for (int i = 0; i < 16; i++) {
            if (logits2[i] > logits2[0]) logits2[0] = logits2[i];
        }
        for (int i = 0; i < 16; i++) {
            if (logits2[i] == logits2[0]) printf("%d ", i);
        }
        printf("(max=%.3f)\n", logits2[0]);
    }
    
    printf("\n=== With high intensity (0.16) ===\n");
    for (int i = 0; i < 16; i++) {
        logits2[i] = (i < 5) ? 5.0f + (i * 0.02f) : 0.0f;
    }
    dither_apply_avx(logits2, 16, 0, 0.16f);
    printf("After: ");
    for (int i = 0; i < 16; i++) {
        printf("%.3f ", logits2[i]);
    }
    printf("\n");
    
    return 0;
}
