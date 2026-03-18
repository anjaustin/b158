#include "dither_avx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    float logits[16] = {0};
    logits[0] = 5.0f;
    logits[1] = 4.95f;
    
    printf("Before dither:\n");
    for (int i = 0; i < 16; i++) printf("  %d: %.4f\n", i, logits[i]);
    
    dither_apply_avx(logits, 16, 0, 0.0809f);
    
    printf("\nAfter dither (pos=0, intensity=0.0809):\n");
    for (int i = 0; i < 16; i++) printf("  %d: %.4f\n", i, logits[i]);
    
    return 0;
}
