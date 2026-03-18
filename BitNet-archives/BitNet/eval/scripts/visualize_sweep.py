#!/usr/bin/env python3
"""
Visualize intensity sweep results.
Creates ASCII plots for terminal display.
"""

import json
from pathlib import Path

# Golden ratio
PHI = 1.618033988749895

def ascii_bar(value: float, max_val: float, width: int = 40, char: str = "█") -> str:
    """Create an ASCII bar."""
    if max_val == 0:
        return ""
    normalized = abs(value) / abs(max_val)
    bar_width = int(normalized * width)
    return char * bar_width


def ascii_plot(data: list, title: str = "") -> str:
    """Create ASCII visualization of sweep data."""
    lines = []
    
    lines.append("=" * 70)
    lines.append(f"  {title}")
    lines.append("=" * 70)
    lines.append("")
    
    # Entropy reduction plot
    lines.append("ENTROPY REDUCTION (bits) - Higher bar = more focusing effect")
    lines.append("-" * 70)
    
    max_entropy = max(abs(d['delta']['entropy']) for d in data)
    
    for d in data:
        intensity = d['intensity']
        entropy_delta = d['delta']['entropy']
        bar = ascii_bar(entropy_delta, max_entropy, width=40)
        
        # Mark golden ratio points
        marker = ""
        if abs(intensity - PHI/100) < 0.001:
            marker = " ← φ/100"
        elif abs(intensity - PHI/20) < 0.001:
            marker = " ← φ/20 (sweet spot)"
        elif abs(intensity - PHI/10) < 0.001:
            marker = " ← φ/10"
        
        lines.append(f"  {intensity:.4f} │{bar}│ {entropy_delta:+.2f}{marker}")
    
    lines.append("")
    lines.append("")
    
    # Unique token reduction plot  
    lines.append("UNIQUE TOKEN REDUCTION - Higher bar = more concentration")
    lines.append("-" * 70)
    
    max_unique = max(abs(d['delta']['unique_tokens']) for d in data)
    
    for d in data:
        intensity = d['intensity']
        unique_delta = d['delta']['unique_tokens']
        bar = ascii_bar(unique_delta, max_unique, width=40)
        lines.append(f"  {intensity:.4f} │{bar}│ {unique_delta:+d}")
    
    lines.append("")
    lines.append("=" * 70)
    
    # Summary statistics
    avg_entropy = sum(d['delta']['entropy'] for d in data) / len(data)
    avg_unique = sum(d['delta']['unique_tokens'] for d in data) / len(data)
    
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Intensity range: [{data[0]['intensity']:.4f}, {data[-1]['intensity']:.4f}]")
    lines.append(f"  Based on: [φ/100, φ/10] where φ = {PHI:.6f}")
    lines.append("")
    lines.append(f"  Average entropy reduction: {avg_entropy:.2f} bits")
    lines.append(f"  Average unique token reduction: {avg_unique:.0f} tokens")
    lines.append("")
    lines.append("  INTERPRETATION:")
    lines.append("  • Negative entropy = MORE focused decisions (less randomness)")
    lines.append("  • Negative unique tokens = FEWER distinct tokens selected")
    lines.append("  • Effect is consistent across the golden ratio intensity range")
    lines.append("  • Ordered dithering creates STRUCTURED preferences, not chaos")
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def plot_response_curve(data: list) -> str:
    """Create response curve visualization."""
    lines = []
    
    lines.append("")
    lines.append("RESPONSE CURVE: Intensity vs Effect")
    lines.append("")
    
    # Create a simple 2D ASCII plot
    height = 15
    width = 50
    
    # Normalize data
    intensities = [d['intensity'] for d in data]
    entropies = [-d['delta']['entropy'] for d in data]  # Negate for "upward is more effect"
    
    min_i, max_i = min(intensities), max(intensities)
    min_e, max_e = min(entropies), max(entropies)
    
    # Create grid
    grid = [[' ' for _ in range(width + 10)] for _ in range(height + 2)]
    
    # Plot points
    for i, e in zip(intensities, entropies):
        x = int((i - min_i) / (max_i - min_i) * (width - 1)) + 8
        y = height - int((e - min_e) / (max_e - min_e) * (height - 1)) - 1
        grid[y][x] = '●'
    
    # Add axes
    for y in range(height):
        grid[y][7] = '│'
    for x in range(8, width + 8):
        grid[height][x] = '─'
    grid[height][7] = '└'
    
    # Add labels
    grid[0][0:7] = list(f"{max_e:.1f}  ")
    grid[height-1][0:7] = list(f"{min_e:.1f}  ")
    
    # Title
    lines.append("    Effect (bits of entropy reduction)")
    lines.append("      ▲")
    
    for row in grid:
        lines.append("".join(row))
    
    lines.append(f"        {min_i:.2f}" + " " * (width - 15) + f"{max_i:.2f}")
    lines.append("                    Intensity (φ-scaled) ──►")
    lines.append("")
    
    return "\n".join(lines)


def main():
    # Load sweep data
    data_path = Path(__file__).parent.parent / "data" / "intensity_sweep.json"
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Run: python dither_inference.py --mode sweep")
        return
    
    with open(data_path) as f:
        data = json.load(f)
    
    # Generate visualizations
    print(ascii_plot(data, "ORDERED DITHERING INTENSITY SWEEP"))
    print(plot_response_curve(data))
    
    # The key insight
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║  KEY INSIGHT:                                                      ║")
    print("║                                                                    ║")
    print("║  Ordered dithering FOCUSES decisions, not scatters them.          ║")
    print("║                                                                    ║")
    print("║  The Bayer pattern creates DETERMINISTIC preferences that          ║")
    print("║  break ties consistently rather than randomly.                     ║")
    print("║                                                                    ║")
    print("║  This is the 'resolution enhancement' effect:                      ║")
    print("║  • Clear winners → unchanged (confidence preserved)                ║")
    print("║  • Ambiguous decisions → structured selection (coherence)          ║")
    print("║                                                                    ║")
    print("║  For language generation, this could mean:                         ║")
    print("║  • More coherent reasoning chains                                  ║")
    print("║  • More consistent voice/style                                     ║")
    print("║  • Less random wandering in semantic space                         ║")
    print("║                                                                    ║")
    print("╚" + "═" * 68 + "╝")


if __name__ == "__main__":
    main()
