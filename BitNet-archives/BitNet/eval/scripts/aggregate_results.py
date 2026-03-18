#!/usr/bin/env python3
"""
Aggregate experiment results and generate summary reports.

Usage:
    python aggregate_results.py [--experiments DIR] [--output FILE]
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Project paths
EVAL_DIR = Path(__file__).parent.parent
EXPERIMENTS_DIR = EVAL_DIR / "experiments"
REGISTRY_PATH = EVAL_DIR / "registry.yaml"
OUTPUT_DIR = EVAL_DIR / "reports"


def load_registry() -> Dict:
    """Load the experiment registry."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return yaml.safe_load(f)
    return {"experiments": [], "metadata": {}}


def parse_experiment_frontmatter(filepath: Path) -> Optional[Dict]:
    """Extract YAML frontmatter from experiment markdown file."""
    try:
        with open(filepath) as f:
            content = f.read()
        
        if not content.startswith("---"):
            return None
        
        # Find end of frontmatter
        end_idx = content.find("---", 3)
        if end_idx == -1:
            return None
        
        frontmatter = content[3:end_idx].strip()
        return yaml.safe_load(frontmatter)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def discover_experiments(experiments_dir: Path) -> List[Dict]:
    """Discover all experiment files and parse their metadata."""
    experiments = []
    
    for md_file in experiments_dir.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue  # Skip templates
        
        metadata = parse_experiment_frontmatter(md_file)
        if metadata:
            metadata["_path"] = str(md_file.relative_to(experiments_dir))
            experiments.append(metadata)
    
    return experiments


def compute_summary_stats(experiments: List[Dict]) -> Dict:
    """Compute summary statistics across experiments."""
    stats = {
        "total": len(experiments),
        "by_status": {},
        "by_algorithm": {},
        "passed_primary": 0,
        "rejected": 0,
        "enhancement_scores": [],
    }
    
    for exp in experiments:
        # Count by status
        status = exp.get("status", "unknown")
        stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
        
        # Count by algorithm
        algo = exp.get("algorithm", "unknown")
        stats["by_algorithm"][algo] = stats["by_algorithm"].get(algo, 0) + 1
        
        # Primary criteria
        primary = exp.get("primary_criteria", {})
        if all(primary.get(k) for k in ["no_regression", "perplexity_preserved", 
                                         "math_accuracy_preserved", "stability_preserved"]):
            stats["passed_primary"] += 1
        elif any(v is False for v in primary.values()):
            stats["rejected"] += 1
        
        # Enhancement scores
        enhancement = exp.get("enhancement_score")
        if enhancement is not None:
            stats["enhancement_scores"].append(enhancement)
    
    return stats


def generate_summary_table(experiments: List[Dict]) -> str:
    """Generate markdown summary table."""
    lines = [
        "| ID | Algorithm | Date | Status | Primary | Enhancement |",
        "|----|-----------|----- |--------|---------|-------------|"
    ]
    
    for exp in sorted(experiments, key=lambda x: x.get("date", "")):
        exp_id = exp.get("id", "?")
        algo = exp.get("algorithm", "?")
        date = exp.get("date", "?")
        status = exp.get("status", "?")
        
        # Compute primary pass/fail
        primary = exp.get("primary_criteria", {})
        if not primary:
            primary_str = "-"
        elif all(primary.get(k) for k in ["no_regression", "perplexity_preserved",
                                           "math_accuracy_preserved", "stability_preserved"]):
            primary_str = "PASS"
        elif any(v is False for v in primary.values()):
            primary_str = "FAIL"
        else:
            primary_str = "pending"
        
        enhancement = exp.get("enhancement_score")
        enhancement_str = f"{enhancement}/5" if enhancement is not None else "-"
        
        lines.append(f"| {exp_id} | {algo} | {date} | {status} | {primary_str} | {enhancement_str} |")
    
    return "\n".join(lines)


def generate_benchmark_comparison(experiments: List[Dict]) -> str:
    """Generate benchmark comparison across experiments."""
    # Find baseline
    baseline = None
    for exp in experiments:
        if exp.get("algorithm") == "baseline" and exp.get("status") == "complete":
            baseline = exp
            break
    
    if not baseline:
        return "*No baseline experiment found.*"
    
    lines = [
        "## Benchmark Comparison vs Baseline",
        "",
        "| Algorithm | MMLU | GSM8K | MATH | HellaSwag | ARC |",
        "|-----------|------|-------|------|-----------|-----|"
    ]
    
    baseline_results = baseline.get("results", {})
    lines.append(f"| baseline | {baseline_results.get('mmlu', {}).get('accuracy', '-')} | "
                 f"{baseline_results.get('gsm8k', {}).get('accuracy', '-')} | "
                 f"{baseline_results.get('math', {}).get('accuracy', '-')} | "
                 f"{baseline_results.get('hellaswag', {}).get('accuracy', '-')} | "
                 f"{baseline_results.get('arc_challenge', {}).get('accuracy', '-')} |")
    
    for exp in experiments:
        if exp.get("algorithm") == "baseline":
            continue
        if exp.get("status") != "complete":
            continue
        
        algo = exp.get("algorithm", "?")
        results = exp.get("results", {})
        
        lines.append(f"| {algo} | {results.get('mmlu', {}).get('accuracy', '-')} | "
                     f"{results.get('gsm8k', {}).get('accuracy', '-')} | "
                     f"{results.get('math', {}).get('accuracy', '-')} | "
                     f"{results.get('hellaswag', {}).get('accuracy', '-')} | "
                     f"{results.get('arc_challenge', {}).get('accuracy', '-')} |")
    
    return "\n".join(lines)


def generate_report(experiments: List[Dict], output_path: Path) -> None:
    """Generate full aggregation report."""
    stats = compute_summary_stats(experiments)
    
    report = f"""# Experiment Aggregation Report

Generated: {datetime.utcnow().isoformat()}Z

## Summary Statistics

- **Total experiments**: {stats['total']}
- **Passed primary criteria**: {stats['passed_primary']}
- **Rejected**: {stats['rejected']}

### By Status

| Status | Count |
|--------|-------|
"""
    
    for status, count in sorted(stats["by_status"].items()):
        report += f"| {status} | {count} |\n"
    
    report += f"""
### By Algorithm

| Algorithm | Count |
|-----------|-------|
"""
    
    for algo, count in sorted(stats["by_algorithm"].items()):
        report += f"| {algo} | {count} |\n"
    
    report += f"""
## Experiment Summary

{generate_summary_table(experiments)}

{generate_benchmark_comparison(experiments)}

## Experiments by Section

"""
    
    # Group by sections
    section_map = {i: [] for i in range(1, 10)}
    for exp in experiments:
        for sec in exp.get("sections", []):
            if sec in section_map:
                section_map[sec].append(exp.get("id", "?"))
    
    for sec, exp_ids in section_map.items():
        report += f"- **Section {sec}**: {', '.join(exp_ids) if exp_ids else 'None'}\n"
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR,
                        help="Path to experiments directory")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "aggregation_report.md",
                        help="Output report path")
    args = parser.parse_args()
    
    print(f"Discovering experiments in: {args.experiments}")
    experiments = discover_experiments(args.experiments)
    print(f"Found {len(experiments)} experiments")
    
    generate_report(experiments, args.output)


if __name__ == "__main__":
    main()
