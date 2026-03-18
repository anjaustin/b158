#!/usr/bin/env python3
"""
Create a new experiment from template.

Usage:
    python new_experiment.py <algorithm_name> [--category CATEGORY] [--tags TAG1,TAG2]
    
Example:
    python new_experiment.py dithering_v1 --category dithering --tags math,reasoning
"""

import os
import sys
import yaml
import shutil
import argparse
from pathlib import Path
from datetime import datetime

EVAL_DIR = Path(__file__).parent.parent
TEMPLATE_PATH = EVAL_DIR / "experiments" / "_template.md"
REGISTRY_PATH = EVAL_DIR / "registry.yaml"


def load_registry() -> dict:
    """Load the experiment registry."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return yaml.safe_load(f)
    return {
        "metadata": {"project": "BitNet Optimization", "created": datetime.now().strftime("%Y-%m-%d")},
        "next_id": 1,
        "experiments": [],
        "algorithms": {},
        "tags": {},
        "sections": {i: [] for i in range(1, 10)},
        "stats": {"total_runs": 0, "passed_primary": 0, "rejected": 0}
    }


def save_registry(registry: dict) -> None:
    """Save the experiment registry."""
    registry["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)


def create_experiment(algorithm: str, category: str = "other", tags: list = None) -> Path:
    """Create a new experiment from template."""
    tags = tags or []
    
    # Load registry and get next ID
    registry = load_registry()
    exp_id = f"exp_{registry['next_id']:03d}"
    registry["next_id"] += 1
    
    # Determine output path
    today = datetime.now().strftime("%Y-%m-%d")
    
    if algorithm == "baseline":
        output_dir = EVAL_DIR / "experiments" / "baseline"
    else:
        output_dir = EVAL_DIR / "experiments" / "algorithms" / algorithm
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{today}_{algorithm}.md"
    
    # Handle existing file
    if output_path.exists():
        counter = 1
        while output_path.exists():
            output_path = output_dir / f"{today}_{algorithm}_{counter}.md"
            counter += 1
    
    # Read template and substitute
    with open(TEMPLATE_PATH) as f:
        template = f.read()
    
    content = template.replace("exp_XXX", exp_id)
    content = content.replace("<algorithm_name>", algorithm)
    content = content.replace("YYYY-MM-DD", today)
    content = content.replace('"HH:MM:SS"', f'"{datetime.now().strftime("%H:%M:%S")}"')
    content = content.replace("tags: []", f"tags: {tags}")
    content = content.replace("<Title>", f"{algorithm.replace('_', ' ').title()}")
    
    # Write experiment file
    with open(output_path, "w") as f:
        f.write(content)
    
    # Update registry
    registry["experiments"].append({
        "id": exp_id,
        "algorithm": algorithm,
        "date": today,
        "status": "pending",
        "path": str(output_path.relative_to(EVAL_DIR)),
        "tags": tags,
        "sections": [],
        "primary_pass": None,
        "enhancement_score": None
    })
    
    # Update algorithm registry
    if algorithm not in registry.get("algorithms", {}):
        registry["algorithms"][algorithm] = {
            "description": f"Algorithm: {algorithm}",
            "category": category,
            "experiments": []
        }
    registry["algorithms"][algorithm]["experiments"].append(exp_id)
    
    # Update tag index
    for tag in tags:
        if tag not in registry.get("tags", {}):
            registry["tags"][tag] = []
        registry["tags"][tag].append(exp_id)
    
    save_registry(registry)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create new experiment from template")
    parser.add_argument("algorithm", help="Algorithm name (e.g., dithering_v1)")
    parser.add_argument("--category", default="other",
                        choices=["control", "dithering", "signal_processing", "sampling", "other"],
                        help="Algorithm category")
    parser.add_argument("--tags", default="",
                        help="Comma-separated tags (e.g., math,reasoning)")
    args = parser.parse_args()
    
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    
    output_path = create_experiment(args.algorithm, args.category, tags)
    
    print(f"Created experiment: {output_path}")
    print(f"Registry updated: {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
