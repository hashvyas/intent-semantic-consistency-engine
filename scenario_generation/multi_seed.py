"""
scenario_generation/multi_seed.py
=================================
Orchestration layer around the held-out scenario generator to support
deterministic, multi-seed evaluation dataset generation.
"""

from __future__ import annotations

import os
import json
import datetime
import subprocess
from typing import Any, Dict, List, Optional

from scenario_generation.generator import generate_held_out_suite


def _get_repository_version() -> str:
    """Attempt to resolve the current git commit hash, falling back to a version string."""
    try:
        # Run git command to get HEAD commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return "v2.0.0-untracked"


class MultiSeedOrchestrator:
    """Orchestrates generation of scenario suites across multiple random seeds."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def generate_suites(
        self,
        seeds: List[int],
        configs_limit: Optional[int] = None,
        message_count_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate isolated scenario suites for each seed and construct master metadata.

        Parameters
        ----------
        seeds : List[int]
            List of deterministic seeds to run.
        configs_limit : Optional[int]
            Limit count of scenarios generated per seed (for testing).
        message_count_override : Optional[int]
            Override count of messages generated per scenario (for testing).

        Returns
        -------
        Dict[str, Any]
            The master metadata dictionary.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        seeds = sorted(list(set(seeds)))
        total_messages_all = 0
        scenarios_per_seed = 0

        for seed in seeds:
            seed_dir = os.path.join(self.output_dir, f"seed_{seed}")
            
            # Call single-seed suite generator library
            suite_meta = generate_held_out_suite(
                output_dir=seed_dir,
                seed=seed,
                configs_limit=configs_limit,
                message_count_override=message_count_override
            )
            
            scenarios_per_seed = len(suite_meta)
            # Add up messages
            for scenario in suite_meta:
                total_messages_all += scenario.get("generated_message_count", 0)

        # Build consolidated master metadata
        master_metadata = {
            "generation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "repository_version": _get_repository_version(),
            "generation_seeds": seeds,
            "scenarios_per_seed": scenarios_per_seed,
            "total_messages": total_messages_all,
            "output_directory": os.path.abspath(self.output_dir),
            "generator_version": "B5-HeldOutGenerator-v1",
        }

        # Write master metadata file
        master_meta_path = os.path.join(self.output_dir, "multi_seed_metadata.json")
        with open(master_meta_path, "w", encoding="utf-8") as f:
            json.dump(master_metadata, f, indent=2)

        return master_metadata


def main() -> None:
    """Command-line interface entry point."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="scenario_generation.multi_seed",
        description="Multi-Seed Scenario Generation Infrastructure for B2 (Partial)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Space-separated list of integer seeds (e.g. 1 7 19 42 99)",
    )
    parser.add_argument(
        "--output-dir",
        default="test_messages/held_out",
        help="Root directory for storing seed-specific folders",
    )
    
    args = parser.parse_args()
    
    orchestrator = MultiSeedOrchestrator(args.output_dir)
    print(f"Starting multi-seed generation for seeds: {args.seeds}...")
    meta = orchestrator.generate_suites(args.seeds)
    
    print("\nGeneration Completed Successfully!")
    print(f"Master metadata saved to: {os.path.join(args.output_dir, 'multi_seed_metadata.json')}")
    print(f"Total scenarios per seed: {meta['scenarios_per_seed']}")
    print(f"Grand total messages:     {meta['total_messages']}")


if __name__ == "__main__":
    main()
