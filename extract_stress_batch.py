"""
extract_stress_batch.py
=======================
Extract a diverse 1 000-object stress-test batch from the local V2X dataset.

Design
------
* Walk every .json file under ./json/Mobile  -> reservoir-sample 500 CAM items.
* Walk every .json file under ./json/Stationary -> reservoir-sample 500 CAM items.
* Combine, shuffle, write to ./b1_stress_test_batch.json.

Memory strategy
---------------
Each file is a dict keyed by ROS topic.  The CAM messages live at key
'/v2x/cam' as a JSON array.  We use ``ijson`` to stream those items one at a
time with the prefix '/v2x/cam.item' so the file is never fully loaded
into RAM.

Reservoir sampling (Vitter's Algorithm R) gives a uniform random sample of
exactly N items from a stream of unknown length, keeping only N items in RAM
at any time, regardless of how large the file or dataset is.

Usage
-----
    python extract_stress_batch.py [--seed SEED]

Dependencies: ijson  (pip install ijson)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Try to import ijson; give a clear error if missing.
# ---------------------------------------------------------------------------
try:
    import ijson
except ImportError:
    sys.exit(
        "ERROR: ijson is not installed.  Run:  pip install ijson\n"
        "ijson is required for memory-safe streaming of large JSON files."
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT    = Path(__file__).resolve().parent
_JSON_ROOT       = _PROJECT_ROOT / "json"
_MOBILE_ROOT     = _JSON_ROOT / "Mobile"
_STATIONARY_ROOT = _JSON_ROOT / "Stationary"
_OUTPUT_PATH     = _PROJECT_ROOT / "b1_stress_test_batch.json"

# Number of samples to draw from each tree
_SAMPLES_PER_TREE = 500

# ijson prefix for individual CAM message objects inside each file.
# File layout: { "/v2x/cam": [ {item}, {item}, ... ], ... }
_CAM_ITEM_PREFIX = "/v2x/cam.item"


# ---------------------------------------------------------------------------
# Reservoir sampling
# ---------------------------------------------------------------------------

def _reservoir_sample_from_tree(
    tree_root: Path,
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Walk every .json file under *tree_root*, stream CAM items via ijson,
    and return a uniform random sample of exactly *n* items using
    Vitter's Algorithm R.

    Parameters
    ----------
    tree_root : Path
        Root directory to walk (Mobile or Stationary).
    n : int
        Reservoir size (target number of items to keep).
    rng : random.Random
        Seeded Random instance for reproducibility.

    Returns
    -------
    list
        Up to *n* sampled dicts (fewer only if the entire tree contains
        fewer than *n* CAM items).
    """
    reservoir: list[dict[str, Any]] = []
    total_seen = 0  # total CAM items streamed so far across all files

    # Discover all json files first so we can report progress
    json_files: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(tree_root):
        for fname in filenames:
            if fname.lower().endswith(".json"):
                json_files.append(os.path.join(dirpath, fname))
    json_files.sort()

    n_files = len(json_files)
    print(f"  Found {n_files} .json file(s) under {tree_root.name}/")

    for file_idx, filepath in enumerate(json_files, start=1):
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        try:
            items_from_file = 0
            with open(filepath, "rb") as fh:
                # Stream only the items inside '/v2x/cam'; ijson yields
                # complete Python dicts one at a time with O(1) RAM overhead.
                parser = ijson.items(fh, _CAM_ITEM_PREFIX)
                for item in parser:
                    total_seen += 1
                    items_from_file += 1

                    if len(reservoir) < n:
                        # Phase 1: fill the reservoir
                        reservoir.append(item)
                    else:
                        # Phase 2: replace a random slot with probability n/total_seen
                        j = rng.randint(0, total_seen - 1)
                        if j < n:
                            reservoir[j] = item

            print(
                f"  [{file_idx:>4}/{n_files}] "
                f"{Path(filepath).name:<50} "
                f"{file_size_mb:>7.1f} MB  "
                f"{items_from_file:>6} CAM  "
                f"reservoir={len(reservoir):>4}  seen={total_seen:>8}"
            )

        except Exception as exc:
            # Unreadable / malformed / empty file: log and continue
            print(
                f"  [{file_idx:>4}/{n_files}] WARNING: skipping {filepath!r}: "
                f"{type(exc).__name__}: {exc}"
            )

    print(
        f"\n  Done: {total_seen:,} total CAM items streamed from {tree_root.name}/, "
        f"{len(reservoir)} sampled into reservoir."
    )
    return reservoir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract a diverse 1000-object stress-test batch from the V2X dataset."
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    t0 = time.time()

    # Validate roots
    for root in (_MOBILE_ROOT, _STATIONARY_ROOT):
        if not root.is_dir():
            sys.exit(f"ERROR: directory not found: {root}")

    # ---- Sample Mobile -------------------------------------------------------
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"Sampling {_SAMPLES_PER_TREE} items from Mobile/  (seed={args.seed})")
    print(sep)
    mobile_samples = _reservoir_sample_from_tree(_MOBILE_ROOT, _SAMPLES_PER_TREE, rng)

    # ---- Sample Stationary ---------------------------------------------------
    print(f"\n{sep}")
    print(f"Sampling {_SAMPLES_PER_TREE} items from Stationary/  (seed={args.seed})")
    print(sep)
    stationary_samples = _reservoir_sample_from_tree(_STATIONARY_ROOT, _SAMPLES_PER_TREE, rng)

    # ---- Combine, tag source, shuffle ----------------------------------------
    print(f"\nCombining and shuffling ...")
    for item in mobile_samples:
        item["_source"] = "Mobile"
    for item in stationary_samples:
        item["_source"] = "Stationary"

    combined = mobile_samples + stationary_samples
    rng.shuffle(combined)
    total = len(combined)

    print(
        f"  {total} items total  "
        f"({len(mobile_samples)} Mobile + {len(stationary_samples)} Stationary)"
    )

    # ---- Write output --------------------------------------------------------
    print(f"\nWriting -> {_OUTPUT_PATH}")
    with _OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2, ensure_ascii=False)

    out_mb = _OUTPUT_PATH.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0

    print(
        f"  Written: {total} objects  |  "
        f"{out_mb:.1f} MB  |  "
        f"{elapsed:.1f}s elapsed"
    )
    print(f"\nOutput saved to: {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
