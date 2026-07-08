"""
pipeline/tests/test_b5_generator.py
=====================================
Unit tests for B5 — Held-Out / Distribution-Shift Scenario Generation.

Verifies:
  1. Determinism: identical seeds generate identical scenarios.
  2. Distinctness: different seeds generate different scenarios.
  3. Schema validation: all generated messages are accepted by the existing B1 parser.
  4. Physical plausibility: benign messages satisfy physical realism constraints.
  5. Isolation: generated scenarios remain isolated in the dedicated output directory.
"""

from __future__ import annotations

import os
import json
import shutil
import tempfile
import pathlib
import pytest

from scenario_generation.generator import (
    ScenarioConfig,
    HeldOutScenarioGenerator,
    generate_held_out_suite,
)
from b1_scsv.models import safe_parse_cam
from b1_scsv.scsv import SCSV


def _calculate_dir_hash(dir_path: str) -> dict[str, str]:
    """Calculate file content mappings to verify exact deterministic generation."""
    path = pathlib.Path(dir_path)
    file_contents = {}
    for p in sorted(path.glob("**/*.json")):
        rel = p.relative_to(path)
        with p.open(encoding="utf-8") as f:
            file_contents[str(rel)] = f.read()
    return file_contents


class TestHeldOutGenerator:
    def test_determinism(self):
        """Identical seeds must generate byte-for-byte identical scenario suites."""
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            generate_held_out_suite(tmp1, seed=42, configs_limit=3, message_count_override=3)
            generate_held_out_suite(tmp2, seed=42, configs_limit=3, message_count_override=3)

            hash1 = _calculate_dir_hash(tmp1)
            hash2 = _calculate_dir_hash(tmp2)

            assert hash1 == hash2, "Deterministic generation failed — identical seeds produced different outputs."
            assert len(hash1) > 0, "No files were generated."

    def test_distinctness(self):
        """Different seeds must generate distinct scenario suites."""
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            generate_held_out_suite(tmp1, seed=42, configs_limit=3, message_count_override=3)
            generate_held_out_suite(tmp2, seed=43, configs_limit=3, message_count_override=3)

            hash1 = _calculate_dir_hash(tmp1)
            hash2 = _calculate_dir_hash(tmp2)

            assert hash1 != hash2, "Distinct generation failed — different seeds produced identical outputs."
            assert set(hash1.keys()) == set(hash2.keys()), "File structure changed unexpectedly between seeds."

    def test_schema_parsing_compatibility(self):
        """All generated CAM messages must successfully parse via `safe_parse_cam`."""
        with tempfile.TemporaryDirectory() as tmp:
            generate_held_out_suite(tmp, seed=42, configs_limit=5, message_count_override=5)
            path = pathlib.Path(tmp)
            
            count = 0
            for json_path in path.glob("**/*.json"):
                if json_path.name == "metadata.json":
                    continue
                
                with json_path.open(encoding="utf-8") as f:
                    msg = json.load(f)
                
                # Verify standard parse succeeds with no fatal warnings
                cam, err = safe_parse_cam(msg)
                assert err is None, f"Parsing failed for {json_path.name}: {err}"
                assert cam is not None, f"No CamMessage produced for {json_path.name}"
                count += 1
                
            assert count > 0, "No scenario messages were parsed."

    def test_physical_plausibility_of_benign_nodes(self):
        """Benign generated messages must satisfy all physical plausibility constraints."""
        scsv = SCSV()
        with tempfile.TemporaryDirectory() as tmp:
            generate_held_out_suite(tmp, seed=42, configs_limit=10, message_count_override=10)
            path = pathlib.Path(tmp)

            # Filter for messages from benign scenarios
            for scenario_dir in path.iterdir():
                if not scenario_dir.is_dir():
                    continue
                
                # Load metadata
                with open(scenario_dir / "metadata.json", encoding="utf-8") as f:
                    meta = json.load(f)
                
                if meta["scenario_family"] != "benign":
                    continue
                
                # Check messages in order to let stateful tracking work
                sorted_files = sorted(scenario_dir.glob("msg_*.json"))
                for json_path in sorted_files:
                    with json_path.open(encoding="utf-8") as f:
                        msg = json.load(f)
                    
                    # Stateful validation check
                    res = scsv.check_stateful(msg)
                    
                    # Benign messages in benign scenarios should be valid and not fatal
                    assert not res.fatal, f"Fatal SCSV check triggered for benign message {json_path}: {res.reason} / {res.details}"
                    assert res.valid, f"Validation failed for benign message {json_path}: {res.reason} / {res.details}"

    def test_isolation(self):
        """Generated scenarios must remain isolated in the target output directory."""
        # Ensure our tests do not write outside their temp directories
        with tempfile.TemporaryDirectory() as tmp:
            generate_held_out_suite(tmp, seed=42, configs_limit=30, message_count_override=1)
            path = pathlib.Path(tmp)
            
            # The suite must generate folders under tmp and a metadata.json
            assert (path / "metadata.json").is_file()
            subdirs = [x for x in path.iterdir() if x.is_dir()]
            assert len(subdirs) == 30, f"Expected 30 scenarios, got {len(subdirs)}"
            
            # Verify that only expected files/directories are inside the temp directory
            contents = set(x.name for x in path.iterdir())
            expected = set(x.name for x in subdirs)
            expected.add("metadata.json")
            assert contents == expected
