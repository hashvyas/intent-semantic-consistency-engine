"""
pipeline/tests/test_b2_multi_seed.py
======================================
Unit tests for B2 (Partial) — Multi-Seed Scenario Generation Infrastructure.

Verifies:
  1. Determinism: identical seed lists produce identical outputs.
  2. Distinctness: different seeds produce different outputs.
  3. Output isolation: seed directories do not overlap or overwrite.
  4. Parsing compatibility: generated messages are accepted by the pipeline parser.
  5. Master metadata correctness: master metadata file contains all expected keys.
"""

from __future__ import annotations

import os
import json
import tempfile
import pathlib
import pytest

from scenario_generation.multi_seed import MultiSeedOrchestrator
from b1_scsv.models import safe_parse_cam


def _calculate_dir_hash(dir_path: str) -> dict[str, str]:
    """Calculate file content mappings to verify exact deterministic generation."""
    path = pathlib.Path(dir_path)
    file_contents = {}
    for p in sorted(path.glob("**/*.json")):
        rel = p.relative_to(path)
        with p.open(encoding="utf-8") as f:
            file_contents[str(rel)] = f.read()
    return file_contents


class TestMultiSeedOrchestrator:
    def test_determinism(self):
        """Identical seeds must generate byte-for-byte identical datasets."""
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            orchestrator1 = MultiSeedOrchestrator(tmp1)
            orchestrator2 = MultiSeedOrchestrator(tmp2)
            
            orchestrator1.generate_suites([10, 20], configs_limit=2, message_count_override=2)
            orchestrator2.generate_suites([10, 20], configs_limit=2, message_count_override=2)

            hash1 = _calculate_dir_hash(tmp1)
            hash2 = _calculate_dir_hash(tmp2)

            # Ignore dynamic timestamp changes in metadata comparison
            meta1 = json.loads(hash1.pop("multi_seed_metadata.json"))
            meta2 = json.loads(hash2.pop("multi_seed_metadata.json"))

            assert hash1 == hash2, "Deterministic generation failed — outputs differ on identical seeds."
            assert meta1["generation_seeds"] == meta2["generation_seeds"]
            assert meta1["total_messages"] == meta2["total_messages"]

    def test_distinctness(self):
        """Different seeds must produce different scenario outputs."""
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            orchestrator1 = MultiSeedOrchestrator(tmp1)
            orchestrator2 = MultiSeedOrchestrator(tmp2)
            
            orchestrator1.generate_suites([10], configs_limit=2, message_count_override=2)
            orchestrator2.generate_suites([20], configs_limit=2, message_count_override=2)

            # Check that files generated under seed_10 differ from seed_20
            # Compare scenario msg contents for corresponding paths
            path1 = pathlib.Path(tmp1) / "seed_10"
            path2 = pathlib.Path(tmp2) / "seed_20"
            
            # Map of relative file paths within the seed folder -> content
            files1 = {str(p.relative_to(path1)): p.read_text(encoding="utf-8") for p in path1.glob("**/*.json")}
            files2 = {str(p.relative_to(path2)): p.read_text(encoding="utf-8") for p in path2.glob("**/*.json")}
            
            # Ignore metadata.json in content comparison
            files1.pop("metadata.json", None)
            files2.pop("metadata.json", None)

            assert files1 != files2, "Distinctness failed — different seeds generated identical outputs."

    def test_output_isolation(self):
        """Suites for different seeds must occupy isolated subdirectories."""
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = MultiSeedOrchestrator(tmp)
            orchestrator.generate_suites([5, 15], configs_limit=2, message_count_override=1)
            
            path = pathlib.Path(tmp)
            assert (path / "seed_5").is_dir()
            assert (path / "seed_15").is_dir()
            assert (path / "multi_seed_metadata.json").is_file()
            
            # Ensure no other untracked folders were created
            folders = sorted([x.name for x in path.iterdir() if x.is_dir()])
            assert folders == ["seed_15", "seed_5"]

    def test_schema_and_parser_compatibility(self):
        """All generated multi-seed messages must successfully parse via `safe_parse_cam`."""
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = MultiSeedOrchestrator(tmp)
            orchestrator.generate_suites([7, 9], configs_limit=3, message_count_override=2)
            
            path = pathlib.Path(tmp)
            count = 0
            for json_path in path.glob("seed_*/**/*.json"):
                if json_path.name == "metadata.json":
                    continue
                
                with json_path.open(encoding="utf-8") as f:
                    msg = json.load(f)
                
                cam, err = safe_parse_cam(msg)
                assert err is None, f"Parsing failed for {json_path}: {err}"
                assert cam is not None
                count += 1
                
            assert count > 0

    def test_master_metadata_correctness(self):
        """The consolidated metadata file must contain all expected key-value descriptors."""
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = MultiSeedOrchestrator(tmp)
            meta = orchestrator.generate_suites([100, 200, 300], configs_limit=3, message_count_override=2)
            
            meta_path = pathlib.Path(tmp) / "multi_seed_metadata.json"
            assert meta_path.is_file()
            
            with meta_path.open(encoding="utf-8") as f:
                loaded_meta = json.load(f)
                
            assert loaded_meta == meta
            assert "generation_timestamp" in meta
            assert "repository_version" in meta
            assert meta["generation_seeds"] == [100, 200, 300]
            assert meta["scenarios_per_seed"] == 3
            assert "total_messages" in meta
            assert "output_directory" in meta
            assert meta["generator_version"] == "B5-HeldOutGenerator-v1"
