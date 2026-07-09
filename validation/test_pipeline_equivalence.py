"""
validation/test_pipeline_equivalence.py
=======================================
Pipeline Equivalence Test.
Asserts that the new orchestrated path (ISCEPipeline) is observationally equivalent
to the legacy pipeline execution path when StubSemanticClassifier is active.
"""

from __future__ import annotations
import sys
import os
import pathlib
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from b1_scsv.scsv import SCSV
from b2_csia.csia import CSIA
from pipeline.orchestrator import ISCEPipeline
from validation.run_phase2_csia_integration_validation import (
    create_v2x_message,
    register_custom_profiles,
    interpret_final_classification
)

class TestPipelineEquivalence(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Configure CSIA with research extensions enabled
        cls.overrides = {
            "research_extensions": {
                "enabled": True,
            },
            "b2_csia": {
                "min_cluster_size": 2,
            }
        }

    def _assert_equivalence(self, messages, target_sid, attacker_ids, context="rural", historical_trust=0.90):
        # Create separate, fresh instances for the legacy run
        scsv_legacy = SCSV()
        csia_legacy = CSIA(config_overrides=self.overrides)
        register_custom_profiles(csia_legacy)
        
        supported = ["highway", "urban", "rural", "residential", "intersection", "roundabout", "tunnel", "bridge", "parking", "rsu_zone"] if context == "all" else [context]
        csia_legacy._raw["motion_context"] = {
            "inference_strategy": "probabilistic",
            "hysteresis": 0.25,
            "supported_contexts": supported,
        }

        # SCSV stateful check attaches validation assessment to target_msg
        import copy
        legacy_msgs = copy.deepcopy(messages)
        for msg in legacy_msgs:
            b1_res = scsv_legacy.check_stateful(msg)
            msg["_validation_assessment"] = b1_res

        # Run legacy B2 (CSIA)
        b2_payload, b2_report = csia_legacy.check_extended(legacy_msgs)
        
        # Create separate, fresh instances for the pipeline run
        from unittest.mock import patch
        with patch("pipeline.orchestrator.classify_text") as mock_classify:
            mock_classify.return_value = {
                "available": False,
                "label": None,
                "confidence": None,
                "status": "B3 integration unavailable"
            }
            scsv_pipeline = SCSV()
            csia_pipeline = CSIA(config_overrides=self.overrides)
            register_custom_profiles(csia_pipeline)
            
            csia_pipeline._raw["motion_context"] = {
                "inference_strategy": "probabilistic",
                "hysteresis": 0.25,
                "supported_contexts": supported,
            }
            pipeline = ISCEPipeline(scsv=scsv_pipeline, csia=csia_pipeline)
            
            pipeline_msgs = copy.deepcopy(messages)
            pipeline_res = pipeline.run(pipeline_msgs, context=context)
        
        # 3. Assert observational equivalence
        self.assertAlmostEqual(b2_payload["trust"], pipeline_res["b2"]["trust"], places=10)
        self.assertAlmostEqual(b2_report.belief, pipeline_res["b2"]["belief"], places=10)
        self.assertAlmostEqual(b2_report.disbelief, pipeline_res["b2"]["disbelief"], places=10)
        self.assertAlmostEqual(b2_report.uncertainty, pipeline_res["b2"]["uncertainty"], places=10)
        self.assertAlmostEqual(b2_report.confidence, pipeline_res["b2"]["confidence"], places=10)
        
        # Assert legacy decision forwarding matches
        legacy_valid = b2_payload["trust"] >= 0.4
        pipeline_valid = pipeline_res["decision"] in ("ACCEPT", "CAUTION")
        self.assertEqual(legacy_valid, pipeline_valid)
        
        # Assert B3 availability status is False
        self.assertFalse(pipeline_res["b3"]["available"])

    def test_scenario_1_benign(self):
        msgs = [
            create_v2x_message(1001 + i, 485512000 + i*20, 96123000 + i*20, 1500 + i*50, 900 + i*10, 4000.0 + i*100.0, 1001 + i, station_type=3 + i)
            for i in range(5)
        ]
        self._assert_equivalence(msgs, target_sid=1001, attacker_ids=[], context="all")

    def test_scenario_2_replay(self):
        msgs = []
        for i in range(3):
            msgs.append(create_v2x_message(1001 + i, 485512000 + i*2000, 96123000 + i*2000, 1500, 900, 4000.0 + i*100.0, 1001 + i))
        for i in range(5):
            msgs.append(create_v2x_message(8001 + i, 485512000, 96123000, 1500, 900, 4000.0 + i*25.0, 8888))
        self._assert_equivalence(msgs, target_sid=8001, attacker_ids=[8001, 8002, 8003, 8004, 8005])

    def test_scenario_3_sybil(self):
        msgs = []
        for i in range(3):
            msgs.append(create_v2x_message(1001 + i, 485512000 + i*2000, 96123000 + i*2000, 1500, 900, 4000.0 + i*100.0, 1001 + i))
        for i in range(5):
            msgs.append(create_v2x_message(9001 + i, 485512000, 96123000, 1500, 900, 4000.0, 9999))
        self._assert_equivalence(msgs, target_sid=9001, attacker_ids=[9001, 9002, 9003, 9004, 9005])

if __name__ == "__main__":
    unittest.main()
