"""
b2_csia/test_research_draft.py
==============================
Draft unit tests verifying the research extensions:
- Evidence Quality Abstraction
- Observability Graph Improvements
- Adaptive Thresholds
- Dempster-Shafer & Yager uncertainty
- Evidence Extractors
- Declarative Attack Profiles & Behavioral Reasoning Engine
- Trust Propagation Engine
- Motion Context Inference Engine
- Research Evaluation Framework (Attack scenarios, ablation runner, metrics, reporting)
- Comparative Benchmarking Engine
"""

from __future__ import annotations

import unittest
import time
import os
from b2_csia.evidence_quality import EvidenceQuality
from b2_csia.observability_graph import ObservabilityGraphBuilder
from b2_csia.adaptive_thresholds import AdaptiveThresholdEngine
from b2_csia.uncertainty import MassFunction, BeliefFusionEngine, combine_dempster, combine_yager, combine_murphy, Provenance
from b2_csia.evidence_extractors import (
    SpatialSimilarityExtractor,
    TemporalSynchronizationExtractor,
    KinematicSimilarityExtractor,
    SemanticSimilarityExtractor,
    GraphConnectivityExtractor,
    IdentityConsistencyExtractor,
    RSUCorroborationExtractor,
    HistoricalTrustExtractor,
)
from b2_csia.behavior_profile import BehaviorEvidence, AttackProfile, AttackProfileRegistry
from b2_csia.behavior_reasoning import BehavioralReasoningEngine, AttackAssessment
from b2_csia.trust_propagation import TrustPropagationEngine
from b2_csia.context_aware import MotionContextInferenceEngine, ContextAssessment, ENVELOPES
from b2_csia.experimental import ExperimentConfig, ExperimentRunner, ReportGenerator
from b2_csia.benchmarking import ComparativeBenchmarkingEngine
from b2_csia import CSIA


class TestResearchDraft(unittest.TestCase):

    # ── 0. Evidence Quality Tests ───────────────────────────────────────────

    def test_evidence_quality_calculation(self) -> None:
        eq = EvidenceQuality(
            gps_accuracy=0.9,
            timestamp_freshness=0.8,
            missing_fields_factor=1.0,
            sensor_reliability=0.9,
            wall_time=time.time()
        )
        self.assertAlmostEqual(eq.score, 0.648)
        self.assertIn("overall_quality_score", eq.to_dict())

    def test_evidence_quality_from_message(self) -> None:
        msg = {
            "cam": {
                "generation_delta_time": time.time() * 1000.0,
                "cam_parameters": {
                    "basic_container": {
                        "station_type": 15,  # RSU
                        "reference_position": {"latitude": 485_512_345, "longitude": 96_123_456}
                    },
                    "high_frequency_container": {
                        "basic_vehicle_container_high_frequency": {
                            "speed": 1400.0,
                            "heading": 900.0,
                            "yaw_rate": 0.0,
                            "longitudinal_acceleration": 0.0
                        }
                    }
                }
            }
        }
        eq = EvidenceQuality.from_message(msg, time.time())
        self.assertEqual(eq.sensor_reliability, 1.0)
        self.assertEqual(eq.gps_accuracy, 1.0)
        self.assertEqual(eq.missing_fields_factor, 1.0)

    # ── 1. Observability Graph Tests ────────────────────────────────────────

    def test_observability_graph_creation(self) -> None:
        builder = ObservabilityGraphBuilder()
        now = time.time()
        
        builder.update_node(
            station_id=1,
            lat_e7=485_512_345,
            lon_e7=96_123_456,
            heading_deg10=900,
            timestamp_ns=1_000_000_000,
            station_type=5,
            wall_time=now,
            context="urban"
        )
        
        builder.update_node(
            station_id=2,
            lat_e7=485_512_445,
            lon_e7=96_123_456,
            heading_deg10=900,
            timestamp_ns=1_000_000_000,
            station_type=5,
            wall_time=now,
            context="urban"
        )

        graph = builder.graph
        self.assertIn(1, graph.nodes)
        self.assertIn(2, graph.nodes)
        
        key = (1, 2)
        self.assertIn(key, graph.edges)
        edge = graph.edges[key]
        self.assertGreater(edge.weight, 0.5)
        self.assertAlmostEqual(edge.confidence, 1.0)

    def test_context_aware_comm_range(self) -> None:
        builder = ObservabilityGraphBuilder()
        self.assertEqual(builder.get_communication_range("highway"), 400.0)
        self.assertEqual(builder.get_communication_range("urban"), 150.0)

    # ── 2. Evidence Extractors Tests ────────────────────────────────────────

    def test_evidence_extractors_extraction(self) -> None:
        cluster = [
            {
                "header": {"station_id": 42, "message_id": 1},
                "cam": {
                    "generation_delta_time": 4000.0,
                    "cam_parameters": {
                        "basic_container": {
                            "station_type": 5,
                            "reference_position": {"latitude": 485_512_345, "longitude": 96_123_456}
                        },
                        "high_frequency_container": {
                            "basic_vehicle_container_high_frequency": {
                                "speed": 1400.0,
                                "heading": 900.0,
                                "yaw_rate": 0.0,
                                "longitudinal_acceleration": 0.0
                            }
                        }
                    }
                }
            },
            {
                "header": {"station_id": 42, "message_id": 1},
                "cam": {
                    "generation_delta_time": 4000.0,
                    "cam_parameters": {
                        "basic_container": {
                            "station_type": 5,
                            "reference_position": {"latitude": 485_512_345, "longitude": 96_123_456}
                        },
                        "high_frequency_container": {
                            "basic_vehicle_container_high_frequency": {
                                "speed": 1400.0,
                                "heading": 900.0,
                                "yaw_rate": 0.0,
                                "longitudinal_acceleration": 0.0
                            }
                        }
                    }
                }
            }
        ]

        spatial_sim, _, _ = SpatialSimilarityExtractor().extract(cluster)
        temporal_sim, _, _ = TemporalSynchronizationExtractor().extract(cluster)
        kinematic_sim, _, _ = KinematicSimilarityExtractor().extract(cluster)
        semantic_sim, _, _ = SemanticSimilarityExtractor().extract(cluster)
        identity_consistency, _, _ = IdentityConsistencyExtractor().extract(cluster)

        self.assertAlmostEqual(spatial_sim, 1.0)
        self.assertAlmostEqual(temporal_sim, 1.0)
        self.assertAlmostEqual(kinematic_sim, 1.0)
        self.assertAlmostEqual(semantic_sim, 1.0)
        self.assertAlmostEqual(identity_consistency, 0.5)

    # ── 3. Behavioral Reasoning Engine Tests ────────────────────────────────

    def test_behavioral_reasoning_sybil_match(self) -> None:
        engine = BehavioralReasoningEngine(fusion_rule="yager")
        prov = Provenance(modules=set(["spatial"]), min_evidence_quality=0.9, min_confidence=0.8)

        evidence_sybil = BehaviorEvidence(
            spatial_similarity=1.0,
            temporal_similarity=1.0,
            kinematic_similarity=1.0,
            semantic_similarity=1.0,
            graph_connectivity=1.0,
            identity_consistency=0.0,
            rsu_corroboration=0.0,
            historical_trust=0.8,
            confidence=0.9,
            provenance=prov
        )

        assessment = engine.evaluate(evidence_sybil, reliability_alpha=0.9)
        
        self.assertEqual(assessment.attack_type, "sybil")
        self.assertGreater(assessment.belief, 0.6)
        self.assertIn("identity_consistency", assessment.explanation["strongest_indicators"])
        self.assertIn("temporal_similarity", assessment.explanation["strongest_indicators"])

    def test_behavioral_reasoning_no_attack(self) -> None:
        engine = BehavioralReasoningEngine(fusion_rule="yager")
        prov = Provenance(modules=set(["spatial"]), min_evidence_quality=0.9, min_confidence=0.8)

        evidence_benign = BehaviorEvidence(
            spatial_similarity=0.2,
            temporal_similarity=0.1,
            kinematic_similarity=0.2,
            semantic_similarity=0.5,
            graph_connectivity=0.1,
            identity_consistency=1.0,
            rsu_corroboration=1.0,
            historical_trust=1.0,
            confidence=0.9,
            provenance=prov
        )

        assessment = engine.evaluate(evidence_benign, reliability_alpha=0.9)
        self.assertEqual(assessment.attack_type, "none")
        self.assertEqual(assessment.belief, 0.0)
        self.assertEqual(assessment.disbelief, 1.0)

    # ── 4. Trust Propagation Tests ──────────────────────────────────────────

    def test_trust_propagation_engine(self) -> None:
        builder = ObservabilityGraphBuilder()
        now = time.time()
        builder.update_node(1, 485_512_345, 96_123_456, 900, 1e9, 5, now)
        builder.update_node(2, 485_512_400, 96_123_456, 900, 1e9, 5, now)
        builder.update_node(3, 485_512_450, 96_123_456, 900, 1e9, 5, now)

        initial_beliefs = {
            1: MassFunction.from_trust_confidence(0.2, 0.9, origin_module="init"),
            2: MassFunction.from_trust_confidence(0.9, 0.9, origin_module="init"),
            3: MassFunction.from_trust_confidence(0.9, 0.9, origin_module="init"),
        }

        engine = TrustPropagationEngine()
        config = {
            "strategy": "belief_diffusion",
            "damping_factor": 0.5,
            "max_iterations": 20,
            "convergence_tolerance": 0.0001,
            "sybil_penalty": 0.0,
            "minimum_edge_confidence": 0.0,
            "minimum_evidence_quality": 0.0,
        }

        propagated, meta = engine.propagate(builder.graph, initial_beliefs, config)

        self.assertIn(1, propagated)
        self.assertIn(2, propagated)
        self.assertIn(3, propagated)

        self.assertGreater(propagated[1].belief, initial_beliefs[1].belief)
        self.assertLess(propagated[2].belief, initial_beliefs[2].belief)
        self.assertIn("iterations_to_converge", meta)

    # ── 5. Motion Context Inference Tests ────────────────────────────────────

    def test_motion_context_inference_highway(self) -> None:
        cluster_highway = [
            {
                "cam": {
                    "cam_parameters": {
                        "basic_container": {"station_type": 5},
                        "high_frequency_container": {
                            "basic_vehicle_container_high_frequency": {
                                "speed": 3000,
                                "heading": 900,
                            }
                        }
                    }
                }
            }
            for _ in range(5)
        ]

        engine = MotionContextInferenceEngine()
        config = {
            "inference_strategy": "probabilistic",
            "hysteresis": 0.25,
            "supported_contexts": ["highway", "urban", "rural"],
        }

        res = engine.infer_context(cluster_highway, cluster_id=1, config=config)
        self.assertEqual(res.context, "highway")
        self.assertGreater(res.confidence, 0.4)

    def test_motion_context_inference_urban(self) -> None:
        cluster_urban = [
            {
                "cam": {
                    "cam_parameters": {
                        "basic_container": {"station_type": 5},
                        "high_frequency_container": {
                            "basic_vehicle_container_high_frequency": {
                                "speed": 800,
                                "heading": (900 + i * 200) % 3600,
                            }
                        }
                    }
                }
            }
            for i in range(5)
        ]

        engine = MotionContextInferenceEngine()
        config = {
            "inference_strategy": "probabilistic",
            "hysteresis": 0.25,
            "supported_contexts": ["highway", "urban", "rural"],
        }

        res = engine.infer_context(cluster_urban, cluster_id=1, config=config)
        self.assertEqual(res.context, "urban")

    def test_motion_envelope_boundaries(self) -> None:
        engine = MotionContextInferenceEngine()
        env = engine.get_envelope("highway")
        self.assertEqual(env.expected_speed_max, 45.0)
        self.assertEqual(env.expected_yaw_max, 15.0)

    # ── 6. Evaluation Framework Tests ────────────────────────────────────────

    def test_experiment_runner_and_ablation(self) -> None:
        csia = CSIA()
        runner = ExperimentRunner(csia)

        # Config: Sybil simulation
        config = ExperimentConfig(
            name="test_sybil_run",
            attack_type="sybil",
            vehicle_count=5,
            attacker_count=2,
            traffic_density="dense",
            context="urban"
        )

        metrics, summary = runner.run_experiment(config)
        self.assertEqual(summary["attack_type"], "sybil")
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertGreaterEqual(metrics.avg_latency_ms, 0.0)

        # Test ablation run
        ablation_results = runner.run_ablation_study(config)
        self.assertIn("baseline", ablation_results)
        self.assertGreater(len(ablation_results), 2)

    def test_comparative_benchmarking_engine(self) -> None:
        csia = CSIA()
        engine = ComparativeBenchmarkingEngine(csia)

        config = ExperimentConfig(
            name="test_bench_run",
            attack_type="replay",
            vehicle_count=5,
            attacker_count=2,
        )

        res = engine.run_benchmark(config)
        self.assertIn("baseline", res)
        self.assertIn("v2_framework", res)
        self.assertIn("differentials", res)
        self.assertIn("accuracy_gain_pct", res["differentials"])

    def test_report_generator_exports(self) -> None:
        results_list = [{"name": "exp1", "f1": 0.95}, {"name": "exp2", "f1": 0.88}]
        ReportGenerator.export_json("temp_report.json", {"results": results_list})
        ReportGenerator.export_csv("temp_report.csv", results_list)

        self.assertTrue(os.path.exists("temp_report.json"))
        self.assertTrue(os.path.exists("temp_report.csv"))

        # Clean up files
        os.remove("temp_report.json")
        os.remove("temp_report.csv")


if __name__ == "__main__":
    unittest.main()
