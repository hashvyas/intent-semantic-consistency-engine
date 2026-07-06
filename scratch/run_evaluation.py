"""
scratch/run_evaluation.py
=========================
A script to execute ablation studies and benchmarks over the ISCE V2 framework.
"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from b2_csia import CSIA
from b2_csia.experimental import ExperimentConfig, ExperimentRunner, ReportGenerator
from b2_csia.benchmarking import ComparativeBenchmarkingEngine

def main():
    print("=====================================================================")
    print("      ISCE V2 TRUST FRAMEWORK RESEARCH EVALUATION RUNNER")
    print("=====================================================================\n")

    # Initialize with research extensions enabled
    config_overrides = {
        "research_extensions": {
            "enabled": True
        },
        "motion_context": {
            "enabled": True,
            "inference_strategy": "probabilistic",
            "hysteresis": 0.25,
            "supported_contexts": ["highway", "urban", "rural"]
        },
        "trust_propagation": {
            "strategy": "belief_diffusion",
            "damping_factor": 0.3,
            "max_iterations": 20,
            "convergence_tolerance": 0.0001
        }
    }
    
    csia = CSIA(config_overrides=config_overrides)
    runner = ExperimentRunner(csia)
    benchmarker = ComparativeBenchmarkingEngine(csia)

    # 1. Ablation Study over Sybil attack
    print("--- Running Ablation Study: Sybil Attack Scenario ---")
    ablation_config = ExperimentConfig(
        name="sybil_ablation",
        attack_type="sybil",
        vehicle_count=10,
        attacker_count=3,
        traffic_density="dense",
        context="urban",
        seed=101
    )
    ablation_results = runner.run_ablation_study(ablation_config)
    
    ablation_report = {}
    for stage, metrics in ablation_results.items():
        print(f"Stage: {stage:<60s} -> F1: {metrics.f1_score:.4f}, Latency: {metrics.avg_latency_ms:.2f} ms")
        ablation_report[stage] = {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "avg_latency_ms": metrics.avg_latency_ms
        }

    # 2. Benchmarking across scenarios
    print("\n--- Running Comparative Benchmarks (Baseline vs V2 Framework) ---")
    scenarios = ["sybil", "replay", "collusion", "fabrication"]
    benchmark_reports = []

    for scenario in scenarios:
        config = ExperimentConfig(
            name=f"benchmark_{scenario}",
            attack_type=scenario,
            vehicle_count=12,
            attacker_count=4,
            seed=202
        )
        res = benchmarker.run_benchmark(config)
        print(f"Scenario: {scenario:<12s} | Baseline F1: {res['baseline']['f1_score']:.4f} -> V2 F1: {res['v2_framework']['f1_score']:.4f} | Gain: {res['differentials']['f1_gain_pct']:.1f}%")
        benchmark_reports.append(res)

    # 3. Export Reports
    output_dir = pathlib.Path(__file__).resolve().parent
    ReportGenerator.export_json(str(output_dir / "ablation_study_results.json"), ablation_report)
    ReportGenerator.export_json(str(output_dir / "benchmark_results.json"), {"benchmarks": benchmark_reports})

    print(f"\n[Success] Results exported to {output_dir}")

if __name__ == "__main__":
    main()
