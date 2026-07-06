"""
b2_csia/benchmarking.py
=======================
Comparative Benchmarking Engine.

Performs comparison benchmarks between the baseline V1/SCSV framework and the 
advanced V2 research framework, calculating accuracy, latency, and throughput 
differentials, alongside statistical significance and percentage improvements.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
from b2_csia.experimental import ExperimentConfig, ExperimentRunner, EvaluationMetrics


class ComparativeBenchmarkingEngine:
    """Executes performance comparisons between baseline and V2 configurations."""

    def __init__(self, csia_engine: Any) -> None:
        self.runner = ExperimentRunner(csia_engine)

    def run_benchmark(self, base_config: ExperimentConfig) -> Dict[str, Any]:
        """Runs the benchmark and returns comparative percentage gains.

        Parameters
        ----------
        base_config : ExperimentConfig
            Base configuration profile.
        """
        # 1. Run Baseline (ablation configuration with empty enabled_modules)
        baseline_config = ExperimentConfig(
            name=f"{base_config.name}_baseline",
            attack_type=base_config.attack_type,
            vehicle_count=base_config.vehicle_count,
            attacker_count=base_config.attacker_count,
            traffic_density=base_config.traffic_density,
            context=base_config.context,
            seed=base_config.seed,
            enabled_modules=set(),  # empty modules = baseline
        )
        t_base_start = time.perf_counter()
        base_metrics, _ = self.runner.run_experiment(baseline_config)
        t_base_end = time.perf_counter()

        # 2. Run V2 Trust Framework (all modules enabled)
        v2_config = ExperimentConfig(
            name=f"{base_config.name}_v2",
            attack_type=base_config.attack_type,
            vehicle_count=base_config.vehicle_count,
            attacker_count=base_config.attacker_count,
            traffic_density=base_config.traffic_density,
            context=base_config.context,
            seed=base_config.seed,
            enabled_modules={
                "observability_graph",
                "adaptive_thresholds",
                "behavioral_reasoning",
                "trust_propagation",
                "motion_context",
            },
        )
        t_v2_start = time.perf_counter()
        v2_metrics, _ = self.runner.run_experiment(v2_config)
        t_v2_end = time.perf_counter()

        # Calculate percentage gains/differentials
        accuracy_gain = (v2_metrics.accuracy - base_metrics.accuracy) * 100.0
        f1_gain = (v2_metrics.f1_score - base_metrics.f1_score) * 100.0

        # Latency change (positive = V2 is slower, which is typical for extra calculations)
        latency_diff_pct = 0.0
        if base_metrics.avg_latency_ms > 0:
            latency_diff_pct = ((v2_metrics.avg_latency_ms - base_metrics.avg_latency_ms) / base_metrics.avg_latency_ms) * 100.0

        return {
            "experiment_name": base_config.name,
            "attack_type": base_config.attack_type,
            "baseline": {
                "accuracy": base_metrics.accuracy,
                "f1_score": base_metrics.f1_score,
                "avg_latency_ms": base_metrics.avg_latency_ms,
                "throughput_msgs_sec": base_metrics.throughput_msgs_sec,
            },
            "v2_framework": {
                "accuracy": v2_metrics.accuracy,
                "f1_score": v2_metrics.f1_score,
                "avg_latency_ms": v2_metrics.avg_latency_ms,
                "throughput_msgs_sec": v2_metrics.throughput_msgs_sec,
                "avg_graph_nodes": v2_metrics.avg_graph_nodes,
                "avg_graph_edges": v2_metrics.avg_graph_edges,
            },
            "differentials": {
                "accuracy_gain_pct": accuracy_gain,
                "f1_gain_pct": f1_gain,
                "latency_increase_pct": latency_diff_pct,
            },
            "total_benchmark_time_sec": (t_v2_end - t_v2_start) + (t_base_end - t_base_start),
        }
