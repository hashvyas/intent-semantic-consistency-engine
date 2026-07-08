"""
b2_csia/experimental.py
=======================
Research Evaluation Framework.

Provides a reproducible evaluation engine for running controlled experiments,
generating attack scenarios (Sybil, Replay, Collusion, Event Fabrication, GPS Spoofing),
executing ablation studies, calculating metrics (Precision, Recall, F1, ROC-AUC, Latency),
and generating structured JSON/CSV reports.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ExperimentConfig:
    """Configures a single evaluation experiment run.

    Parameters
    ----------
    name : str
        Descriptive experiment name.
    attack_type : str
        The attack scenario to simulate ('sybil', 'replay', 'collusion', 'fabrication', 'none').
    vehicle_count : int
        Number of simulated vehicles.
    attacker_count : int
        Number of simulated attacker nodes.
    traffic_density : str
        Density mode ('sparse', 'dense').
    context : str
        Operating environment context ('highway', 'urban').
    seed : int
        Random seed for deterministic generation.
    enabled_modules : Set[str]
        List of pipeline modules to enable (for ablation).
    """

    name: str
    attack_type: str = "none"
    vehicle_count: int = 10
    attacker_count: int = 0
    traffic_density: str = "sparse"
    context: str = "urban"
    seed: int = 42
    enabled_modules: Set[str] = field(
        default_factory=lambda: {
            "observability_graph",
            "adaptive_thresholds",
            "behavioral_reasoning",
            "trust_propagation",
            "motion_context",
        }
    )


@dataclass
class EvaluationMetrics:
    """Stores computed metrics for an experiment."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    fpr: float = 0.0
    fnr: float = 0.0
    roc_auc: float = 0.5
    avg_latency_ms: float = 0.0
    throughput_msgs_sec: float = 0.0
    max_memory_kb: float = 0.0
    avg_graph_nodes: float = 0.0
    avg_graph_edges: float = 0.0


# ===========================================================================
# Attack Scenario Generator
# ===========================================================================

class AttackScenarioGenerator:
    """Generates deterministic, parameterized traffic datasets for experiments."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def generate_scenario(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Generates a list of CAM messages matching the experiment configuration."""
        self.rng.seed(config.seed)

        messages: List[Dict[str, Any]] = []
        base_lat = 485_512_345
        base_lon = 96_123_456
        now_ns = 1_000_000_000

        # Determine density scaling
        spacing = 3000 if config.traffic_density == "sparse" else 1000
        speed_base = 3000 if config.context == "highway" else 1000

        # 1. Generate Benign Vehicles
        benign_count = max(0, config.vehicle_count - config.attacker_count)
        for i in range(benign_count):
            station_id = 1000 + i
            # Offset position based on spacing (approx 30m / 10m spacing)
            lat = base_lat + i * spacing
            lon = base_lon + i * spacing
            speed = speed_base + self.rng.randint(-200, 200)
            heading = 900 + self.rng.randint(-50, 50)
            yaw = self.rng.randint(-20, 20)

            msg = self._create_cam_message(
                station_id=station_id,
                station_type=5,  # passengerCar
                lat=lat,
                lon=lon,
                speed=speed,
                heading=heading,
                yaw=yaw,
                ts_ns=4000.0 + i * 100.0,  # 100 ms spacing
                is_attacker=False,
            )
            messages.append(msg)

        # 2. Inject Attackers
        if config.attacker_count > 0:
            attack_type = config.attack_type.lower().strip()

            if attack_type == "sybil":
                # Sybil: same station ID (clones) broadcasting identical kinematics
                sybil_speed = speed_base + 100
                sybil_heading = 900
                sybil_yaw = 0
                for a in range(config.attacker_count):
                    # Identical station_id = 9000 to trigger low identity consistency
                    station_id = 9000
                    msg = self._create_cam_message(
                        station_id=station_id,
                        station_type=5,
                        lat=base_lat + 20,
                        lon=base_lon + 20,
                        speed=sybil_speed,
                        heading=sybil_heading,
                        yaw=sybil_yaw,
                        ts_ns=4000.0 + a * 2.0,  # highly synchronized (2ms apart)
                        is_attacker=True,
                    )
                    messages.append(msg)

            elif attack_type == "replay":
                # Replay: repeat benign message values with time delay
                target_msg = messages[0] if messages else self._create_cam_message(1000, 5, base_lat, base_lon, speed_base, 900, 0, 4000.0, False)
                for a in range(config.attacker_count):
                    station_id = 8000 + a
                    # Duplicate kinematics but delay timestamp
                    msg = dict(target_msg)
                    msg["header"] = {"station_id": station_id, "message_id": 99}
                    msg["cam"] = dict(target_msg["cam"])
                    # Slight delay (e.g. 5 ms)
                    msg["cam"]["generation_delta_time"] = target_msg["cam"]["generation_delta_time"] + (a + 1) * 5.0
                    msg["is_attacker"] = True
                    messages.append(msg)

            elif attack_type == "collusion":
                # Collusion: different IDs reporting near-identical dynamics in a cluster
                col_speed = speed_base - 100
                for a in range(config.attacker_count):
                    station_id = 7000 + a
                    msg = self._create_cam_message(
                        station_id=station_id,
                        station_type=5,
                        lat=base_lat + a * 10,
                        lon=base_lon + a * 10,
                        speed=col_speed + self.rng.randint(-5, 5),
                        heading=900,
                        yaw=0,
                        ts_ns=4000.0 + a * 10.0,
                        is_attacker=True,
                    )
                    messages.append(msg)

            elif attack_type == "fabrication":
                # Fabrication: fake warning message types with low corroboration
                for a in range(config.attacker_count):
                    station_id = 6000 + a
                    msg = self._create_cam_message(
                        station_id=station_id,
                        station_type=5,
                        lat=base_lat + 20000,  # far away
                        lon=base_lon + 20000,
                        speed=speed_base,
                        heading=900,
                        yaw=0,
                        ts_ns=4000.0 + a * 100.0,
                        is_attacker=True,
                    )
                    msg["message_type"] = "DENM"
                    msg["is_attacker"] = True
                    messages.append(msg)

        # Sort messages by timestamp
        messages.sort(key=lambda m: m["cam"]["generation_delta_time"])
        return messages

    @staticmethod
    def _create_cam_message(
        station_id: int,
        station_type: int,
        lat: int,
        lon: int,
        speed: int,
        heading: int,
        yaw: int,
        ts_ns: float,
        is_attacker: bool,
    ) -> Dict[str, Any]:
        return {
            "header": {"station_id": station_id, "message_id": 1},
            "cam": {
                "generation_delta_time": ts_ns,
                "cam_parameters": {
                    "basic_container": {
                        "station_type": station_type,
                        "reference_position": {"latitude": lat, "longitude": lon},
                    },
                    "high_frequency_container": {
                        "basic_vehicle_container_high_frequency": {
                            "speed": speed,
                            "heading": heading,
                            "yaw_rate": yaw,
                            "steering_wheel_angle": 0,
                            "lateral_acceleration": 0,
                            "longitudinal_acceleration": 0,
                        }
                    },
                },
            },
            "is_attacker": is_attacker,
        }


# ===========================================================================
# Experiment Runner & Ablation Engine
# ===========================================================================

class ExperimentRunner:
    """Executes controlled experiments and computes precision/recall metrics."""

    def __init__(self, csia_engine: Any) -> None:
        self.csia = csia_engine
        self.generator = AttackScenarioGenerator()

    def run_experiment(self, config: ExperimentConfig) -> Tuple[EvaluationMetrics, Dict[str, Any]]:
        """Runs the experiment and calculates performance and detection metrics."""
        # Reset CSIA engine state to avoid state pollution between runs
        if hasattr(self.csia, "reset"):
            self.csia.reset()

        # 1. Generate dataset
        messages = self.generator.generate_scenario(config)

        # Ground truth labels
        y_true = [1 if m.get("is_attacker", False) else 0 for m in messages]
        y_pred = []

        latencies = []
        node_counts = []
        edge_counts = []

        # Set enabled modules on the engine for ablation
        if hasattr(self.csia, "enabled_modules"):
            self.csia.enabled_modules = config.enabled_modules

        # Run pipeline over rolling windows of messages
        window_size = 5
        for i in range(len(messages)):
            start_idx = max(0, i - window_size + 1)
            window = messages[start_idx:i+1]

            t0 = time.perf_counter()
            # Call pipeline check (representing predictions)
            # In V2, check() outputs a dict of trust scores.
            res = self.csia.check(window)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            # Record graph stats if available
            if hasattr(self.csia, "_graph_builder"):
                node_counts.append(len(self.csia._graph_builder.graph.nodes))
                edge_counts.append(len(self.csia._graph_builder.graph.edges))

            # Classify as attacker if trust score falls below 0.4
            trust = res.get("trust", 1.0)
            pred_attacker = 1 if trust < 0.4 else 0
            y_pred.append(pred_attacker)

        # 2. Calculate detection metrics
        metrics = self._calculate_metrics(y_true, y_pred, latencies, node_counts, edge_counts)

        # Explainability metadata summary
        summary = {
            "experiment_name": config.name,
            "attack_type": config.attack_type,
            "total_messages": len(messages),
            "attacker_messages": sum(y_true),
            "detected_messages": sum(y_pred),
            "enabled_modules": list(config.enabled_modules),
        }

        return metrics, summary

    def run_ablation_study(self, base_config: ExperimentConfig) -> Dict[str, EvaluationMetrics]:
        """Runs ablation study by incrementally adding research modules."""
        modules = [
            "observability_graph",
            "adaptive_thresholds",
            "behavioral_reasoning",
            "trust_propagation",
            "motion_context",
        ]

        results = {}
        # Start with empty/baseline and add one by one
        current_modules: Set[str] = set()

        # Step 0: Baseline SCSV only (empty csia V2 features)
        config = ExperimentConfig(
            name=f"{base_config.name}_baseline",
            attack_type=base_config.attack_type,
            vehicle_count=base_config.vehicle_count,
            attacker_count=base_config.attacker_count,
            traffic_density=base_config.traffic_density,
            context=base_config.context,
            seed=base_config.seed,
            enabled_modules=set(),
        )
        results["baseline"] = self.run_experiment(config)[0]

        # Step 1-N: Accumulate modules
        for mod in modules:
            current_modules.add(mod)
            config = ExperimentConfig(
                name=f"{base_config.name}_{mod}",
                attack_type=base_config.attack_type,
                vehicle_count=base_config.vehicle_count,
                attacker_count=base_config.attacker_count,
                traffic_density=base_config.traffic_density,
                context=base_config.context,
                seed=base_config.seed,
                enabled_modules=set(current_modules),
            )
            results[" + ".join(current_modules)] = self.run_experiment(config)[0]

        return results

    @staticmethod
    def _calculate_metrics(
        y_true: List[int],
        y_pred: List[int],
        latencies: List[float],
        nodes: List[int],
        edges: List[int],
    ) -> EvaluationMetrics:
        tp = fp = tn = fn = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                tp += 1
            elif yt == 0 and yp == 1:
                fp += 1
            elif yt == 0 and yp == 0:
                tn += 1
            elif yt == 1 and yp == 0:
                fn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        throughput = 1000.0 / avg_lat if avg_lat > 0.0 else 0.0

        avg_n = sum(nodes) / len(nodes) if nodes else 0.0
        avg_e = sum(edges) / len(edges) if edges else 0.0

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            fpr=fpr,
            fnr=fnr,
            roc_auc=0.5 + 0.5 * (recall - fpr),  # approximation
            avg_latency_ms=avg_lat,
            throughput_msgs_sec=throughput,
            avg_graph_nodes=avg_n,
            avg_graph_edges=avg_e,
        )


# ===========================================================================
# Report Generator
# ===========================================================================

class ReportGenerator:
    """Generates structured CSV and JSON reports of experiment results."""

    @staticmethod
    def export_json(filepath: str, results: Dict[str, Any]) -> None:
        """Write evaluation results dict to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

    @staticmethod
    def export_csv(filepath: str, results: List[Dict[str, Any]]) -> None:
        """Write a list of evaluation result dicts to a CSV file."""
        if not results:
            return
        keys = results[0].keys()
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
