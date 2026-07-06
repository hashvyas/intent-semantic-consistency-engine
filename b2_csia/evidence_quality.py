"""
b2_csia/evidence_quality.py
===========================
Evidence Quality Abstraction.

Quantifies input reliability based on GPS accuracy, timestamp freshness,
missing field factor, and sensor reliability. The resulting score is used 
to parameterize Shafer reliability discounting in uncertainty fusion.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class EvidenceQuality:
    """Immutable representation of evidence quality metrics.

    Parameters
    ----------
    gps_accuracy : float
        Normalized GPS accuracy ∈ [0.0, 1.0].
    timestamp_freshness : float
        Normalized timestamp freshness ∈ [0.0, 1.0].
    missing_fields_factor : float
        Normalized field completeness ratio ∈ [0.0, 1.0].
    sensor_reliability : float
        Opaque sensor source confidence ∈ [0.0, 1.0].
    """

    gps_accuracy: float = 1.0
    timestamp_freshness: float = 1.0
    missing_fields_factor: float = 1.0
    sensor_reliability: float = 1.0
    wall_time: float = field(default_factory=time.time)

    @property
    def score(self) -> float:
        """Computes the overall joint evidence quality score ∈ [0.0, 1.0].

        Using a multiplicative formulation to represent joint probability:
        Q = gps_accuracy * timestamp_freshness * missing_fields_factor * sensor_reliability
        """
        score = (
            self.gps_accuracy
            * self.timestamp_freshness
            * self.missing_fields_factor
            * self.sensor_reliability
        )
        return float(max(0.0, min(1.0, score)))

    def to_dict(self) -> Dict[str, float]:
        """Convert quality metrics to a serializable dictionary."""
        return {
            "gps_accuracy": self.gps_accuracy,
            "timestamp_freshness": self.timestamp_freshness,
            "missing_fields_factor": self.missing_fields_factor,
            "sensor_reliability": self.sensor_reliability,
            "overall_quality_score": self.score,
        }

    @classmethod
    def from_message(
        cls,
        message: Dict[str, Any],
        now_wall_s: float,
        freshness_decay_lambda: float = 0.5,
    ) -> EvidenceQuality:
        """Estimate EvidenceQuality from a decoded message.

        Parameters
        ----------
        message : Dict[str, Any]
            Raw or decoded message dictionary.
        now_wall_s : float
            Current system wall time in seconds.
        freshness_decay_lambda : float
            Lambda value governing exponential age decay of quality.
        """
        if not isinstance(message, dict):
            return cls(0.0, 0.0, 0.0, 0.0)

        # 1. GPS Accuracy
        # Check standard ETSI reference position fields
        rp = message.get("cam", {}).get("cam_parameters", {}).get("basic_container", {}).get("reference_position", {})
        # If positionConfidenceEllipse is available, use it. Otherwise assume basic default:
        gps_acc = 1.0
        if not rp:
            gps_acc = 0.5  # lower if no position container
        else:
            lat = rp.get("latitude")
            lon = rp.get("longitude")
            if lat is None or lon is None or not math.isfinite(lat) or not math.isfinite(lon):
                gps_acc = 0.0

        # 2. Timestamp Freshness
        gen_time = message.get("cam", {}).get("generation_delta_time")
        freshness = 1.0
        if gen_time is not None:
            try:
                # If timestamp is absolute epoch ms:
                if gen_time > 1_000_000_000:
                    age_s = abs(now_wall_s - (gen_time / 1000.0))
                    freshness = math.exp(-freshness_decay_lambda * age_s)
            except Exception:
                freshness = 0.5

        # 3. Missing Fields Factor
        # Ratio of populated kinematic fields to total expected kinematic fields
        basic_vehicle = message.get("cam", {}).get("cam_parameters", {}).get("high_frequency_container", {}).get("basic_vehicle_container_high_frequency", {})
        expected_fields = ["speed", "heading", "yaw_rate", "longitudinal_acceleration"]
        populated = 0
        for field_name in expected_fields:
            if basic_vehicle.get(field_name) is not None:
                populated += 1
        missing_factor = (populated / len(expected_fields)) if expected_fields else 1.0

        # 4. Sensor Reliability
        # Derived from station type. RSUs are highly reliable (1.0). Cars are standard (0.9).
        st = message.get("cam", {}).get("cam_parameters", {}).get("basic_container", {}).get("station_type", 5)
        sensor_rel = 0.9
        if st == 15:  # RSU
            sensor_rel = 1.0
        elif st == 1:   # Pedestrian / VRU (lower reliability sensor)
            sensor_rel = 0.7

        return cls(
            gps_accuracy=gps_acc,
            timestamp_freshness=freshness,
            missing_fields_factor=missing_factor,
            sensor_reliability=sensor_rel,
            wall_time=now_wall_s
        )
