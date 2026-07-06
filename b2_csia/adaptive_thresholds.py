"""
b2_csia/adaptive_thresholds.py
==============================
Adaptive Threshold Engine.

Calculates dynamic, context-aware kinematic thresholds using robust statistical
estimators (Mean/Std, Median/MAD, Percentile), exponential forgetting, and 
returns a structured threshold result with confidence estimation and explainability metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AdaptiveThresholdResult:
    """Belief/Result object from an adaptive threshold calculation.

    Parameters
    ----------
    threshold_value : float
        Calculated threshold value.
    confidence : float
        Confidence ∈ [0.0, 1.0] in the calculated threshold.
    sample_count : int
        Number of historical samples used.
    variance_estimate : float
        Variance estimate of the historical samples.
    estimation_method : str
        Method name ('mean_std', 'median_mad', or 'percentile').
    statistics : Dict[str, float]
        Internal statistics (e.g. median, MAD, mean, std) used for calculation.
    """

    threshold_value: float
    confidence: float
    sample_count: int
    variance_estimate: float
    estimation_method: str
    statistics: Dict[str, float]


class AdaptiveThresholdEngine:
    """Calculates dynamic, context-aware statistical thresholds for trust evaluation.

    Why Adaptive Thresholds Outperform Static Thresholds
    ---------------------------------------------------
    Static thresholds suffer from a fundamental trade-off: if set too loose, they fail to
    detect subtle misbehaviors (such as coordinate spoofing) in highly structured traffic;
    if set too tight, they trigger high false alarm rates in complex urban scenarios.
    Adaptive thresholds resolve this by dynamically adjusting kinematic envelopes based on
    speed context, traffic density, vehicle type heterogeneity, and historical noise.

    Mathematical Intuition Behind Adaptive Factors
    ---------------------------------------------
    1. Context (Speed): Selects the base threshold (city fallback 0.50 vs highway fallback 0.20)
       to constrain absolute spatial-temporal deviation margins at high speeds.
    2. Traffic Density: As traffic density N increases, vehicles are physically constrained
       by traffic flow dynamics. The density factor f_density decreases the threshold baseline:
       f_density = 1.0 / (1.0 + (N / density_scale))
       enforcing tighter vehicle synchronization requirements in dense traffic.
    3. Vehicle Diversity: Quantifies the kinematic heterogeneity of the traffic mix using
       Shannon entropy. Diverse vehicle classes (e.g. buses, trucks, passenger cars) exhibit
       different acceleration and braking envelopes. Higher entropy relaxes thresholds:
       f_diversity = 1.0 + 0.3 * normalized_entropy
    4. Historical Variance: Previous observations adapt the threshold to localized, historical
       noise levels (measured via Standard Deviation or MAD) to prevent false alerts.

    Why Shannon Entropy for Heterogeneous Traffic
    ---------------------------------------------
    Shannon entropy measures the information uncertainty/diversity of the station type distribution.
    A well-mixed environment of passenger cars, buses, heavy trucks, and infrastructure (RSUs)
    has higher entropy than a homogeneous car-only stream. The engine uses normalized entropy
    to scale up thresholds proportionally to the kinematic diversity of the cluster.

    Why Median + MAD (Median Absolute Deviation) vs Mean + Standard Deviation
    ------------------------------------------------------------------------
    Mean and Standard Deviation have a breakdown point of 0%, meaning a single extreme outlier
    (e.g., a buggy or malicious telemetry transmission) can heavily bias the mean and balloon
    the standard deviation, widening the threshold envelope and letting subsequent spoofed data
    pass. Median and MAD are non-parametric estimators with a high breakdown point (50%),
    ensuring that extreme telemetry outliers are completely ignored, keeping thresholds stable.

    Threshold Value Interpretation
    -----------------------------
    - <0.25     : Very strict threshold (appropriate for highway traffic / dense platoon flows).
    - 0.25-0.50 : Normal operating threshold (standard urban/rural environment bounds).
    - 0.50-0.75 : Relaxed threshold (sparse traffic or diverse vehicle groups).
    - >0.75     : Highly tolerant threshold (extreme historical noise or highly diverse flow).

    Parameters
    ----------
    k_factor : float
        Scaling factor for standard deviation / MAD adjustments.
    density_scale : float
        Normalising constant for traffic density scaling.
    fallback_city_threshold : float
        Default threshold when history is empty or traffic is low speed.
    fallback_highway_threshold : float
        Default threshold when history is empty or traffic is high speed.
    estimation_method : str
        The estimation strategy to use ('median_mad', 'mean_std', 'percentile').
    forgetting_factor : float
        Exponential forgetting factor β ∈ (0, 1] for streaming updates.
    percentile_value : float
        The percentile target [0.0, 100.0] for the percentile estimator (default 90.0).
    """

    def __init__(
        self,
        k_factor: float = 1.5,
        density_scale: float = 10.0,
        fallback_city_threshold: float = 0.50,
        fallback_highway_threshold: float = 0.20,
        estimation_method: str = "median_mad",
        forgetting_factor: float = 0.95,
        percentile_value: float = 90.0,
    ) -> None:
        self.k_factor = k_factor
        self.density_scale = density_scale
        self.fallback_city_threshold = fallback_city_threshold
        self.fallback_highway_threshold = fallback_highway_threshold
        self.estimation_method = estimation_method.lower().strip()
        self.forgetting_factor = max(0.01, min(1.0, forgetting_factor))
        self.percentile_value = max(0.0, min(100.0, percentile_value))

        # We keep a rolling window for non-parametric calculations (median, percentiles)
        self._history_distances: List[float] = []
        self._max_history_len = 100

        # Streaming state variables for mean/variance/MAD
        self._running_mean: float = 0.0
        self._running_var: float = 0.0
        self._running_count: int = 0

    def record_distance(self, distance: float) -> None:
        """Record a calculated distance into the streaming engine with exponential forgetting."""
        if not math.isfinite(distance):
            return

        # 1. Update rolling window (for median/MAD and percentiles)
        self._history_distances.append(distance)
        if len(self._history_distances) > self._max_history_len:
            self._history_distances.pop(0)

        # 2. Update streaming stats with exponential forgetting
        self._running_count += 1
        if self._running_count == 1:
            self._running_mean = distance
            self._running_var = 0.0
        else:
            # Apply forgetting factor to running statistics
            beta = self.forgetting_factor
            old_mean = self._running_mean
            self._running_mean = beta * old_mean + (1.0 - beta) * distance
            diff = distance - old_mean
            self._running_var = beta * self._running_var + (1.0 - beta) * (diff ** 2)

    def calculate_threshold(
        self,
        cluster: List[Dict[str, Any]],
        median_speed: float,
        highway_speed_threshold: float,
        message_arrival_rate: float,
        observation_duration_s: float,
    ) -> AdaptiveThresholdResult:
        """Dynamically computes the kinematic threshold.

        Parameters
        ----------
        cluster : List[Dict[str, Any]]
            The current spatio-temporal cluster of messages.
        median_speed : float
            Median speed of the cluster in ETSI 0.01 m/s units.
        highway_speed_threshold : float
            Speed boundary (ETSI 0.01 m/s) separating city from highway.
        message_arrival_rate : float
            Average incoming message frequency (Hz).
        observation_duration_s : float
            Time span of messages in this cluster (seconds).

        Returns
        -------
        AdaptiveThresholdResult
            The calculated adaptive threshold result with confidence and metrics.
        """
        is_highway = median_speed >= highway_speed_threshold
        base_threshold = self.fallback_highway_threshold if is_highway else self.fallback_city_threshold

        N = len(cluster)
        sample_count = len(self._history_distances)

        # Confidence calculation: decreases when history/sample count is sparse
        # c = 1 - exp(-0.05 * sample_count)
        threshold_confidence = float(1.0 - math.exp(-0.05 * max(1, sample_count)))

        # 1. Traffic Density Factor
        f_density = 1.0 / (1.0 + (N / self.density_scale))

        # 2. Vehicle Diversity Factor (Entropy of station types)
        station_types = []
        for msg in cluster:
            if not isinstance(msg, dict):
                continue
            st = msg.get("cam", {}).get("cam_parameters", {}).get("basic_container", {}).get("station_type")
            if st is not None:
                station_types.append(st)

        f_diversity = 1.0
        if station_types:
            type_counts = {}
            for t in station_types:
                type_counts[t] = type_counts.get(t, 0) + 1
            entropy = 0.0
            total = len(station_types)
            for c in type_counts.values():
                p = c / total
                entropy -= p * math.log2(p)
            max_entropy = math.log2(4)
            normalized_entropy = min(1.0, entropy / max_entropy) if max_entropy > 0.0 else 0.0
            f_diversity = 1.0 + 0.3 * normalized_entropy

        # 3. Stability Adjustment
        f_stability = 1.0
        if observation_duration_s > 0.0:
            f_stability = max(0.9, 1.0 - (observation_duration_s / 10.0))

        # Multiplicative baseline calculation
        baseline_adaptive = base_threshold * f_density * f_diversity * f_stability

        # 4. Statistical Estimation Strategy
        threshold_value = baseline_adaptive
        stats: Dict[str, float] = {}

        if sample_count >= 5:
            if self.estimation_method == "median_mad":
                # Compute Median and Median Absolute Deviation
                sorted_history = sorted(self._history_distances)
                median = self._percentile(sorted_history, 50.0)
                deviations = [abs(x - median) for x in self._history_distances]
                mad = self._percentile(sorted(deviations), 50.0)

                # Adjusted Threshold: median + k * MAD * (1 + 1/sqrt(N))
                adjustment = self.k_factor * mad * (1.0 + 1.0 / math.sqrt(N))
                threshold_value = median + adjustment

                stats["median"] = median
                stats["mad"] = mad
                stats["adjustment"] = adjustment

            elif self.estimation_method == "percentile":
                # Percentile thresholding
                sorted_history = sorted(self._history_distances)
                target_perc = self._percentile(sorted_history, self.percentile_value)
                threshold_value = target_perc
                stats["percentile_value"] = target_perc

            else:  # 'mean_std'
                mean = self._running_mean
                std = math.sqrt(self._running_var)
                adjustment = self.k_factor * std
                threshold_value = mean + adjustment

                stats["mean"] = mean
                stats["std"] = std
                stats["adjustment"] = adjustment
        else:
            # Fallback when history is sparse
            stats["fallback_used"] = 1.0
            threshold_value = baseline_adaptive

        # Clamp threshold to reasonable boundaries
        clamped_value = float(max(0.05, min(0.95, threshold_value)))

        return AdaptiveThresholdResult(
            threshold_value=clamped_value,
            confidence=threshold_confidence,
            sample_count=sample_count,
            variance_estimate=self._running_var,
            estimation_method=self.estimation_method,
            statistics=stats,
        )

    @staticmethod
    def _percentile(sorted_data: List[float], percentile: float) -> float:
        """Find the percentile of a sorted list of values (linear interpolation)."""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * (percentile / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return d0 + d1
