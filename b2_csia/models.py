"""
b2_csia/models.py
=================
Strongly-typed value objects, enums, and statistical accumulators
for the B2 CSIA layer (V2).

These types extend the CSIA pipeline with:

* ``VehicleProfile`` – kinematic envelope for a specific station type.
* ``VehicleProfileRegistry`` – selects the correct profile from a cluster.
* ``ExplainabilityReport`` – structured, machine-readable trust explanation.
* ``TrustHistory`` – bounded history of per-vehicle trust observations.
* ``StreamingStats`` – incremental mean/variance/entropy without batch
  recomputation (Welford's online algorithm + streaming histogram).
* ``AnalysisPlugin`` + ``AnalysisRegistry`` – extensible plugin system for
  future analysis engines.

Design notes
------------
* ``ExplainabilityReport`` is frozen (immutable) – it is the final verdict.
* ``TrustHistory`` is mutable – updated in place on every observation.
* ``StreamingStats`` is mutable – designed for a single-pass update loop.
* ``VehicleProfile`` is frozen – profiles are configuration constants.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Protocol, Tuple


# ---------------------------------------------------------------------------
# VehicleProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VehicleProfile:
    """Physical kinematic envelope for a specific ITS station type.

    All thresholds are in SI units (m/s, m/s², °/s) unless noted.

    Parameters
    ----------
    station_type:
        ETSI station_type integer code that this profile applies to.
    label:
        Human-readable label (e.g. ``"passenger_car"``).
    max_acceleration:
        Maximum plausible longitudinal acceleration (m/s²).
    max_deceleration:
        Maximum plausible deceleration magnitude (m/s²).
    max_yaw_rate:
        Maximum plausible yaw rate (°/s).
    expected_update_hz:
        Expected CAM broadcast rate (Hz).  Used for jitter scoring.
    heading_tolerance:
        Acceptable heading deviation within a window (°).
    max_speed:
        Maximum plausible speed (m/s).
    """

    station_type: int
    label: str
    max_acceleration: float
    max_deceleration: float
    max_yaw_rate: float
    expected_update_hz: float
    heading_tolerance: float
    max_speed: float


# ---------------------------------------------------------------------------
# Default built-in profiles (override-able from YAML)
# ---------------------------------------------------------------------------

#: Default profiles keyed by station_type integer.
DEFAULT_PROFILES: Dict[int, VehicleProfile] = {
    1: VehicleProfile(1,  "pedestrian",    3.0,  5.0, 180.0,  2.0, 30.0,  3.0),
    4: VehicleProfile(4,  "motorcycle",   10.0, 15.0,  90.0, 10.0, 10.0, 83.3),
    5: VehicleProfile(5,  "passenger_car", 8.0, 12.0,  45.0, 10.0,  5.0, 55.6),
    6: VehicleProfile(6,  "bus",           3.5,  8.0,  25.0,  5.0,  3.0, 33.3),
    7: VehicleProfile(7,  "light_truck",   5.0,  9.0,  30.0,  5.0,  4.0, 44.4),
    8: VehicleProfile(8,  "heavy_truck",   3.0,  7.0,  20.0,  5.0,  3.0, 36.1),
    15: VehicleProfile(15, "rsu",           0.0,  0.0,   0.0,  1.0,  0.0,  0.0),
}

#: Fallback profile used when no station_type-specific profile is found.
_UNKNOWN_PROFILE = VehicleProfile(0, "unknown", 15.0, 15.0, 180.0, 10.0, 30.0, 100.0)


class VehicleProfileRegistry:
    """Registry of ``VehicleProfile`` objects, selectable by station_type.

    Callers can override or extend the built-in defaults by calling
    ``register()``.  The registry is deterministic: for a given station_type
    the same profile is always returned.

    Parameters
    ----------
    profiles:
        Initial mapping of station_type → VehicleProfile.  If ``None``,
        the built-in ``DEFAULT_PROFILES`` are used.
    """

    def __init__(
        self,
        profiles: Optional[Dict[int, VehicleProfile]] = None,
    ) -> None:
        self._profiles: Dict[int, VehicleProfile] = dict(
            DEFAULT_PROFILES if profiles is None else profiles
        )

    def register(self, profile: VehicleProfile) -> None:
        """Add or replace a profile for ``profile.station_type``.

        Parameters
        ----------
        profile:
            The profile to register.
        """
        self._profiles[profile.station_type] = profile

    def get(self, station_type: Optional[int]) -> VehicleProfile:
        """Return the profile for *station_type*, or the unknown fallback.

        Parameters
        ----------
        station_type:
            ETSI station_type integer.

        Returns
        -------
        VehicleProfile
            Matched profile, or ``_UNKNOWN_PROFILE`` if not found.
        """
        if station_type is None:
            return _UNKNOWN_PROFILE
        return self._profiles.get(station_type, _UNKNOWN_PROFILE)

    def dominant_profile(
        self,
        cluster: List[Dict[str, Any]],
        station_type_field: str = "cam.cam_parameters.basic_container.station_type",
    ) -> VehicleProfile:
        """Determine the dominant profile for a message cluster.

        Scans each message for its station_type value, counts occurrences,
        and returns the profile for the most common station_type.

        Parameters
        ----------
        cluster:
            List of decoded message dicts.
        station_type_field:
            Dot-path to the station_type field within each message.

        Returns
        -------
        VehicleProfile
            Profile for the most frequent station_type in the cluster,
            or ``_UNKNOWN_PROFILE`` if the cluster is empty or the field
            is absent from all messages.
        """
        counts: Dict[int, int] = {}
        for msg in cluster:
            if not isinstance(msg, dict):
                continue
            st = _nested_get_int(msg, station_type_field)
            if st is not None:
                counts[st] = counts.get(st, 0) + 1
        if not counts:
            return _UNKNOWN_PROFILE
        dominant_st = max(counts, key=lambda k: counts[k])
        return self.get(dominant_st)


def _nested_get_int(obj: Any, dotted_key: str) -> Optional[int]:
    """Traverse *obj* using *dotted_key* and return an int leaf value."""
    parts = dotted_key.split(".")
    node: Any = obj
    for part in parts:
        if not isinstance(node, dict):
            return None
        node = node.get(part)
        if node is None:
            return None
    try:
        return int(node)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# StreamingStats – Welford online mean/variance + streaming entropy
# ---------------------------------------------------------------------------


class StreamingStats:
    """Online incremental statistics accumulator.

    Implements Welford's single-pass algorithm for mean and variance.
    Also maintains a fixed-bin histogram for Shannon entropy estimation.

    This avoids recomputing statistics from scratch on every new observation.
    The running entropy is an approximation based on the current histogram;
    it converges to the exact batch entropy as observations accumulate.

    Parameters
    ----------
    n_bins:
        Number of histogram bins for entropy computation (default 8).
    window:
        Maximum number of recent samples retained for exact statistics.
        If ``None``, all samples are retained (unbounded).
    """

    def __init__(self, n_bins: int = 8, window: Optional[int] = None) -> None:
        self._n_bins = n_bins
        self._window = window

        # Welford accumulators
        self._count: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0  # sum of squared deviations

        # Bounded sample window (for IQR-based robust statistics)
        self._samples: Deque[float] = deque(maxlen=window)

        # Histogram bins; re-scaled when the value range expands
        self._hist: List[int] = [0] * n_bins
        self._min_val: float = float("inf")
        self._max_val: float = float("-inf")
        self._needs_rebuild: bool = False

    # ------------------------------------------------------------------
    # Public update / query API
    # ------------------------------------------------------------------

    def update(self, value: float) -> None:
        """Incorporate a new observation into the running statistics.

        Parameters
        ----------
        value:
            The new observation.  Must be a finite float.
        """
        if not math.isfinite(value):
            return  # silently skip non-finite values

        # Welford update
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._M2 += delta * delta2

        self._samples.append(value)

        # Histogram update – re-build if range expanded
        if value < self._min_val or value > self._max_val:
            old_min, old_max = self._min_val, self._max_val
            self._min_val = min(self._min_val, value)
            self._max_val = max(self._max_val, value)
            if old_min != self._min_val or old_max != self._max_val:
                self._needs_rebuild = True
        if self._needs_rebuild:
            self._rebuild_histogram()
        else:
            self._insert_bin(value)

    def _rebuild_histogram(self) -> None:
        """Recompute histogram from the retained sample window."""
        self._hist = [0] * self._n_bins
        self._needs_rebuild = False
        span = self._max_val - self._min_val
        if span == 0.0:
            self._hist[0] = len(self._samples)
            return
        for s in self._samples:
            self._insert_bin(s)

    def _insert_bin(self, value: float) -> None:
        span = self._max_val - self._min_val
        if span == 0.0:
            self._hist[0] += 1
            return
        idx = int((value - self._min_val) / span * (self._n_bins - 1))
        idx = max(0, min(self._n_bins - 1, idx))
        self._hist[idx] += 1

    @property
    def count(self) -> int:
        """Number of observations incorporated so far."""
        return self._count

    @property
    def mean(self) -> float:
        """Running mean (0.0 when no observations)."""
        return self._mean if self._count > 0 else 0.0

    @property
    def variance(self) -> float:
        """Running sample variance (0.0 when fewer than 2 observations)."""
        if self._count < 2:
            return 0.0
        return self._M2 / (self._count - 1)

    @property
    def std(self) -> float:
        """Running sample standard deviation."""
        v = self.variance
        return math.sqrt(v) if v > 0.0 else 0.0

    @property
    def entropy(self) -> float:
        """Normalised Shannon entropy ∈ [0.0, 1.0] from the running histogram.

        Returns 0.0 when fewer than 2 unique bins are populated or when
        there are fewer than 2 observations.
        """
        total = sum(self._hist)
        if total < 2:
            return 0.0
        h = 0.0
        for b in self._hist:
            if b > 0:
                p = b / total
                h -= p * math.log2(p)
        max_h = math.log2(self._n_bins) if self._n_bins > 1 else 1.0
        return min(1.0, max(0.0, h / max_h))

    def reset(self) -> None:
        """Reset all accumulators to their initial state."""
        self._count = 0
        self._mean = 0.0
        self._M2 = 0.0
        self._samples.clear()
        self._hist = [0] * self._n_bins
        self._min_val = float("inf")
        self._max_val = float("-inf")
        self._needs_rebuild = False


# ---------------------------------------------------------------------------
# TrustHistory – bounded history of per-vehicle trust scores
# ---------------------------------------------------------------------------


@dataclass
class TrustHistory:
    """Bounded rolling history of trust scores for a single ITS station.

    Trust evolves smoothly over time:
    * When a new score is below the running average → exponential decay
      (parameterised by ``decay_alpha``).
    * When a new score is above the running average → gradual recovery
      (parameterised by ``recovery_beta``).

    Parameters
    ----------
    station_id:
        The ITS station this history belongs to.
    window:
        Maximum number of trust scores retained (default 20).
    decay_alpha:
        Decay rate per suspicious observation (default 0.1).
        Higher values make trust drop faster.
    recovery_beta:
        Recovery rate per benign observation (default 0.05).
        Higher values make trust recover faster.
    """

    station_id: int
    window: int = 20
    decay_alpha: float = 0.1
    recovery_beta: float = 0.05

    _scores: Deque[float] = field(init=False)
    _current: float = field(init=False, default=1.0)
    _last_updated: float = field(init=False, default_factory=time.time)
    _observation_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._scores = deque(maxlen=self.window)
        self._current = 1.0
        self._last_updated = time.time()
        self._observation_count = 0

    def update(self, new_score: float) -> float:
        """Incorporate *new_score* and return the evolved trust value.

        Parameters
        ----------
        new_score:
            Raw trust score ∈ [0.0, 1.0] from the latest analysis.

        Returns
        -------
        float
            The evolved (smoothed) trust value ∈ [0.0, 1.0].
        """
        new_score = float(max(0.0, min(1.0, new_score)))
        self._scores.append(new_score)
        self._observation_count += 1
        self._last_updated = time.time()

        if new_score < self._current:
            # Suspicious observation – apply decay
            self._current = (1.0 - self.decay_alpha) * self._current + self.decay_alpha * new_score
        else:
            # Benign or recovering – apply gradual recovery
            self._current = (1.0 - self.recovery_beta) * self._current + self.recovery_beta * new_score

        return float(max(0.0, min(1.0, self._current)))

    @property
    def current(self) -> float:
        """The current smoothed trust value ∈ [0.0, 1.0]."""
        return self._current

    @property
    def observation_count(self) -> int:
        """Total number of trust observations incorporated."""
        return self._observation_count

    @property
    def statistical_stability(self) -> float:
        """Estimate of how stable/confident the trust estimate is ∈ [0.0, 1.0].

        Returns higher values when more observations have been accumulated
        (asymptotically approaches 1.0).  Uses a logistic-like curve.
        """
        n = self._observation_count
        return float(n / (n + 10.0))  # 10 observations → 0.5 stability

    @property
    def history(self) -> List[float]:
        """Copy of the rolling score history (oldest first)."""
        return list(self._scores)


# ---------------------------------------------------------------------------
# ExplainabilityReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplainabilityReport:
    """Structured, machine-readable explanation of a B2 trust decision.

    Returned by ``CSIA.check_extended()`` (the opt-in extended API).
    The standard ``CSIA.check()`` output is always a plain ``dict`` with the
    original 5 keys; this report is *additional* information.

    Parameters
    ----------
    trust_score:
        Final fused trust probability ∈ [0.0, 1.0].  Identical to
        ``check()["trust"]``.
    confidence:
        Estimate of how confident the engine is in this trust score
        ∈ [0.0, 1.0].  Based on cluster size, statistical stability,
        and the number of historical observations for these vehicles.
    statistical_stability:
        Measure of how stable the input data was during this window
        ∈ [0.0, 1.0].  1.0 means all sub-scores were in agreement;
        0.0 means the sub-scores were maximally contradictory.
    contributing_factors:
        Mapping of engine name → contribution to the final score.
        Example: ``{"kinematic": 0.55, "semantic": 0.12, "temporal": 0.18}``.
    anomaly_reasons:
        Ordered list of human-readable anomaly descriptions, most severe
        first.  Empty when no anomaly is detected.
    decision_summary:
        One-sentence natural-language verdict
        (e.g. ``"High Sybil confidence: kinematic clone + identical station_id"``).
    cluster_size:
        Number of messages in the analysed cluster.
    vehicle_profile_label:
        Label of the dominant vehicle profile selected for this cluster.
    raw_scores:
        Dictionary of raw sub-engine scores before fusion weighting:
        ``{"kinematic": float, "semantic": float, "temporal": float}``.
    """

    trust_score: float
    confidence: float
    statistical_stability: float
    contributing_factors: Dict[str, float]
    anomaly_reasons: List[str]
    decision_summary: str
    cluster_size: int
    vehicle_profile_label: str
    raw_scores: Dict[str, float]

    validation_score: float = 1.0
    validation_confidence: float = 1.0
    fatal: bool = False
    validation_reasons: List[str] = field(default_factory=list)
    applied_penalties: Dict[str, float] = field(default_factory=dict)
    belief: float = 1.0
    disbelief: float = 0.0
    uncertainty: float = 0.0
    evidence_summary: Dict[str, str] = field(default_factory=dict)
    evidence_reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AnalysisPlugin protocol
# ---------------------------------------------------------------------------


class AnalysisPlugin(Protocol):
    """Protocol that all pluggable analysis engines must implement.

    Register custom engines with ``AnalysisRegistry.register()`` to
    extend the B2 pipeline without modifying core CSIA code.

    Attributes
    ----------
    name:
        Unique string identifier for this plugin
        (used as the key in ``ExplainabilityReport.contributing_factors``).
    weight:
        Fusion weight ∈ (0, 1].  The registry normalises weights so they
        sum to 1.0 before computing the fused score.
    """

    name: str
    weight: float

    def analyse(self, cluster: List[Dict[str, Any]], config: Dict[str, Any]) -> float:
        """Run analysis on *cluster* and return a trust score ∈ [0.0, 1.0].

        Parameters
        ----------
        cluster:
            List of decoded message dicts forming a spatio-temporal cluster.
        config:
            The full ``b2_csia`` configuration section from ``isce_config.yaml``.

        Returns
        -------
        float
            Trust score ∈ [0.0, 1.0].  1.0 = benign, 0.0 = suspicious.
        """
        ...


# ---------------------------------------------------------------------------
# AnalysisRegistry
# ---------------------------------------------------------------------------


class AnalysisRegistry:
    """Registry of ``AnalysisPlugin`` objects.

    Manages plugin registration, weight normalisation, and dispatches
    analysis to all registered engines in deterministic order.

    Parameters
    ----------
    config:
        The ``b2_csia`` config section dict.  Passed to each plugin's
        ``analyse()`` call.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._plugins: Dict[str, AnalysisPlugin] = {}

    def register(self, plugin: AnalysisPlugin) -> None:
        """Register a new analysis plugin.

        Existing plugins with the same ``name`` are replaced.

        Parameters
        ----------
        plugin:
            An object implementing the ``AnalysisPlugin`` protocol.
        """
        self._plugins[plugin.name] = plugin

    def run_all(
        self, cluster: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Run all registered plugins and fuse their scores.

        Weights are taken from each plugin's ``weight`` attribute and
        normalised so they sum to 1.0.

        Parameters
        ----------
        cluster:
            List of decoded message dicts.

        Returns
        -------
        fused_score : float
            Weighted sum of all plugin scores, clamped to [0.0, 1.0].
        raw_scores : dict[str, float]
            Per-plugin trust scores before weight application.
        contributions : dict[str, float]
            Per-plugin weighted contributions to the fused score.
        """
        if not self._plugins:
            return 1.0, {}, {}

        total_weight = sum(p.weight for p in self._plugins.values())
        if total_weight <= 0.0:
            total_weight = 1.0

        raw_scores: Dict[str, float] = {}
        contributions: Dict[str, float] = {}
        fused = 0.0

        for name, plugin in sorted(self._plugins.items()):
            try:
                score = float(plugin.analyse(cluster, self._config))
                score = max(0.0, min(1.0, score))
            except Exception:
                score = 1.0  # defensive: treat errors as benign
            raw_scores[name] = score
            w = plugin.weight / total_weight
            contributions[name] = w * score
            fused += w * score

        return float(max(0.0, min(1.0, fused))), raw_scores, contributions


__all__ = [
    "VehicleProfile",
    "VehicleProfileRegistry",
    "DEFAULT_PROFILES",
    "StreamingStats",
    "TrustHistory",
    "ExplainabilityReport",
    "AnalysisPlugin",
    "AnalysisRegistry",
]
