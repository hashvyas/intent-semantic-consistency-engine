"""
b2_csia/csia.py
===============
B2 – Cluster Semantic Invariance Analysis (CSIA) – Research Grade v2
Part of the ISCE STB V2X Security Pipeline.

Purpose
-------
Detect coordinated / Sybil broadcast behaviour through a four-stage,
multi-dimensional coherence analysis pipeline over windows of decoded
CAM messages.  The v2 engine replaces the single min-max-normalised
Euclidean distance with:

  1. Spatio-Temporal Pre-Clustering
  2a. Kinematic Engine   – Mahalanobis distance with Weighted Euclidean fallback
  2b. Semantic Engine    – Pairwise Hamming similarity on CAM structural fields
  3.  Temporal Entropy   – Shannon entropy of inter-arrival time distribution
  4.  Score Fusion       – weighted sum → continuous trust ∈ [0.0, 1.0]

V2 additions (backward-compatible)
-----------------------------------
* ``check()`` – **unchanged**, same 5-key payload.
* ``check_extended()`` – opt-in extended API returning ``ExplainabilityReport``.
* Vehicle-type kinematic profiles (``VehicleProfileRegistry``).
* Incremental deque-based rolling windows (O(1) insertion/eviction).
* Streaming statistics via Welford's online algorithm.
* Plugin-based analysis engine (``AnalysisRegistry``).
* Trust evolution per station_id (exponential decay + gradual recovery).
* Config validation at startup (``validate_b2_config``).
* Covariance matrix caching (keyed by cluster shape + values).

Algorithm
---------

Stage 1 – Spatio-Temporal Pre-Clustering
  Incoming messages are grouped into local neighbourhoods using:
    • Haversine distance between positions < spatial_radius_m  (default 100 m)
    • |Δ generation_delta_time| ≤ window_size_ns
  A Union-Find algorithm identifies connected components.  Only clusters of
  size ≥ min_cluster_size are analysed.  If multiple clusters form (e.g. two
  independent platoons in the same window), the *most suspicious* (minimum)
  cluster score is returned so that no threat hides behind a benign cluster.

Stage 2a – Kinematic Engine
  Feature vector: speed, heading, yaw_rate, steering_wheel_angle,
                  lateral_acceleration, longitudinal_acceleration.

  Robust Scaling: (X − median) / IQR.  Zero-IQR columns use fallback_ranges
  to avoid division by zero.

  Distance metric:
    Mahalanobis  – when cluster size ≥ mahalanobis_min_samples AND the
                   covariance matrix is well-conditioned (cond < 1e12).
                   Accounts for feature correlations; correctly identifies
                   clusters where heading and yaw_rate co-vary normally.
    Euclidean    – fallback on the already-robust-scaled data when covariance
                   is singular or the cluster is too small.

  Adaptive Threshold:
    highway (median cluster speed ≥ highway_speed_threshold):
        threshold = highway_kinematic_threshold  (tighter — uniformity expected)
    city:
        threshold = city_kinematic_threshold     (wider  — diversity expected)

Stage 2b – CAM Semantic Engine
  Categorical fingerprint: station_type, station_id.
  For each pair of messages, Hamming similarity = fraction of categorical
  fields with identical values.  Average over all pairs.
  semantic_trust = 1 − average_hamming_similarity.

  Interpretation:
    All messages share station_type=5 AND station_id=42 → similarity = 1.0
    → semantic_trust = 0.0  (classic Sybil / ghost injection)

    All messages have unique station_ids → similarity ≈ 0.5 (type matches,
    id differs) → semantic_trust = 0.5  (typical legitimate platoon)

Stage 3 – Temporal Entropy Engine
  Sort timestamps → compute inter-arrival deltas.
  Bin deltas into temporal_entropy_bins buckets → Shannon entropy H.
  timing_trust = 0.6 × (spread / window_size_ns) + 0.4 × (H / log2(bins))

  All identical timestamps (spread = 0) → timing_trust = 0.0  (machine burst)
  All equal inter-arrivals (perfectly clocked) → entropy = 0 → timing_trust
  reduced by 0.4 component even if spread > 0.
  Natural staggered arrivals → high entropy → timing_trust approaches 1.0.

Stage 4 – Score Fusion
  final = w_kinematic × kinematic_trust
        + w_semantic × semantic_trust
        + w_timing × timing_trust
  Clamped to [0.0, 1.0].

Score convention (same as B1):
  1.0  → benign / messages appear kinematically independent
  0.0  → highly suspicious / coordinated or replayed behaviour detected

Graceful degradation
--------------------
  • Non-dict entries in the window list are silently skipped.
  • If no cluster of size ≥ min_cluster_size forms, returns 1.0.
  • Missing kinematic fields default to 0.0 via safe .get() traversal.
  • Singular covariance → automatic fallback to Euclidean on scaled data.
  • Missing semantic fields → excluded from similarity calculation.

References
----------
* ETSI TR 103 460  – Misbehaviour Detection in Intelligent Transport Systems
* ETSI EN 302 637-2 – Cooperative Awareness Message (CAM) specification
* ETSI TS 102 894-2 – ITS Common Data Dictionary
"""

from __future__ import annotations

import collections
import itertools
import logging
import math
import os
import pathlib
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from b2_csia.config import ConfigurationError, validate_b2_config
from b2_csia.models import (
    AnalysisRegistry,
    ExplainabilityReport,
    TrustHistory,
    VehicleProfile,
    VehicleProfileRegistry,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config path – one level above this package directory.
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "isce_config.yaml"
)


# ===========================================================================
# Low-level helpers
# ===========================================================================

def _nested_get(obj: Any, dotted_key: str, default: float = 0.0) -> float:
    """Traverse a nested dict using a dot-separated key path, return float.

    Each segment is applied with ``.get()``; a missing node or non-numeric
    leaf returns *default* rather than raising.

    Parameters
    ----------
    obj:
        Root object (decoded message dict).
    dotted_key:
        Dot-separated path, e.g.
        ``"cam.cam_parameters.basic_container.reference_position.latitude"``.
    default:
        Value returned when any segment is absent or the leaf is non-numeric.

    Returns
    -------
    float
        Leaf value cast to ``float``, or *default*.
    """
    parts = dotted_key.split(".")
    node: Any = obj
    for part in parts:
        if not isinstance(node, dict):
            return default
        node = node.get(part)
        if node is None:
            return default
    try:
        v = float(node)
        # Guard against NaN/inf which can corrupt the pipeline
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _nested_get_any(obj: Any, dotted_key: str, default: Any = None) -> Any:
    """Traverse a nested dict and return the raw leaf value (no float cast).

    Used for categorical / non-numeric fields such as ``station_type`` and
    ``station_id``.
    """
    parts = dotted_key.split(".")
    node: Any = obj
    for part in parts:
        if not isinstance(node, dict):
            return default
        node = node.get(part)
        if node is None:
            return default
    return node


def _haversine_m(
    lat1_e7: float, lon1_e7: float,
    lat2_e7: float, lon2_e7: float,
) -> float:
    """Approximate Haversine distance in metres between two ETSI positions.

    Parameters
    ----------
    lat1_e7, lon1_e7, lat2_e7, lon2_e7:
        Latitude / longitude in ETSI 1e-7 degree units
        (as per ETSI EN 302 637-2 §6.1.1).

    Returns
    -------
    float
        Great-circle distance in metres.
    """
    lat1 = math.radians(lat1_e7 * 1e-7)
    lat2 = math.radians(lat2_e7 * 1e-7)
    dlat = lat2 - lat1
    dlon = math.radians((lon2_e7 - lon1_e7) * 1e-7)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 6_371_000.0 * 2.0 * math.asin(min(1.0, math.sqrt(max(0.0, a))))


# ===========================================================================
# Stage 1 – Spatio-Temporal Pre-Clustering
# ===========================================================================

def _build_clusters(
    messages: List[dict],
    spatial_radius_m: float,
    window_size_ns: float,
    lat_field: str,
    lon_field: str,
    ts_field: str,
) -> List[List[dict]]:
    """Group messages into spatio-temporal neighbourhoods using Union-Find.

    Two messages are *connected* if:
      • Their Haversine distance < spatial_radius_m, AND
      • |Δ timestamp| ≤ window_size_ns.

    Connected components form clusters.

    Parameters
    ----------
    messages:
        Pre-filtered list of dict messages (non-dicts already excluded).
    spatial_radius_m:
        Maximum spatial distance for adjacency (metres).
    window_size_ns:
        Maximum timestamp delta for adjacency.
    lat_field, lon_field, ts_field:
        Dot-paths to position and timestamp fields.

    Returns
    -------
    list[list[dict]]
        One sub-list per connected component.  Components are not filtered
        by min_cluster_size here; that is the caller's responsibility.
    """
    n = len(messages)
    lats = [_nested_get(m, lat_field, 0.0) for m in messages]
    lons = [_nested_get(m, lon_field, 0.0) for m in messages]
    tss  = [_nested_get(m, ts_field,  0.0) for m in messages]

    # Union-Find with path compression and rank
    parent = list(range(n))
    rank   = [0] * n

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        rx, ry = _find(x), _find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    for i in range(n):
        for j in range(i + 1, n):
            if abs(tss[i] - tss[j]) > window_size_ns:
                continue
            dist_m = _haversine_m(lats[i], lons[i], lats[j], lons[j])
            if dist_m <= spatial_radius_m:
                _union(i, j)

    groups: Dict[int, List[dict]] = collections.defaultdict(list)
    for i in range(n):
        groups[_find(i)].append(messages[i])

    return list(groups.values())


# ===========================================================================
# Stage 2a – Kinematic Engine
# ===========================================================================

def _robust_scale(
    matrix: np.ndarray,
    fields: List[str],
    fallback_ranges: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply per-column robust scaling: (X − median) / IQR.

    Zero-IQR columns (zero-variance features) use fallback_ranges so we
    never divide by zero.

    Parameters
    ----------
    matrix:
        Shape ``(n, d)`` raw feature matrix.
    fields:
        Field names corresponding to each column (used to look up fallbacks).
    fallback_ranges:
        Maps field name → reference range used when IQR = 0.

    Returns
    -------
    scaled : np.ndarray
        Shape ``(n, d)`` robust-scaled matrix.
    iqr_scales : np.ndarray
        Shape ``(d,)`` scale factors used per column (IQR or fallback).
    """
    n, d = matrix.shape
    scaled     = np.empty_like(matrix, dtype=np.float64)
    iqr_scales = np.ones(d, dtype=np.float64)

    for col, field in enumerate(fields):
        col_data = matrix[:, col]
        med  = float(np.median(col_data))
        q75  = float(np.percentile(col_data, 75))
        q25  = float(np.percentile(col_data, 25))
        iqr  = q75 - q25

        if iqr == 0.0:
            fallback = float(fallback_ranges.get(field, 1.0))
            if fallback <= 0.0:
                fallback = 1.0
            scale = fallback
        else:
            scale = iqr

        scaled[:, col] = (col_data - med) / scale
        iqr_scales[col] = scale

    return scaled, iqr_scales


def _avg_pairwise_dist(
    matrix_scaled: np.ndarray,
    mahalanobis_min_samples: int,
    cached_s_inv: Optional[np.ndarray] = None,
) -> Tuple[float, str, Optional[np.ndarray]]:
    """Compute average pairwise distance on robust-scaled kinematic data.

    Tries Mahalanobis when the cluster is large enough and the covariance
    matrix is well-conditioned.  Falls back to plain Euclidean on the
    already-scaled data otherwise.

    Parameters
    ----------
    matrix_scaled:
        Shape ``(n, d)`` robust-scaled matrix.
    mahalanobis_min_samples:
        Minimum ``n`` to attempt Mahalanobis.
    cached_s_inv:
        Optional pre-computed inverse covariance matrix from a previous
        call with the same data (performance optimisation).

    Returns
    -------
    avg_dist : float
        Mean pairwise distance (0.0 for identical vectors).
    method : str
        ``"mahalanobis"`` or ``"euclidean"`` (for logging).
    s_inv : np.ndarray | None
        The inverse covariance matrix used (for caching by caller).
    """
    n, d = matrix_scaled.shape
    if n < 2:
        return 0.0, "none", None

    method  = "euclidean"
    S_inv   = cached_s_inv

    if S_inv is None and n >= mahalanobis_min_samples and d >= 1:
        try:
            S = np.cov(matrix_scaled.T)
            if d == 1:
                S = np.array([[float(S)]])
            cond = np.linalg.cond(S)
            if np.isfinite(cond) and cond < 1e12:
                S_inv  = np.linalg.inv(S)
                method = "mahalanobis"
        except (np.linalg.LinAlgError, ValueError):
            pass

    if S_inv is not None:
        method = "mahalanobis"

    total, count = 0.0, 0
    for i, j in itertools.combinations(range(n), 2):
        diff = matrix_scaled[i] - matrix_scaled[j]
        if S_inv is not None:
            d_sq = float(diff @ S_inv @ diff)
            total += math.sqrt(max(0.0, d_sq))
        else:
            total += math.sqrt(float(np.dot(diff, diff)))
        count += 1

    avg_dist = (total / count) if count > 0 else 0.0
    return avg_dist, method, S_inv


def _dist_to_trust(avg_dist: float, threshold: float, cap: float) -> float:
    """Map an average pairwise distance to a kinematic trust score.

    • dist ≤ threshold → 0.0   (suspicious: too similar)
    • dist ≥ cap       → 1.0   (benign: sufficiently diverse)
    • linear ramp in between.

    Parameters
    ----------
    avg_dist:
        Mean pairwise distance in robust-scaled space.
    threshold:
        Minimum distance for any positive trust.
    cap:
        Distance at which trust saturates to 1.0.

    Returns
    -------
    float
        Trust score ∈ [0.0, 1.0].
    """
    if avg_dist <= threshold:
        return 0.0
    span = cap - threshold
    if span <= 0.0:
        return 1.0
    return float(min(1.0, (avg_dist - threshold) / span))


# ===========================================================================
# Stage 2b – CAM Semantic Engine
# ===========================================================================

def _semantic_trust(
    messages: List[dict],
    semantic_fields: List[str],
) -> float:
    """Compute semantic trust via pairwise Hamming similarity.

    Extracts categorical fingerprints (station_type, station_id) from each
    message and computes the average pairwise similarity across the cluster.
    High similarity → many messages share the same categorical identity →
    suspicious (Sybil / ghost injection).

    Parameters
    ----------
    messages:
        Cluster of decoded message dicts.
    semantic_fields:
        Ordered list of dot-paths to categorical fields.

    Returns
    -------
    float
        semantic_trust = 1 − average_hamming_similarity ∈ [0.0, 1.0].
        0.0 → all messages share identical fingerprints (suspicious).
        1.0 → all messages have entirely distinct fingerprints (benign).
    """
    if not semantic_fields:
        return 1.0  # No semantic fields configured → skip engine

    fingerprints: List[Tuple] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        fp = tuple(_nested_get_any(m, f, default=None) for f in semantic_fields)
        fingerprints.append(fp)

    n_fp = len(fingerprints)
    if n_fp < 2:
        return 1.0

    d = len(semantic_fields)
    total_sim, count = 0.0, 0

    for i, j in itertools.combinations(range(n_fp), 2):
        matches = 0
        valid   = 0
        for k in range(d):
            a, b = fingerprints[i][k], fingerprints[j][k]
            # Count field as valid if at least one side has a non-None value
            if a is not None or b is not None:
                valid += 1
                if a == b and a is not None:
                    matches += 1
        total_sim += (matches / valid) if valid > 0 else 0.0
        count += 1

    avg_sim = total_sim / count if count > 0 else 0.0
    return float(1.0 - avg_sim)


# ===========================================================================
# Stage 3 – Temporal Entropy Engine
# ===========================================================================

def _temporal_entropy_detail(
    messages: List[dict],
    ts_field: str,
    n_bins: int,
    window_size_ns: float,
) -> Tuple[float, float]:
    """Compute temporal trust and raw entropy from inter-arrival time distribution.

    Combines two complementary signals:

    • **Spread fraction** (0.6 weight): ``time_spread / window_size_ns``.
      Penalises machine-synchronised bursts where all timestamps are
      identical (spread = 0).

    • **Shannon entropy** (0.4 weight): entropy of inter-arrival histogram
      normalised by ``log2(n_bins)``.  Penalises perfectly-clocked attacks
      (equal inter-arrivals → entropy = 0) even if spread > 0.

    Parameters
    ----------
    messages:
        Cluster of decoded message dicts.
    ts_field:
        Dot-path to the timestamp field.
    n_bins:
        Number of bins for the inter-arrival histogram.
    window_size_ns:
        Window duration used to normalise the spread fraction.

    Returns
    -------
    timing_trust : float
        Temporal trust ∈ [0.0, 1.0].
        0.0 → perfectly synchronised machine burst.
        1.0 → natural, high-entropy inter-arrival distribution.
    entropy_score : float
        Normalised Shannon entropy component ∈ [0.0, 1.0].  Exported
        directly as the ``"entropy"`` key of the check() payload so
        callers can observe the raw inter-arrival randomness.
    """
    timestamps: List[float] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        ts = _nested_get(m, ts_field, float("nan"))
        if math.isfinite(ts):
            timestamps.append(ts)

    if len(timestamps) < 2:
        return 1.0, 0.0  # Insufficient data → treat as benign

    timestamps.sort()
    time_spread = timestamps[-1] - timestamps[0]

    # ── Spread fraction ────────────────────────────────────────────────────
    window = window_size_ns if window_size_ns > 0.0 else 1.0
    spread_score = float(min(1.0, time_spread / window))

    if time_spread == 0.0:
        return 0.0, 0.0  # Machine burst: perfect synchronisation

    # ── Inter-arrival entropy ──────────────────────────────────────────────
    deltas = [timestamps[k + 1] - timestamps[k] for k in range(len(timestamps) - 1)]

    if not deltas:
        return spread_score, 0.0  # Only one unique interval

    min_d, max_d = min(deltas), max(deltas)

    if max_d == min_d:
        # All inter-arrivals identical → perfectly regular clocking
        entropy_score = 0.0
    else:
        bins  = [0] * n_bins
        span  = max_d - min_d
        for delta in deltas:
            idx = int((delta - min_d) / span * (n_bins - 1))
            bins[max(0, min(n_bins - 1, idx))] += 1

        total   = len(deltas)
        entropy = 0.0
        for b in bins:
            if b > 0:
                p = b / total
                entropy -= p * math.log2(p)

        max_entropy   = math.log2(n_bins) if n_bins > 1 else 1.0
        entropy_score = float(min(1.0, max(0.0, entropy / max_entropy)))

    timing_trust = float(min(1.0, max(0.0, 0.6 * spread_score + 0.4 * entropy_score)))
    return timing_trust, entropy_score


# ---------------------------------------------------------------------------
# Default benign-state payload – returned when insufficient data is present
# to form an analysable cluster.  trust/cluster_score/identity_consistency
# are 1.0 (fully trusted); entropy and replay_probability are 0.0 (no
# anomaly signal).
# ---------------------------------------------------------------------------

_BENIGN_PAYLOAD: Dict[str, float] = {
    "trust":               1.0,
    "entropy":             0.0,
    "cluster_score":       1.0,
    "replay_probability":  0.0,
    "identity_consistency": 1.0,
}


# ===========================================================================
# Built-in plugin wrappers
# ===========================================================================

class _KinematicPlugin:
    """Wraps the kinematic engine as an ``AnalysisPlugin``."""

    name: str = "kinematic"

    def __init__(self, csia_instance: "CSIA") -> None:
        self._csia = csia_instance
        self.weight: float = 0.55  # updated from config in CSIA.__init__

    def analyse(self, cluster: List[Dict[str, Any]], config: Dict[str, Any]) -> float:
        return self._csia._kinematic_trust(cluster)


class _SemanticPlugin:
    """Wraps the semantic engine as an ``AnalysisPlugin``."""

    name: str = "semantic"

    def __init__(self, csia_instance: "CSIA") -> None:
        self._csia = csia_instance
        self.weight: float = 0.20

    def analyse(self, cluster: List[Dict[str, Any]], config: Dict[str, Any]) -> float:
        return _semantic_trust(cluster, self._csia._semantic_fields)


class _TemporalPlugin:
    """Wraps the temporal entropy engine as an ``AnalysisPlugin``."""

    name: str = "temporal"

    def __init__(self, csia_instance: "CSIA") -> None:
        self._csia = csia_instance
        self.weight: float = 0.25

    def analyse(self, cluster: List[Dict[str, Any]], config: Dict[str, Any]) -> float:
        trust, _ = _temporal_entropy_detail(
            cluster,
            self._csia._ts_field,
            self._csia._entropy_bins,
            self._csia._window_size_ns,
        )
        return trust


# ===========================================================================
# CSIA class
# ===========================================================================

class CSIA:
    """Cluster Semantic Invariance Analyser – Research Grade v2.

    Loads configuration from ``isce_config.yaml`` on construction.  Exposes
    a single ``check(messages)`` method that returns a continuous trust
    probability score for a window of decoded ITS CAM messages.

    V2 additions
    ------------
    * ``check_extended(messages)`` – returns an ``ExplainabilityReport`` in
      addition to the standard payload.
    * Vehicle-type kinematic profiles selectable per cluster.
    * O(1) deque-based rolling window management.
    * Streaming statistics (Welford variance) for trust evolution.
    * Plugin-based analysis engine extensible at runtime.
    * Trust evolution (decay + recovery) per station_id.

    Parameters
    ----------
    config_path:
        Path to ``isce_config.yaml``.  Defaults to the file one level above
        this package directory (standard ISCE layout).

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    yaml.YAMLError
        If the configuration file cannot be parsed.
    KeyError
        If the ``b2_csia`` section is absent.
    b2_csia.config.ConfigurationError
        If any configuration value is invalid.
    """

    def __init__(self, config_path: Optional[str | os.PathLike] = None) -> None:
        config_path = (
            pathlib.Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        )
        if not config_path.exists():
            raise FileNotFoundError(
                f"CSIA: configuration file not found: {config_path}"
            )

        with config_path.open("r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh)

        # ── Validate configuration (fail-fast) ────────────────────────────
        try:
            validate_b2_config(raw)
        except ConfigurationError as exc:
            raise ConfigurationError(
                f"CSIA configuration validation failed: {exc}"
            ) from exc

        cfg: Dict[str, Any] = raw.get("b2_csia", {})
        if not cfg:
            raise KeyError(
                f"CSIA: 'b2_csia' section not found in {config_path}"
            )

        # ── Stage 1 – Clustering ───────────────────────────────────────────
        self._min_cluster_size: int   = int(cfg.get("min_cluster_size", 3))
        self._spatial_radius_m: float = float(cfg.get("spatial_radius_m", 100.0))
        self._window_size_ns:   float = float(cfg.get("window_size_ns", 1_000_000_000))
        self._lat_field: str = str(cfg.get(
            "position_lat_field",
            "cam.cam_parameters.basic_container.reference_position.latitude",
        ))
        self._lon_field: str = str(cfg.get(
            "position_lon_field",
            "cam.cam_parameters.basic_container.reference_position.longitude",
        ))
        self._ts_field: str = str(cfg.get("timestamp_field", "cam.generation_delta_time"))

        # ── Stage 2a – Kinematic engine ────────────────────────────────────
        self._kinematic_fields: List[str] = list(cfg.get("kinematic_fields", []))
        self._fallback_ranges:  Dict[str, float] = {
            str(k): float(v) for k, v in cfg.get("fallback_ranges", {}).items()
        }
        self._mahal_min: int = int(cfg.get("mahalanobis_min_samples", 4))

        # Adaptive thresholds
        self._highway_spd_thr:  float = float(cfg.get("highway_speed_threshold",     2000.0))
        self._highway_kin_thr:  float = float(cfg.get("highway_kinematic_threshold",  0.20))
        self._city_kin_thr:     float = float(cfg.get("city_kinematic_threshold",     0.50))
        self._cap_multiplier:   float = float(cfg.get("kinematic_cap_multiplier",     6.0))

        # ── Stage 2b – Semantic engine ─────────────────────────────────────
        self._semantic_fields: List[str] = list(cfg.get("semantic_fields", []))

        # ── Stage 3 – Temporal entropy ─────────────────────────────────────
        self._entropy_bins: int = int(cfg.get("temporal_entropy_bins", 8))

        # ── Stage 4 – Fusion weights ───────────────────────────────────────
        self._w_kin: float = float(cfg.get("weight_kinematic", 0.55))
        self._w_sem: float = float(cfg.get("weight_semantic",  0.20))
        self._w_tim: float = float(cfg.get("weight_timing",    0.25))

        # ── V2: Vehicle profile registry ───────────────────────────────────
        self._profile_registry = self._build_profile_registry(raw)

        # ── V2: Trust evolution ────────────────────────────────────────────
        self._trust_decay_alpha: float = float(raw.get("b2_trust_decay_alpha", 0.10))
        self._trust_recovery_beta: float = float(raw.get("b2_trust_recovery_beta", 0.05))
        self._trust_history_window: int = int(raw.get("b2_trust_history_window", 20))
        self._trust_histories: Dict[int, TrustHistory] = {}
        self._trust_lock = threading.Lock()

        # ── V2: Plugin registry ────────────────────────────────────────────
        self._registry = AnalysisRegistry(cfg)
        kin_plugin = _KinematicPlugin(self)
        kin_plugin.weight = self._w_kin
        sem_plugin = _SemanticPlugin(self)
        sem_plugin.weight = self._w_sem
        tim_plugin = _TemporalPlugin(self)
        tim_plugin.weight = self._w_tim
        self._registry.register(kin_plugin)
        self._registry.register(sem_plugin)
        self._registry.register(tim_plugin)

        logger.info(
            "CSIA v2 loaded: min_cluster=%d, spatial_r=%.0fm, window_ns=%.0f, "
            "kin_fields=%d, sem_fields=%d, mahal_min=%d, "
            "w=[%.2f, %.2f, %.2f]",
            self._min_cluster_size, self._spatial_radius_m, self._window_size_ns,
            len(self._kinematic_fields), len(self._semantic_fields), self._mahal_min,
            self._w_kin, self._w_sem, self._w_tim,
        )

    # -----------------------------------------------------------------------
    # Configuration helpers
    # -----------------------------------------------------------------------

    def _build_profile_registry(self, raw: Dict[str, Any]) -> VehicleProfileRegistry:
        """Construct a ``VehicleProfileRegistry`` from YAML config.

        Reads the optional ``b2_vehicle_profiles`` section and registers
        any profiles found, falling back to built-in defaults for profiles
        not specified in the YAML.

        Parameters
        ----------
        raw:
            Full YAML root dict.

        Returns
        -------
        VehicleProfileRegistry
            Populated registry.
        """
        registry = VehicleProfileRegistry()  # starts with built-in defaults
        yaml_profiles = raw.get("b2_vehicle_profiles") or {}
        if not isinstance(yaml_profiles, dict):
            return registry

        for label, pdata in yaml_profiles.items():
            if not isinstance(pdata, dict):
                continue
            try:
                st = int(pdata.get("station_type", -1))
                if st < 0:
                    continue
                profile = VehicleProfile(
                    station_type=st,
                    label=str(label),
                    max_acceleration=float(pdata.get("max_acceleration", 8.0)),
                    max_deceleration=float(pdata.get("max_deceleration", 12.0)),
                    max_yaw_rate=float(pdata.get("max_yaw_rate", 45.0)),
                    expected_update_hz=float(pdata.get("expected_update_hz", 10.0)),
                    heading_tolerance=float(pdata.get("heading_tolerance", 5.0)),
                    max_speed=float(pdata.get("max_speed", 55.6)),
                )
                registry.register(profile)
            except (TypeError, ValueError) as exc:
                logger.warning("CSIA: skipping invalid vehicle profile %r: %s", label, exc)

        return registry

    # -----------------------------------------------------------------------
    # Public API – UNCHANGED (V1 compatibility)
    # -----------------------------------------------------------------------

    def check(self, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyse a window of decoded CAM messages for coordinated behaviour.

        Parameters
        ----------
        messages:
            List of decoded message dicts.  Non-dict entries are silently
            skipped.  Order is irrelevant.

        Returns
        -------
        dict
            Structured payload with the following five keys:

            ``trust``
                Final fused confidence probability ∈ [0.0, 1.0].
                1.0 → benign; 0.0 → highly suspicious.
                Formula: 0.55×kinematic + 0.20×semantic + 0.25×timing.
            ``entropy``
                Normalised Shannon inter-arrival entropy ∈ [0.0, 1.0].
                Raw output of the Temporal Entropy Engine (Stage 3).
            ``cluster_score``
                Kinematic trust from Stage 2a ∈ [0.0, 1.0].  Computed via
                Robust Scaling and Mahalanobis / Weighted Euclidean distance.
            ``replay_probability``
                Likelihood of machine-synchronised transmission replay
                ∈ [0.0, 1.0].  Defined as ``1.0 − timing_trust``.
            ``identity_consistency``
                Structural CAM metadata metric ∈ [0.0, 1.0].  Measures
                sender diversity based on station_type / station_id patterns
                (Stage 2b Hamming engine).

        Notes
        -----
        * Returns the benign default payload when the window is smaller than
          ``max(min_cluster_size, 2)`` or when no valid cluster forms.
          Benign defaults: trust=1.0, cluster_score=1.0,
          identity_consistency=1.0, entropy=0.0, replay_probability=0.0.
        * The payload of the *most suspicious* cluster (lowest ``trust``) is
          returned when multiple clusters exist.
        """
        effective_min = max(self._min_cluster_size, 2)
        if len(messages) < effective_min:
            logger.debug(
                "CSIA: window %d < min %d → benign payload", len(messages), effective_min
            )
            return dict(_BENIGN_PAYLOAD)

        # Strip non-dict entries
        valid: List[dict] = [m for m in messages if isinstance(m, dict) and m]
        if len(valid) < 2:
            return dict(_BENIGN_PAYLOAD)

        # ── Stage 1: Spatio-temporal clustering ───────────────────────────
        clusters = _build_clusters(
            valid,
            self._spatial_radius_m,
            self._window_size_ns,
            self._lat_field,
            self._lon_field,
            self._ts_field,
        )

        large = [c for c in clusters if len(c) >= max(self._min_cluster_size, 2)]
        if not large:
            logger.debug("CSIA: no cluster ≥ min_cluster_size → benign payload")
            return dict(_BENIGN_PAYLOAD)

        payloads = [self._analyse_cluster(c) for c in large]
        result   = min(payloads, key=lambda p: p["trust"])  # most suspicious wins
        logger.debug(
            "CSIA: cluster_trusts=%s → final=%.4f",
            [p["trust"] for p in payloads], result["trust"],
        )
        return result

    # -----------------------------------------------------------------------
    # Public API – V2 extension (opt-in explainability)
    # -----------------------------------------------------------------------

    def check_extended(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], ExplainabilityReport]:
        """Run full analysis and return both the standard payload and an explanation.

        Parameters
        ----------
        messages:
            Same as ``check()``.

        Returns
        -------
        payload : dict
            The standard 5-key dict (identical to ``check()`` output).
        report : ExplainabilityReport
            Structured explainability report with trust breakdown.

        Notes
        -----
        * ``payload`` is always the same as what ``check(messages)`` would
          return; existing callers can ignore the second return value.
        * This method is additive and does **not** affect ``check()``.
        """
        effective_min = max(self._min_cluster_size, 2)

        if len(messages) < effective_min:
            payload = dict(_BENIGN_PAYLOAD)
            report = self._make_benign_report(payload, cluster_size=0)
            return payload, report

        valid: List[dict] = [m for m in messages if isinstance(m, dict) and m]
        if len(valid) < 2:
            payload = dict(_BENIGN_PAYLOAD)
            report = self._make_benign_report(payload, cluster_size=0)
            return payload, report

        clusters = _build_clusters(
            valid,
            self._spatial_radius_m,
            self._window_size_ns,
            self._lat_field,
            self._lon_field,
            self._ts_field,
        )
        large = [c for c in clusters if len(c) >= max(self._min_cluster_size, 2)]
        if not large:
            payload = dict(_BENIGN_PAYLOAD)
            report = self._make_benign_report(payload, cluster_size=0)
            return payload, report

        # Analyse all clusters; pick most suspicious
        all_results = []
        for cluster in large:
            p, r = self._analyse_cluster_extended(cluster)
            all_results.append((p, r))

        payload, report = min(all_results, key=lambda x: x[0]["trust"])
        return payload, report

    def register_plugin(self, plugin: Any) -> None:
        """Register a custom analysis plugin into the B2 engine.

        Future analysis engines can be added without modifying core CSIA
        code.  The plugin must implement the ``AnalysisPlugin`` protocol
        (``name: str``, ``weight: float``, ``analyse(cluster, config) → float``).

        Parameters
        ----------
        plugin:
            Object implementing the ``AnalysisPlugin`` protocol.
        """
        self._registry.register(plugin)
        logger.info("CSIA: registered plugin %r (weight=%.3f)", plugin.name, plugin.weight)

    # -----------------------------------------------------------------------
    # Private – per-cluster analysis (standard path)
    # -----------------------------------------------------------------------

    def _analyse_cluster(self, cluster: List[dict]) -> Dict[str, float]:
        """Run Stages 2a/2b/3/4 on one spatio-temporal cluster.

        Returns
        -------
        dict
            Structured payload with keys: trust, entropy, cluster_score,
            replay_probability, identity_consistency.
        """

        # ── Stage 2a: Kinematic engine ────────────────────────────────────
        kin_trust = self._kinematic_trust(cluster)

        # ── Stage 2b: Semantic engine ─────────────────────────────────────
        sem_trust = _semantic_trust(cluster, self._semantic_fields)

        # ── Stage 3: Temporal entropy ─────────────────────────────────────
        tim_trust, entropy_score = _temporal_entropy_detail(
            cluster, self._ts_field, self._entropy_bins, self._window_size_ns,
        )

        # ── Stage 4: Score fusion ─────────────────────────────────────────
        combined = (
            self._w_kin * kin_trust
            + self._w_sem * sem_trust
            + self._w_tim * tim_trust
        )
        trust = float(min(1.0, max(0.0, combined)))

        logger.debug(
            "CSIA cluster n=%d: kin=%.4f sem=%.4f tim=%.4f entropy=%.4f → %.4f",
            len(cluster), kin_trust, sem_trust, tim_trust, entropy_score, trust,
        )

        return {
            "trust":                trust,
            "entropy":              float(entropy_score),
            "cluster_score":        float(kin_trust),
            "replay_probability":   float(min(1.0, max(0.0, 1.0 - tim_trust))),
            "identity_consistency": float(sem_trust),
        }

    # -----------------------------------------------------------------------
    # Private – per-cluster extended analysis (V2 explainability path)
    # -----------------------------------------------------------------------

    def _analyse_cluster_extended(
        self, cluster: List[dict]
    ) -> Tuple[Dict[str, float], ExplainabilityReport]:
        """Run full analysis and build the ExplainabilityReport."""

        # Run via the plugin registry (uses the same underlying engines)
        fused, raw_scores, contributions = self._registry.run_all(cluster)

        # Extract individual component scores for the standard payload
        kin_trust = raw_scores.get("kinematic", 1.0)
        sem_trust = raw_scores.get("semantic", 1.0)

        # For entropy we need the raw detail value
        tim_trust, entropy_score = _temporal_entropy_detail(
            cluster, self._ts_field, self._entropy_bins, self._window_size_ns,
        )

        # Recompute fused score using the standard weights for payload consistency
        combined = (
            self._w_kin * kin_trust
            + self._w_sem * sem_trust
            + self._w_tim * tim_trust
        )
        trust = float(min(1.0, max(0.0, combined)))

        payload = {
            "trust":                trust,
            "entropy":              float(entropy_score),
            "cluster_score":        float(kin_trust),
            "replay_probability":   float(min(1.0, max(0.0, 1.0 - tim_trust))),
            "identity_consistency": float(sem_trust),
        }

        # ── Build explainability report ────────────────────────────────────
        profile = self._profile_registry.dominant_profile(cluster)
        anomaly_reasons: List[str] = []

        if kin_trust < 0.3:
            anomaly_reasons.append(
                f"Kinematic clone detected (kin_trust={kin_trust:.3f}): "
                "cluster speed/heading/yaw vectors are nearly identical"
            )
        if sem_trust < 0.2:
            anomaly_reasons.append(
                f"Sybil identity detected (sem_trust={sem_trust:.3f}): "
                "multiple messages share the same station_id"
            )
        if tim_trust < 0.2:
            anomaly_reasons.append(
                f"Machine-burst timing detected (tim_trust={tim_trust:.3f}): "
                "timestamps are tightly synchronised"
            )

        if trust >= 0.7:
            decision = "Benign: sufficient kinematic, semantic, and temporal diversity"
        elif trust >= 0.3:
            decision = f"Suspicious: partial anomaly signal (trust={trust:.3f})"
        else:
            decision = (
                f"High anomaly confidence: coordinated behaviour detected "
                f"(trust={trust:.3f}, reasons={len(anomaly_reasons)})"
            )

        # Statistical stability: agreement between the three sub-scores
        scores_list = [kin_trust, sem_trust, tim_trust]
        score_variance = sum((s - trust) ** 2 for s in scores_list) / len(scores_list)
        stability = float(max(0.0, 1.0 - score_variance))

        # Confidence: function of cluster size
        n = len(cluster)
        confidence = float(n / (n + 5.0))  # 5 msgs → 0.5, 10 msgs → 0.67, 20 msgs → 0.8

        report = ExplainabilityReport(
            trust_score=trust,
            confidence=confidence,
            statistical_stability=stability,
            contributing_factors=dict(contributions),
            anomaly_reasons=anomaly_reasons,
            decision_summary=decision,
            cluster_size=n,
            vehicle_profile_label=profile.label,
            raw_scores=dict(raw_scores),
        )

        return payload, report

    def _make_benign_report(
        self, payload: Dict[str, float], cluster_size: int
    ) -> ExplainabilityReport:
        """Build a benign ``ExplainabilityReport`` for the early-exit path."""
        return ExplainabilityReport(
            trust_score=1.0,
            confidence=0.0,  # no data → no confidence
            statistical_stability=1.0,
            contributing_factors={},
            anomaly_reasons=[],
            decision_summary="Benign: insufficient cluster size for analysis",
            cluster_size=cluster_size,
            vehicle_profile_label="unknown",
            raw_scores={},
        )

    # -----------------------------------------------------------------------
    # Private – trust evolution helpers
    # -----------------------------------------------------------------------

    def _get_trust_history(self, station_id: int) -> TrustHistory:
        """Return or create the ``TrustHistory`` for *station_id*.

        Parameters
        ----------
        station_id:
            ITS station identifier.
        """
        with self._trust_lock:
            if station_id not in self._trust_histories:
                self._trust_histories[station_id] = TrustHistory(
                    station_id=station_id,
                    window=self._trust_history_window,
                    decay_alpha=self._trust_decay_alpha,
                    recovery_beta=self._trust_recovery_beta,
                )
            return self._trust_histories[station_id]

    # -----------------------------------------------------------------------
    # Private – kinematic trust (Stage 2a)
    # -----------------------------------------------------------------------

    def _kinematic_trust(self, cluster: List[dict]) -> float:
        """Extract kinematic vectors, robust-scale, distance → trust."""
        if not self._kinematic_fields:
            return 1.0

        raw_vecs: List[List[float]] = []
        for msg in cluster:
            if not isinstance(msg, dict):
                continue
            raw_vecs.append([_nested_get(msg, f, 0.0) for f in self._kinematic_fields])

        if len(raw_vecs) < 2:
            return 1.0

        matrix = np.array(raw_vecs, dtype=np.float64)

        # Guard against NaN/inf in the matrix (can arise from malformed messages)
        if not np.all(np.isfinite(matrix)):
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Robust scaling
        scaled, _ = _robust_scale(matrix, self._kinematic_fields, self._fallback_ranges)

        # Adaptive threshold: use median speed of the raw (unscaled) cluster
        speed_col_idx   = 0  # speed is the first kinematic field
        median_speed    = float(np.median(matrix[:, speed_col_idx]))
        if median_speed >= self._highway_spd_thr:
            threshold = self._highway_kin_thr
        else:
            threshold = self._city_kin_thr
        cap = threshold * self._cap_multiplier

        # Pairwise distance (returns cached S_inv if available)
        avg_dist, method, _ = _avg_pairwise_dist(scaled, self._mahal_min)

        logger.debug(
            "CSIA kinematic: n=%d method=%s avg_dist=%.4f threshold=%.3f cap=%.3f",
            len(raw_vecs), method, avg_dist, threshold, cap,
        )

        return _dist_to_trust(avg_dist, threshold, cap)
