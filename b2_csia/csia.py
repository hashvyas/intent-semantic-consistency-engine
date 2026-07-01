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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

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
        return float(node)
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
) -> Tuple[float, str]:
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

    Returns
    -------
    avg_dist : float
        Mean pairwise distance (0.0 for identical vectors).
    method : str
        ``"mahalanobis"`` or ``"euclidean"`` (for logging).
    """
    n, d = matrix_scaled.shape
    if n < 2:
        return 0.0, "none"

    method  = "euclidean"
    S_inv   = None

    if n >= mahalanobis_min_samples and d >= 1:
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

    total, count = 0.0, 0
    for i, j in itertools.combinations(range(n), 2):
        diff = matrix_scaled[i] - matrix_scaled[j]
        if S_inv is not None:
            d_sq = float(diff @ S_inv @ diff)
            total += math.sqrt(max(0.0, d_sq))
        else:
            total += math.sqrt(float(np.dot(diff, diff)))
        count += 1

    return (total / count) if count > 0 else 0.0, method


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

def _temporal_entropy_trust(
    messages: List[dict],
    ts_field: str,
    n_bins: int,
    window_size_ns: float,
) -> float:
    """Compute temporal trust from inter-arrival time distribution.

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
    float
        Temporal trust ∈ [0.0, 1.0].
        0.0 → perfectly synchronised machine burst.
        1.0 → natural, high-entropy inter-arrival distribution.
    """
    timestamps: List[float] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        ts = _nested_get(m, ts_field, float("nan"))
        if not math.isnan(ts):
            timestamps.append(ts)

    if len(timestamps) < 2:
        return 1.0  # Insufficient data → treat as benign

    timestamps.sort()
    time_spread = timestamps[-1] - timestamps[0]

    # ── Spread fraction ────────────────────────────────────────────────────
    window = window_size_ns if window_size_ns > 0.0 else 1.0
    spread_score = float(min(1.0, time_spread / window))

    if time_spread == 0.0:
        return 0.0  # Machine burst: perfect synchronisation

    # ── Inter-arrival entropy ──────────────────────────────────────────────
    deltas = [timestamps[k + 1] - timestamps[k] for k in range(len(timestamps) - 1)]

    if not deltas:
        return spread_score  # Only one unique interval

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

    return float(min(1.0, max(0.0, 0.6 * spread_score + 0.4 * entropy_score)))


# ===========================================================================
# CSIA class
# ===========================================================================

class CSIA:
    """Cluster Semantic Invariance Analyser – Research Grade v2.

    Loads configuration from ``isce_config.yaml`` on construction.  Exposes
    a single ``check(messages)`` method that returns a continuous trust
    probability score for a window of decoded ITS CAM messages.

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

        logger.info(
            "CSIA v2 loaded: min_cluster=%d, spatial_r=%.0fm, window_ns=%.0f, "
            "kin_fields=%d, sem_fields=%d, mahal_min=%d, "
            "w=[%.2f, %.2f, %.2f]",
            self._min_cluster_size, self._spatial_radius_m, self._window_size_ns,
            len(self._kinematic_fields), len(self._semantic_fields), self._mahal_min,
            self._w_kin, self._w_sem, self._w_tim,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def check(self, messages: List[Dict[str, Any]]) -> float:
        """Analyse a window of decoded CAM messages for coordinated behaviour.

        Parameters
        ----------
        messages:
            List of decoded message dicts.  Non-dict entries are silently
            skipped.  Order is irrelevant.

        Returns
        -------
        float
            Trust score ∈ [0.0, 1.0].
            1.0 → benign; messages appear kinematically independent.
            0.0 → highly suspicious; coordinated / Sybil behaviour detected.

        Notes
        -----
        * Returns 1.0 (insufficient data) when the window is smaller than
          ``max(min_cluster_size, 2)``.
        * If spatio-temporal clustering yields no cluster ≥ min_cluster_size,
          returns 1.0.
        * The score of the *most suspicious* cluster is returned when multiple
          clusters exist.
        """
        effective_min = max(self._min_cluster_size, 2)
        if len(messages) < effective_min:
            logger.debug("CSIA: window %d < min %d → 1.0", len(messages), effective_min)
            return 1.0

        # Strip non-dict entries
        valid: List[dict] = [m for m in messages if isinstance(m, dict) and m]
        if len(valid) < 2:
            return 1.0

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
            logger.debug("CSIA: no cluster ≥ min_cluster_size → 1.0")
            return 1.0

        scores = [self._analyse_cluster(c) for c in large]
        final  = float(min(scores))  # most suspicious cluster wins
        logger.debug("CSIA: cluster_scores=%s → final=%.4f", scores, final)
        return final

    # -----------------------------------------------------------------------
    # Private – per-cluster analysis
    # -----------------------------------------------------------------------

    def _analyse_cluster(self, cluster: List[dict]) -> float:
        """Run Stages 2a/2b/3/4 on one spatio-temporal cluster."""

        # ── Stage 2a: Kinematic engine ────────────────────────────────────
        kin_trust = self._kinematic_trust(cluster)

        # ── Stage 2b: Semantic engine ─────────────────────────────────────
        sem_trust = _semantic_trust(cluster, self._semantic_fields)

        # ── Stage 3: Temporal entropy ─────────────────────────────────────
        tim_trust = _temporal_entropy_trust(
            cluster, self._ts_field, self._entropy_bins, self._window_size_ns,
        )

        # ── Stage 4: Score fusion ─────────────────────────────────────────
        combined = (
            self._w_kin * kin_trust
            + self._w_sem * sem_trust
            + self._w_tim * tim_trust
        )
        result = float(min(1.0, max(0.0, combined)))

        logger.debug(
            "CSIA cluster n=%d: kin=%.4f sem=%.4f tim=%.4f → %.4f",
            len(cluster), kin_trust, sem_trust, tim_trust, result,
        )
        return result

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

        # Pairwise distance
        avg_dist, method = _avg_pairwise_dist(scaled, self._mahal_min)

        logger.debug(
            "CSIA kinematic: n=%d method=%s avg_dist=%.4f threshold=%.3f cap=%.3f",
            len(raw_vecs), method, avg_dist, threshold, cap,
        )

        return _dist_to_trust(avg_dist, threshold, cap)
