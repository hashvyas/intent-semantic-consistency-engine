"""
b1_scsv/scsv.py
===============
B1 – Sender Certificate Semantic Validation (SCSV)
Part of the ISCE STB V2X Security Pipeline.

Architectural Separation of Responsibilities
---------------------------------------------
* **SCSV (Validation Layer)**
  - *Question*: Is this message technically valid enough to continue through the pipeline?
  - *Focus*: Handles objective physical sanity checks on single messages (GPS coordinate ranges,
    invalid heading encoding, negative timestamps, impossible absolute speed, and impossible
    absolute acceleration) as well as protocol validity checks (certificate anomalies, replay, and freshness).
  - *Examples*: Coordinates valid, Timestamp fresh, Certificate valid, Replay detected.
* **Misbehavior Detection (MBD Layer)**
  - *Question*: Is this sender behaving maliciously despite sending a technically valid message?
  - *Focus*: Analyzes vehicle motion context and trajectory/speed anomalies relative to physics profiles,
    abnormal motion, Sybil behavior, or spoofing. SCSV delegates these behavioral checks to MBD.
  - *Examples*: Abnormal acceleration, Speed anomaly, Possible replay pattern.
* **CSIA (Trust Reasoning Layer)**
  - *Question*: Given the outputs of SCSV and MBD together with cooperative observations, how trustworthy is this sender?
  - *Focus*: Reasons about cooperative observations and propagates trust over the vehicle observability graph.
  - *Examples*: Neighbour corroboration, Belief, Disbelief, Uncertainty, Trust, Confidence.

Purpose
-------
Validate that the *kind* of message being sent is semantically consistent
with the *type of station* (vehicle, pedestrian, RSU, etc.) that claims to
be sending it.  This is a lightweight, rule-based check that can catch:

  * Impersonation: a passenger car sending RSU-exclusive messages.
  * Misconfiguration or spoofed headers: an "unknown" station type
    broadcasting anything at all.
  * Message-type mismatch: a VRU device originating infrastructure messages.

V2 extensions (backward-compatible additions)
---------------------------------------------
The original ``check(station_type, message_type) → float`` API is
**unchanged**.  V2 adds:

  * ``check_stateful(message)`` – full stateful validation including replay
    detection, timestamp freshness, certificate continuity, and physical
    plausibility.  Accepts a raw dict or a parsed ``CamMessage``.
  * ``VehicleStateManager`` – private, internal state engine tracking
    rolling per-vehicle history (positions, speeds, cert IDs, …).
  * ``_ReplayCache`` – bounded TTL-based cache that rejects duplicate frames.
  * ``PhysicalPlausibilityValidator`` – checks kinematic sanity before
    forwarding to the rule table.

These extensions are fully invisible to existing callers.  Existing code
that calls only ``check()`` is completely unaffected.

Data interface
--------------
Messages arrive pre-decoded as nested Python dicts (typically from JSON).
Callers are expected to extract:

    station_type  ← cam.cam_parameters.basic_container.station_type
    message_type  ← header.message_id  (resolved to string, e.g. "CAM")

and call ``SCSV.check(station_type, message_type)``.

Return value
------------
``SCSV.check()`` returns a float score:

    1.0  → message passes (allow)
    0.0  → message is blocked
    intermediate values are supported for future confidence-weighted rules

References
----------
* ETSI EN 302 637-2  – Cooperative Awareness Message (CAM)
* ETSI EN 302 637-3  – Decentralised Environmental Notification (DENM)
* ETSI TS 103 900    – ITS Station Types
* ETSI TS 102 894-2  – Common Data Dictionary
"""

from __future__ import annotations

import logging
import math
import os
import pathlib
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

from b1_scsv.config import ConfigurationError, validate_b1_config
from b1_scsv.models import (
    CamMessage,
    ValidationFailureReason,
    ValidationResult,
    ValidationAssessment,
    VehicleState,
    safe_parse_cam,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel score values – callers may compare against these constants.
# ---------------------------------------------------------------------------
SCORE_ALLOW: float = 1.0
SCORE_BLOCK: float = 0.0

# Wildcard token used in rule definitions
_WILDCARD = "*"

# Path to the shared pipeline config, relative to this file's package root.
_DEFAULT_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "isce_config.yaml"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _policy_to_score(policy: str) -> float:
    """Convert the string policy token ('allow' / 'block') to a numeric score.

    Parameters
    ----------
    policy:
        A string, expected to be ``"allow"`` or ``"block"``.

    Returns
    -------
    float
        ``SCORE_ALLOW`` for "allow", ``SCORE_BLOCK`` for anything else.
    """
    return SCORE_ALLOW if str(policy).strip().lower() == "allow" else SCORE_BLOCK


# ===========================================================================
# _ReplayCache – bounded TTL-based replay detection cache
# ===========================================================================

class _ReplayCache:
    """Thread-safe bounded cache for replay attack detection.

    Stores ``(station_id, message_id, timestamp)`` triples with an
    associated expiry wall-clock time.  Entries are evicted lazily on
    each ``check()`` call so that the cache never grows unboundedly.

    Parameters
    ----------
    ttl_s:
        Time-to-live in seconds for each cache entry.  A value of 0
        disables replay detection (all checks return ``False``).
    max_size:
        Maximum number of entries retained before forced eviction of
        the oldest entries.
    """

    def __init__(self, ttl_s: float, max_size: int = 10_000) -> None:
        self._ttl = float(ttl_s)
        self._max_size = int(max_size)
        self._lock = threading.Lock()
        # {key: expiry_wall_time}
        self._cache: Dict[Tuple, float] = {}

    def is_replay(
        self,
        station_id: Optional[int],
        message_id: Optional[int],
        timestamp: Optional[float],
    ) -> bool:
        """Check whether this (station_id, message_id, timestamp) triple is a replay.

        Also registers the triple in the cache if it is not a replay.

        Parameters
        ----------
        station_id:
            ITS station identifier.  If ``None``, replay detection is skipped.
        message_id:
            Message type identifier.
        timestamp:
            Message timestamp value.  If ``None``, replay detection is skipped.

        Returns
        -------
        bool
            ``True`` if this triple was already seen within the TTL window.
        """
        if self._ttl <= 0.0 or station_id is None or timestamp is None:
            return False

        key = (station_id, message_id, timestamp)
        now = time.monotonic()
        expiry = now + self._ttl

        with self._lock:
            # Lazy eviction of expired entries
            self._evict_expired(now)

            if key in self._cache:
                # Entry still alive → replay detected
                return True

            # Forced eviction when cache is full
            if len(self._cache) >= self._max_size:
                self._evict_oldest()

            self._cache[key] = expiry
            return False

    def _evict_expired(self, now: float) -> None:
        """Remove all entries whose expiry time has passed."""
        expired = [k for k, exp in self._cache.items() if exp <= now]
        for k in expired:
            del self._cache[k]

    def _evict_oldest(self) -> None:
        """Remove the entry with the smallest expiry time."""
        if not self._cache:
            return
        oldest = min(self._cache, key=lambda k: self._cache[k])
        del self._cache[oldest]

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        with self._lock:
            return len(self._cache)


# ===========================================================================
# PhysicalPlausibilityValidator
# ===========================================================================

class PhysicalPlausibilityValidator:
    """Validate that a parsed CAM message is physically plausible.

    Checks speed, acceleration, yaw rate, heading, and GPS coordinate
    bounds against configurable thresholds.  All thresholds use ETSI
    native units (0.01 m/s for speed, 0.1° for heading, etc.) to avoid
    unit-conversion bugs.

    Parameters
    ----------
    max_speed:
        Maximum plausible speed in ETSI 0.01 m/s units.
    max_acceleration:
        Maximum plausible |acceleration| in ETSI 0.01 m/s² units.
    max_jerk:
        Maximum plausible |jerk| per frame delta (same units).
    max_heading_change:
        Maximum plausible heading change between consecutive messages
        (ETSI 0.1° units).
    max_yaw_rate:
        Maximum plausible |yaw rate| (ETSI 0.01 °/s units).
    lat_min, lat_max:
        Valid latitude range in ETSI 1e-7 degree units.
    lon_min, lon_max:
        Valid longitude range in ETSI 1e-7 degree units.
    """

    def __init__(
        self,
        max_speed: float = 8330.0,
        max_acceleration: float = 1500.0,
        max_jerk: float = 3000.0,
        max_heading_change: float = 900.0,
        max_yaw_rate: float = 7500.0,
        lat_min: float = -900_000_000.0,
        lat_max: float = 900_000_000.0,
        lon_min: float = -1_800_000_000.0,
        lon_max: float = 1_800_000_000.0,
    ) -> None:
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)
        self.max_jerk = float(max_jerk)
        self.max_heading_change = float(max_heading_change)
        self.max_yaw_rate = float(max_yaw_rate)
        self.lat_min = float(lat_min)
        self.lat_max = float(lat_max)
        self.lon_min = float(lon_min)
        self.lon_max = float(lon_max)

    def validate(
        self,
        msg: CamMessage,
        prev_state: Optional[VehicleState] = None,
    ) -> Optional[str]:
        """Check *msg* for physical plausibility.

        This method performs objective physical sanity validation on a message,
        ignoring behavioral features (jerk, heading change, yaw rate, lateral acceleration)
        which are delegated to the Misbehavior Detection (MBD) layer.

        Parameters
        ----------
        msg:
            The parsed CAM message to validate.
        prev_state:
            Optional prior ``VehicleState`` for this station. Kept for backward compatibility
            but ignored to avoid dependency on historical behavior in SCSV.

        Returns
        -------
        str | None
            A human-readable violation description, or ``None`` if the
            message is plausible.
        """
        # ── GPS coordinates ───────────────────────────────────────────────
        if msg.latitude is not None:
            if not (self.lat_min <= msg.latitude <= self.lat_max):
                return (
                    f"latitude {msg.latitude} out of valid range "
                    f"[{self.lat_min}, {self.lat_max}]"
                )
        if msg.longitude is not None:
            if not (self.lon_min <= msg.longitude <= self.lon_max):
                return (
                    f"longitude {msg.longitude} out of valid range "
                    f"[{self.lon_min}, {self.lon_max}]"
                )

        # ── Heading encoding ──────────────────────────────────────────────
        if msg.heading is not None:
            if not (0.0 <= msg.heading <= 3600.0):
                return (
                    f"heading {msg.heading} out of valid range [0, 3600] "
                    f"(ETSI 0.1° units)"
                )

        # ── Speed ─────────────────────────────────────────────────────────
        if msg.speed is not None:
            if msg.speed < 0.0 or msg.speed > self.max_speed:
                return (
                    f"speed {msg.speed} outside valid range [0, {self.max_speed}] "
                    f"(ETSI 0.01 m/s units)"
                )

        # ── Longitudinal Acceleration ─────────────────────────────────────
        if msg.longitudinal_acceleration is not None:
            if abs(msg.longitudinal_acceleration) > self.max_acceleration:
                return (
                    f"longitudinal_acceleration |{msg.longitudinal_acceleration}| exceeds max plausible {self.max_acceleration} "
                    f"(ETSI 0.01 m/s² units)"
                )

        return None  # all checks passed


# ===========================================================================
# _VehicleStateManager (private)
# ===========================================================================

class _VehicleStateManager:
    """Internal rolling state engine for per-vehicle behaviour tracking.

    Maintains a ``{station_id: VehicleState}`` registry.  All mutation
    is protected by a per-station lock to support concurrent pipelines.

    Parameters
    ----------
    window:
        Number of observations to retain in each history deque (per vehicle).
    max_vehicles:
        Maximum number of vehicle records to retain simultaneously.
        Oldest-seen entries are evicted when this limit is reached.
    cert_rotation_window_s:
        Time window (seconds) over which certificate changes are counted.
    cert_max_rotations:
        Maximum allowed certificate changes within the window.
    """

    def __init__(
        self,
        window: int = 50,
        max_vehicles: int = 10_000,
        cert_rotation_window_s: float = 60.0,
        cert_max_rotations: int = 3,
    ) -> None:
        self._window = int(window)
        self._max_vehicles = int(max_vehicles)
        self._cert_window = float(cert_rotation_window_s)
        self._cert_max = int(cert_max_rotations)
        self._states: Dict[int, VehicleState] = {}
        self._lock = threading.Lock()

    def get_or_create(self, station_id: int) -> VehicleState:
        """Return the ``VehicleState`` for *station_id*, creating one if absent.

        Parameters
        ----------
        station_id:
            ITS station identifier.

        Returns
        -------
        VehicleState
            Mutable state record for this vehicle.
        """
        with self._lock:
            if station_id not in self._states:
                if len(self._states) >= self._max_vehicles:
                    self._evict_oldest()
                self._states[station_id] = VehicleState(
                    station_id=station_id,
                    window=self._window,
                )
            return self._states[station_id]

    def _evict_oldest(self) -> None:
        """Evict the vehicle state that was seen least recently."""
        if not self._states:
            return
        oldest_sid = min(self._states, key=lambda s: self._states[s].last_seen)
        del self._states[oldest_sid]
        logger.debug("VehicleStateManager: evicted state for station_id=%d", oldest_sid)

    def check_cert_rotation(self, state: VehicleState) -> bool:
        """Return ``True`` if the station has rotated certificates too frequently.

        Counts how many certificate changes occurred within
        ``cert_rotation_window_s`` of the current wall clock.

        Parameters
        ----------
        state:
            The vehicle's current state record.

        Returns
        -------
        bool
            ``True`` when excessive rotation is detected.
        """
        if not state.cert_change_times:
            return False
        now = time.time()
        cutoff = now - self._cert_window
        recent_rotations = sum(1 for t in state.cert_change_times if t >= cutoff)
        return recent_rotations > self._cert_max

    def record(self, msg: CamMessage, wall_time: float) -> None:
        """Update the state for *msg.station_id* with the new observation.

        Parameters
        ----------
        msg:
            Parsed and validated CAM message.
        wall_time:
            Current Unix wall-clock time.
        """
        if msg.station_id is None:
            return
        state = self.get_or_create(msg.station_id)
        state.record_observation(msg, wall_time)

    @property
    def vehicle_count(self) -> int:
        """Number of tracked vehicles currently in the registry."""
        with self._lock:
            return len(self._states)


# ===========================================================================
# Main class
# ===========================================================================

class SCSV:
    """Sender Certificate Semantic Validator.

    Loads a YAML rule table on construction and exposes a single
    ``check(station_type, message_type)`` method that returns a score
    indicating whether the combination is acceptable.

    V2 also exposes ``check_stateful(message)`` for full stateful
    validation.  Existing callers using only ``check()`` are unaffected.

    Parameters
    ----------
    config_path:
        Absolute or relative path to ``isce_config.yaml``.  Defaults to the
        file sitting one level above this package directory.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    yaml.YAMLError
        If the configuration file cannot be parsed.
    b1_scsv.config.ConfigurationError
        If the configuration fails validation.
    """

    def __init__(self, config_path: Optional[str | os.PathLike] = None) -> None:
        config_path = pathlib.Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

        if not config_path.exists():
            raise FileNotFoundError(
                f"SCSV: configuration file not found: {config_path}"
            )

        with config_path.open("r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh)

        # ── Validate configuration at startup (fail-fast) ──────────────────
        try:
            validate_b1_config(raw)
        except ConfigurationError as exc:
            raise ConfigurationError(
                f"SCSV configuration validation failed: {exc}"
            ) from exc

        b1_cfg: Dict[str, Any] = raw.get("b1_scsv", {})

        # Default policy when no rule matches
        default_raw: str = b1_cfg.get("default_policy", "allow")
        self._default_score: float = _policy_to_score(default_raw)

        # Known station types (name → integer code)
        self._station_types: Dict[str, int] = raw.get("station_types", {})
        # Known message types (name → integer code)
        self._message_types: Dict[str, int] = raw.get("message_types", {})

        # Rule list: each entry is a dict with keys
        #   station_type, message_type, action, score (optional), note (optional)
        self._rules: List[Dict[str, Any]] = b1_cfg.get("rules", [])

        # ── V2: Replay cache ──────────────────────────────────────────────
        replay_ttl = float(raw.get("b1_replay_cache_ttl_s", 30))
        self._replay_cache = _ReplayCache(ttl_s=replay_ttl)

        # ── V2: Timestamp freshness ───────────────────────────────────────
        self._freshness_ms: float = float(raw.get("b1_timestamp_freshness_ms", 5000))

        # ── V2: Certificate rotation ──────────────────────────────────────
        cert_window = float(raw.get("b1_cert_rotation_window_s", 60.0))
        cert_max = int(raw.get("b1_cert_max_rotations", 3))

        # ── V2: Physical plausibility ─────────────────────────────────────
        plaus_cfg: Dict[str, Any] = raw.get("b1_plausibility", {})
        self._plausibility = PhysicalPlausibilityValidator(
            max_speed=float(plaus_cfg.get("max_speed_etsi", 8330)),
            max_acceleration=float(plaus_cfg.get("max_acceleration_etsi", 1500)),
            max_jerk=float(plaus_cfg.get("max_jerk_etsi", 3000)),
            max_heading_change=float(plaus_cfg.get("max_heading_change_etsi", 900)),
            max_yaw_rate=float(plaus_cfg.get("max_yaw_rate_etsi", 7500)),
            lat_min=float(plaus_cfg.get("lat_min", -900_000_000)),
            lat_max=float(plaus_cfg.get("lat_max", 900_000_000)),
            lon_min=float(plaus_cfg.get("lon_min", -1_800_000_000)),
            lon_max=float(plaus_cfg.get("lon_max", 1_800_000_000)),
        )

        # ── V2: Vehicle state manager ─────────────────────────────────────
        self._state_manager = _VehicleStateManager(
            cert_rotation_window_s=cert_window,
            cert_max_rotations=cert_max,
        )

        # ── V2 / B3 Transition: Validation Parameters ────────────────────
        val_cfg = raw.get("validation", {})
        self._fatal_checks = val_cfg.get("fatal", {
            "malformed_json": True,
            "invalid_schema": True,
            "parser_failure": True
        })
        self._penalties = val_cfg.get("penalties", {
            "replay": 0.30,
            "stale_timestamp": 0.20,
            "certificate_rotation": 0.15,
            "physics": 0.25
        })
        self._min_val_score = float(val_cfg.get("minimum_validation_score", 0.40))

        logger.info(
            "SCSV v2 loaded: %d rules, default_policy=%s, "
            "replay_ttl=%.0fs, freshness_ms=%.0f, "
            "cert_window=%.0fs, cert_max=%d",
            len(self._rules),
            "allow" if self._default_score == SCORE_ALLOW else "block",
            replay_ttl,
            self._freshness_ms,
            cert_window,
            cert_max,
        )

    # ------------------------------------------------------------------
    # Public API – UNCHANGED (V1 compatibility)
    # ------------------------------------------------------------------

    def check(self, station_type: Any, message_type: Any) -> float:
        """Evaluate whether a (station_type, message_type) combination is valid.

        This is the primary entry point for B1.  Call it once per decoded
        message after extracting the station_type from the basic container
        and the message_type string from the header.

        Parameters
        ----------
        station_type:
            The station type string as decoded from the message, e.g.
            ``"passengerCar"``, ``"roadSideUnit"``.  May also be the raw
            integer station_type value; will be resolved to its string name
            if recognised.
        message_type:
            The message type string, e.g. ``"CAM"``, ``"DENM"``.  May also
            be the raw integer message_id; will be resolved if recognised.

        Returns
        -------
        float
            ``1.0`` (SCORE_ALLOW) if the combination is explicitly allowed or
            falls through to an "allow" default.
            ``0.0`` (SCORE_BLOCK) if the combination is explicitly blocked or
            falls through to a "block" default.
            Intermediate values are possible when a rule specifies a custom
            ``score`` field.

        Notes
        -----
        * Unknown or malformed inputs are handled gracefully: the method
          never raises; it falls back to ``default_policy``.
        * No text scanning or regex is used anywhere in this method.
        """
        # -- Normalise inputs to canonical string names --------------------
        st = self._resolve_station_type(station_type)
        mt = self._resolve_message_type(message_type)

        logger.debug("SCSV.check: station_type=%r → %r, message_type=%r → %r", station_type, st, message_type, mt)

        # -- Walk rule list top-to-bottom; first match wins ----------------
        for rule in self._rules:
            rule_st: str = str(rule.get("station_type", _WILDCARD))
            rule_mt: str = str(rule.get("message_type", _WILDCARD))

            st_match = rule_st == _WILDCARD or rule_st == st
            mt_match = rule_mt == _WILDCARD or rule_mt == mt

            if st_match and mt_match:
                # Rule matched – return the rule's score (or derive from action)
                score = self._rule_score(rule)
                logger.debug(
                    "SCSV rule matched: station_type=%r message_type=%r → score=%.2f (note: %s)",
                    rule_st,
                    rule_mt,
                    score,
                    rule.get("note", ""),
                )
                return score

        # No rule matched – apply default policy
        logger.debug(
            "SCSV no rule matched for station_type=%r message_type=%r → default score=%.2f",
            st,
            mt,
            self._default_score,
        )
        return self._default_score

    # ------------------------------------------------------------------
    # Public API – V2 extension (opt-in stateful validation)
    # ------------------------------------------------------------------

    def check_stateful(self, message: Any) -> ValidationAssessment:
        """Full stateful validation of a decoded CAM message.

        Refactored to transform B1 into a confidence-aware validation layer.
        Fatal protocol and parsing errors terminate processing (fatal=True,
        B2 bypassed). Recoverable validation anomalies deduct configurable
        penalties from the validation score, returning a complete
        ValidationAssessment for subsequent B2 evaluation.
        """
        wall_time = time.time()

        try:
            return self._check_stateful_impl(message, wall_time)
        except Exception as exc:
            logger.warning("SCSV.check_stateful: unexpected error: %s", exc, exc_info=True)
            return ValidationAssessment(
                fatal=True,
                validation_score=SCORE_BLOCK,
                confidence=1.0,
                reasons=[str(exc)],
                checks={
                    "structure": False,
                    "replay": False,
                    "timestamp": False,
                    "certificate": False,
                    "physics": False
                },
                details={"error": str(exc), "minimum_validation_score": self._min_val_score, "reason": ValidationFailureReason.PARSE_ERROR},
                wall_time=wall_time,
            )

    def _check_stateful_impl(self, message: Any, wall_time: float) -> ValidationAssessment:
        """Implementation of ``check_stateful`` (separated for testability)."""

        # ── Step 1: Parse & Structural Fatal Checks ────────────────────────
        if isinstance(message, CamMessage):
            cam = message
            parse_error = None
        else:
            cam, parse_error = safe_parse_cam(message)

        fatal = False
        reasons = []
        checks = {
            "structure": True,
            "replay": True,
            "timestamp": True,
            "certificate": True,
            "physics": True
        }
        primary_reason = None

        if cam is None:
            fatal = True
            reasons.append(parse_error or "unknown parse failure")
            checks["structure"] = False
            primary_reason = ValidationFailureReason.PARSE_ERROR
        else:
            # Check mandatory fields for stateful verification
            missing_fields = []
            if cam.station_id is None: missing_fields.append("station_id")
            if cam.timestamp is None: missing_fields.append("timestamp")
            if cam.latitude is None: missing_fields.append("latitude")
            if cam.longitude is None: missing_fields.append("longitude")
            if cam.station_type is None: missing_fields.append("station_type")

            if missing_fields:
                fatal = True
                checks["structure"] = False
                reasons.append(f"Missing mandatory fields: {', '.join(missing_fields)}")
                primary_reason = ValidationFailureReason.PARSE_ERROR

            # If invalid_numeric_values is configured as fatal
            if self._fatal_checks.get("invalid_numeric_values", False):
                if cam.parse_warnings:
                    fatal = True
                    checks["structure"] = False
                    reasons.extend(cam.parse_warnings)
                    primary_reason = ValidationFailureReason.PARSE_ERROR
                elif cam.timestamp is not None and cam.timestamp < 0:
                    fatal = True
                    checks["timestamp"] = False
                    reasons.append("Negative timestamp")
                    primary_reason = ValidationFailureReason.STALE_TIMESTAMP

        if fatal:
            return ValidationAssessment(
                fatal=True,
                validation_score=SCORE_BLOCK,
                confidence=1.0,
                reasons=reasons,
                checks=checks,
                details={
                    "error": reasons[0] if reasons else "Parser failure",
                    "minimum_validation_score": self._min_val_score,
                    "reason": primary_reason or ValidationFailureReason.PARSE_ERROR
                },
                wall_time=wall_time,
            )

        if cam.parse_warnings:
            logger.debug(
                "SCSV.check_stateful: parse warnings for station_id=%s: %s",
                cam.station_id,
                cam.parse_warnings,
            )

        base_details: Dict[str, Any] = {
            "station_id": cam.station_id,
            "message_id": cam.message_id,
            "timestamp": cam.timestamp,
            "parse_warnings": cam.parse_warnings,
            "minimum_validation_score": self._min_val_score,
        }

        validation_score = 1.0
        primary_reason = None

        # ── Step 2: Replay detection (Recoverable) ─────────────────────────
        is_replay = self._replay_cache.is_replay(cam.station_id, cam.message_id, cam.timestamp)
        if is_replay:
            checks["replay"] = False
            validation_score -= self._penalties.get("replay", 0.30)
            reasons.append("Replay detected")
            if primary_reason is None:
                primary_reason = ValidationFailureReason.REPLAY

        # ── Step 3: Timestamp freshness (Recoverable) ──────────────────────
        now_ms = wall_time * 1000.0
        age_ms = 0.0
        is_stale_ts = False
        if cam.timestamp is not None:
            if cam.timestamp < 0:
                checks["timestamp"] = False
                validation_score -= self._penalties.get("stale_timestamp", 0.20)
                reasons.append("Negative timestamp")
                if primary_reason is None:
                    primary_reason = ValidationFailureReason.STALE_TIMESTAMP
                base_details["age_ms"] = 0.0
                base_details["freshness_ms"] = self._freshness_ms
                is_stale_ts = True
            elif self._freshness_ms > 0:
                if cam.timestamp > 1_000_000:  # likely an absolute ms-epoch value
                    age_ms = abs(now_ms - cam.timestamp)
                    if age_ms > self._freshness_ms:
                        checks["timestamp"] = False
                        validation_score -= self._penalties.get("stale_timestamp", 0.20)
                        reasons.append(f"Timestamp stale (age {age_ms:.1f} ms)")
                        if primary_reason is None:
                            primary_reason = ValidationFailureReason.STALE_TIMESTAMP
                        base_details["age_ms"] = age_ms
                        base_details["freshness_ms"] = self._freshness_ms
                        is_stale_ts = True

        # ── Step 4: Certificate rotation check (Recoverable) ───────────────
        state = self._state_manager.get_or_create(cam.station_id)
        if cam.certificate_id is not None:
            prev_cert = state.cert_ids[-1] if state.cert_ids else None
            if prev_cert is not None and cam.certificate_id != prev_cert:
                state.cert_change_times.append(wall_time)
            state.cert_ids.append(cam.certificate_id)

        is_cert_anomaly = self._state_manager.check_cert_rotation(state)
        if is_cert_anomaly:
            checks["certificate"] = False
            validation_score -= self._penalties.get("certificate_rotation", 0.15)
            reasons.append("Suspicious certificate rotation")
            if primary_reason is None:
                primary_reason = ValidationFailureReason.CERT_ROTATION_ANOMALY
            base_details["cert_id"] = cam.certificate_id
            base_details["cert_change_count"] = len(state.cert_change_times)

        # ── Step 5: Physical plausibility (Recoverable / Fatal) ────────────
        plausibility_violation = self._plausibility.validate(cam, prev_state=state)
        is_fatal_phys = False
        phys_reason = None
        
        if plausibility_violation:
            if "latitude" in plausibility_violation or "longitude" in plausibility_violation:
                if self._fatal_checks.get("invalid_coordinates", False):
                    is_fatal_phys = True
                    phys_reason = ValidationFailureReason.INVALID_COORDINATES
            elif "speed" in plausibility_violation:
                if self._fatal_checks.get("impossible_speed", False):
                    is_fatal_phys = True
                    phys_reason = ValidationFailureReason.IMPOSSIBLE_KINEMATICS
            elif "longitudinal_acceleration" in plausibility_violation:
                if self._fatal_checks.get("impossible_acceleration", False):
                    is_fatal_phys = True
                    phys_reason = ValidationFailureReason.IMPOSSIBLE_KINEMATICS
            elif "heading" in plausibility_violation:
                if self._fatal_checks.get("invalid_heading", False):
                    is_fatal_phys = True
                    phys_reason = ValidationFailureReason.INVALID_HEADING

        # ── Step 6: Rule-table check (Recoverable) ─────────────────────────
        rule_score = self.check(cam.station_type, cam.message_id)
        is_policy_blocked = rule_score == SCORE_BLOCK

        # ── Calculate Validation Confidence (Evidence-Based Model) ──────────
        # 1. Structural Completeness
        missing_count = sum(1 for field_name in ["speed", "heading", "longitudinal_acceleration", "yaw_rate"] if getattr(cam, field_name, None) is None)
        c_struct = max(0.0, 1.0 - 0.1 * missing_count)
        
        # 2. Historical Evidence (Continuous Logarithmic Growth Model)
        max_history = 50
        c_hist = (
            0.30
            + 0.70
            * min(
                math.log1p(state.message_count) / math.log1p(max_history),
                1.0
            )
        )
            
        # 3. Certificate Stability
        if is_cert_anomaly:
            c_cert = 0.60
        elif not state.cert_ids:
            c_cert = 0.80
        else:
            rotations = len(state.cert_change_times)
            if rotations == 0:
                c_cert = 1.00
            elif rotations == 1:
                c_cert = 0.95
            elif rotations == 2:
                c_cert = 0.80
            else:
                c_cert = 0.60
                
        # 4. Replay Certainty
        c_replay = 1.00 # exact match is a definitive observation
        
        # 5. Timestamp Reliability
        if is_stale_ts:
            c_time = 0.85
        elif cam.timestamp is None or cam.timestamp < 0:
            c_time = 0.50
        else:
            if cam.timestamp < 65536: # relative delta timestamp
                c_time = 1.00
            else:
                c_time = 1.00 if age_ms <= 1000.0 else 0.85
                
        # Combine using weights
        confidence = (0.15 * c_struct +
                      0.40 * c_hist +
                      0.15 * c_replay +
                      0.15 * c_time +
                      0.15 * c_cert)
        confidence = max(0.0, min(1.0, confidence))
        
        # If it's a fatal physical sanity check or replay detection, confidence is 1.00
        if is_fatal_phys or is_replay:
            confidence = 1.00
            
        # Formulate contributors
        contributors = []
        if c_struct == 1.0:
            contributors.append("✓ Structure Valid")
        else:
            contributors.append("⚠ Missing Optional Kinematics")
            
        if c_cert == 1.0:
            contributors.append("✓ Stable Certificate History")
        elif is_cert_anomaly:
            contributors.append("✗ Certificate Rotation Anomaly Detected")
        else:
            contributors.append(f"⚠ Certificate Rotated {len(state.cert_change_times)} Times")
            
        if state.message_count > 15:
            contributors.append(f"✓ {state.message_count} Previous Observations")
        else:
            contributors.append(f"⚠ Sparse Observation History ({state.message_count} messages)")
            
        if not is_stale_ts and c_time >= 0.99:
            contributors.append("✓ Timestamp Reliable")
        elif is_stale_ts:
            contributors.append("✗ Timestamp Stale/Unreliable")
        else:
            contributors.append("⚠ Borderline Timestamp Age")
            
        if is_replay:
            contributors.append("✗ Replay Detected (Cache Match)")
        else:
            contributors.append("✓ Replay Cache Checked (No Match)")
            
        confidence_breakdown = {
            "Structural Completeness": c_struct,
            "Historical Evidence": c_hist,
            "Certificate Stability": c_cert,
            "Replay Certainty": c_replay,
            "Timestamp Reliability": c_time
        }

        # Add history details to base details
        base_details.update({
            "history_count": state.message_count,
            "historical_confidence": c_hist,
            "max_history": max_history
        })

        if is_fatal_phys:
            checks["physics"] = False
            fatal_details = dict(base_details)
            fatal_details.update({
                "kinematic_violation": plausibility_violation,
                "reason": phys_reason,
            })
            return ValidationAssessment(
                fatal=True,
                validation_score=SCORE_BLOCK,
                confidence=confidence,
                reasons=[plausibility_violation],
                checks=checks,
                details=fatal_details,
                wall_time=wall_time,
                confidence_breakdown=confidence_breakdown,
                confidence_contributors=contributors
            )
        else:
            if plausibility_violation:
                checks["physics"] = False
                validation_score -= self._penalties.get("physics", 0.25)
                reasons.append(f"Borderline physics anomaly: {plausibility_violation}")
                if primary_reason is None:
                    if "latitude" in plausibility_violation or "longitude" in plausibility_violation:
                        primary_reason = ValidationFailureReason.INVALID_COORDINATES
                    elif "heading" in plausibility_violation:
                        primary_reason = ValidationFailureReason.INVALID_HEADING
                    else:
                        primary_reason = ValidationFailureReason.IMPOSSIBLE_KINEMATICS
                base_details["kinematic_violation"] = plausibility_violation

        # Rule-table check block
        if is_policy_blocked:
            validation_score = 0.0
            reasons.append("Blocked by rule table policy")
            if primary_reason is None:
                primary_reason = ValidationFailureReason.BLOCKED_BY_POLICY
            base_details["station_type"] = cam.station_type

        # Ensure validation score is bounded [0.0, 1.0]
        validation_score = max(0.0, min(1.0, validation_score))

        if primary_reason is not None:
            base_details["reason"] = primary_reason

        # ── Record observation state ──────────────────────────────────────
        self._state_manager.record(cam, wall_time)

        # Cache the assessment on the state
        assessment = ValidationAssessment(
            fatal=False,
            validation_score=validation_score,
            confidence=confidence,
            reasons=reasons,
            checks=checks,
            details=base_details,
            wall_time=wall_time,
            confidence_breakdown=confidence_breakdown,
            confidence_contributors=contributors
        )
        state.last_validation_result = assessment

        return assessment

    # ------------------------------------------------------------------
    # Introspection helpers (useful for dashboards / B2 hand-off)
    # ------------------------------------------------------------------

    @property
    def default_score(self) -> float:
        """The numeric score that applies when no rule matches."""
        return self._default_score

    @property
    def known_station_types(self) -> List[str]:
        """Return the list of station type names defined in the config."""
        return list(self._station_types.keys())

    @property
    def known_message_types(self) -> List[str]:
        """Return the list of message type names defined in the config."""
        return list(self._message_types.keys())

    @property
    def tracked_vehicle_count(self) -> int:
        """Number of vehicle state records currently held in memory."""
        return self._state_manager.vehicle_count

    @property
    def replay_cache_size(self) -> int:
        """Number of entries currently in the replay cache."""
        return self._replay_cache.size

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_station_type(self, raw: Any) -> str:
        """Normalise a raw station_type value to its canonical string name.

        Resolution order
        ----------------
        1. ``int``        – reverse-lookup by integer in the station_types enum.
        2. digit string   – a string whose stripped form is all digits is
                           converted to ``int`` first, then reverse-looked-up.
                           This handles decoders that serialise enums as
                           ``"5"`` instead of ``5`` or ``"passengerCar"``.
        3. string         – returned as-is (stripped) for direct name matching.
        4. anything else  – converted via ``str()``; if the result is empty or
                           does not match any rule the default policy applies.

        Malformed / unrecognised values are returned as an empty string so
        that no rule can spuriously match them; the default policy will apply.

        Parameters
        ----------
        raw:
            Raw value from the decoded message field.

        Returns
        -------
        str
            Canonical station type name, or ``""`` if unresolvable.
        """
        if raw is None:
            return ""
        # (1) True integer → reverse-lookup
        if isinstance(raw, int):
            reverse = {v: k for k, v in self._station_types.items()}
            return reverse.get(raw, "")
        # (2) Digit-only string → convert to int and reverse-lookup
        try:
            s = str(raw).strip()
        except Exception:  # pragma: no cover – truly bizarre input
            return ""
        if s.isdigit():
            reverse = {v: k for k, v in self._station_types.items()}
            return reverse.get(int(s), "")
        # (3) Treat as a literal name string
        return s

    def _resolve_message_type(self, raw: Any) -> str:
        """Normalise a raw message_type value to its canonical string name.

        Resolution order
        ----------------
        1. ``int``        – reverse-lookup by integer in the message_types enum.
        2. digit string   – a string whose stripped form is all digits is
                           converted to ``int`` first, then reverse-looked-up.
                           This handles decoders that emit ``"1"`` instead of
                           ``1`` or ``"CAM"``.
        3. string         – returned as-is (stripped) for direct name matching.
        4. anything else  – converted via ``str()``; unresolvable values fall
                           through to the default policy.

        Parameters
        ----------
        raw:
            Raw value from the decoded header field.

        Returns
        -------
        str
            Canonical message type name, or ``""`` if unresolvable.
        """
        if raw is None:
            return ""
        # (1) True integer → reverse-lookup
        if isinstance(raw, int):
            reverse = {v: k for k, v in self._message_types.items()}
            return reverse.get(raw, "")
        # (2) Digit-only string → convert to int and reverse-lookup
        try:
            s = str(raw).strip()
        except Exception:  # pragma: no cover
            return ""
        if s.isdigit():
            reverse = {v: k for k, v in self._message_types.items()}
            return reverse.get(int(s), "")
        # (3) Treat as a literal name string
        return s

    @staticmethod
    def _rule_score(rule: Dict[str, Any]) -> float:
        """Extract the numeric score from a rule dict.

        If the rule has an explicit ``score`` key, that value is used
        (clamped to [0, 1]).  Otherwise the ``action`` key is converted
        via ``_policy_to_score``.

        Parameters
        ----------
        rule:
            A rule dict from the YAML configuration.

        Returns
        -------
        float
            Score in [0.0, 1.0].
        """
        if "score" in rule:
            try:
                return float(max(0.0, min(1.0, rule["score"])))
            except (TypeError, ValueError):
                pass
        return _policy_to_score(rule.get("action", "block"))
