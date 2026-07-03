"""
b1_scsv/models.py
=================
Strongly-typed value objects and enums for the B1 SCSV layer (V2).

These types are used internally by the ``SCSV`` class and its new
stateful extensions.  They are also exported so that callers and test
code can pattern-match on failure reasons without comparing magic strings.

Design notes
------------
* ``CamMessage`` is *not* frozen because the parser may need to populate
  fields incrementally before handing the object to downstream validators.
* ``ValidationResult`` is frozen (immutable) because it is the final
  verdict returned to callers; post-creation mutation would be a bug.
* ``VehicleState`` is mutable because the ``VehicleStateManager`` updates
  it in place on every new observation.
* All dataclasses use ``__slots__=True`` for memory efficiency in
  long-running pipelines handling millions of messages.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto, unique
from typing import Any, Deque, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


@unique
class ValidationFailureReason(Enum):
    """Machine-readable reason codes for B1 validation failures.

    These codes are attached to ``ValidationResult`` objects so that
    downstream components can react programmatically to specific failure
    modes without parsing free-text messages.
    """

    #: A message with an identical (station_id, message_id, timestamp)
    #: triple was already processed within the replay-cache TTL window.
    REPLAY = auto()

    #: The message timestamp lies outside the configured freshness window
    #: (``|now − msg_timestamp| > freshness_tolerance_ms``).
    STALE_TIMESTAMP = auto()

    #: One or more kinematic fields exceed physically plausible bounds
    #: (speed, acceleration, jerk, heading change, yaw rate, coordinates).
    IMPOSSIBLE_KINEMATICS = auto()

    #: The station changed its certificate identifier more times than the
    #: configured maximum within the certificate-rotation tracking window.
    CERT_ROTATION_ANOMALY = auto()

    #: The ``SCSV`` rule-table returned ``SCORE_BLOCK`` for the
    #: (station_type, message_type) combination.
    BLOCKED_BY_POLICY = auto()

    #: The input could not be parsed into a valid ``CamMessage``
    #: (malformed JSON, wrong types, NaN, infinity, missing required fields).
    PARSE_ERROR = auto()

    #: Latitude or longitude fields are outside the ETSI-defined valid range.
    INVALID_COORDINATES = auto()


# ---------------------------------------------------------------------------
# CamMessage – parsed, validated representation of a raw CAM dict
# ---------------------------------------------------------------------------


@dataclass
class CamMessage:
    """Parsed and sanitised representation of a decoded CAM message.

    Produced by ``safe_parse_cam()``.  All numeric fields have been
    validated to be finite and within ETSI-defined ranges.  Fields that
    could not be parsed are set to ``None`` so downstream validators can
    distinguish "legitimately zero" from "missing".

    Parameters
    ----------
    raw:
        The original unmodified message dict (retained for forensic logging).
    station_id:
        ITS station identifier.
    message_id:
        Message type code (e.g. 1 = CAM, 2 = DENM).
    station_type:
        Station type code (e.g. 5 = passengerCar, 15 = roadSideUnit).
    timestamp:
        ``generation_delta_time`` field.  ``None`` if absent or non-numeric.
    latitude:
        Reference position latitude in ETSI 1e-7 degree units.
    longitude:
        Reference position longitude in ETSI 1e-7 degree units.
    speed:
        Speed in 0.01 m/s units (ETSI range 0–16 383).
    heading:
        Heading in 0.1° units (ETSI range 0–3 600).
    yaw_rate:
        Yaw rate in 0.01 °/s units (ETSI range −32 766 … +32 766).
    longitudinal_acceleration:
        Longitudinal acceleration in 0.01 m/s² units (±2 000).
    lateral_acceleration:
        Lateral acceleration in 0.01 m/s² units (±2 000).
    steering_wheel_angle:
        Steering-wheel angle in 0.1° units (±1 023).
    certificate_id:
        Opaque certificate identifier string (e.g. from a SPAT/signed PDU).
        ``None`` when not present in the decoded message.
    parse_warnings:
        List of non-fatal issues encountered during parsing (e.g. a field
        that defaulted to ``None`` because its value was NaN).
    """

    raw: Dict[str, Any]
    station_id: Optional[int] = None
    message_id: Optional[int] = None
    station_type: Optional[int] = None
    timestamp: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    yaw_rate: Optional[float] = None
    longitudinal_acceleration: Optional[float] = None
    lateral_acceleration: Optional[float] = None
    steering_wheel_angle: Optional[float] = None
    certificate_id: Optional[str] = None
    parse_warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ValidationResult – immutable verdict returned to callers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """Immutable result of a B1 stateful validation check.

    Returned by ``SCSV.check_stateful()``.  All fields are populated
    regardless of whether the message passed or failed.

    Parameters
    ----------
    valid:
        ``True`` if the message passed all checks; ``False`` otherwise.
    score:
        The SCSV rule-table score ∈ [0.0, 1.0].  ``0.0`` when the message
        was rejected before reaching the rule table.
    reason:
        Primary failure reason, or ``None`` if the message is valid.
    details:
        Supplementary key-value pairs for logging and forensics.  Typical
        keys include ``"station_id"``, ``"timestamp"``, ``"reject_stage"``,
        ``"cert_id"``, ``"kinematic_violation"``.
    wall_time:
        Unix timestamp (float) at the moment the validation completed.
        Useful for latency profiling.
    """

    valid: bool
    score: float
    reason: Optional[ValidationFailureReason]
    details: Dict[str, Any]
    wall_time: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# VehicleState – mutable rolling state maintained per station_id
# ---------------------------------------------------------------------------


@dataclass
class VehicleState:
    """Rolling behavioural state for a single ITS station.

    Maintained by ``VehicleStateManager``.  All history queues have a
    bounded ``maxlen`` so memory usage is O(window_size) per vehicle.

    Parameters
    ----------
    station_id:
        The ITS station identifier this record belongs to.
    window:
        Maximum number of observations retained in each history deque.
    first_seen:
        Unix wall-clock time when the first message from this station
        was processed.
    last_seen:
        Unix wall-clock time of the most recently processed message.
    latest_timestamp:
        The ``generation_delta_time`` of the most recent message.
    timestamps:
        Deque of recent ``generation_delta_time`` values (newest last).
    positions:
        Deque of recent ``(latitude, longitude)`` pairs in ETSI 1e-7°.
    headings:
        Deque of recent heading values (0.1° units).
    speeds:
        Deque of recent speed values (0.01 m/s units).
    accelerations:
        Deque of recent longitudinal acceleration values (0.01 m/s²).
    cert_ids:
        Deque of recent certificate identifier strings.
    cert_change_times:
        Deque of Unix timestamps when certificate changes were detected.
    message_count:
        Total number of messages processed for this station.
    trust_score:
        Running trust estimate ∈ [0.0, 1.0] for this station.
        Updated via exponential smoothing each time ``check_stateful``
        processes a message from this station.
    last_validation_result:
        The ``ValidationResult`` produced by the most recent check.
    """

    station_id: int
    window: int = 50

    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    latest_timestamp: Optional[float] = None
    message_count: int = 0
    trust_score: float = 1.0
    last_validation_result: Optional[ValidationResult] = None

    timestamps: Deque[float] = field(init=False)
    positions: Deque[Tuple[float, float]] = field(init=False)
    headings: Deque[float] = field(init=False)
    speeds: Deque[float] = field(init=False)
    accelerations: Deque[float] = field(init=False)
    cert_ids: Deque[str] = field(init=False)
    cert_change_times: Deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.timestamps = deque(maxlen=self.window)
        self.positions = deque(maxlen=self.window)
        self.headings = deque(maxlen=self.window)
        self.speeds = deque(maxlen=self.window)
        self.accelerations = deque(maxlen=self.window)
        self.cert_ids = deque(maxlen=self.window)
        self.cert_change_times = deque(maxlen=self.window)

    def record_observation(self, msg: CamMessage, wall_time: float) -> None:
        """Append fields from *msg* to the rolling history deques.

        Parameters
        ----------
        msg:
            Parsed CAM message to record.
        wall_time:
            Current Unix wall-clock time.
        """
        self.last_seen = wall_time
        self.message_count += 1

        if msg.timestamp is not None:
            self.latest_timestamp = msg.timestamp
            self.timestamps.append(msg.timestamp)

        if msg.latitude is not None and msg.longitude is not None:
            self.positions.append((msg.latitude, msg.longitude))

        if msg.heading is not None:
            self.headings.append(msg.heading)

        if msg.speed is not None:
            self.speeds.append(msg.speed)

        if msg.longitudinal_acceleration is not None:
            self.accelerations.append(msg.longitudinal_acceleration)

        if msg.certificate_id is not None:
            prev = self.cert_ids[-1] if self.cert_ids else None
            if prev is None or msg.certificate_id != prev:
                self.cert_change_times.append(wall_time)
            self.cert_ids.append(msg.certificate_id)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any, field_name: str, warnings: List[str]) -> Optional[float]:
    """Try to convert *value* to a finite float; append to *warnings* on failure.

    Parameters
    ----------
    value:
        Raw value from the decoded message.
    field_name:
        Human-readable name used in warning messages.
    warnings:
        Mutable list; any parse issue is appended here.

    Returns
    -------
    float | None
        The converted value, or ``None`` if conversion failed or the result
        is not finite (NaN / ±∞).
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        warnings.append(f"{field_name}: cannot convert {type(value).__name__!r} to float")
        return None
    if not math.isfinite(f):
        warnings.append(f"{field_name}: non-finite value ({f!r})")
        return None
    return f


def _safe_int(value: Any, field_name: str, warnings: List[str]) -> Optional[int]:
    """Try to convert *value* to an int; append to *warnings* on failure.

    Parameters
    ----------
    value:
        Raw value from the decoded message.
    field_name:
        Human-readable name used in warning messages.
    warnings:
        Mutable list; any parse issue is appended here.

    Returns
    -------
    int | None
        The converted value, or ``None`` if conversion failed.
    """
    if value is None:
        return None
    # Accept numeric-only strings (e.g. some decoders emit "5" for passengerCar)
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    # Float-like values (e.g. 5.0) are accepted if they have no fractional part
    try:
        f = float(value)
        if math.isfinite(f) and f == int(f):
            return int(f)
    except (TypeError, ValueError):
        pass
    warnings.append(f"{field_name}: cannot convert {type(value).__name__!r} to int")
    return None


def safe_parse_cam(raw: Any) -> Tuple[Optional[CamMessage], Optional[str]]:
    """Parse a raw message dict into a ``CamMessage``.

    Handles all forms of malformed input gracefully:

    * Non-dict input → returns ``(None, error_reason)``
    * Missing nested keys → fields default to ``None``
    * NaN / infinity → fields set to ``None``, warning recorded
    * Wrong types → fields set to ``None``, warning recorded
    * Valid input → returns ``(CamMessage(...), None)``

    Parameters
    ----------
    raw:
        Raw message from the pipeline.  Typically a ``dict`` but may be
        anything (``None``, ``str``, ``list``, …) due to upstream bugs.

    Returns
    -------
    (cam_message, error_reason)
        ``cam_message`` is a ``CamMessage`` on success, ``None`` on hard
        failure (non-dict input).
        ``error_reason`` is a human-readable string on hard failure, ``None``
        on success (soft warnings are inside ``CamMessage.parse_warnings``).
    """
    if not isinstance(raw, dict):
        return None, f"expected dict, got {type(raw).__name__}"

    warnings: List[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    header = raw.get("header") or {}
    if not isinstance(header, dict):
        header = {}
        warnings.append("header: not a dict")

    station_id = _safe_int(header.get("station_id"), "header.station_id", warnings)
    message_id = _safe_int(header.get("message_id"), "header.message_id", warnings)

    # ── CAM root ──────────────────────────────────────────────────────────
    cam = raw.get("cam") or {}
    if not isinstance(cam, dict):
        cam = {}
        warnings.append("cam: not a dict")

    timestamp = _safe_float(cam.get("generation_delta_time"), "cam.generation_delta_time", warnings)

    # ── cam_parameters ────────────────────────────────────────────────────
    cp = cam.get("cam_parameters") or {}
    if not isinstance(cp, dict):
        cp = {}
        warnings.append("cam.cam_parameters: not a dict")

    bc = cp.get("basic_container") or {}
    if not isinstance(bc, dict):
        bc = {}

    station_type = _safe_int(bc.get("station_type"), "station_type", warnings)

    rp = bc.get("reference_position") or {}
    if not isinstance(rp, dict):
        rp = {}

    latitude = _safe_float(rp.get("latitude"), "latitude", warnings)
    longitude = _safe_float(rp.get("longitude"), "longitude", warnings)

    # ── High-frequency container ───────────────────────────────────────────
    hfc = cp.get("high_frequency_container") or {}
    if not isinstance(hfc, dict):
        hfc = {}

    bvhf = hfc.get("basic_vehicle_container_high_frequency") or {}
    if not isinstance(bvhf, dict):
        bvhf = {}

    speed = _safe_float(bvhf.get("speed"), "speed", warnings)
    heading = _safe_float(bvhf.get("heading"), "heading", warnings)
    yaw_rate = _safe_float(bvhf.get("yaw_rate"), "yaw_rate", warnings)
    lon_acc = _safe_float(bvhf.get("longitudinal_acceleration"), "longitudinal_acceleration", warnings)
    lat_acc = _safe_float(bvhf.get("lateral_acceleration"), "lateral_acceleration", warnings)
    steering = _safe_float(bvhf.get("steering_wheel_angle"), "steering_wheel_angle", warnings)

    # ── Certificate identifier (optional, provider-specific) ─────────────
    cert_id: Optional[str] = None
    raw_cert = raw.get("certificate_id") or raw.get("cert_id")
    if raw_cert is not None:
        try:
            cert_id = str(raw_cert).strip() or None
        except Exception:
            pass

    return CamMessage(
        raw=raw,
        station_id=station_id,
        message_id=message_id,
        station_type=station_type,
        timestamp=timestamp,
        latitude=latitude,
        longitude=longitude,
        speed=speed,
        heading=heading,
        yaw_rate=yaw_rate,
        longitudinal_acceleration=lon_acc,
        lateral_acceleration=lat_acc,
        steering_wheel_angle=steering,
        certificate_id=cert_id,
        parse_warnings=warnings,
    ), None


__all__ = [
    "ValidationFailureReason",
    "CamMessage",
    "ValidationResult",
    "VehicleState",
    "safe_parse_cam",
]
