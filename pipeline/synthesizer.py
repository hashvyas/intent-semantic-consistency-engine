"""
pipeline/synthesizer.py
========================
Translates a cooperative V2X message cluster into a deterministic, natural-language
scene description for B3 semantic reasoning.

Contract
--------
* The generated text contains ONLY objective, pre-B2-reasoning evidence.
* No B2-derived value (trust, belief, disbelief, uncertainty, cluster_score,
  entropy, replay_probability, identity_consistency, confidence, or any variable
  whose value depends on B2 computation) may appear anywhere in the output text.
* The ``b2_result`` parameter is accepted for API stability and forward
  compatibility (A3, B4, B6 integration) but is intentionally never read inside
  this function.
* The synthesizer describes; it never infers, counts agreements, or derives
  conclusions. Contradictions between sources emerge naturally from the
  individual observations listed. B3 is responsible for all semantic inference.
* Identical structured input always produces identical output (deterministic).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nested_get(obj: Any, dotted_key: str, default: Any = None) -> Any:
    """Traverse a nested dict using a dot-separated key path.

    Parameters
    ----------
    obj:
        Root object to traverse.
    dotted_key:
        Dot-separated key path, e.g. ``"cam.cam_parameters.basic_container"``.
    default:
        Value returned when any intermediate key is absent or non-dict.

    Returns
    -------
    Any
        The value found at the key path, or ``default``.
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
    lat1_e7: float,
    lon1_e7: float,
    lat2_e7: float,
    lon2_e7: float,
) -> float:
    """Compute the Haversine great-circle distance in metres between two
    positions expressed as ETSI integer-scaled (×10⁻⁷ degree) coordinates.

    Parameters
    ----------
    lat1_e7, lon1_e7:
        Reference position (integer-scaled degrees × 10⁷).
    lat2_e7, lon2_e7:
        Comparison position (integer-scaled degrees × 10⁷).

    Returns
    -------
    float
        Distance in metres.
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


#: ETSI ITS station_type integer → readable name.
_STATION_TYPE_NAMES: Dict[int, str] = {
    0:  "unknown",
    1:  "pedestrian",
    2:  "cyclist",
    3:  "moped",
    4:  "motorcycle",
    5:  "passengerCar",
    6:  "bus",
    7:  "lightTruck",
    8:  "heavyTruck",
    9:  "trailer",
    10: "specialVehicle",
    11: "tram",
    12: "lightVruVehicle",
    13: "animal",
    14: "agricultural",
    15: "roadSideUnit",
}


def _station_type_name(station_type: Optional[int]) -> str:
    """Return the human-readable station type label for an ETSI station_type code.

    Parameters
    ----------
    station_type:
        ETSI ITS-S station_type integer, or ``None``.

    Returns
    -------
    str
        Readable label, e.g. ``"passengerCar"``, or ``"unknown"`` when the
        code is absent or unrecognised.
    """
    if station_type is None:
        return "unknown"
    return _STATION_TYPE_NAMES.get(station_type, f"unknown ({station_type})")


def _fmt_optional(value: Any, unit: str = "") -> str:
    """Format an optional scalar for display.

    Parameters
    ----------
    value:
        The value to format.  ``None`` renders as ``"N/A"``.
    unit:
        Optional unit string appended after the value (e.g. ``" m/s"``).

    Returns
    -------
    str
        Formatted string.
    """
    if value is None:
        return "N/A"
    return f"{value}{unit}"


def _extract_cam_telemetry(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all objective CAM telemetry fields from a single message dict.

    Only raw CAM fields are read.  No B2 fields are accessed.

    Parameters
    ----------
    msg:
        A single V2X message dictionary.

    Returns
    -------
    dict
        Flat dictionary of extracted telemetry values.  Missing fields are
        represented as ``None``.
    """
    station_id = (
        _nested_get(msg, "header.station_id")
        or msg.get("station_id")
    )
    station_type = (
        _nested_get(msg, "cam.cam_parameters.basic_container.station_type")
        or msg.get("station_type")
    )
    lat = (
        _nested_get(msg, "cam.cam_parameters.basic_container.reference_position.latitude")
        or msg.get("latitude")
    )
    lon = (
        _nested_get(msg, "cam.cam_parameters.basic_container.reference_position.longitude")
        or msg.get("longitude")
    )
    hfc = _nested_get(
        msg,
        "cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency",
    ) or {}
    speed       = hfc.get("speed")       or msg.get("speed")
    heading     = hfc.get("heading")     or msg.get("heading")
    yaw_rate    = hfc.get("yaw_rate")
    long_accel  = hfc.get("longitudinal_acceleration")
    gen_dt      = _nested_get(msg, "cam.generation_delta_time")

    return {
        "station_id":   station_id,
        "station_type": station_type,
        "lat":          lat,
        "lon":          lon,
        "speed":        speed,
        "heading":      heading,
        "yaw_rate":     yaw_rate,
        "long_accel":   long_accel,
        "gen_dt":       gen_dt,
    }


def _serialize_peer_report(report: Any, index: int) -> str:
    """Serialize a single peer report entry into an objective observation sentence.

    The peer report is taken verbatim from the ``scene_context.peer_reports``
    list in the target message.  No inference is performed on the reported values.

    Parameters
    ----------
    report:
        A peer report entry.  May be a dict or a raw string.
    index:
        Zero-based index of this report within the peer_reports list.

    Returns
    -------
    str
        A single natural-language sentence describing the raw report.
    """
    if isinstance(report, str):
        return f"Peer report {index + 1}: {report}"

    if not isinstance(report, dict):
        return f"Peer report {index + 1}: (non-standard format)"

    parts: List[str] = []

    peer_id = report.get("station_id") or report.get("peer_id") or report.get("id")
    if peer_id is not None:
        parts.append(f"Station {peer_id}")

    peer_type = report.get("station_type")
    if peer_type is not None:
        parts.append(f"({_station_type_name(peer_type)})")

    event = report.get("event") or report.get("event_type") or report.get("cause")
    if event is not None:
        parts.append(f"reports: {event}")

    pos = report.get("position") or {}
    p_lat = pos.get("latitude") or report.get("latitude")
    p_lon = pos.get("longitude") or report.get("longitude")
    if p_lat is not None and p_lon is not None:
        parts.append(f"at position (lat={p_lat}, lon={p_lon})")

    spd = report.get("speed")
    if spd is not None:
        parts.append(f"speed={spd}")

    hdg = report.get("heading")
    if hdg is not None:
        parts.append(f"heading={hdg} deg")

    dist = report.get("distance_m")
    if dist is not None:
        parts.append(f"distance={dist:.1f} m")

    if not parts:
        return f"Peer report {index + 1}: (no structured fields)"

    body = " ".join(parts)
    return f"Peer report {index + 1}: {body}."


def _serialize_rsu_message(rsu_msg: Any, index: int) -> str:
    """Serialize a single RSU message entry into an objective observation sentence.

    The RSU message is taken verbatim from ``scene_context.rsu_messages``.
    No inference is performed on the reported values.

    Parameters
    ----------
    rsu_msg:
        A single RSU message entry.  May be a dict or a raw string.
    index:
        Zero-based index of this message within the rsu_messages list.

    Returns
    -------
    str
        A single natural-language sentence describing the raw RSU report.
    """
    if isinstance(rsu_msg, str):
        return f"RSU message {index + 1}: {rsu_msg}"

    if not isinstance(rsu_msg, dict):
        return f"RSU message {index + 1}: (non-standard format)"

    parts: List[str] = []

    rsu_id = rsu_msg.get("station_id") or rsu_msg.get("rsu_id") or rsu_msg.get("id")
    if rsu_id is not None:
        parts.append(f"RSU {rsu_id}")

    event = rsu_msg.get("event") or rsu_msg.get("event_type") or rsu_msg.get("cause") or rsu_msg.get("hazard")
    if event is not None:
        parts.append(f"reports: {event}")

    pos = rsu_msg.get("position") or {}
    r_lat = pos.get("latitude") or rsu_msg.get("latitude")
    r_lon = pos.get("longitude") or rsu_msg.get("longitude")
    if r_lat is not None and r_lon is not None:
        parts.append(f"at position (lat={r_lat}, lon={r_lon})")

    msg_text = rsu_msg.get("message") or rsu_msg.get("text") or rsu_msg.get("advisory")
    if msg_text is not None:
        parts.append(f"advisory: \"{msg_text}\"")

    if not parts:
        return f"RSU message {index + 1}: (no structured fields)"

    body = " ".join(parts)
    return f"RSU message {index + 1}: {body}."


def _serialize_cluster_peer(
    msg: Dict[str, Any],
    target_lat: Optional[float],
    target_lon: Optional[float],
    index: int,
) -> str:
    """Serialize a single non-target cluster member as an objective kinematic observation.

    Parameters
    ----------
    msg:
        The cluster peer message dictionary.
    target_lat, target_lon:
        ETSI integer-scaled coordinates of the target (ego) vehicle, used to
        compute haversine distance.  If either is ``None``, distance is omitted.
    index:
        One-based index of this peer within the cluster peer list.

    Returns
    -------
    str
        A sentence describing the raw kinematic state of this cluster peer.
    """
    tel = _extract_cam_telemetry(msg)
    parts: List[str] = [f"Cluster peer {index}"]

    if tel["station_id"] is not None:
        st_name = _station_type_name(tel["station_type"])
        parts.append(f"(station {tel['station_id']}, type={st_name})")

    if tel["lat"] is not None and tel["lon"] is not None:
        parts.append(f"position=(lat={tel['lat']}, lon={tel['lon']})")
        if target_lat is not None and target_lon is not None:
            dist_m = _haversine_m(target_lat, target_lon, tel["lat"], tel["lon"])
            parts.append(f"distance={dist_m:.1f} m from ego")

    if tel["speed"] is not None:
        parts.append(f"speed={tel['speed']}")

    if tel["heading"] is not None:
        parts.append(f"heading={tel['heading']} deg")

    if tel["yaw_rate"] is not None:
        parts.append(f"yaw_rate={tel['yaw_rate']}")

    if tel["gen_dt"] is not None:
        parts.append(f"timestamp={tel['gen_dt']}")

    return ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_message(
    cluster: List[Dict[str, Any]],
    b2_result: Dict[str, Any],
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """Translate a cooperative V2X message cluster into an objective, deterministic
    natural-language scene description suitable for B3 semantic reasoning.

    The output text contains only observable scene evidence available before
    any trust or semantic reasoning has been applied.  Specifically:

    * Raw CAM fields of the target (ego) vehicle: station identity, position,
      speed, heading, yaw rate, longitudinal acceleration, timestamp.
    * Local sensor readings: camera, radar, lidar observations as reported.
    * Per-peer-report observations from ``scene_context.peer_reports``, each
      listed individually.
    * Per-RSU-message observations from ``scene_context.rsu_messages``, each
      listed individually.
    * Kinematic state and haversine distance of every other vehicle in the
      cooperative cluster.

    The ``b2_result`` parameter is accepted for API stability (A3, B4, B6
    forward compatibility) but is intentionally never read inside this function.
    No B2-derived value — trust, belief, disbelief, uncertainty, cluster_score,
    entropy, replay_probability, identity_consistency, or any derived inference —
    may appear in the output text.

    This function is deterministic: identical ``cluster`` and ``context`` inputs
    always produce identical output text.

    Parameters
    ----------
    cluster:
        Window of V2X message dicts.  ``cluster[-1]`` is the target message
        being evaluated.
    b2_result:
        B2 CSIA result dictionary.  Accepted for API stability; not read.
    context:
        Operational context label (e.g. ``"urban"``, ``"highway"``).
        ``None`` renders as ``"unknown"``.

    Returns
    -------
    dict
        ``{"text": str, "template": str, "sources": list}``
    """
    # b2_result is intentionally unused.  It is accepted only to maintain the
    # stable public interface shared with orchestrator.py and future callers.
    _ = b2_result

    if not cluster:
        return {
            "text": "V2X Scene Report: No cooperative scene information available.",
            "template": "cooperative_scene_report",
            "sources": [],
        }

    target_msg = cluster[-1]
    ctx_name = context or "unknown"

    # ------------------------------------------------------------------
    # 1. Extract ego vehicle (target) CAM telemetry
    # ------------------------------------------------------------------
    tel = _extract_cam_telemetry(target_msg)
    station_id   = tel["station_id"]   if tel["station_id"]   is not None else "unknown"
    st_type_name = _station_type_name(tel["station_type"])
    lat          = tel["lat"]
    lon          = tel["lon"]

    # ------------------------------------------------------------------
    # 2. Extract local sensor observations
    # ------------------------------------------------------------------
    local_perception: Dict[str, Any] = target_msg.get("local_perception") or {}
    camera = local_perception.get("camera", "UNKNOWN")
    radar  = local_perception.get("radar",  "UNKNOWN")
    lidar  = local_perception.get("lidar",  "UNKNOWN")

    # Capture any additional sensor keys beyond the canonical three
    extra_sensor_parts: List[str] = []
    for key, val in local_perception.items():
        if key not in {"camera", "radar", "lidar"}:
            extra_sensor_parts.append(f"{key}={val}")

    # ------------------------------------------------------------------
    # 3. Extract scene context: peer reports and RSU messages
    # ------------------------------------------------------------------
    scene_context: Dict[str, Any] = target_msg.get("scene_context") or {}
    peer_reports: List[Any]  = scene_context.get("peer_reports")  or []
    rsu_messages: List[Any]  = scene_context.get("rsu_messages")  or []

    # ------------------------------------------------------------------
    # 4. Build cluster peer observations (all messages except the target)
    # ------------------------------------------------------------------
    cluster_peers = cluster[:-1]

    # ------------------------------------------------------------------
    # 5. Assemble the scene description
    # ------------------------------------------------------------------
    lines: List[str] = []

    # — Ego vehicle header
    lines.append(
        f"V2X Scene Report: context={ctx_name}. "
        f"Ego vehicle: station {station_id} (type={st_type_name}), "
        f"position=(lat={_fmt_optional(lat)}, lon={_fmt_optional(lon)}), "
        f"speed={_fmt_optional(tel['speed'])}, "
        f"heading={_fmt_optional(tel['heading'])} deg, "
        f"yaw_rate={_fmt_optional(tel['yaw_rate'])}, "
        f"longitudinal_acceleration={_fmt_optional(tel['long_accel'])}, "
        f"timestamp={_fmt_optional(tel['gen_dt'])}."
    )

    # — Local sensor observations
    sensor_line = f"Local sensor observations: camera={camera}, radar={radar}, lidar={lidar}"
    if extra_sensor_parts:
        sensor_line += ", " + ", ".join(extra_sensor_parts)
    sensor_line += "."
    lines.append(sensor_line)

    # — Individual peer report observations
    if peer_reports:
        for i, report in enumerate(peer_reports):
            lines.append(_serialize_peer_report(report, i))
    else:
        lines.append("No peer reports received.")

    # — Individual RSU message observations
    if rsu_messages:
        for i, rsu_msg in enumerate(rsu_messages):
            lines.append(_serialize_rsu_message(rsu_msg, i))
    else:
        lines.append("No RSU messages received.")

    # — Cooperative cluster peer kinematic observations
    if cluster_peers:
        for i, peer_msg in enumerate(cluster_peers):
            lines.append(_serialize_cluster_peer(peer_msg, lat, lon, i + 1))
    else:
        lines.append("No other vehicles in cooperative cluster.")

    text = " ".join(lines)

    sources = [
        "local_perception",
        "peer_reports",
        "rsu_messages",
        "cooperative_observations",
    ]

    return {
        "text": text,
        "template": "cooperative_scene_report",
        "sources": sources,
    }
