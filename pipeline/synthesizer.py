"""
pipeline/synthesizer.py
========================
Translates a cooperative V2X message cluster into a deterministic, natural-language
scene description for B3 semantic reasoning.

B6 Template Support
-------------------
The synthesizer implements three interchangeable rendering templates, all driven
from a single shared evidence extraction pipeline:

* ``TemplateStyle.DEFAULT``    — Compact key=value format (A2-compatible, default).
* ``TemplateStyle.NARRATIVE``  — Flowing prose, different vocabulary.
* ``TemplateStyle.STRUCTURED`` — Sectioned report with bracketed headers.

All three templates describe **exactly the same facts** extracted from the cluster.
They differ only in linguistic presentation. This is required so that the B3 owner
can measure template sensitivity without any change in evidence content.

A2 Contract (preserved)
------------------------
* The generated text contains ONLY objective, pre-B2-reasoning evidence.
* No B2-derived value (trust, belief, disbelief, uncertainty, cluster_score,
  entropy, replay_probability, identity_consistency, confidence, or any variable
  whose value depends on B2 computation) may appear anywhere in the output text.
* The ``b2_result`` parameter is accepted for API stability and forward
  compatibility (A3, B4, B6 integration) but is intentionally never read inside
  this function.
* The synthesizer describes; it never infers, counts agreements, or derives
  conclusions. Contradictions between sources emerge naturally from the individual
  observations listed. B3 is responsible for all semantic inference.
* Identical structured input always produces identical output (deterministic).

Modularity
----------
Evidence extraction runs exactly once per ``synthesize_message()`` call and
populates a ``SceneEvidence`` object. Each renderer is a pure function of that
object. Adding a new template requires implementing only a new renderer function
and registering it in ``_RENDERERS`` — no changes to extraction logic.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Internal helpers (unchanged from A2)
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


# ---------------------------------------------------------------------------
# Template style selection
# ---------------------------------------------------------------------------

class TemplateStyle(enum.Enum):
    """Strongly-typed selector for the synthesizer rendering template.

    Attributes
    ----------
    DEFAULT:
        Compact ``key=value`` format.  Output is byte-for-byte identical to the
        A2 synthesizer.  This is the default when no template is specified.
    NARRATIVE:
        Flowing natural-language prose.  Same facts, different vocabulary and
        sentence structure.
    STRUCTURED:
        Sectioned report with bracketed headers and indented key-value pairs.
        Same facts, different visual layout.

    All styles produce identical factual content from identical inputs and are
    fully deterministic.  They are designed for B6 template sensitivity
    experiments — the only difference is linguistic presentation.
    """
    DEFAULT    = "default"
    NARRATIVE  = "narrative"
    STRUCTURED = "structured"


# ---------------------------------------------------------------------------
# Intermediate evidence representation
# ---------------------------------------------------------------------------

@dataclass
class _PeerReportFact:
    """Pre-parsed facts extracted from a single peer report entry.

    Populated once by ``_parse_peer_report_fact`` and shared by all renderers.
    No renderer may add, remove, or infer any field.

    Attributes
    ----------
    index:
        1-based position of this report in the peer_reports list.
    is_verbatim:
        ``True`` when the raw entry was a plain string rather than a structured
        dict.  Only ``verbatim_text`` is populated in this case.
    is_nonstandard:
        ``True`` when the raw entry was neither a string nor a dict.
    verbatim_text:
        The original string when ``is_verbatim`` is ``True``; ``None`` otherwise.
    station_id:
        Peer station identifier, or ``None`` when absent.
    station_type:
        ETSI station_type integer, or ``None`` when absent.
    station_type_name:
        Pre-computed human-readable label from ``_station_type_name``.
    event:
        Reported event or cause string, or ``None`` when absent.
    position_lat, position_lon:
        Reported position coordinates, or ``None`` when absent.
    speed:
        Reported speed, or ``None`` when absent.
    heading:
        Reported heading, or ``None`` when absent.
    distance_m:
        Peer-reported distance from ego in metres, or ``None`` when absent.
    """
    index:             int
    is_verbatim:       bool
    is_nonstandard:    bool
    verbatim_text:     Optional[str]
    station_id:        Optional[Any]
    station_type:      Optional[int]
    station_type_name: str
    event:             Optional[str]
    position_lat:      Optional[Any]
    position_lon:      Optional[Any]
    speed:             Optional[Any]
    heading:           Optional[Any]
    distance_m:        Optional[float]


@dataclass
class _RSUMessageFact:
    """Pre-parsed facts extracted from a single RSU message entry.

    Attributes
    ----------
    index:
        1-based position of this message in the rsu_messages list.
    is_verbatim:
        ``True`` when the raw entry was a plain string.
    is_nonstandard:
        ``True`` when the raw entry was neither a string nor a dict.
    verbatim_text:
        The original string when ``is_verbatim`` is ``True``; ``None`` otherwise.
    station_id:
        RSU station identifier, or ``None`` when absent.
    event:
        Reported event, cause, or hazard string, or ``None`` when absent.
    position_lat, position_lon:
        Reported position coordinates, or ``None`` when absent.
    advisory:
        Optional free-text advisory string, or ``None`` when absent.
    """
    index:          int
    is_verbatim:    bool
    is_nonstandard: bool
    verbatim_text:  Optional[str]
    station_id:     Optional[Any]
    event:          Optional[str]
    position_lat:   Optional[Any]
    position_lon:   Optional[Any]
    advisory:       Optional[str]


@dataclass
class _ClusterPeerFact:
    """Pre-parsed kinematics of a single non-target cluster member.

    The haversine distance from the ego vehicle is pre-computed once during
    evidence extraction and stored here so that no renderer needs to re-compute
    or re-access raw message coordinates.

    Attributes
    ----------
    index:
        1-based position of this peer in the cluster (excluding target).
    station_id:
        Peer station identifier, or ``None`` when absent.
    station_type:
        ETSI station_type integer, or ``None`` when absent.
    station_type_name:
        Pre-computed human-readable label from ``_station_type_name``.
    lat, lon:
        Peer position coordinates (ETSI integer-scaled), or ``None``.
    distance_m:
        Haversine distance from ego vehicle in metres, or ``None`` when either
        ego or peer position was unavailable.
    speed, heading, yaw_rate, gen_dt:
        Raw CAM kinematic fields, or ``None`` when absent.
    """
    index:             int
    station_id:        Optional[Any]
    station_type:      Optional[int]
    station_type_name: str
    lat:               Optional[Any]
    lon:               Optional[Any]
    distance_m:        Optional[float]
    speed:             Optional[Any]
    heading:           Optional[Any]
    yaw_rate:          Optional[Any]
    gen_dt:            Optional[Any]


@dataclass
class SceneEvidence:
    """Complete structured evidence extracted from a V2X message cluster.

    This object is the single source of truth for all template renderers.
    It is populated once by ``_extract_evidence()`` and then passed to the
    selected renderer.  All three rendering templates (DEFAULT, NARRATIVE,
    STRUCTURED) consume this same object — no renderer may access the raw
    cluster messages directly.

    Attributes
    ----------
    context:
        Operational context label (e.g. ``"urban"``, ``"highway"``).
    station_id:
        Ego vehicle station identifier, or ``"unknown"`` when absent.
    station_type_name:
        Pre-computed human-readable ETSI station type label.
    lat, lon:
        Ego vehicle position (ETSI integer-scaled), or ``None``.
    speed, heading, yaw_rate, long_accel, gen_dt:
        Raw ego vehicle CAM kinematic fields, or ``None`` when absent.
    camera, radar, lidar:
        Local on-board sensor readings, or ``"UNKNOWN"`` when absent.
    extra_sensors:
        List of ``(key, value)`` tuples for any local_perception keys beyond
        the canonical ``camera``, ``radar``, ``lidar`` set.
    peer_report_facts:
        Ordered list of pre-parsed peer report facts.
    rsu_message_facts:
        Ordered list of pre-parsed RSU message facts.
    cluster_peer_facts:
        Ordered list of pre-parsed kinematic facts for non-target cluster
        members, with haversine distances pre-computed.
    """
    context:            str
    station_id:         Any
    station_type_name:  str
    lat:                Optional[Any]
    lon:                Optional[Any]
    speed:              Optional[Any]
    heading:            Optional[Any]
    yaw_rate:           Optional[Any]
    long_accel:         Optional[Any]
    gen_dt:             Optional[Any]
    camera:             str
    radar:              str
    lidar:              str
    extra_sensors:      List[Tuple[str, Any]]
    peer_report_facts:  List[_PeerReportFact]
    rsu_message_facts:  List[_RSUMessageFact]
    cluster_peer_facts: List[_ClusterPeerFact]


# ---------------------------------------------------------------------------
# Evidence extraction helpers
# ---------------------------------------------------------------------------

def _parse_peer_report_fact(report: Any, index: int) -> _PeerReportFact:
    """Parse a raw peer report entry into a ``_PeerReportFact``.

    Uses the exact same field lookup priority as the A2 synthesizer.

    Parameters
    ----------
    report:
        A single entry from ``scene_context.peer_reports``.  May be a dict,
        a plain string, or any other type.
    index:
        1-based index of this report within the peer_reports list.

    Returns
    -------
    _PeerReportFact
    """
    if isinstance(report, str):
        return _PeerReportFact(
            index=index,
            is_verbatim=True,
            is_nonstandard=False,
            verbatim_text=report,
            station_id=None,
            station_type=None,
            station_type_name="unknown",
            event=None,
            position_lat=None,
            position_lon=None,
            speed=None,
            heading=None,
            distance_m=None,
        )

    if not isinstance(report, dict):
        return _PeerReportFact(
            index=index,
            is_verbatim=False,
            is_nonstandard=True,
            verbatim_text=None,
            station_id=None,
            station_type=None,
            station_type_name="unknown",
            event=None,
            position_lat=None,
            position_lon=None,
            speed=None,
            heading=None,
            distance_m=None,
        )

    station_id   = report.get("station_id") or report.get("peer_id") or report.get("id")
    station_type = report.get("station_type")
    event        = report.get("event") or report.get("event_type") or report.get("cause")
    pos          = report.get("position") or {}
    position_lat = pos.get("latitude")   or report.get("latitude")
    position_lon = pos.get("longitude")  or report.get("longitude")
    speed        = report.get("speed")
    heading      = report.get("heading")
    raw_dist     = report.get("distance_m")

    return _PeerReportFact(
        index=index,
        is_verbatim=False,
        is_nonstandard=False,
        verbatim_text=None,
        station_id=station_id,
        station_type=station_type,
        station_type_name=_station_type_name(station_type),
        event=event,
        position_lat=position_lat,
        position_lon=position_lon,
        speed=speed,
        heading=heading,
        distance_m=raw_dist,
    )


def _parse_rsu_message_fact(rsu_msg: Any, index: int) -> _RSUMessageFact:
    """Parse a raw RSU message entry into a ``_RSUMessageFact``.

    Uses the exact same field lookup priority as the A2 synthesizer.

    Parameters
    ----------
    rsu_msg:
        A single entry from ``scene_context.rsu_messages``.  May be a dict,
        a plain string, or any other type.
    index:
        1-based index of this message within the rsu_messages list.

    Returns
    -------
    _RSUMessageFact
    """
    if isinstance(rsu_msg, str):
        return _RSUMessageFact(
            index=index,
            is_verbatim=True,
            is_nonstandard=False,
            verbatim_text=rsu_msg,
            station_id=None,
            event=None,
            position_lat=None,
            position_lon=None,
            advisory=None,
        )

    if not isinstance(rsu_msg, dict):
        return _RSUMessageFact(
            index=index,
            is_verbatim=False,
            is_nonstandard=True,
            verbatim_text=None,
            station_id=None,
            event=None,
            position_lat=None,
            position_lon=None,
            advisory=None,
        )

    station_id   = rsu_msg.get("station_id") or rsu_msg.get("rsu_id") or rsu_msg.get("id")
    event        = (
        rsu_msg.get("event") or rsu_msg.get("event_type")
        or rsu_msg.get("cause") or rsu_msg.get("hazard")
    )
    pos          = rsu_msg.get("position") or {}
    position_lat = pos.get("latitude")  or rsu_msg.get("latitude")
    position_lon = pos.get("longitude") or rsu_msg.get("longitude")
    advisory     = rsu_msg.get("message") or rsu_msg.get("text") or rsu_msg.get("advisory")

    return _RSUMessageFact(
        index=index,
        is_verbatim=False,
        is_nonstandard=False,
        verbatim_text=None,
        station_id=station_id,
        event=event,
        position_lat=position_lat,
        position_lon=position_lon,
        advisory=advisory,
    )


def _parse_cluster_peer_fact(
    peer_msg: Dict[str, Any],
    target_lat: Optional[Any],
    target_lon: Optional[Any],
    index: int,
) -> _ClusterPeerFact:
    """Parse a cluster peer message into a ``_ClusterPeerFact``.

    Haversine distance is pre-computed here so that renderers never need to
    re-access raw message data or re-run the distance calculation.

    Parameters
    ----------
    peer_msg:
        The cluster peer message dictionary.
    target_lat, target_lon:
        Ego vehicle position (ETSI integer-scaled).  Used for haversine
        calculation.  ``None`` when ego position is unavailable.
    index:
        1-based index of this peer within the cluster (excluding target).

    Returns
    -------
    _ClusterPeerFact
    """
    tel = _extract_cam_telemetry(peer_msg)

    distance_m: Optional[float] = None
    if (
        tel["lat"] is not None and tel["lon"] is not None
        and target_lat is not None and target_lon is not None
    ):
        distance_m = _haversine_m(target_lat, target_lon, tel["lat"], tel["lon"])

    return _ClusterPeerFact(
        index=index,
        station_id=tel["station_id"],
        station_type=tel["station_type"],
        station_type_name=_station_type_name(tel["station_type"]),
        lat=tel["lat"],
        lon=tel["lon"],
        distance_m=distance_m,
        speed=tel["speed"],
        heading=tel["heading"],
        yaw_rate=tel["yaw_rate"],
        gen_dt=tel["gen_dt"],
    )


def _extract_evidence(
    cluster: List[Dict[str, Any]],
    context: str,
) -> SceneEvidence:
    """Extract all structured evidence from a non-empty cluster.

    This function executes **exactly once** per ``synthesize_message()`` call.
    It populates a ``SceneEvidence`` object that is then passed to the selected
    renderer.  No renderer may call this function again or access raw cluster
    messages directly.

    Parameters
    ----------
    cluster:
        Non-empty window of V2X message dicts.  ``cluster[-1]`` is the target.
    context:
        Operational context label (never ``None`` at this point).

    Returns
    -------
    SceneEvidence
    """
    target_msg = cluster[-1]

    # Ego vehicle CAM telemetry
    tel          = _extract_cam_telemetry(target_msg)
    station_id   = tel["station_id"] if tel["station_id"] is not None else "unknown"
    st_type_name = _station_type_name(tel["station_type"])
    lat          = tel["lat"]
    lon          = tel["lon"]

    # Local sensor observations
    local_perception: Dict[str, Any] = target_msg.get("local_perception") or {}
    camera = local_perception.get("camera", "UNKNOWN")
    radar  = local_perception.get("radar",  "UNKNOWN")
    lidar  = local_perception.get("lidar",  "UNKNOWN")
    extra_sensors: List[Tuple[str, Any]] = [
        (k, v) for k, v in local_perception.items()
        if k not in {"camera", "radar", "lidar"}
    ]

    # Peer reports and RSU messages
    scene_context: Dict[str, Any] = target_msg.get("scene_context") or {}
    raw_peer_reports: List[Any]   = scene_context.get("peer_reports")  or []
    raw_rsu_messages: List[Any]   = scene_context.get("rsu_messages")  or []

    peer_report_facts: List[_PeerReportFact] = [
        _parse_peer_report_fact(r, i + 1)
        for i, r in enumerate(raw_peer_reports)
    ]
    rsu_message_facts: List[_RSUMessageFact] = [
        _parse_rsu_message_fact(m, i + 1)
        for i, m in enumerate(raw_rsu_messages)
    ]

    # Cooperative cluster peers (all messages except the target)
    cluster_peer_facts: List[_ClusterPeerFact] = [
        _parse_cluster_peer_fact(peer_msg, lat, lon, i + 1)
        for i, peer_msg in enumerate(cluster[:-1])
    ]

    return SceneEvidence(
        context=context,
        station_id=station_id,
        station_type_name=st_type_name,
        lat=lat,
        lon=lon,
        speed=tel["speed"],
        heading=tel["heading"],
        yaw_rate=tel["yaw_rate"],
        long_accel=tel["long_accel"],
        gen_dt=tel["gen_dt"],
        camera=camera,
        radar=radar,
        lidar=lidar,
        extra_sensors=extra_sensors,
        peer_report_facts=peer_report_facts,
        rsu_message_facts=rsu_message_facts,
        cluster_peer_facts=cluster_peer_facts,
    )


# ---------------------------------------------------------------------------
# DEFAULT renderer — byte-for-byte identical to A2 output
# ---------------------------------------------------------------------------

def _default_peer_report(fact: _PeerReportFact) -> str:
    """Render one peer report fact in DEFAULT format.

    Output is byte-for-byte identical to the A2 ``_serialize_peer_report``
    function operating on the same raw entry.
    """
    if fact.is_verbatim:
        return f"Peer report {fact.index}: {fact.verbatim_text}"
    if fact.is_nonstandard:
        return f"Peer report {fact.index}: (non-standard format)"

    parts: List[str] = []
    if fact.station_id is not None:
        parts.append(f"Station {fact.station_id}")
    if fact.station_type is not None:
        parts.append(f"({fact.station_type_name})")
    if fact.event is not None:
        parts.append(f"reports: {fact.event}")
    if fact.position_lat is not None and fact.position_lon is not None:
        parts.append(f"at position (lat={fact.position_lat}, lon={fact.position_lon})")
    if fact.speed is not None:
        parts.append(f"speed={fact.speed}")
    if fact.heading is not None:
        parts.append(f"heading={fact.heading} deg")
    if fact.distance_m is not None:
        parts.append(f"distance={fact.distance_m:.1f} m")
    if not parts:
        return f"Peer report {fact.index}: (no structured fields)"
    return f"Peer report {fact.index}: {' '.join(parts)}."


def _default_rsu_message(fact: _RSUMessageFact) -> str:
    """Render one RSU message fact in DEFAULT format.

    Output is byte-for-byte identical to the A2 ``_serialize_rsu_message``
    function operating on the same raw entry.
    """
    if fact.is_verbatim:
        return f"RSU message {fact.index}: {fact.verbatim_text}"
    if fact.is_nonstandard:
        return f"RSU message {fact.index}: (non-standard format)"

    parts: List[str] = []
    if fact.station_id is not None:
        parts.append(f"RSU {fact.station_id}")
    if fact.event is not None:
        parts.append(f"reports: {fact.event}")
    if fact.position_lat is not None and fact.position_lon is not None:
        parts.append(f"at position (lat={fact.position_lat}, lon={fact.position_lon})")
    if fact.advisory is not None:
        parts.append(f'advisory: "{fact.advisory}"')
    if not parts:
        return f"RSU message {fact.index}: (no structured fields)"
    return f"RSU message {fact.index}: {' '.join(parts)}."


def _default_cluster_peer(fact: _ClusterPeerFact) -> str:
    """Render one cluster peer fact in DEFAULT format.

    Output is byte-for-byte identical to the A2 ``_serialize_cluster_peer``
    function operating on the same raw message.
    """
    parts: List[str] = [f"Cluster peer {fact.index}"]
    if fact.station_id is not None:
        parts.append(f"(station {fact.station_id}, type={fact.station_type_name})")
    if fact.lat is not None and fact.lon is not None:
        parts.append(f"position=(lat={fact.lat}, lon={fact.lon})")
        if fact.distance_m is not None:
            parts.append(f"distance={fact.distance_m:.1f} m from ego")
    if fact.speed is not None:
        parts.append(f"speed={fact.speed}")
    if fact.heading is not None:
        parts.append(f"heading={fact.heading} deg")
    if fact.yaw_rate is not None:
        parts.append(f"yaw_rate={fact.yaw_rate}")
    if fact.gen_dt is not None:
        parts.append(f"timestamp={fact.gen_dt}")
    return ", ".join(parts) + "."


def _render_default(evidence: SceneEvidence) -> str:
    """Render a scene description in DEFAULT (A2-compatible) format.

    Produces output byte-for-byte identical to the A2 synthesizer for all
    inputs.  Uses compact ``key=value`` notation with observations joined
    by spaces.

    Parameters
    ----------
    evidence:
        Pre-extracted scene evidence shared by all renderers.

    Returns
    -------
    str
        Compact natural-language scene description.
    """
    lines: List[str] = []

    # Ego vehicle header
    lines.append(
        f"V2X Scene Report: context={evidence.context}. "
        f"Ego vehicle: station {evidence.station_id} (type={evidence.station_type_name}), "
        f"position=(lat={_fmt_optional(evidence.lat)}, lon={_fmt_optional(evidence.lon)}), "
        f"speed={_fmt_optional(evidence.speed)}, "
        f"heading={_fmt_optional(evidence.heading)} deg, "
        f"yaw_rate={_fmt_optional(evidence.yaw_rate)}, "
        f"longitudinal_acceleration={_fmt_optional(evidence.long_accel)}, "
        f"timestamp={_fmt_optional(evidence.gen_dt)}."
    )

    # Local sensor observations
    sensor_line = (
        f"Local sensor observations: "
        f"camera={evidence.camera}, radar={evidence.radar}, lidar={evidence.lidar}"
    )
    if evidence.extra_sensors:
        sensor_line += ", " + ", ".join(f"{k}={v}" for k, v in evidence.extra_sensors)
    sensor_line += "."
    lines.append(sensor_line)

    # Peer reports
    if evidence.peer_report_facts:
        for fact in evidence.peer_report_facts:
            lines.append(_default_peer_report(fact))
    else:
        lines.append("No peer reports received.")

    # RSU messages
    if evidence.rsu_message_facts:
        for fact in evidence.rsu_message_facts:
            lines.append(_default_rsu_message(fact))
    else:
        lines.append("No RSU messages received.")

    # Cooperative cluster peers
    if evidence.cluster_peer_facts:
        for fact in evidence.cluster_peer_facts:
            lines.append(_default_cluster_peer(fact))
    else:
        lines.append("No other vehicles in cooperative cluster.")

    return " ".join(lines)


# ---------------------------------------------------------------------------
# NARRATIVE renderer — flowing prose, same facts, different vocabulary
# ---------------------------------------------------------------------------

def _narrative_peer_report(fact: _PeerReportFact) -> str:
    """Render one peer report fact as a flowing prose sentence."""
    if fact.is_verbatim:
        return f"A neighbouring vehicle reports: {fact.verbatim_text}."
    if fact.is_nonstandard:
        return "A neighbouring vehicle submitted a report in an unrecognised format."

    if fact.station_id is not None:
        id_part = f"station {fact.station_id}"
    else:
        id_part = "an unidentified station"

    type_part = f", {fact.station_type_name}" if fact.station_type is not None else ""
    subject = f"A neighbouring vehicle ({id_part}{type_part})"

    if fact.event is not None:
        body = f"{subject} reports {fact.event}"
    else:
        body = f"{subject} submitted an observation"

    if fact.position_lat is not None and fact.position_lon is not None:
        body += f", at position (lat={fact.position_lat}, lon={fact.position_lon})"

    tail: List[str] = []
    if fact.speed is not None:
        tail.append(f"speed {fact.speed}")
    if fact.heading is not None:
        tail.append(f"heading {fact.heading} degrees")
    if fact.distance_m is not None:
        tail.append(f"distance {fact.distance_m:.1f} m")
    if tail:
        body += " (" + ", ".join(tail) + ")"

    return body + "."


def _narrative_rsu_message(fact: _RSUMessageFact) -> str:
    """Render one RSU message fact as a flowing prose sentence."""
    if fact.is_verbatim:
        return f"An infrastructure message was received: {fact.verbatim_text}."
    if fact.is_nonstandard:
        return "An infrastructure message was received in an unrecognised format."

    subject = (
        f"The roadside unit (RSU {fact.station_id})"
        if fact.station_id is not None
        else "A roadside unit"
    )

    if fact.event is not None:
        body = f"{subject} reports {fact.event}"
    else:
        body = f"{subject} transmitted a message"

    if fact.position_lat is not None and fact.position_lon is not None:
        body += f", at position (lat={fact.position_lat}, lon={fact.position_lon})"

    if fact.advisory is not None:
        body += f'; advisory: "{fact.advisory}"'

    return body + "."


def _narrative_cluster_peer(fact: _ClusterPeerFact) -> str:
    """Render one cluster peer fact as a flowing prose sentence."""
    if fact.station_id is not None:
        subject = f"Vehicle (station {fact.station_id}, {fact.station_type_name})"
    else:
        subject = "A cooperative vehicle"

    if fact.distance_m is not None:
        intro = f"{subject} was observed {fact.distance_m:.1f} metres from the ego vehicle"
    else:
        intro = f"{subject} was observed in the cooperative cluster"

    if fact.lat is not None and fact.lon is not None:
        intro += f", at position latitude {fact.lat}, longitude {fact.lon}"

    kinematic: List[str] = []
    if fact.speed is not None:
        kinematic.append(f"speed {fact.speed}")
    if fact.heading is not None:
        kinematic.append(f"heading {fact.heading} degrees")
    if fact.yaw_rate is not None:
        kinematic.append(f"yaw rate {fact.yaw_rate}")

    sentence = intro
    if kinematic:
        sentence += ", moving at " + ", ".join(kinematic)

    if fact.gen_dt is not None:
        sentence += f". The CAM timestamp is {fact.gen_dt}"

    return sentence + "."


def _render_narrative(evidence: SceneEvidence) -> str:
    """Render a scene description in NARRATIVE format.

    Produces flowing prose using different vocabulary from DEFAULT while
    expressing exactly the same factual observations.  Each peer report and RSU
    message is listed as an individual prose sentence.  No inference, no
    aggregation, no reasoning.

    Parameters
    ----------
    evidence:
        Pre-extracted scene evidence shared by all renderers.

    Returns
    -------
    str
        Flowing prose scene description.
    """
    article = "an" if evidence.context and evidence.context[0].lower() in "aeiou" else "a"

    parts: List[str] = []

    # Ego vehicle
    parts.append(
        f"V2X Scene Narrative: "
        f"The ego vehicle (station {evidence.station_id}, {evidence.station_type_name}) "
        f"is operating in {article} {evidence.context} environment. "
        f"Its reported position is latitude {_fmt_optional(evidence.lat)}, "
        f"longitude {_fmt_optional(evidence.lon)}. "
        f"It is travelling at speed {_fmt_optional(evidence.speed)}, "
        f"on heading {_fmt_optional(evidence.heading)} degrees, "
        f"with yaw rate {_fmt_optional(evidence.yaw_rate)} "
        f"and longitudinal acceleration {_fmt_optional(evidence.long_accel)}. "
        f"The CAM message timestamp is {_fmt_optional(evidence.gen_dt)}."
    )

    # Local sensors
    sensor_items = [
        f"the camera sensor reports {evidence.camera}",
        f"the radar sensor reports {evidence.radar}",
        f"the lidar sensor reports {evidence.lidar}",
    ]
    for k, v in evidence.extra_sensors:
        sensor_items.append(f"the {k} sensor reports {v}")
    parts.append("On-board perception: " + "; ".join(sensor_items) + ".")

    # Peer reports
    if not evidence.peer_report_facts:
        parts.append("No cooperative peer observations were received in this window.")
    else:
        sentences = [_narrative_peer_report(f) for f in evidence.peer_report_facts]
        parts.append("Cooperative peer observations: " + " ".join(sentences))

    # RSU messages
    if not evidence.rsu_message_facts:
        parts.append("No infrastructure messages were received.")
    else:
        sentences = [_narrative_rsu_message(f) for f in evidence.rsu_message_facts]
        parts.append("Infrastructure observations: " + " ".join(sentences))

    # Cluster peers
    if not evidence.cluster_peer_facts:
        parts.append("No other cooperative cluster members were observed.")
    else:
        sentences = [_narrative_cluster_peer(f) for f in evidence.cluster_peer_facts]
        parts.append("Cooperative cluster observations: " + " ".join(sentences))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# STRUCTURED renderer — sectioned report with bracketed headers
# ---------------------------------------------------------------------------

def _structured_peer_report_lines(fact: _PeerReportFact) -> List[str]:
    """Render one peer report fact as indented STRUCTURED lines."""
    if fact.is_verbatim:
        return [f"[{fact.index}] (verbatim) {fact.verbatim_text}"]
    if fact.is_nonstandard:
        return [f"[{fact.index}] (non-standard format)"]

    header_parts: List[str] = []
    if fact.station_id is not None:
        type_suffix = f" ({fact.station_type_name})" if fact.station_type is not None else ""
        header_parts.append(f"Station {fact.station_id}{type_suffix}")
    if fact.event is not None:
        header_parts.append(f"reports: {fact.event}")

    header = " | ".join(header_parts) if header_parts else "(no structured fields)"
    lines = [f"[{fact.index}] {header}"]

    if fact.position_lat is not None and fact.position_lon is not None:
        lines.append(f"    Position: lat={fact.position_lat}, lon={fact.position_lon}")

    kin: List[str] = []
    if fact.speed is not None:
        kin.append(f"speed={fact.speed}")
    if fact.heading is not None:
        kin.append(f"heading={fact.heading} deg")
    if kin:
        lines.append("    Kinematics: " + " | ".join(kin))

    if fact.distance_m is not None:
        lines.append(f"    Distance: {fact.distance_m:.1f} m")

    return lines


def _structured_rsu_message_lines(fact: _RSUMessageFact) -> List[str]:
    """Render one RSU message fact as indented STRUCTURED lines."""
    if fact.is_verbatim:
        return [f"[{fact.index}] (verbatim) {fact.verbatim_text}"]
    if fact.is_nonstandard:
        return [f"[{fact.index}] (non-standard format)"]

    header_parts: List[str] = []
    if fact.station_id is not None:
        header_parts.append(f"RSU {fact.station_id}")
    if fact.event is not None:
        header_parts.append(f"reports: {fact.event}")

    header = " | ".join(header_parts) if header_parts else "(no structured fields)"
    lines = [f"[{fact.index}] {header}"]

    if fact.position_lat is not None and fact.position_lon is not None:
        lines.append(f"    Position: lat={fact.position_lat}, lon={fact.position_lon}")
    if fact.advisory is not None:
        lines.append(f'    Advisory: "{fact.advisory}"')

    return lines


def _structured_cluster_peer_lines(fact: _ClusterPeerFact) -> List[str]:
    """Render one cluster peer fact as indented STRUCTURED lines."""
    if fact.station_id is not None:
        type_suffix = f" ({fact.station_type_name})" if fact.station_type is not None else ""
        dist_suffix = f" | {fact.distance_m:.1f} m from ego" if fact.distance_m is not None else ""
        header = f"Station {fact.station_id}{type_suffix}{dist_suffix}"
    else:
        dist_suffix = f" | {fact.distance_m:.1f} m from ego" if fact.distance_m is not None else ""
        header = f"(unknown station){dist_suffix}"

    lines = [f"[{fact.index}] {header}"]

    if fact.lat is not None and fact.lon is not None:
        lines.append(f"    Position: lat={fact.lat}, lon={fact.lon}")

    kin: List[str] = []
    if fact.speed is not None:
        kin.append(f"speed={fact.speed}")
    if fact.heading is not None:
        kin.append(f"heading={fact.heading} deg")
    if fact.yaw_rate is not None:
        kin.append(f"yaw_rate={fact.yaw_rate}")
    if kin:
        lines.append("    Kinematics: " + " | ".join(kin))

    if fact.gen_dt is not None:
        lines.append(f"    Timestamp: {fact.gen_dt}")

    return lines


def _render_structured(evidence: SceneEvidence) -> str:
    """Render a scene description in STRUCTURED format.

    Produces a multi-line, section-delimited report using bracketed headers and
    indented key-value pairs.  Expresses exactly the same factual observations
    as DEFAULT and NARRATIVE.

    Parameters
    ----------
    evidence:
        Pre-extracted scene evidence shared by all renderers.

    Returns
    -------
    str
        Multi-line sectioned scene report.
    """
    lines: List[str] = []

    lines.append("=== V2X Scene Report ===")
    lines.append(f"Context: {evidence.context}")
    lines.append("")

    # Ego vehicle
    lines.append("[Ego Vehicle]")
    lines.append(f"Station: {evidence.station_id} | Type: {evidence.station_type_name}")
    lines.append(f"Position: lat={_fmt_optional(evidence.lat)}, lon={_fmt_optional(evidence.lon)}")
    lines.append(
        f"Kinematics: speed={_fmt_optional(evidence.speed)} | "
        f"heading={_fmt_optional(evidence.heading)} deg | "
        f"yaw_rate={_fmt_optional(evidence.yaw_rate)} | "
        f"longitudinal_acceleration={_fmt_optional(evidence.long_accel)}"
    )
    lines.append(f"Timestamp: {_fmt_optional(evidence.gen_dt)}")
    lines.append("")

    # Local sensor observations
    lines.append("[Local Sensor Observations]")
    lines.append(f"Camera: {evidence.camera}")
    lines.append(f"Radar: {evidence.radar}")
    lines.append(f"Lidar: {evidence.lidar}")
    for k, v in evidence.extra_sensors:
        lines.append(f"{k.capitalize()}: {v}")
    lines.append("")

    # Peer reports
    lines.append("[Peer Reports]")
    if not evidence.peer_report_facts:
        lines.append("(none received)")
    else:
        for fact in evidence.peer_report_facts:
            lines.extend(_structured_peer_report_lines(fact))
    lines.append("")

    # RSU messages
    lines.append("[RSU Messages]")
    if not evidence.rsu_message_facts:
        lines.append("(none received)")
    else:
        for fact in evidence.rsu_message_facts:
            lines.extend(_structured_rsu_message_lines(fact))
    lines.append("")

    # Cooperative cluster
    lines.append("[Cooperative Cluster]")
    if not evidence.cluster_peer_facts:
        lines.append("(no additional vehicles observed)")
    else:
        for fact in evidence.cluster_peer_facts:
            lines.extend(_structured_cluster_peer_lines(fact))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Renderer dispatch table
# ---------------------------------------------------------------------------

#: Maps each ``TemplateStyle`` to its renderer function.
#: To add a new template, implement a ``_render_<name>(evidence) -> str``
#: function and add a single entry here.  No other code requires modification.
_RENDERERS: Dict[TemplateStyle, Callable[[SceneEvidence], str]] = {
    TemplateStyle.DEFAULT:    _render_default,
    TemplateStyle.NARRATIVE:  _render_narrative,
    TemplateStyle.STRUCTURED: _render_structured,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_message(
    cluster: List[Dict[str, Any]],
    b2_result: Dict[str, Any],
    context: Optional[str] = None,
    template: Optional[TemplateStyle] = None,
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

    This function is deterministic: identical ``cluster``, ``context``, and
    ``template`` inputs always produce identical output text.

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
    template:
        Rendering template to use.  ``None`` or ``TemplateStyle.DEFAULT``
        produces output identical to the A2 synthesizer.  Passing a different
        ``TemplateStyle`` selects an alternative linguistic representation of
        the same facts.

    Returns
    -------
    dict
        Keys:

        ``"text"`` (str)
            The synthesized scene description, ready for B3 input.
        ``"template"`` (str)
            Always ``"cooperative_scene_report"`` (unchanged from A2).
        ``"template_style"`` (str)
            The ``TemplateStyle`` value used (``"default"``, ``"narrative"``,
            or ``"structured"``).
        ``"sources"`` (list)
            List of evidence source identifiers included in the text.
    """
    # b2_result is intentionally unused.  It is accepted only to maintain the
    # stable public interface shared with orchestrator.py and future callers.
    _ = b2_result

    chosen = template if template is not None else TemplateStyle.DEFAULT

    if not cluster:
        return {
            "text": "V2X Scene Report: No cooperative scene information available.",
            "template": "cooperative_scene_report",
            "template_style": chosen.value,
            "sources": [],
        }

    ctx_name = context or "unknown"
    evidence = _extract_evidence(cluster, ctx_name)
    text     = _RENDERERS[chosen](evidence)

    return {
        "text":           text,
        "template":       "cooperative_scene_report",
        "template_style": chosen.value,
        "sources": [
            "local_perception",
            "peer_reports",
            "rsu_messages",
            "cooperative_observations",
        ],
    }
