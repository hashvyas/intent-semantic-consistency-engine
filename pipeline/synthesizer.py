"""
pipeline/synthesizer.py
========================
Serializes cooperative scene information and B2 outputs into a natural-language description.
Acts as a serialization layer only.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

def _nested_get(obj: Any, dotted_key: str, default: Any = None) -> Any:
    """Traverse a nested dict using a dot-separated key path."""
    parts = dotted_key.split(".")
    node: Any = obj
    for part in parts:
        if not isinstance(node, dict):
            return default
        node = node.get(part)
        if node is None:
            return default
    return node

def _haversine_m(lat1_e7: float, lon1_e7: float, lat2_e7: float, lon2_e7: float) -> float:
    lat1 = math.radians(lat1_e7 * 1e-7)
    lat2 = math.radians(lat2_e7 * 1e-7)
    dlat = lat2 - lat1
    dlon = math.radians((lon2_e7 - lon1_e7) * 1e-7)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 6_371_000.0 * 2.0 * math.asin(min(1.0, math.sqrt(max(0.0, a))))

def synthesize_message(
    cluster: List[Dict[str, Any]],
    b2_result: Dict[str, Any],
    context: Optional[str] = None
) -> Dict[str, Any]:
    """Convert cooperative scene information and B2 results into a deterministic,
    template-based V2X natural-language message.
    
    This function acts as a serialization layer only. It does not perform inference,
    extract features, or modify trust scores.
    """
    if not cluster:
        return {
            "text": "V2X Scene Report: No cooperative scene information available.",
            "template": "cooperative_scene_report",
            "sources": []
        }

    target_msg = cluster[-1]
    
    # 1. Extract basic telemetry
    station_id = _nested_get(target_msg, "header.station_id") or target_msg.get("station_id") or "unknown"
    station_type = (
        _nested_get(target_msg, "cam.cam_parameters.basic_container.station_type")
        or target_msg.get("station_type")
    )
    
    station_type_mapping = {
        0: "unknown",
        1: "pedestrian",
        2: "cyclist",
        3: "moped",
        4: "motorcycle",
        5: "passengerCar",
        6: "bus",
        7: "lightTruck",
        8: "heavyTruck",
        9: "trailer",
        10: "specialVehicle",
        11: "tram",
        12: "lightVruVehicle",
        13: "animal",
        14: "agricultural",
        15: "roadSideUnit",
    }
    station_type_name = station_type_mapping.get(station_type, f"unknown ({station_type})") if station_type is not None else "unknown"

    lat = (
        _nested_get(target_msg, "cam.cam_parameters.basic_container.reference_position.latitude")
        or target_msg.get("latitude")
    )
    lon = (
        _nested_get(target_msg, "cam.cam_parameters.basic_container.reference_position.longitude")
        or target_msg.get("longitude")
    )
    speed = (
        _nested_get(target_msg, "cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.speed")
        or target_msg.get("speed")
    )
    heading = (
        _nested_get(target_msg, "cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.heading")
        or target_msg.get("heading")
    )

    # 2. Extract perception and scene context
    local_perception = target_msg.get("local_perception") or {}
    camera = local_perception.get("camera", "UNKNOWN")
    radar = local_perception.get("radar", "UNKNOWN")
    lidar = local_perception.get("lidar", "UNKNOWN")

    scene_context = target_msg.get("scene_context") or {}
    peer_reports = scene_context.get("peer_reports") or []
    rsu_messages = scene_context.get("rsu_messages") or []
    peer_reports_count = len(peer_reports)
    rsu_messages_count = len(rsu_messages)

    cluster_size = len(cluster)
    other_station_ids = []
    for msg in cluster:
        sid = _nested_get(msg, "header.station_id") or msg.get("station_id")
        if sid is not None and sid != station_id:
            other_station_ids.append(str(sid))
    other_stations_str = ", ".join(other_station_ids) if other_station_ids else "none"

    # 3. Calculate max spatial separation
    max_d = 0.0
    if len(cluster) > 1 and lat is not None and lon is not None:
        dists = []
        for msg in cluster[:-1]:
            olat = (
                _nested_get(msg, "cam.cam_parameters.basic_container.reference_position.latitude")
                or msg.get("latitude")
            )
            olon = (
                _nested_get(msg, "cam.cam_parameters.basic_container.reference_position.longitude")
                or msg.get("longitude")
            )
            if olat is not None and olon is not None:
                dists.append(_haversine_m(lat, lon, olat, olon))
        if dists:
            max_d = max(dists)

    # 4. Extract B2 metrics
    trust = b2_result.get("trust", 1.0)
    belief = b2_result.get("belief", 1.0)
    disbelief = b2_result.get("disbelief", 0.0)
    uncertainty = b2_result.get("uncertainty", 0.0)
    cluster_score = b2_result.get("cluster_score", 1.0)
    entropy = b2_result.get("entropy", 0.0)
    replay_probability = b2_result.get("replay_probability", 0.0)
    identity_consistency = b2_result.get("identity_consistency", 1.0)

    obstacle_detected = "YES" if trust < 0.3 else "NO"
    ctx_name = context or "unknown"

    # 5. Serialization using a deterministic template
    # Future templates (e.g. DENM, CPM, Cooperative Summary, Hazard Notification, Infrastructure Advisory)
    # may be added here or selected based on scene context metadata.
    text = (
        f"V2X Scene Report: Station {station_id} ({station_type_name}) in {ctx_name} context. "
        f"Position: Lat={lat if lat is not None else 'N/A'}, Lon={lon if lon is not None else 'N/A'}, "
        f"Speed={speed if speed is not None else 'N/A'} m/s, Heading={heading if heading is not None else 'N/A'} deg. "
        f"Local Sensors: Camera={camera}, Radar={radar}, Lidar={lidar}. "
        f"Scene Context: {peer_reports_count} peer reports, {rsu_messages_count} RSU messages. "
        f"Cooperative Cluster: size={cluster_size}, peers=[{other_stations_str}], max_spatial_separation={max_d:.2f}m. "
        f"Trust Metadata: TrustScore={trust:.4f}, Belief={belief:.4f}, Disbelief={disbelief:.4f}, Uncertainty={uncertainty:.4f}. "
        f"Behavioral Evidence: KinematicSimilarity={cluster_score:.4f}, TemporalEntropy={entropy:.4f}, "
        f"ReplayProbability={replay_probability:.4f}, IdentityConsistency={identity_consistency:.4f}. "
        f"Obstacle Alert: {obstacle_detected}."
    )

    sources = ["local_perception", "peer_reports", "rsu_messages", "cooperative_observations"]

    return {
        "text": text,
        "template": "cooperative_scene_report",
        "sources": sources
    }
