"""
b2_csia/observability_graph.py
==============================
Observability Graph Module.

Models the traffic scene as a weighted observability graph where nodes represent
senders (ITS stations) and edge weights represent mutual observability probability
calculated via a multiplicative formulation. Tracks node evidence quality,
edge confidence, temporal exponential edge decay, context-aware V2X communication
ranges, and detailed edge explainability components.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set, Tuple, Optional
from b2_csia.evidence_quality import EvidenceQuality


class ObservabilityEdge:
    """Represents a weighted edge between two nodes in the observability graph.

    Attributes
    ----------
    u, v : int
        Nodes connected by this edge.
    weight : float
        Edge weight ∈ [0.0, 1.0].
    confidence : float
        Edge confidence ∈ [0.0, 1.0] derived from node quality.
    contributing_factors : Dict[str, float]
        Individual normalized factors that make up the weight.
    last_updated_wall : float
        System wall timestamp when this edge was last updated.
    """

    def __init__(
        self,
        u: int,
        v: int,
        weight: float,
        confidence: float,
        factors: Dict[str, float],
        wall_time: float,
    ) -> None:
        self.u = min(u, v)
        self.v = max(u, v)
        self.weight = weight
        self.confidence = confidence
        self.contributing_factors = factors
        self.last_updated_wall = wall_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to a serializable dictionary representation."""
        return {
            "source": self.u,
            "target": self.v,
            "weight": self.weight,
            "confidence": self.confidence,
            "contributing_factors": self.contributing_factors,
            "last_updated_wall": self.last_updated_wall,
        }


class ObservabilityGraph:
    """Represents a weighted observability graph for a traffic scene."""

    def __init__(self) -> None:
        self.nodes: Set[int] = set()
        self.edges: Dict[Tuple[int, int], ObservabilityEdge] = {}
        self.node_metadata: Dict[int, Dict[str, Any]] = {}
        self.node_qualities: Dict[int, EvidenceQuality] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph into a standard JSON-compatible dict."""
        return {
            "nodes": list(self.nodes),
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "node_metadata": {str(k): v for k, v in self.node_metadata.items()},
            "node_qualities": {str(k): v.to_dict() for k, v in self.node_qualities.items()},
        }


class ObservabilityGraphBuilder:
    """Constructs and maintains a weighted observability graph incrementally.

    Parameters
    ----------
    los_threshold_m : float
        Distance threshold (meters) to consider a third vehicle as obstructing the line-of-sight.
    dist_decay_lambda : float
        Exponential decay rate for distance-based weight.
    rsu_boost : float
        Booster value ∈ [0.0, 1.0] when both nodes are within range of the same RSU.
    temporal_tolerance_s : float
        Maximum time delta (seconds) for observations to overlap.
    edge_decay_alpha : float
        Exponential decay rate governing temporal edge weight decay.
    node_expiry_s : float
        Time threshold (seconds) to expire inactive nodes.
    """

    def __init__(
        self,
        los_threshold_m: float = 2.0,
        dist_decay_lambda: float = 0.005,
        rsu_boost: float = 0.15,
        temporal_tolerance_s: float = 1.0,
        edge_decay_alpha: float = 0.1,  # temporal decay factor
        node_expiry_s: float = 5.0,
    ) -> None:
        self.los_threshold_m = los_threshold_m
        self.dist_decay_lambda = dist_decay_lambda
        self.rsu_boost = rsu_boost
        self.temporal_tolerance_s = temporal_tolerance_s
        self.edge_decay_alpha = edge_decay_alpha
        self.node_expiry_s = node_expiry_s
        self.graph = ObservabilityGraph()

    def get_communication_range(self, context: str) -> float:
        """Context-aware communication ranges configuration lookup.

        Parameters
        ----------
        context : str
            Traffic environment context (e.g. 'highway', 'urban', 'rural', 'rsu').
        """
        c = context.lower().strip()
        if "highway" in c:
            return 400.0
        elif "urban" in c:
            return 150.0
        elif "rural" in c:
            return 300.0
        elif "rsu" in c:
            return 500.0
        return 300.0  # default range

    def update_node(
        self,
        station_id: int,
        lat_e7: float,
        lon_e7: float,
        heading_deg10: float,
        timestamp_ns: float,
        station_type: int,
        wall_time: float,
        context: str = "urban",
        lane_position: Optional[int] = None,
        raw_msg: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Incrementally updates a node and recalculates its edges.

        Parameters
        ----------
        station_id : int
            The unique ID of the sender.
        lat_e7 : float
            Latitude in 1e-7 degree units.
        lon_e7 : float
            Longitude in 1e-7 degree units.
        heading_deg10 : float
            Heading in 0.1 degree units.
        timestamp_ns : float
            Generation timestamp (nanoseconds).
        station_type : int
            ETSI station type code.
        wall_time : float
            Current system wall time (seconds).
        context : str
            Traffic context ('highway', 'urban', 'rural', 'rsu').
        lane_position : int, optional
            Lane identifier if available.
        raw_msg : dict, optional
            The raw message dictionary to extract EvidenceQuality.
        """
        # Calculate node quality
        if raw_msg is not None:
            quality = EvidenceQuality.from_message(raw_msg, wall_time)
        else:
            quality = EvidenceQuality(1.0, 1.0, 1.0, 1.0, wall_time)

        self.graph.nodes.add(station_id)
        self.graph.node_qualities[station_id] = quality
        self.graph.node_metadata[station_id] = {
            "lat": lat_e7,
            "lon": lon_e7,
            "heading": heading_deg10,
            "timestamp": timestamp_ns,
            "station_type": station_type,
            "wall_time": wall_time,
            "lane": lane_position,
            "context": context,
        }

        # Recalculate edges for this node (O(N) update)
        for other_id in list(self.graph.nodes):
            if other_id == station_id:
                continue
            self._update_edge_weight(station_id, other_id, wall_time)

    def expire_nodes(self, current_wall_time: float) -> None:
        """Expires nodes and applies temporal edge weight decay for inactive edges.

        Parameters
        ----------
        current_wall_time : float
            The current system wall time (seconds).
        """
        expired_nodes = set()
        for node_id, meta in self.graph.node_metadata.items():
            if current_wall_time - meta["wall_time"] > self.node_expiry_s:
                expired_nodes.add(node_id)

        for node_id in expired_nodes:
            self.graph.nodes.remove(node_id)
            del self.graph.node_metadata[node_id]
            self.graph.node_qualities.pop(node_id, None)

        # Decay edges incrementally
        edges_to_remove = []
        for key, edge in self.graph.edges.items():
            if key[0] in expired_nodes or key[1] in expired_nodes:
                edges_to_remove.append(key)
                continue

            # Apply temporal edge decay to existing edges
            dt = current_wall_time - edge.last_updated_wall
            if dt > 0.0:
                decayed_w = edge.weight * math.exp(-self.edge_decay_alpha * dt)
                if decayed_w < 0.01:
                    edges_to_remove.append(key)
                else:
                    edge.weight = decayed_w

        for key in edges_to_remove:
            self.graph.edges.pop(key, None)

    def _update_edge_weight(self, u: int, v: int, wall_time: float) -> None:
        """Calculate and store the mutual edge weight between nodes u and v.
        
        The final edge weight is computed as a strictly multiplicative combination of the 
        individual normalized factors. This multiplicative fusion ensures that if any single 
        critical condition (e.g. communication range or temporal overlap) fails or drops to zero, 
        the overall observability falls to zero.

        Parameters
        ----------
        u, v : int
            The node IDs of the two traffic participants.
        wall_time : float
            Current system wall time in seconds.
        """
        meta_u = self.graph.node_metadata[u]
        meta_v = self.graph.node_metadata[v]

        q_u = self.graph.node_qualities.get(u)
        q_v = self.graph.node_qualities.get(v)
        q_u_score = q_u.score if q_u else 1.0
        q_v_score = q_v.score if q_v else 1.0

        # Joint Edge Confidence = product of nodes' evidence quality scores
        edge_confidence = q_u_score * q_v_score

        # Get context-aware communication range
        range_u = self.get_communication_range(meta_u["context"])
        range_v = self.get_communication_range(meta_v["context"])
        comm_range = min(range_u, range_v)

        # 1. Distance factor (d_m / comm_range)
        # Reflects that closer vehicles observe each other more reliably.
        d_m = self._haversine_m(meta_u["lat"], meta_u["lon"], meta_v["lat"], meta_v["lon"])
        if d_m > comm_range:
            f_dist = 0.0
            f_comm = 0.0
        else:
            f_dist = math.exp(-self.dist_decay_lambda * d_m)
            f_comm = 1.0 - (d_m / comm_range)

        # 2. Heading alignment factor (abs(cos(delta_theta)))
        # Why abs(cos(delta_theta)) is used instead of cos(delta_theta):
        # Heading similarity measures road-axis alignment, not direction of travel.
        # - 0° (same direction) -> cos(0) = 1.0 (High similarity).
        # - 180° (opposite direction, same road axis) -> abs(cos(pi)) = abs(-1.0) = 1.0 (High similarity).
        #   Vehicles traveling in opposite directions on the same road axis observe the same physical
        #   traffic environment and possess strong cooperative observability of the roadway.
        # - 90° (perpendicular traffic) -> abs(cos(pi/2)) = 0.0 (Low similarity).
        #   Vehicles traveling perpendicular to each other cross paths momentarily and do not share
        #   continuous road-axis cooperative observability.
        hu = math.radians(meta_u["heading"] * 0.1)
        hv = math.radians(meta_v["heading"] * 0.1)
        f_heading = abs(math.cos(hu - hv))

        # 3. Temporal overlap factor
        # Requires observations to be simultaneous. Falls to 0.0 if the time difference exceeds
        # temporal_tolerance_s.
        dt_s = abs(meta_u["timestamp"] - meta_v["timestamp"]) / 1e9
        if dt_s > self.temporal_tolerance_s:
            f_time = 0.0
        else:
            f_time = 1.0 - (dt_s / self.temporal_tolerance_s)

        # 4. Lane factor
        # Rewards proximity in adjacent/same-lane structures to boost confidence for local interactions.
        f_lane = 1.0
        if meta_u["lane"] is not None and meta_v["lane"] is not None:
            if meta_u["lane"] == meta_v["lane"]:
                f_lane = 1.0
            elif abs(meta_u["lane"] - meta_v["lane"]) == 1:
                f_lane = 0.9
            else:
                f_lane = 0.7

        # 5. Line-of-Sight factor (LOS Factor)
        # Evaluates whether a third vehicle (node k) is physically located on the segment connecting 
        # u and v within los_threshold_m. If so, a confidence reduction (LOS factor = 0.70) is applied.
        f_los = 1.0
        for k in self.graph.nodes:
            if k == u or k == v:
                continue
            meta_k = self.graph.node_metadata[k]
            dist_to_segment = self._point_to_segment_distance(
                meta_u["lat"], meta_u["lon"],
                meta_v["lat"], meta_v["lon"],
                meta_k["lat"], meta_k["lon"]
            )
            if dist_to_segment < self.los_threshold_m:
                if self._is_point_between(
                    meta_u["lat"], meta_u["lon"],
                    meta_v["lat"], meta_v["lon"],
                    meta_k["lat"], meta_k["lon"]
                ):
                    f_los *= 0.7

        # 6. RSU Visibility boost
        # Applies a positive infrastructure multiplier (e.g. 1.15) if either node is an RSU 
        # or if both nodes are within range of a shared RSU, confirming ground truth visibility.
        rsu_boost_val = 1.0
        is_u_rsu = meta_u["station_type"] == 15
        is_v_rsu = meta_v["station_type"] == 15
        if is_u_rsu or is_v_rsu:
            rsu_boost_val = 1.0 + self.rsu_boost
        else:
            for k in self.graph.nodes:
                meta_k = self.graph.node_metadata[k]
                if meta_k["station_type"] == 15:
                    d_uk = self._haversine_m(meta_u["lat"], meta_u["lon"], meta_k["lat"], meta_k["lon"])
                    d_vk = self._haversine_m(meta_v["lat"], meta_v["lon"], meta_k["lat"], meta_k["lon"])
                    if d_uk <= comm_range and d_vk <= comm_range:
                        rsu_boost_val = 1.0 + self.rsu_boost
                        break

        # Strictly Multiplicative Edge Weight Formulation
        w = f_dist * f_comm * f_heading * f_time * f_lane * f_los * rsu_boost_val
        w = max(0.0, min(1.0, w))

        factors = {
            "distance": f_dist,
            "communication": f_comm,
            "heading": f_heading,
            "temporal": f_time,
            "lane": f_lane,
            "line_of_sight": f_los,
            "rsu_visibility_multiplier": rsu_boost_val,
        }

        key = (u, v) if u < v else (v, u)
        if w > 0.005:
            self.graph.edges[key] = ObservabilityEdge(
                u=u,
                v=v,
                weight=w,
                confidence=edge_confidence,
                factors=factors,
                wall_time=wall_time,
            )
        else:
            self.graph.edges.pop(key, None)

    @staticmethod
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

    @staticmethod
    def _point_to_segment_distance(
        lat_u: float, lon_u: float,
        lat_v: float, lon_v: float,
        lat_p: float, lon_p: float
    ) -> float:
        lat_scale = 111_139.0
        lon_scale = 111_139.0 * math.cos(math.radians(lat_u * 1e-7))

        dx = (lon_v - lon_u) * 1e-7 * lon_scale
        dy = (lat_v - lat_u) * 1e-7 * lat_scale

        px = (lon_p - lon_u) * 1e-7 * lon_scale
        py = (lat_p - lat_u) * 1e-7 * lat_scale

        l2 = dx * dx + dy * dy
        if l2 == 0.0:
            return math.sqrt(px * px + py * py)

        t = max(0.0, min(1.0, (px * dx + py * dy) / l2))
        proj_x = t * dx
        proj_y = t * dy

        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    @staticmethod
    def _is_point_between(
        lat_u: float, lon_u: float,
        lat_v: float, lon_v: float,
        lat_p: float, lon_p: float
    ) -> bool:
        lat_scale = 111_139.0
        lon_scale = 111_139.0 * math.cos(math.radians(lat_u * 1e-7))

        dx = (lon_v - lon_u) * 1e-7 * lon_scale
        dy = (lat_v - lat_u) * 1e-7 * lat_scale

        px = (lon_p - lon_u) * 1e-7 * lon_scale
        py = (lat_p - lat_u) * 1e-7 * lat_scale

        l2 = dx * dx + dy * dy
        if l2 == 0.0:
            return False

        t = (px * dx + py * dy) / l2
        return 0.0 <= t <= 1.0
