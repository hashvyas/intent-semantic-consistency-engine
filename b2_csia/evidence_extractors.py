"""
b2_csia/evidence_extractors.py
==============================
Evidence Extractors.

Provides a pluggable registry of independent extractors that compute behavioral
evidence dimensions (Spatial, Temporal, Kinematic, Semantic, Graph, Identity,
RSU, History) over a message cluster, yielding normalized scores and confidence values.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set, Tuple, Optional, Protocol
from b2_csia.observability_graph import ObservabilityGraph


class EvidenceExtractor(Protocol):
    """Protocol for pluggable evidence extractors."""

    name: str

    def extract(
        self,
        cluster: List[Dict[str, Any]],
        graph: Optional[ObservabilityGraph] = None,
        history_trust_scores: Optional[Dict[int, float]] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Extract a behavioral evidence dimension from the cluster.

        Returns
        -------
        score : float
            Normalized evidence score ∈ [0.0, 1.0].
        confidence : float
            Confidence in this extraction ∈ [0.0, 1.0].
        metadata : Dict[str, Any]
            Explainability key-value pairs.
        """
        ...


class EvidenceExtractorRegistry:
    """Registry managing auto-registration of evidence extractors."""

    def __init__(self) -> None:
        self._extractors: Dict[str, EvidenceExtractor] = {}

    def register(self, extractor: EvidenceExtractor) -> None:
        """Register an evidence extractor."""
        self._extractors[extractor.name] = extractor

    def get_all(self) -> List[EvidenceExtractor]:
        """Get all registered extractors sorted by name."""
        return [self._extractors[name] for name in sorted(self._extractors.keys())]


# ===========================================================================
# Individual Extractor Implementations
# ===========================================================================

class SpatialSimilarityExtractor:
    """Measures spatial proximity of nodes in the cluster."""

    name: str = "spatial_similarity"

    def extract(
        self,
        cluster: List[Dict[str, Any]],
        graph: Optional[ObservabilityGraph] = None,
        history_trust_scores: Optional[Dict[int, float]] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        N = len(cluster)
        if N < 2:
            return 0.0, 1.0, {"info": "Single node cluster"}

        # Extract positions
        lats = []
        lons = []
        for msg in cluster:
            rp = msg.get("cam", {}).get("cam_parameters", {}).get("basic_container", {}).get("reference_position", {})
            if rp:
                lat = rp.get("latitude")
                lon = rp.get("longitude")
                if lat is not None and lon is not None:
                    lats.append(lat)
                    lons.append(lon)

        if len(lats) < 2:
            return 0.0, 0.5, {"warning": "Insufficient spatial position data"}

        # Max pairwise distance in meters
        max_d = 0.0
        for i in range(len(lats)):
            for j in range(i + 1, len(lats)):
                d = self._haversine_m(lats[i], lons[i], lats[j], lons[j])
                if d > max_d:
                    max_d = d

        # Map distance to similarity: closer -> higher similarity
        # Exp decay: similarity = exp(-0.01 * max_d)
        similarity = math.exp(-0.01 * max_d)
        confidence = 1.0 - math.exp(-0.5 * N)  # more samples -> higher confidence

        return float(similarity), float(confidence), {"max_pairwise_distance_m": max_d}

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


class TemporalSynchronizationExtractor:
    """Measures temporal alignment / synchronization of timestamps."""

    name: str = "temporal_similarity"

    def extract(
        self,
        cluster: List[Dict[str, Any]],
        graph: Optional[ObservabilityGraph] = None,
        history_trust_scores: Optional[Dict[int, float]] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        N = len(cluster)
        if N < 2:
            return 0.0, 1.0, {"info": "Single node cluster"}

        timestamps = []
        for msg in cluster:
            ts = msg.get("cam", {}).get("generation_delta_time")
            if ts is not None:
                timestamps.append(ts)

        if len(timestamps) < 2:
            return 0.0, 0.5, {"warning": "Insufficient timestamps"}

        # Calculate time span (spread) in milliseconds
        # generation_delta_time is in ms. If it's absolute, it's ms-epoch.
        spread = max(timestamps) - min(timestamps)

        # Close timestamps -> high synchronization
        # If spread is less than 50 ms, it is a highly synchronized burst
        similarity = 1.0 / (1.0 + (spread / 100.0))  # decay with spread scale of 100ms
        confidence = 1.0 - math.exp(-0.5 * N)

        return float(similarity), float(confidence), {"temporal_spread_ms": spread}


class KinematicSimilarityExtractor:
    """Measures similarity of speed, heading, and acceleration."""

    name: str = "kinematic_similarity"

    def extract(
        self,
        cluster: List[Dict[str, Any]],
        graph: Optional[ObservabilityGraph] = None,
        history_trust_scores: Optional[Dict[int, float]] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        N = len(cluster)
        if N < 2:
            return 0.0, 1.0, {"info": "Single node cluster"}

        speeds = []
        headings = []
        for msg in cluster:
            bvhf = msg.get("cam", {}).get("cam_parameters", {}).get("high_frequency_container", {}).get("basic_vehicle_container_high_frequency", {})
            if bvhf:
                speed = bvhf.get("speed")
                heading = bvhf.get("heading")
                if speed is not None and heading is not None:
                    speeds.append(speed)
                    headings.append(heading)

        if len(speeds) < 2:
            return 0.0, 0.5, {"warning": "Insufficient kinematic details"}

        # Normalize speeds (typical max 50 m/s = 5000 in ETSI units)
        # Normalize headings (0 - 3600 ETSI units)
        speed_var = self._variance(speeds)
        # Heading circular difference variance
        heading_rads = [math.radians(h * 0.1) for h in headings]
        heading_var = self._circular_variance(heading_rads)

        # High variance -> low similarity; low variance -> high similarity (clones)
        # We transform to similarity:
        sim_speed = 1.0 / (1.0 + math.sqrt(speed_var) / 100.0)  # scale speed std of 1 m/s (100 ETSI)
        sim_heading = 1.0 - heading_var  # circular var ranges [0, 1]

        similarity = 0.5 * sim_speed + 0.5 * sim_heading
        confidence = 1.0 - math.exp(-0.5 * N)

        return float(similarity), float(confidence), {
            "speed_standard_deviation_etsi": math.sqrt(speed_var),
            "heading_circular_variance": heading_var,
        }

    @staticmethod
    def _variance(data: List[float]) -> float:
        n = len(data)
        if n < 2:
            return 0.0
        mean = sum(data) / n
        return sum((x - mean) ** 2 for x in data) / (n - 1)

    @staticmethod
    def _circular_variance(angles_rad: List[float]) -> float:
        n = len(angles_rad)
        if n == 0:
            return 0.0
        sum_sin = sum(math.sin(a) for a in angles_rad)
        sum_cos = sum(math.cos(a) for a in angles_rad)
        r = math.sqrt(sum_sin ** 2 + sum_cos ** 2) / n
        return 1.0 - r


class SemanticSimilarityExtractor:
    """Measures correlation of station types and message types."""

    name: str = "semantic_similarity"

    def extract(
        self,
        cluster: List[Dict[str, Any]],
        graph: Optional[ObservabilityGraph] = None,
        history_trust_scores: Optional[Dict[int, float]] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        N = len(cluster)
        if N < 2:
            return 0.0, 1.0, {"info": "Single node cluster"}

        station_types = []
        for msg in cluster:
            st = msg.get("cam", {}).get("cam_parameters", {}).get("basic_container", {}).get("station_type")
            if st is not None:
                station_types.append(st)

        if not station_types:
            return 0.0, 0.5, {"warning": "No station types"}

        # Similarity of station type across senders: if all have identical station types
        type_counts = {}
        for st in station_types:
            type_counts[st] = type_counts.get(st, 0) + 1
        max_type_cnt = max(type_counts.values())
        similarity = max_type_cnt / len(station_types)
        confidence = 1.0 - math.exp(-0.5 * N)

        return float(similarity), float(confidence), {"dominant_station_type_ratio": similarity}


class GraphConnectivityExtractor:
    """Measures average edge weight/connectivity of the cluster in the Observability Graph."""

    name: str = "graph_connectivity"

    def extract(
        self,
        cluster: List[Dict[str, Any]],
        graph: Optional[ObservabilityGraph] = None,
        history_trust_scores: Optional[Dict[int, float]] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        N = len(cluster)
        if N < 2 or graph is None:
            return 0.0, 1.0, {"info": "No graph or single node"}

        station_ids = []
        for msg in cluster:
            sid = msg.get("header", {}).get("station_id")
            if sid is not None:
                station_ids.append(sid)

        if len(station_ids) < 2:
            return 0.0, 0.5, {"warning": "Insufficient station IDs"}

        # Compute average edge weight among station_ids in the graph
        total_w = 0.0
        count = 0
        for i in range(len(station_ids)):
            for j in range(i + 1, len(station_ids)):
                u, v = station_ids[i], station_ids[j]
                key = (u, v) if u < v else (v, u)
                if key in graph.edges:
                    total_w += graph.edges[key].weight
                    count += 1

        similarity = (total_w / count) if count > 0 else 0.0
        confidence = 1.0 - math.exp(-0.5 * N)

        return float(similarity), float(confidence), {
            "graph_edge_count": count,
            "average_graph_observability_weight": similarity,
        }


class IdentityConsistencyExtractor:
    """Measures identical station_id/certificate_id usage (indicator of cloning)."""

    name: str = "identity_consistency"

    def extract(
        self,
        cluster: List[Dict[str, Any]],
        graph: Optional[ObservabilityGraph] = None,
        history_trust_scores: Optional[Dict[int, float]] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        N = len(cluster)
        if N < 2:
            return 1.0, 1.0, {"info": "Single node cluster - consistent identity"}

        station_ids = []
        cert_ids = []
        for msg in cluster:
            sid = msg.get("header", {}).get("station_id")
            cert = msg.get("certificate_id") or msg.get("cert_id")
            if sid is not None:
                station_ids.append(sid)
            if cert is not None:
                cert_ids.append(cert)

        unique_sids = len(set(station_ids))
        unique_certs = len(set(cert_ids))

        # Identity consistency: 1.0 means all are unique, 0.0 means all have identical IDs (bad)
        sid_ratio = (unique_sids / len(station_ids)) if station_ids else 1.0
        cert_ratio = (unique_certs / len(cert_ids)) if cert_ids else 1.0

        # Overall identity diversity. High duplication -> low score
        diversity = min(sid_ratio, cert_ratio)
        confidence = 1.0 - math.exp(-0.5 * N)

        return float(diversity), float(confidence), {
            "unique_station_id_ratio": sid_ratio,
            "unique_certificate_id_ratio": cert_ratio,
        }


class RSUCorroborationExtractor:
    """Measures RSU support or corroboration for nodes in the cluster."""

    name: str = "rsu_corroboration"

    def extract(
        self,
        cluster: List[Dict[str, Any]],
        graph: Optional[ObservabilityGraph] = None,
        history_trust_scores: Optional[Dict[int, float]] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        N = len(cluster)
        if graph is None:
            return 0.0, 0.5, {"info": "No graph available for RSU check"}

        # Find if any active node in the graph is an RSU (station_type=15)
        # Check if RSU is close to the cluster nodes
        cluster_sids = []
        for msg in cluster:
            sid = msg.get("header", {}).get("station_id")
            if sid is not None:
                cluster_sids.append(sid)

        rsu_connected = False
        rsu_id = None
        for rsu_node in graph.nodes:
            meta = graph.node_metadata.get(rsu_node)
            if meta and meta.get("station_type") == 15:
                # Check connection to any cluster node in graph edges
                for cl_sid in cluster_sids:
                    key = (rsu_node, cl_sid) if rsu_node < cl_sid else (cl_sid, rsu_node)
                    if key in graph.edges and graph.edges[key].weight > 0.1:
                        rsu_connected = True
                        rsu_id = rsu_node
                        break
            if rsu_connected:
                break

        score = 1.0 if rsu_connected else 0.0
        confidence = 1.0 if rsu_connected else 0.5

        return float(score), float(confidence), {"rsu_observed": rsu_connected, "rsu_node_id": rsu_id}


class HistoricalTrustExtractor:
    """Measures historical trust values of cluster senders."""

    name: str = "historical_trust"

    def extract(
        self,
        cluster: List[Dict[str, Any]],
        graph: Optional[ObservabilityGraph] = None,
        history_trust_scores: Optional[Dict[int, float]] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        N = len(cluster)
        if history_trust_scores is None:
            return 1.0, 0.5, {"info": "No historical trust records"}

        station_ids = []
        for msg in cluster:
            sid = msg.get("header", {}).get("station_id")
            if sid is not None:
                station_ids.append(sid)

        trust_sum = 0.0
        count = 0
        for sid in station_ids:
            if sid in history_trust_scores:
                trust_sum += history_trust_scores[sid]
                count += 1

        score = (trust_sum / count) if count > 0 else 1.0
        confidence = 0.9 if count > 0 else 0.5

        return float(score), float(confidence), {
            "tracked_history_count": count,
            "average_historical_trust": score,
        }
