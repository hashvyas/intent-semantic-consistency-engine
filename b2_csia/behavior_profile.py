"""
b2_csia/behavior_profile.py
===========================
Behavior Evidence & Attack Profiles.

Defines the strongly-typed BehaviorEvidence data model and declarative attack profiles.
Enables matching extracted evidence against profiles using distance metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from b2_csia.uncertainty import Provenance


@dataclass(frozen=True)
class BehaviorEvidence:
    """Strongly-typed data model capturing all extracted evidence dimensions.

    Parameters
    ----------
    spatial_similarity : float
        Proximity score in [0.0, 1.0].
    temporal_similarity : float
        Transmission timestamp synchronization in [0.0, 1.0].
    kinematic_similarity : float
        Kinematic profile alignment score in [0.0, 1.0].
    semantic_similarity : float
        Correlation of message type metadata in [0.0, 1.0].
    graph_connectivity : float
        Observability graph edge density in [0.0, 1.0].
    identity_consistency : float
        Uniqueness / diversity of node identities (low = duplicates/clones) in [0.0, 1.0].
    rsu_corroboration : float
        RSU observation confirmation score in [0.0, 1.0].
    historical_trust : float
        Prior average trust score of senders in [0.0, 1.0].
    confidence : float
        Unified extraction confidence rating in [0.0, 1.0].
    provenance : Provenance
        Provenance metadata of the source elements.
    """

    spatial_similarity: float
    temporal_similarity: float
    kinematic_similarity: float
    semantic_similarity: float
    graph_connectivity: float
    identity_consistency: float
    rsu_corroboration: float
    historical_trust: float
    confidence: float
    provenance: Provenance
    validation_score: float = 1.0
    validation_confidence: float = 1.0

    def get_value(self, feature_name: str) -> float:
        """Helper to dynamically fetch a feature value by its string name."""
        val = getattr(self, feature_name, None)
        if isinstance(val, (int, float)):
            return float(val)
        return 0.5


class AttackProfile:
    """Declarative template representing expectations of specific attack vectors.

    Parameters
    ----------
    name : str
        Target attack type (e.g. 'sybil', 'replay', 'collusion', 'fabrication').
    targets : Dict[str, str]
        Mappings of feature name -> target ('low', 'medium', 'high', 'any').
    """

    def __init__(self, name: str, targets: Dict[str, str]) -> None:
        self.name = name
        self.targets = {k: v.lower().strip() for k, v in targets.items()}

    def get_target_value(self, feature_name: str) -> Optional[float]:
        """Convert categorical targets into numeric values for distance comparison."""
        target = self.targets.get(feature_name, "any")
        if target == "high":
            return 0.9
        elif target == "medium":
            return 0.5
        elif target == "low":
            return 0.1
        return None  # 'any' is ignored

    def match_similarity(self, evidence: BehaviorEvidence) -> float:
        """Compute structural matching similarity ∈ [0.0, 1.0] between evidence and profile."""
        squared_diff = 0.0
        count = 0
        for feature in [
            "spatial_similarity",
            "temporal_similarity",
            "kinematic_similarity",
            "semantic_similarity",
            "graph_connectivity",
            "identity_consistency",
            "rsu_corroboration",
            "historical_trust",
        ]:
            target_val = self.get_target_value(feature)
            if target_val is not None:
                x = evidence.get_value(feature)
                squared_diff += (x - target_val) ** 2
                count += 1

        if count == 0:
            return 1.0

        rmse = math.sqrt(squared_diff / count)
        # Convert distance to similarity score
        return float(max(0.0, min(1.0, 1.0 - rmse)))


class AttackProfileRegistry:
    """Registry loading and holding declarative attack profiles."""

    def __init__(self) -> None:
        self._profiles: Dict[str, AttackProfile] = {}
        self._load_default_profiles()

    def register(self, profile: AttackProfile) -> None:
        """Register an attack profile."""
        self._profiles[profile.name] = profile

    def get(self, name: str) -> Optional[AttackProfile]:
        """Fetch a registered profile by name."""
        return self._profiles.get(name)

    def get_all(self) -> List[AttackProfile]:
        """Get all registered profiles sorted by name."""
        return [self._profiles[name] for name in sorted(self._profiles.keys())]

    def _load_default_profiles(self) -> None:
        """Seed the registry with standard V2X coordinated attack profiles."""
        # 1. Sybil Profile: Nodes share kinematics, timestamps, graph connections
        # but have highly homogenous identities (low identity consistency/diversity).
        self.register(AttackProfile("sybil", {
            "identity_consistency": "low",
            "temporal_similarity": "high",
            "kinematic_similarity": "high",
            "graph_connectivity": "high",
            "spatial_similarity": "high",
        }))

        # 2. Replay Profile: Messages share identical semantic values, duplicate signatures,
        # and are slightly spread out in time (medium temporal alignment).
        self.register(AttackProfile("replay", {
            "identity_consistency": "low",
            "temporal_similarity": "medium",
            "kinematic_similarity": "high",
            "semantic_similarity": "high",
        }))

        # 3. Collusion Profile: Multiple nodes coordinate to report identical semantic contexts
        # and share high graph edge connections, but have different station_ids.
        self.register(AttackProfile("collusion", {
            "semantic_similarity": "high",
            "graph_connectivity": "high",
            "temporal_similarity": "high",
            "identity_consistency": "high",
        }))

        # 4. Event Fabrication Profile: Fake event reports with low RSU corroboration
        # and low baseline historical trust.
        self.register(AttackProfile("fabrication", {
            "semantic_similarity": "high",
            "rsu_corroboration": "low",
            "historical_trust": "low",
        }))
