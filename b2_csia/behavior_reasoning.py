"""
b2_csia/behavior_reasoning.py
=============================
Behavioral Reasoning Engine.

Fuses behavioral evidence dimensions into structured AttackAssessments
using Dempster-Shafer theory of evidence. Supports profile matching,
Shafer reliability discounting, and explainable decision paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from b2_csia.uncertainty import MassFunction, BeliefFusionEngine, Provenance
from b2_csia.behavior_profile import BehaviorEvidence, AttackProfile, AttackProfileRegistry


@dataclass(frozen=True)
class AttackAssessment:
    """Rich structured assessment of a suspected coordinated attack.

    Parameters
    ----------
    attack_type : str
        The identified attack type (e.g. 'sybil', 'replay', 'none').
    confidence : float
        Overall confidence rating ∈ [0.0, 1.0].
    belief : float
        Belief mass assigned to the attack match.
    disbelief : float
        Disbelief/distrust mass assigned (evidence against).
    uncertainty : float
        Remaining ignorance/uncertainty mass.
    conflict : float
        Degree of conflict K observed during evidence combination.
    matched_profile : str
        Name of the matching declarative profile.
    evidence : BehaviorEvidence
        The extracted behavior evidence object.
    explanation : Dict[str, Any]
        Machine-readable diagnostic explanation structure.
    """

    attack_type: str
    confidence: float
    belief: float
    disbelief: float
    uncertainty: float
    conflict: float
    matched_profile: str
    evidence: BehaviorEvidence
    explanation: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to a serializable dictionary."""
        return {
            "attack_type": self.attack_type,
            "confidence": self.confidence,
            "belief": self.belief,
            "disbelief": self.disbelief,
            "uncertainty": self.uncertainty,
            "conflict": self.conflict,
            "matched_profile": self.matched_profile,
            "evidence": {
                "spatial": self.evidence.spatial_similarity,
                "temporal": self.evidence.temporal_similarity,
                "kinematic": self.evidence.kinematic_similarity,
                "semantic": self.evidence.semantic_similarity,
                "graph": self.evidence.graph_connectivity,
                "identity": self.evidence.identity_consistency,
                "rsu": self.evidence.rsu_corroboration,
                "history": self.evidence.historical_trust,
            },
            "explanation": self.explanation,
        }


class BehavioralReasoningEngine:
    """Generic engine reasoning over evidence using attack profiles and DS theory.

    Parameters
    ----------
    fusion_rule : str
        The Dempster-Shafer combination rule ('yager', 'dempster', 'murphy', 'dubois_prade').
    """

    def __init__(self, fusion_rule: str = "yager") -> None:
        self.fusion_rule = fusion_rule
        self.profile_registry = AttackProfileRegistry()
        self.fusion_engine = BeliefFusionEngine(fusion_rule=fusion_rule)

    def evaluate(
        self,
        evidence: BehaviorEvidence,
        reliability_alpha: float = 1.0,
    ) -> AttackAssessment:
        """Evaluate BehaviorEvidence against all profiles and return the strongest assessment.

        Parameters
        ----------
        evidence : BehaviorEvidence
            The extracted behavior evidence.
        reliability_alpha : float
            Reliability coefficient used to discount incoming evidence.
        """
        profiles = self.profile_registry.get_all()
        if not profiles:
            return self._make_none_assessment(evidence)

        assessments: List[AttackAssessment] = []

        # Combined reliability alpha using validation confidence
        fused_reliability = reliability_alpha * evidence.validation_confidence

        for profile in profiles:
            # 1. Construct MassFunctions for each feature defined in the profile
            feature_masses: List[MassFunction] = []
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
                target = profile.get_target_value(feature)
                if target is not None:
                    val = evidence.get_value(feature)
                    # Feature match: closer to target -> higher match
                    match_score = 1.0 - abs(val - target)

                    # Create MassFunction using standard D-S module
                    m = MassFunction.from_trust_confidence(
                        trust=match_score,
                        confidence=evidence.confidence,
                        origin_module=feature,
                        evidence_quality=evidence.provenance.min_evidence_quality,
                    )

                    # Apply Reliability Discounting step
                    discounted_m = m.discount(fused_reliability)
                    feature_masses.append(discounted_m)

            # 1b. Inject B1 validation score as an independent source of evidence if there are validation anomalies
            if evidence.validation_score < 1.0:
                m_val = MassFunction.from_trust_confidence(
                    trust=1.0 - evidence.validation_score,
                    confidence=evidence.validation_confidence,
                    origin_module="validation",
                    evidence_quality=evidence.provenance.min_evidence_quality,
                )
                discounted_m_val = m_val.discount(fused_reliability)
                feature_masses.append(discounted_m_val)

            if not feature_masses:
                continue

            # 2. Fuse the discounted feature masses
            fused_mass, K = self.fusion_engine.fuse(feature_masses)

            # 3. Build diagnostic explanation
            strongest_indicators = []
            weakest_indicators = []
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
                target = profile.get_target_value(feature)
                if target is not None:
                    val = evidence.get_value(feature)
                    diff = abs(val - target)
                    if diff < 0.15:
                        strongest_indicators.append(feature)
                    elif diff > 0.4:
                        weakest_indicators.append(feature)

            explanation = {
                "strongest_indicators": strongest_indicators,
                "weakest_indicators": weakest_indicators,
                "reliability_discount_applied": reliability_alpha,
                "combination_rule_used": self.fusion_rule,
                "evidence_conflict": K,
                "fused_belief_breakdown": fused_mass.to_dict(),
            }

            # Trust is mapped from fused belief mass (higher belief -> higher match)
            assessments.append(
                AttackAssessment(
                    attack_type=profile.name,
                    confidence=evidence.confidence * reliability_alpha,
                    belief=fused_mass.belief,
                    disbelief=fused_mass.disbelief,
                    uncertainty=fused_mass.uncertainty,
                    conflict=K,
                    matched_profile=profile.name,
                    evidence=evidence,
                    explanation=explanation,
                )
            )

        if not assessments:
            return self._make_none_assessment(evidence)

        # Matched profile is the one with the HIGHEST belief (strongest match)
        # Filter assessments where belief > disbelief and belief >= 0.5 to prevent false matches
        valid_matches = [a for a in assessments if a.belief > a.disbelief and a.belief >= 0.5]
        if not valid_matches:
            return self._make_none_assessment(evidence)

        best_match = max(valid_matches, key=lambda a: a.belief)
        return best_match

    def _make_none_assessment(self, evidence: BehaviorEvidence) -> AttackAssessment:
        explanation = {
            "strongest_indicators": [],
            "weakest_indicators": [],
            "reliability_discount_applied": 1.0,
            "combination_rule_used": self.fusion_rule,
            "evidence_conflict": 0.0,
            "info": "No suspicious attack profile matches the behavior evidence",
        }
        return AttackAssessment(
            attack_type="none",
            confidence=evidence.confidence,
            belief=0.0,
            disbelief=1.0,
            uncertainty=0.0,
            conflict=0.0,
            matched_profile="none",
            evidence=evidence,
            explanation=explanation,
        )
