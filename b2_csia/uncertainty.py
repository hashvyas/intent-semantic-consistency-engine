"""
b2_csia/uncertainty.py
======================
Formal Uncertainty Representation & Evidence Fusion.

Implements Dempster–Shafer Theory of Evidence to represent belief objects
possessing explicit trust (belief), distrust (disbelief), and uncertainty (ignorance),
and combines evidence using Dempster's, Yager's, Murphy's, or Dubois-Prade's rule
of combination. Features reliability discounting, provenance tracking, and rolling
conflict history tracking.
"""

from __future__ import annotations

import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


@dataclass(frozen=True)
class Provenance:
    """Tracks the origin and evidence quality of a MassFunction.

    Parameters
    ----------
    modules : Set[str]
        Set of modules that contributed to this belief.
    min_evidence_quality : float
        The lowest evidence quality score ∈ [0.0, 1.0] in the combination history.
    min_confidence : float
        The lowest confidence value ∈ [0.0, 1.0] in the combination history.
    creation_time : float
        Unix wall timestamp of creation.
    """

    modules: Set[str]
    min_evidence_quality: float
    min_confidence: float
    creation_time: float = field(default_factory=time.time)

    def merge(self, other: Provenance) -> Provenance:
        """Merge two provenances by taking union of modules and minimum values."""
        return Provenance(
            modules=self.modules.union(other.modules),
            min_evidence_quality=min(self.min_evidence_quality, other.min_evidence_quality),
            min_confidence=min(self.min_confidence, other.min_confidence),
            creation_time=min(self.creation_time, other.creation_time),
        )


class MassFunction:
    """Represents a basic probability assignment (mass function) under Dempster-Shafer theory.

    Focal elements:
      - 'A': Benign (Trust)
      - 'not_A': Suspicious (Distrust)
      - 'Theta': Uncertain (Ignorance)
    """

    def __init__(
        self,
        m_A: float,
        m_not_A: float,
        m_Theta: float,
        provenance: Optional[Provenance] = None,
    ) -> None:
        # Normalize focal elements to sum to exactly 1.0
        total = m_A + m_not_A + m_Theta
        if total <= 0:
            self.m_A = 0.0
            self.m_not_A = 0.0
            self.m_Theta = 1.0
        else:
            self.m_A = m_A / total
            self.m_not_A = m_not_A / total
            self.m_Theta = m_Theta / total

        # Setup provenance tracking
        if provenance is None:
            self.provenance = Provenance(
                modules=set(["unknown"]),
                min_evidence_quality=1.0,
                min_confidence=1.0,
                creation_time=time.time(),
            )
        else:
            self.provenance = provenance

    @classmethod
    def from_trust_confidence(
        cls,
        trust: float,
        confidence: float,
        origin_module: str = "unknown",
        evidence_quality: float = 1.0,
    ) -> MassFunction:
        """Create a MassFunction from trust, confidence, and quality metrics.

        Parameters
        ----------
        trust : float
            Trust probability ∈ [0.0, 1.0].
        confidence : float
            Confidence value ∈ [0.0, 1.0].
        origin_module : str
            Name of the generating module (for provenance).
        evidence_quality : float
            The estimated quality score of the source message/data.
        """
        trust = max(0.0, min(1.0, trust))
        confidence = max(0.0, min(1.0, confidence))

        m_A = trust * confidence
        m_not_A = (1.0 - trust) * confidence
        m_Theta = 1.0 - confidence

        prov = Provenance(
            modules=set([origin_module]),
            min_evidence_quality=evidence_quality,
            min_confidence=confidence,
            creation_time=time.time(),
        )
        return cls(m_A, m_not_A, m_Theta, prov)

    @property
    def belief(self) -> float:
        """Belief of Benign = m(A)."""
        return self.m_A

    @property
    def disbelief(self) -> float:
        """Disbelief of Benign (Distrust) = m(not_A)."""
        return self.m_not_A

    @property
    def uncertainty(self) -> float:
        """Uncertainty (Ignorance) = m(Theta)."""
        return self.m_Theta

    @property
    def plausibility(self) -> float:
        """Plausibility of Benign = m(A) + m(Theta) = 1 - m(not_A)."""
        return self.m_A + self.m_Theta

    def discount(self, reliability: float) -> MassFunction:
        """Applies Shafer's reliability discounting to this MassFunction.

        Parameters
        ----------
        reliability : float
            The reliability coefficient α ∈ [0.0, 1.0] (often derived from EvidenceQuality).
        """
        alpha = max(0.0, min(1.0, reliability))

        # Shafer Discounting Formulations:
        # m_discounted(A) = alpha * m(A)
        # m_discounted(not_A) = alpha * m(not_A)
        # m_discounted(Theta) = 1 - alpha + alpha * m(Theta)
        new_m_A = alpha * self.m_A
        new_m_not_A = alpha * self.m_not_A
        new_m_Theta = 1.0 - alpha + alpha * self.m_Theta

        # Update provenance to reflect this discounting step
        new_prov = Provenance(
            modules=self.provenance.modules,
            min_evidence_quality=min(self.provenance.min_evidence_quality, alpha),
            min_confidence=self.provenance.min_confidence,
            creation_time=self.provenance.creation_time,
        )

        return MassFunction(new_m_A, new_m_not_A, new_m_Theta, new_prov)

    def to_dict(self) -> Dict[str, Any]:
        """Convert mass function and its provenance into a dictionary."""
        return {
            "trust_belief": self.m_A,
            "distrust_disbelief": self.m_not_A,
            "uncertainty_ignorance": self.m_Theta,
            "plausibility": self.plausibility,
            "provenance": {
                "modules": list(self.provenance.modules),
                "min_evidence_quality": self.provenance.min_evidence_quality,
                "min_confidence": self.provenance.min_confidence,
                "creation_time": self.provenance.creation_time,
            },
        }


# ===========================================================================
# Evidence combination rules
# ===========================================================================

def combine_dempster(m1: MassFunction, m2: MassFunction) -> Tuple[MassFunction, float]:
    """Combines two mass functions using Dempster's Rule of Combination.

    Normalizes the joint mass to distribute conflict.
    """
    K = m1.m_A * m2.m_not_A + m1.m_not_A * m2.m_A

    if K >= 0.999:
        # Total contradiction. Fallback to Yager combination to preserve safety
        return combine_yager(m1, m2)

    normalization = 1.0 - K
    m_A = (m1.m_A * m2.m_A + m1.m_A * m2.m_Theta + m1.m_Theta * m2.m_A) / normalization
    m_not_A = (m1.m_not_A * m2.m_not_A + m1.m_not_A * m2.m_Theta + m1.m_Theta * m2.m_not_A) / normalization
    m_Theta = (m1.m_Theta * m2.m_Theta) / normalization

    merged_prov = m1.provenance.merge(m2.provenance)
    return MassFunction(m_A, m_not_A, m_Theta, merged_prov), K


def combine_yager(m1: MassFunction, m2: MassFunction) -> Tuple[MassFunction, float]:
    """Combines two mass functions using Yager's Rule of Combination.

    Assigns all conflict K to the uncertainty element 'Theta' rather than normalizing.
    Note: Under binary frame of discernment {A, not_A}, the Dubois-Prade rule 
    assigns conflict to union (A ∪ not_A = Theta), which is mathematically 
    equivalent to Yager's rule.
    """
    K = m1.m_A * m2.m_not_A + m1.m_not_A * m2.m_A

    m_A = m1.m_A * m2.m_A + m1.m_A * m2.m_Theta + m1.m_Theta * m2.m_A
    m_not_A = m1.m_not_A * m2.m_not_A + m1.m_not_A * m2.m_Theta + m1.m_Theta * m2.m_not_A
    m_Theta = m1.m_Theta * m2.m_Theta + K

    merged_prov = m1.provenance.merge(m2.provenance)
    return MassFunction(m_A, m_not_A, m_Theta, merged_prov), K


def combine_murphy(masses: List[MassFunction]) -> Tuple[MassFunction, float]:
    """Combines a list of mass functions using Murphy's Combination Rule.

    Averages the mass functions, then combines the average function with
    itself N-1 times using Dempster's rule. Excellent for high-conflict scenarios.
    """
    if not masses:
        return MassFunction(1.0, 0.0, 0.0), 0.0
    N = len(masses)
    if N == 1:
        return masses[0], 0.0

    # 1. Compute Simple Average Mass Function
    avg_m_A = sum(m.m_A for m in masses) / N
    avg_m_not_A = sum(m.m_not_A for m in masses) / N
    avg_m_Theta = sum(m.m_Theta for m in masses) / N

    # Merge all provenances
    prov = masses[0].provenance
    for m in masses[1:]:
        prov = prov.merge(m.provenance)

    m_avg = MassFunction(avg_m_A, avg_m_not_A, avg_m_Theta, prov)

    # 2. Combine average mass function with itself N-1 times using Dempster's rule
    current = m_avg
    total_conflict = 0.0
    for _ in range(N - 1):
        current, K = combine_dempster(current, m_avg)
        total_conflict += K

    avg_conflict = total_conflict / (N - 1) if N > 1 else 0.0
    return current, avg_conflict


# ===========================================================================
# BeliefFusionEngine
# ===========================================================================

class BeliefFusionEngine:
    """Manages multi-source evidence combination and conflict history tracking.

    Parameters
    ----------
    fusion_rule : str
        The fusion strategy to use ('dempster', 'yager', 'murphy', 'dubois_prade').
    conflict_history_len : int
        Maximum size of the conflict history ring buffer.
    """

    def __init__(self, fusion_rule: str = "yager", conflict_history_len: int = 50) -> None:
        self.fusion_rule = fusion_rule.lower().strip()
        self.conflict_history = deque(maxlen=conflict_history_len)

    def fuse(self, masses: List[MassFunction]) -> Tuple[MassFunction, float]:
        """Fuse a list of MassFunctions using the configured fusion rule.

        Parameters
        ----------
        masses : List[MassFunction]
            The list of belief mass functions.

        Returns
        -------
        combined_belief : MassFunction
            The unified belief state.
        conflict : float
            The conflict value(s) encountered.
        """
        if not masses:
            return MassFunction(1.0, 0.0, 0.0), 0.0
        if len(masses) == 1:
            return masses[0], 0.0

        if self.fusion_rule == "murphy":
            combined, K = combine_murphy(masses)
            self.conflict_history.append(K)
            return combined, K

        # Sequential combination loop for Dempster, Yager, and Dubois-Prade
        # (Dubois-Prade is equivalent to Yager under binary frame)
        combine_func = combine_yager if self.fusion_rule in ("yager", "dubois_prade") else combine_dempster

        current = masses[0]
        total_conflict = 0.0
        steps = 0

        for next_mass in masses[1:]:
            current, K = combine_func(current, next_mass)
            total_conflict += K
            steps += 1

        avg_conflict = total_conflict / steps if steps > 0 else 0.0
        self.conflict_history.append(avg_conflict)

        return current, avg_conflict
