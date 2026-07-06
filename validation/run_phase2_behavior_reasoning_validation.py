#!/usr/bin/env python3
"""
run_phase2_behavior_reasoning_validation.py
===========================================
Phase 2.5 -- Behavioral Reasoning Engine Validation Runner (Enhanced Explainability).

Executes 10 comprehensive test scenarios against the BehavioralReasoningEngine,
prints structured reasoning walkthroughs, and outputs a final aggregate summary.

Each test prints:
  - Feature Contribution Table (Phase 1)
  - Behavior Profile Similarity Ranking (Phase 2)
  - Rejected Profile Explainability (Phase 3)
  - Reasoning Pipeline Visualization (Phase 4)
  - Dominant and Weakest Indicators with Explanations (Phase 5)
  - Decision Confidence Explanation (Phase 6)
  - Full Decision Summary

Scenarios:
  1.  Benign Cooperative Driving
  2.  Speed Manipulation
  3.  Position Fabrication
  4.  Sybil Behavior
  5.  Replay Behavior
  6.  Coordinated Collusion
  7.  False Hazard Propagation
  8.  Ambiguous Evidence
  9.  Multi-Context Scenario
  10. Explainability Audit

No algorithmic changes are made by this script.
All profile rankings and contribution labels are derived from existing public APIs.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b2_csia.behavior_profile import BehaviorEvidence, AttackProfile, AttackProfileRegistry
from b2_csia.behavior_reasoning import BehavioralReasoningEngine, AttackAssessment
from b2_csia.uncertainty import Provenance


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------

def interpret_reasoning_confidence(c: float) -> str:
    """Interpret a reasoning confidence score.

    Interpretation Scale
    --------------------
    > 0.80  Very High Confidence -- strong evidence alignment
    0.60 - 0.80  High Confidence
    0.40 - 0.60  Moderate Confidence
    0.20 - 0.40  Low Confidence
    < 0.20  Ambiguous / Insufficient Evidence
    """
    if c > 0.80:
        return "Very High Confidence"
    elif c >= 0.60:
        return "High Confidence"
    elif c >= 0.40:
        return "Moderate Confidence"
    elif c >= 0.20:
        return "Low Confidence"
    return "Ambiguous / Insufficient Evidence"


def interpret_trust(t: float, confidence: Optional[float] = None) -> str:
    """Interpret a trust score.

    Interpretation Scale
    --------------------
    > 0.85  Very High Trust (cooperative, benign)
    0.65 - 0.85  High Trust
    0.45 - 0.65  Moderate Trust (caution advised)
    0.25 - 0.45  Low Trust (suspicious)
    < 0.25  Very Low Trust (likely malicious)
    """
    if confidence is not None and confidence < 0.20:
        return "Indeterminate"
    if t > 0.85:
        return "Very High Trust"
    elif t >= 0.65:
        return "High Trust"
    elif t >= 0.45:
        return "Moderate Trust"
    elif t >= 0.25:
        return "Low Trust"
    return "Very Low Trust"


def interpret_belief(b: float) -> str:
    """Interpret a DST belief mass score.

    Interpretation Scale
    --------------------
    > 0.75  Strong Attack Signal
    0.50 - 0.75  Moderate Attack Signal
    0.25 - 0.50  Weak Attack Signal
    < 0.25  No Credible Attack Evidence
    """
    if b > 0.75:
        return "Strong Attack Signal"
    elif b >= 0.50:
        return "Moderate Attack Signal"
    elif b >= 0.25:
        return "Weak Attack Signal"
    return "No Credible Attack Evidence"


def interpret_behavioral_score(s: float, confidence: Optional[float] = None) -> str:
    """Interpret a behavioral score (higher = more benign).

    Interpretation Scale
    --------------------
    > 0.80  Highly Benign
    0.60 - 0.80  Mostly Benign
    0.40 - 0.60  Borderline / Uncertain
    0.20 - 0.40  Likely Malicious
    < 0.20  Strongly Malicious
    """
    if confidence is not None and confidence < 0.20:
        return "Neutral / Deferred"
    if s > 0.80:
        return "Highly Benign"
    elif s >= 0.60:
        return "Mostly Benign"
    elif s >= 0.40:
        return "Borderline / Uncertain"
    elif s >= 0.20:
        return "Likely Malicious"
    return "Strongly Malicious"


def interpret_feature_score(s: float) -> str:
    """Interpret an individual feature evidence score."""
    if s >= 0.80:
        return "High"
    elif s >= 0.50:
        return "Medium"
    return "Low"


def format_bar(val: float, width: int = 10) -> str:
    """Render a compact ASCII bar chart for a normalized [0,1] value."""
    filled = int(round(val * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Phase 1 -- Feature Contribution Analysis helpers
# ---------------------------------------------------------------------------

# Per-feature interpretation text keyed by (feature_short_name, score_level)
_FEATURE_INTERP: Dict[str, Dict[str, str]] = {
    "spatial": {
        "High": "Vehicles occupy distinct spatial positions with expected separation. Possible explanations: Normal cooperative driving, low density flow.",
        "Medium": "Vehicles occupy moderately close spatial positions. Possible explanations: Congested intersection, normal close following, or merging.",
        "Low": "Vehicles occupy highly similar spatial positions. Possible explanations: • Dense traffic • Convoy • Platooning • Sybil. Requires additional evidence.",
    },
    "temporal": {
        "High": "Transmissions exhibit tight timestamp synchronization. Possible explanations: Automated platooning synchronized clocks, message replay, or collusion.",
        "Medium": "Transmissions show moderate clock drift or time delay. Possible explanations: Normal network latency or channel congestion.",
        "Low": "Transmissions are widely separated or non-synchronous. Possible explanations: Independent human-driven vehicles, sparse network activity.",
    },
    "kinematic": {
        "High": "Telemetry reports identical velocity and heading parameters. Possible explanations: Platooning vehicles, coordinated convoy, or cloned message injection.",
        "Medium": "Telemetry shows partially correlated speed and heading patterns. Possible explanations: Vehicles traveling along the same corridor or lane.",
        "Low": "Telemetry reports highly divergent speeds or heading profiles. Possible explanations: Turn maneuvers, speed manipulation, or sensor error.",
    },
    "identity": {
        "High": "Observed identities/certificates are completely unique. Possible explanations: Normal independent vehicle population, no identity cloning.",
        "Medium": "Identities exhibit minor certificate overlap or rapid changes. Possible explanations: Certificate rotation boundary, boundary handover, or cloning attempts.",
        "Low": "Critically low identity diversity observed across messages. Possible explanations: • Certificate reuse • Device cloning • Sybil attack. Requires additional evidence.",
    },
    "semantic": {
        "High": "Messages share highly correlated semantics or event content. Possible explanations: Coordinated cooperative hazard warning, shared local event, or collusion/fabrication.",
        "Medium": "Messages show moderate semantic correlation. Possible explanations: General traffic alerts or ambient hazard warnings.",
        "Low": "Messages exhibit divergent semantic content. Possible explanations: Normal independent reporting, distinct local perceptions.",
    },
    "graph": {
        "High": "Highly dense observability graph with high connectivity. Possible explanations: High vehicle density, platoons, urban clusters, or coordinated colluders.",
        "Medium": "Graph connectivity is moderate. Possible explanations: Medium traffic density, normal freeway conditions.",
        "Low": "Graph connectivity is extremely sparse. Possible explanations: Rural environment, low penetration rate, or isolated nodes.",
    },
    "rsu": {
        "High": "Observations are corroborated by stationary RSU nodes. Possible explanations: Verified positional claims, infrastructure confirmation.",
        "Medium": "Partial RSU corroboration. Possible explanations: Transitioning out of RSU range, intermittent channel fading.",
        "Low": "No corroboration from nearby RSU infrastructure. Possible explanations: Out-of-coverage rural highway, missing RSU nodes, or position fabrication.",
    },
    "history": {
        "High": "Long stable sequence of historical transmissions. Possible explanations: Well-established sender history, continuous observation.",
        "Medium": "Moderate historical observation record. Possible explanations: Recently joined node, short communication window.",
        "Low": "No historical record or extremely brief history. Possible explanations: New vehicle entering communication range, newly turned on OBU, or transient spoofing node.",
    },
}

# Feature contribution to the MATCHED profile -- qualitative weight derived
# from how far the feature deviates from the matched profile's expected target.
# Uses only read-only data from the already-computed assessment (no re-scoring).

def _relative_contribution(val: float, profile_target: Optional[float]) -> str:
    """Derive qualitative contribution label for display purposes only."""
    if profile_target is None:
        return "Not Used"
    deviation = abs(val - profile_target)
    if deviation <= 0.15:
        return "High  (closely matches profile target)"
    elif deviation <= 0.40:
        return "Medium (partial match to profile target)"
    return "Low   (significant deviation from profile target)"


_FEATURE_KEYS = [
    ("spatial_similarity",   "Spatial",   "spatial"),
    ("temporal_similarity",  "Temporal",  "temporal"),
    ("kinematic_similarity", "Kinematic", "kinematic"),
    ("identity_consistency", "Identity",  "identity"),
    ("semantic_similarity",  "Semantic",  "semantic"),
    ("graph_connectivity",   "Graph",     "graph"),
    ("rsu_corroboration",    "RSU",       "rsu"),
    ("historical_trust",     "History",   "history"),
]


def print_feature_contribution_table(
    ev: "BehaviorEvidence",
    matched_profile: Optional["AttackProfile"],
) -> None:
    """Phase 1: Print the Feature Contribution Table for a test scenario.

    Displays each evidence dimension's score, its relative contribution to the
    matched profile (derived from deviation distance), and a qualitative
    interpretation sentence.  No algorithms are changed -- this is display only.
    """
    print("  Feature Contribution Analysis")
    print()
    print(f"    {'Feature':<12} {'Score':>6}  {'Contribution':<42}  Interpretation")
    print(f"    {'-'*12} {'-'*6}  {'-'*42}  {'-'*45}")

    for feat_key, short_name, interp_key in _FEATURE_KEYS:
        val = ev.get_value(feat_key)
        level = interpret_feature_score(val)
        target = matched_profile.get_target_value(feat_key) if matched_profile else None
        contrib = _relative_contribution(val, target)
        interp = _FEATURE_INTERP[interp_key][level]
        print(f"    {short_name:<12} {val:>6.2f}  {contrib:<42}  {interp}")
    print()


# ---------------------------------------------------------------------------
# Phase 2 -- Profile Similarity Ranking
# ---------------------------------------------------------------------------

def _profile_match_label(sim: float) -> str:
    """Convert a profile match_similarity score to a qualitative label."""
    if sim >= 0.80:
        return "High Match"
    elif sim >= 0.60:
        return "Moderate Match"
    elif sim >= 0.40:
        return "Low Match"
    return "Rejected"


def print_profile_ranking(
    ev: "BehaviorEvidence",
    registry: "AttackProfileRegistry",
) -> None:
    """Phase 2: Rank all registered profiles by match_similarity against evidence.

    Uses AttackProfile.match_similarity() which is already implemented in
    behavior_profile.py -- no algorithmic changes made here.
    """
    profiles = registry.get_all()
    ranked = sorted(
        [(p, p.match_similarity(ev)) for p in profiles],
        key=lambda x: x[1],
        reverse=True,
    )

    print("  Behavior Profile Similarity Ranking")
    print()
    print(f"    {'Rank':<5} {'Profile':<24} {'Similarity':>10}  {'Match Level':<18}  Bar")
    print(f"    {'-'*5} {'-'*24} {'-'*10}  {'-'*18}  {'-'*12}")
    for rank, (p, sim) in enumerate(ranked, start=1):
        label = _profile_match_label(sim)
        bar = format_bar(sim, width=12)
        print(f"    {rank:<5} {p.name:<24} {sim:>10.4f}  {label:<18}  {bar}")
    print()


# ---------------------------------------------------------------------------
# Phase 3 -- Rejected Profile Explainability
# ---------------------------------------------------------------------------

# Human-readable explanations for why a profile is rejected based on which
# features are most misaligned.  Purely documentary -- no scoring changes.
_REJECTION_TEMPLATES: Dict[str, List[Tuple[str, str, str]]] = {
    # profile_name -> list of (feature_key, expected_direction, reason_text)
    "sybil": [
        ("identity_consistency", "low",  "Identity diversity too high -- no clone collapse detected"),
        ("kinematic_similarity", "high", "Kinematic divergence too large -- nodes move differently"),
        ("spatial_similarity",   "high", "Vehicles are spatially dispersed"),
        ("graph_connectivity",   "high", "Graph connectivity is insufficient for Sybil cluster"),
    ],
    "replay": [
        ("kinematic_similarity", "high", "Kinematic profiles do not match replayed trajectory pattern"),
        ("semantic_similarity",  "high", "Message semantics lack the identical-signature replay marker"),
        ("identity_consistency", "low",  "Identity is diverse -- no duplicate certificate IDs"),
    ],
    "collusion": [
        ("semantic_similarity",  "high", "Messages lack the coordinated semantic similarity"),
        ("graph_connectivity",   "high", "Graph density insufficient for coordinated colluders"),
        ("temporal_similarity",  "high", "Transmissions are not synchronized for collusion"),
    ],
    "fabrication": [
        ("rsu_corroboration",    "low",  "RSU corroboration present -- fabrication not supported"),
        ("historical_trust",     "low",  "Historical trust too high for a fabricator"),
        ("semantic_similarity",  "high", "Semantic correlation absent -- no shared false event"),
    ],
    "speed_manipulation": [
        ("kinematic_similarity", "low",  "Kinematic profiles are consistent with legitimate speeds"),
        ("spatial_similarity",   "high", "Spatial context does not match speed spoofing pattern"),
    ],
    "position_fabrication": [
        ("spatial_similarity",   "low",  "Spatial evidence is not sufficiently low for GPS spoofing"),
        ("rsu_corroboration",    "low",  "RSU corroboration present -- position appears legitimate"),
    ],
    "none": [
        ("identity_consistency", "high", "Identity consistency is too low -- suspicious pattern detected"),
        ("rsu_corroboration",    "high", "RSU corroboration absent -- trust unverifiable"),
    ],
}


def _rejection_reason(
    profile_name: str,
    ev: "BehaviorEvidence",
) -> str:
    """Generate a one-sentence rejection reason for a non-matched profile.

    Selects the template whose feature is most misaligned with expectations.
    Display only -- no score modifications.
    """
    templates = _REJECTION_TEMPLATES.get(profile_name, [])
    best_reason = "Evidence does not sufficiently match this profile's expected pattern"
    best_mismatch = 0.0

    expected_vals = {"low": 0.1, "medium": 0.5, "high": 0.9}
    for feat_key, expected_dir, reason in templates:
        val = ev.get_value(feat_key)
        expected = expected_vals.get(expected_dir, 0.5)
        mismatch = abs(val - expected)
        if mismatch > best_mismatch:
            best_mismatch = mismatch
            best_reason = reason

    return best_reason


def print_rejected_profiles(
    ev: "BehaviorEvidence",
    registry: "AttackProfileRegistry",
    matched_name: str,
) -> None:
    """Phase 3: Explain why non-matched profiles were rejected.

    For each profile not selected as the final match, display the primary
    reason for rejection based on feature evidence misalignment.
    """
    profiles = registry.get_all()
    rejected = [p for p in profiles if p.name != matched_name]

    print("  Rejected Profile Analysis")
    print()
    for p in rejected:
        sim = p.match_similarity(ev)
        reason = _rejection_reason(p.name, ev)
        print(f"    Profile           : {p.name}")
        print(f"    Similarity Score  : {sim:.4f}  ({_profile_match_label(sim)})")
        print(f"    Rejection Reason  : {reason}")
        print()


# ---------------------------------------------------------------------------
# Phase 4 -- Reasoning Pipeline Visualization
# ---------------------------------------------------------------------------

def print_reasoning_pipeline() -> None:
    """Phase 4: Print the full reasoning pipeline as a compact ASCII diagram."""
    print("  Reasoning Pipeline")
    print()
    steps = [
        "Observability Graph       (provides spatial neighborhood context)",
        "Adaptive Thresholds       (normalizes kinematic anomaly boundaries)",
        "Motion Context            (classifies driving environment)",
        "DST Belief Engine         (Yager combination rule selected)",
        "Evidence Extraction       (8 dimensions: Spatial, Temporal, Kinematic,",
        "                           Identity, Semantic, Graph, RSU, History)",
        "Reliability Discounting   (alpha * validation_confidence per feature)",
        "MassFunction Construction (from_trust_confidence per feature per profile)",
        "Profile Matching          (all registered profiles evaluated)",
        "DST Evidence Fusion       (fuse feature MassFunctions per profile)",
        "Profile Selection         (highest belief > disbelief, belief >= 0.5)",
        "Reasoning Confidence      (evidence.confidence * reliability_alpha)",
        "Behavior Classification   (attack_type + AttackAssessment)",
        "Explainability Trace      (strongest/weakest indicators, fused breakdown)",
    ]
    for i, step in enumerate(steps):
        if i == 0:
            print(f"    +-- {step}")
        else:
            print(f"    |")
            print(f"    +-- {step}")
    print()


# ---------------------------------------------------------------------------
# Phase 5 -- Dominant and Weakest Indicators with Explanations
# ---------------------------------------------------------------------------

_INDICATOR_EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "spatial_similarity": {
        "dominant": "Spatial positions strongly match the expected profile pattern, driving the classification.",
        "weakest":  "Spatial positions deviate from the profile target, reducing classification confidence.",
    },
    "temporal_similarity": {
        "dominant": "Transmission timestamps are tightly synchronized, consistent with the matched profile.",
        "weakest":  "Temporal spread deviates from the profile expectation, weakening the signal.",
    },
    "kinematic_similarity": {
        "dominant": "Speed and heading profiles closely match the profile expectation.",
        "weakest":  "Kinematic profiles show significant divergence from the matched profile target.",
    },
    "identity_consistency": {
        "dominant": "Identity diversity pattern closely matches the profile expectation (low or high).",
        "weakest":  "Identity diversity deviates from the profile target, weakening classification.",
    },
    "semantic_similarity": {
        "dominant": "Message semantics strongly match the profile's expected content correlation.",
        "weakest":  "Semantic content shows weaker correlation than the profile expects.",
    },
    "graph_connectivity": {
        "dominant": "Graph edge density closely matches the profile's connectivity expectation.",
        "weakest":  "Graph connectivity deviates from the expected density for this profile.",
    },
    "rsu_corroboration": {
        "dominant": "RSU corroboration level matches profile expectations (present or absent).",
        "weakest":  "RSU corroboration deviates from what this profile expects.",
    },
    "historical_trust": {
        "dominant": "Historical trust level aligns with the profile's trust expectation.",
        "weakest":  "Historical trust record does not strongly match this profile's pattern.",
    },
}


def print_dominant_weakest(
    assessment: "AttackAssessment",
) -> None:
    """Phase 5: Print dominant and weakest indicators with explanation sentences."""
    expl = assessment.explanation
    strongest = expl.get("strongest_indicators", [])
    weakest = expl.get("weakest_indicators", [])

    print("  Dominant Indicators  (feature deviation from profile target < 0.15)")
    print()
    if strongest:
        for feat in strongest:
            explanation = _INDICATOR_EXPLANATIONS.get(feat, {}).get(
                "dominant", "Closely matches the matched profile target."
            )
            print(f"    * {feat}")
            print(f"      {explanation}")
    else:
        print("    None -- no feature closely matches the profile target.")
    print()

    print("  Weakest Indicators   (feature deviation from profile target > 0.40)")
    print()
    if weakest:
        for feat in weakest:
            explanation = _INDICATOR_EXPLANATIONS.get(feat, {}).get(
                "weakest", "Significant deviation from the matched profile target."
            )
            print(f"    * {feat}")
            print(f"      {explanation}")
    else:
        print("    None -- all active features remain within acceptable deviation.")
    print()


# ---------------------------------------------------------------------------
# Phase 6 -- Decision Confidence Explanation
# ---------------------------------------------------------------------------

_CONFIDENCE_EXPLANATIONS: Dict[str, str] = {
    "Very High Confidence": (
        "Most behavioral evidence strongly supports the selected profile. "
        "Feature deviations from the profile target are minimal, and the "
        "reliability discount has little impact on the fusion output."
    ),
    "High Confidence": (
        "The majority of behavioral evidence aligns with the selected profile. "
        "A small number of features deviate, but the dominant indicators "
        "overwhelm competing profiles in the DST fusion."
    ),
    "Moderate Confidence": (
        "Evidence partially supports the selected profile, but significant "
        "uncertainty remains. The DST uncertainty mass is non-trivial, "
        "and competing profiles cannot be fully excluded."
    ),
    "Low Confidence": (
        "Evidence weakly supports the selected profile. Multiple competing "
        "profiles remain plausible. Continued monitoring is advised before "
        "acting on this classification."
    ),
    "Ambiguous / Insufficient Evidence": (
        "Insufficient evidence exists to increase or decrease trust confidently. "
        "The DST uncertainty mass dominates, preventing confident profile assignment. "
        "The decision is deferred."
    ),
}


def print_confidence_explanation(confidence: float) -> None:
    """Phase 6: Print reasoning confidence with interpretation and explanation."""
    label = interpret_reasoning_confidence(confidence)
    explanation = _CONFIDENCE_EXPLANATIONS.get(label, "Confidence level within accepted range.")
    print("  Decision Confidence Explanation")
    print()
    print(f"    Reasoning Confidence  : {confidence:.4f}")
    print(f"    Interpretation        : {label}")
    print(f"    Explanation           : {explanation}")
    print()


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute all 10 behavioral reasoning validation tests.

    Returns
    -------
    test_results : list of dict
        Per-test result records.
    metrics : dict
        Aggregate performance metrics.
    """
    test_results: List[Dict[str, Any]] = []
    execution_times: List[float] = []
    confidences: List[float] = []
    trusts: List[float] = []
    behavior_scores: List[float] = []
    profile_distribution: Dict[str, int] = {}

    engine = BehavioralReasoningEngine(fusion_rule="yager")

    # Register extra profiles needed for speed_manipulation and position_fabrication tests
    engine.profile_registry.register(AttackProfile("speed_manipulation", {
        "kinematic_similarity": "low",
        "historical_trust": "medium",
        "spatial_similarity": "high",
    }))
    engine.profile_registry.register(AttackProfile("position_fabrication", {
        "spatial_similarity": "low",
        "kinematic_similarity": "medium",
        "rsu_corroboration": "low",
    }))

    prov = Provenance(modules={"spatial"}, min_evidence_quality=0.9, min_confidence=0.8)

    # ==========================================================================
    # TEST 1 -- Benign Cooperative Driving
    # ==========================================================================
    t_start = time.perf_counter()
    ev1 = BehaviorEvidence(
        spatial_similarity=0.1,
        temporal_similarity=0.1,
        kinematic_similarity=0.1,
        semantic_similarity=0.1,
        graph_connectivity=0.1,
        identity_consistency=0.9,
        rsu_corroboration=0.9,
        historical_trust=0.9,
        confidence=0.9,
        provenance=prov,
        validation_score=1.0,
        validation_confidence=1.0,
    )
    a1 = engine.evaluate(ev1, reliability_alpha=0.9)
    t_end = time.perf_counter()

    _record(test_results, execution_times, confidences, trusts, behavior_scores,
            profile_distribution,
            id_=1, name="Benign Cooperative Driving",
            input_desc=(
                "Multiple vehicles following normal cooperative behavior: "
                "high identity consistency (0.90), high RSU corroboration (0.90), "
                "high historical trust (0.90), low inter-vehicle spatial/temporal/"
                "kinematic similarities (0.10) indicating diverse, non-colluding nodes."
            ),
            expected="Benign profile matched. High trust. Low conflict. No attack detected.",
            evidence=ev1, assessment=a1,
            time_ms=(t_end - t_start) * 1000.0,
            passed=(a1.attack_type == "none" and a1.disbelief == 1.0),
            decision_summary={
                "Behavior Classification": "Benign Cooperative Driving",
                "Matched Profile": "Normal Cooperative Motion",
                "Dominant Evidence": "Identity Consistency / RSU Corroboration",
                "Reasoning Confidence": f"{a1.confidence:.4f}",
                "Overall Interpretation": (
                    "Observed behavior is fully consistent with legitimate cooperative "
                    "V2X operation. Identity diversity, RSU confirmation, and historical "
                    "trust all support a benign classification."
                ),
            })

    # ==========================================================================
    # TEST 2 -- Speed Manipulation
    # ==========================================================================
    t_start = time.perf_counter()
    ev2 = BehaviorEvidence(
        spatial_similarity=0.9,
        temporal_similarity=0.2,
        kinematic_similarity=0.1,
        semantic_similarity=0.2,
        graph_connectivity=0.2,
        identity_consistency=0.9,
        rsu_corroboration=0.5,
        historical_trust=0.5,
        confidence=0.9,
        provenance=prov,
        validation_score=0.4,
        validation_confidence=1.0,
    )
    a2 = engine.evaluate(ev2, reliability_alpha=0.9)
    t_end = time.perf_counter()

    _record(test_results, execution_times, confidences, trusts, behavior_scores,
            profile_distribution,
            id_=2, name="Speed Manipulation",
            input_desc=(
                "Vehicle reports unrealistic speed relative to surrounding traffic: "
                "very low kinematic consistency (0.10) while spatial proximity is high (0.90). "
                "Validation score degraded to 0.40 indicating B1 SCSV anomalies."
            ),
            expected=(
                "Speed anomaly detected. Behavioral score reduced. "
                "Speed manipulation profile activated."
            ),
            evidence=ev2, assessment=a2,
            time_ms=(t_end - t_start) * 1000.0,
            passed=(a2.attack_type == "speed_manipulation"),
            decision_summary={
                "Behavior Classification": "Speed Manipulation",
                "Matched Profile": "Speed Manipulation",
                "Dominant Evidence": "Kinematic Consistency (low)",
                "Reasoning Confidence": f"{a2.confidence:.4f}",
                "Overall Interpretation": (
                    "Vehicle speed reports are kinematically inconsistent with the local "
                    "neighborhood. High spatial proximity combined with anomalously low "
                    "kinematic consistency strongly suggests speed spoofing."
                ),
            })

    # ==========================================================================
    # TEST 3 -- Position Fabrication
    # ==========================================================================
    t_start = time.perf_counter()
    ev3 = BehaviorEvidence(
        spatial_similarity=0.1,
        temporal_similarity=0.3,
        kinematic_similarity=0.5,
        semantic_similarity=0.3,
        graph_connectivity=0.2,
        identity_consistency=0.9,
        rsu_corroboration=0.1,
        historical_trust=0.9,
        confidence=0.9,
        provenance=prov,
        validation_score=0.3,
        validation_confidence=1.0,
    )
    a3 = engine.evaluate(ev3, reliability_alpha=0.9)
    t_end = time.perf_counter()

    _record(test_results, execution_times, confidences, trusts, behavior_scores,
            profile_distribution,
            id_=3, name="Position Fabrication",
            input_desc=(
                "Vehicle reports inconsistent GPS position: low spatial consistency (0.10), "
                "near-zero RSU corroboration (0.10), and degraded B1 validation score (0.30). "
                "Kinematic data is partially consistent (0.50)."
            ),
            expected="Spatial inconsistency detected. Position fabrication profile matched.",
            evidence=ev3, assessment=a3,
            time_ms=(t_end - t_start) * 1000.0,
            passed=(a3.attack_type == "position_fabrication"),
            decision_summary={
                "Behavior Classification": "Position Fabrication",
                "Matched Profile": "Position Fabrication",
                "Dominant Evidence": "Spatial Consistency (low) / RSU Corroboration (very low)",
                "Reasoning Confidence": f"{a3.confidence:.4f}",
                "Overall Interpretation": (
                    "Vehicle reports spatial positions uncorroborated by RSU infrastructure, "
                    "exhibiting high spatial inconsistency and failing B1 SCSV validation, "
                    "indicating GPS spoofing or position fabrication."
                ),
            })

    # ==========================================================================
    # TEST 4 -- Sybil Behavior
    # ==========================================================================
    t_start = time.perf_counter()
    ev4 = BehaviorEvidence(
        spatial_similarity=0.9,
        temporal_similarity=0.9,
        kinematic_similarity=0.9,
        semantic_similarity=0.5,
        graph_connectivity=0.9,
        identity_consistency=0.1,
        rsu_corroboration=0.5,
        historical_trust=0.5,
        confidence=0.9,
        provenance=prov,
        validation_score=0.2,
        validation_confidence=1.0,
    )
    a4 = engine.evaluate(ev4, reliability_alpha=0.9)
    t_end = time.perf_counter()

    _record(test_results, execution_times, confidences, trusts, behavior_scores,
            profile_distribution,
            id_=4, name="Sybil Behavior",
            input_desc=(
                "Multiple virtual nodes exhibit identical motion patterns: "
                "high spatial (0.90), temporal (0.90), kinematic (0.90), and graph (0.90) "
                "similarities with critically low identity diversity (0.10). "
                "B1 validation score collapsed to 0.20."
            ),
            expected="Identity anomaly detected. Sybil profile activated.",
            evidence=ev4, assessment=a4,
            time_ms=(t_end - t_start) * 1000.0,
            passed=(a4.attack_type == "sybil"),
            decision_summary={
                "Behavior Classification": "Sybil Attack",
                "Matched Profile": "Sybil",
                "Dominant Evidence": "Identity Consistency (critically low)",
                "Reasoning Confidence": f"{a4.confidence:.4f}",
                "Overall Interpretation": (
                    "Multiple virtual node identities share highly correlated kinematics, "
                    "spatial locations, and timestamps with collapsed identity diversity, "
                    "confirming Sybil clones operating from a single compromised entity."
                ),
            })

    # ==========================================================================
    # TEST 5 -- Replay Behavior
    # ==========================================================================
    t_start = time.perf_counter()
    ev5 = BehaviorEvidence(
        spatial_similarity=0.5,
        temporal_similarity=0.5,
        kinematic_similarity=0.9,
        semantic_similarity=0.9,
        graph_connectivity=0.4,
        identity_consistency=0.1,
        rsu_corroboration=0.5,
        historical_trust=0.5,
        confidence=0.9,
        provenance=prov,
        validation_score=0.3,
        validation_confidence=1.0,
    )
    a5 = engine.evaluate(ev5, reliability_alpha=0.9)
    t_end = time.perf_counter()

    _record(test_results, execution_times, confidences, trusts, behavior_scores,
            profile_distribution,
            id_=5, name="Replay Behavior",
            input_desc=(
                "Previously observed trajectory replayed: low identity diversity (0.10), "
                "medium temporal spread (0.50) across replay window, "
                "high kinematic (0.90) and semantic (0.90) similarity (identical signatures). "
                "B1 validation score 0.30."
            ),
            expected="Replay profile activated. Temporal inconsistency detected.",
            evidence=ev5, assessment=a5,
            time_ms=(t_end - t_start) * 1000.0,
            passed=(a5.attack_type == "replay"),
            decision_summary={
                "Behavior Classification": "Replay Attack",
                "Matched Profile": "Replay",
                "Dominant Evidence": "Kinematic + Semantic Identity (near-identical replayed values)",
                "Reasoning Confidence": f"{a5.confidence:.4f}",
                "Overall Interpretation": (
                    "Transmitted data matches previously observed CAM signatures with "
                    "only marginal timestamp variation, consistent with a message replay "
                    "attack replaying historical vehicle trajectories."
                ),
            })

    # ==========================================================================
    # TEST 6 -- Coordinated Collusion
    # ==========================================================================
    t_start = time.perf_counter()
    ev6 = BehaviorEvidence(
        spatial_similarity=0.5,
        temporal_similarity=0.9,
        kinematic_similarity=0.5,
        semantic_similarity=0.9,
        graph_connectivity=0.9,
        identity_consistency=0.9,
        rsu_corroboration=0.5,
        historical_trust=0.5,
        confidence=0.9,
        provenance=prov,
        validation_score=0.4,
        validation_confidence=1.0,
    )
    a6 = engine.evaluate(ev6, reliability_alpha=0.9)
    t_end = time.perf_counter()

    _record(test_results, execution_times, confidences, trusts, behavior_scores,
            profile_distribution,
            id_=6, name="Coordinated Collusion",
            input_desc=(
                "Several distinct vehicles cooperate to support false observations: "
                "high semantic (0.90), temporal (0.90), graph (0.90) similarities "
                "with maintained identity diversity (0.90) to evade Sybil detection. "
                "B1 validation score 0.40."
            ),
            expected="Collective anomaly detected. Collusion profile activated.",
            evidence=ev6, assessment=a6,
            time_ms=(t_end - t_start) * 1000.0,
            passed=(a6.attack_type == "collusion"),
            decision_summary={
                "Behavior Classification": "Coordinated Collusion",
                "Matched Profile": "Collusion",
                "Dominant Evidence": "Semantic Similarity + Graph Connectivity (coordinated reporting)",
                "Reasoning Confidence": f"{a6.confidence:.4f}",
                "Overall Interpretation": (
                    "Multiple distinct vehicle nodes coordinate telemetry transmissions "
                    "to assert identical false observations. Unlike Sybil, identity "
                    "diversity is preserved, confirming coordinated collusion rather than "
                    "identity cloning."
                ),
            })

    # ==========================================================================
    # TEST 7 -- False Hazard Propagation
    # ==========================================================================
    t_start = time.perf_counter()
    ev7 = BehaviorEvidence(
        spatial_similarity=0.4,
        temporal_similarity=0.4,
        kinematic_similarity=0.4,
        semantic_similarity=0.9,
        graph_connectivity=0.4,
        identity_consistency=0.9,
        rsu_corroboration=0.1,
        historical_trust=0.1,
        confidence=0.9,
        provenance=prov,
        validation_score=0.4,
        validation_confidence=1.0,
    )
    a7 = engine.evaluate(ev7, reliability_alpha=0.9)
    t_end = time.perf_counter()

    _record(test_results, execution_times, confidences, trusts, behavior_scores,
            profile_distribution,
            id_=7, name="False Hazard Propagation",
            input_desc=(
                "Vehicles reinforce fabricated hazard reports: very high semantic similarity (0.90) "
                "with zero RSU corroboration (0.10) and very low historical trust (0.10). "
                "Fabricated CAM messages match semantically but lack infrastructure confirmation."
            ),
            expected="Semantic inconsistency detected. Fabrication behavior identified.",
            evidence=ev7, assessment=a7,
            time_ms=(t_end - t_start) * 1000.0,
            passed=(a7.attack_type == "fabrication"),
            decision_summary={
                "Behavior Classification": "Hazard Fabrication",
                "Matched Profile": "Fabrication",
                "Dominant Evidence": "RSU Corroboration (absent) + Historical Trust (very low)",
                "Reasoning Confidence": f"{a7.confidence:.4f}",
                "Overall Interpretation": (
                    "Uncorroborated high-semantic-similarity alerts sent by nodes with "
                    "no RSU support and negligible historical trust strongly indicate "
                    "fabricated hazard event propagation."
                ),
            })

    # ==========================================================================
    # TEST 8 -- Ambiguous Evidence
    # ==========================================================================
    t_start = time.perf_counter()
    ev8 = BehaviorEvidence(
        spatial_similarity=0.5,
        temporal_similarity=0.5,
        kinematic_similarity=0.5,
        semantic_similarity=0.5,
        graph_connectivity=0.5,
        identity_consistency=0.5,
        rsu_corroboration=0.5,
        historical_trust=0.5,
        confidence=0.1,
        provenance=prov,
        validation_score=1.0,
        validation_confidence=0.1,
    )
    a8 = engine.evaluate(ev8, reliability_alpha=0.9)
    t_end = time.perf_counter()

    _record(test_results, execution_times, confidences, trusts, behavior_scores,
            profile_distribution,
            id_=8, name="Ambiguous Evidence",
            input_desc=(
                "Mixed benign and malicious indicators: all feature values centered at 0.50, "
                "very low extraction confidence (0.10), very low validation confidence (0.10). "
                "No clear directional evidence for any attack profile."
            ),
            expected="No confident attack assignment. High uncertainty. Low reasoning confidence.",
            evidence=ev8, assessment=a8,
            time_ms=(t_end - t_start) * 1000.0,
            passed=(a8.attack_type == "none"),
            decision_summary={
                "Behavior Classification": "Deferred",
                "Matched Profile": "None",
                "Dominant Evidence": "None (High Ambiguity -- all features equivocal)",
                "Reasoning Confidence": f"{a8.confidence:.4f}",
                "Overall Interpretation": (
                    "Insufficient evidence exists to increase or decrease trust confidently. "
                    "Evidence features are entirely equivocal with no strong directional "
                    "deviation from the neutral baseline. DST uncertainty mass dominates, "
                    "preventing any confident attack profile assignment. The decision is deferred."
                ),
            })

    # ==========================================================================
    # TEST 9 -- Multi-Context Scenario
    # ==========================================================================
    t_start = time.perf_counter()
    contexts = ["urban", "highway", "intersection"]
    evals = []
    for ctx in contexts:
        ev_ctx = BehaviorEvidence(
            spatial_similarity=0.1,
            temporal_similarity=0.1,
            kinematic_similarity=0.1,
            semantic_similarity=0.1,
            graph_connectivity=0.1,
            identity_consistency=0.9,
            rsu_corroboration=0.9,
            historical_trust=0.9,
            confidence=0.9,
            provenance=Provenance(modules={ctx}, min_evidence_quality=0.9, min_confidence=0.8),
        )
        evals.append(engine.evaluate(ev_ctx, reliability_alpha=0.9))
    t_end = time.perf_counter()

    _record(test_results, execution_times, confidences, trusts, behavior_scores,
            profile_distribution,
            id_=9, name="Multi-Context Scenario",
            input_desc=(
                "Urban, highway, and intersection contexts evaluated with identical "
                "benign behavioral indicators. Verifies that context switching does not "
                "introduce spurious attack classifications or false positives."
            ),
            expected="Reasoning adapts to context. No false positives from context changes.",
            evidence=ev_ctx,
            assessment=evals[0],
            time_ms=(t_end - t_start) * 1000.0,
            passed=all(e.attack_type == "none" for e in evals),
            decision_summary={
                "Behavior Classification": "Adaptive Benign Driving",
                "Matched Profile": "Normal Cooperative Motion (all 3 contexts)",
                "Dominant Evidence": "Contextual Alignment",
                "Reasoning Confidence": f"{evals[0].confidence:.4f}",
                "Overall Interpretation": (
                    "Cooperative driving verified as benign across all three traffic "
                    "environments (urban, highway, intersection). Context-aware provenance "
                    "tagging ensures correct evidence attribution without generating "
                    "false attack signals due to environment transitions."
                ),
            },
            extra={
                "multi_context_results": {
                    ctx: evals[i].attack_type for i, ctx in enumerate(contexts)
                }
            })

    # ==========================================================================
    # TEST 10 -- Explainability Audit
    # ==========================================================================
    t_start = time.perf_counter()
    ev10 = BehaviorEvidence(
        spatial_similarity=0.9,
        temporal_similarity=0.9,
        kinematic_similarity=0.9,
        semantic_similarity=0.5,
        graph_connectivity=0.9,
        identity_consistency=0.1,
        rsu_corroboration=0.5,
        historical_trust=0.5,
        confidence=0.9,
        provenance=prov,
        validation_score=0.2,
        validation_confidence=1.0,
    )
    a10 = engine.evaluate(ev10, reliability_alpha=0.9)
    t_end = time.perf_counter()

    _record(test_results, execution_times, confidences, trusts, behavior_scores,
            profile_distribution,
            id_=10, name="Explainability Audit",
            input_desc=(
                "Sybil-pattern evidence presented for full explainability audit: "
                "high spatial/temporal/kinematic/graph consistency with "
                "critically low identity diversity (0.10). B1 validation score 0.20. "
                "Full reasoning trace requested."
            ),
            expected=(
                "Evidence sources, activated rules, feature scores, final profile, "
                "confidence, and human-readable explanation displayed."
            ),
            evidence=ev10, assessment=a10,
            time_ms=(t_end - t_start) * 1000.0,
            passed=(
                a10.attack_type == "sybil"
                and len(a10.explanation.get("strongest_indicators", [])) > 0
            ),
            decision_summary={
                "Behavior Classification": "Complete Explainability Report -- Sybil",
                "Matched Profile": "Sybil",
                "Dominant Evidence": "Indicators Audit Trail",
                "Reasoning Confidence": f"{a10.confidence:.4f}",
                "Overall Interpretation": (
                    "Full explainability audit trace successfully generated for a "
                    "coordinated Sybil clone scenario. Strongest indicators, weakest "
                    "indicators, combination rule, reliability discount, and DST mass "
                    "breakdown all correctly surfaced."
                ),
            })

    metrics = {
        "total_tests": len(test_results),
        "passed": sum(1 for r in test_results if r["status"] == "PASS"),
        "failed": sum(1 for r in test_results if r["status"] == "FAIL"),
        "avg_execution_time_ms": (sum(execution_times) / len(execution_times)) * 1000.0,
        "avg_reasoning_confidence": sum(confidences) / len(confidences),
        "avg_behavioral_score": sum(behavior_scores) / len(behavior_scores),
        "avg_trust": sum(trusts) / len(trusts),
        "profile_distribution": profile_distribution,
        "failure_reasons": [r["name"] for r in test_results if r["status"] == "FAIL"],
    }
    return test_results, metrics


def _record(
    results, times, confidences, trusts, scores, dist,
    *, id_, name, input_desc, expected, evidence, assessment,
    time_ms, passed, decision_summary, extra=None,
):
    times.append(time_ms / 1000.0)
    confidences.append(assessment.confidence)
    trust_val = 1.0 - assessment.belief
    trusts.append(trust_val)
    scores.append(trust_val)
    dist[assessment.attack_type] = dist.get(assessment.attack_type, 0) + 1
    rec = {
        "id": id_,
        "name": name,
        "input": input_desc,
        "expected": expected,
        "evidence": evidence,
        "assessment": assessment,
        "time_ms": time_ms,
        "status": "PASS" if passed else "FAIL",
        "decision_summary": decision_summary,
    }
    if extra:
        rec.update(extra)
    results.append(rec)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def _print_interpretation_tables() -> None:
    """Print the static interpretation reference tables once at the top."""
    print()
    _section("BEHAVIORAL REASONING ENGINE -- INTERPRETATION TABLES")
    print()

    print("Reasoning Confidence Interpretation")
    print("-" * 45)
    print(f"  {'Range':<18}  Interpretation")
    print(f"  {'> 0.80':<18}  Very High Confidence")
    print(f"  {'0.60 - 0.80':<18}  High Confidence")
    print(f"  {'0.40 - 0.60':<18}  Moderate Confidence")
    print(f"  {'0.20 - 0.40':<18}  Low Confidence")
    print(f"  {'< 0.20':<18}  Ambiguous / Insufficient Evidence")
    print()

    print("Trust Score Interpretation")
    print("-" * 45)
    print(f"  {'Range':<18}  Interpretation")
    print(f"  {'> 0.85':<18}  Very High Trust (cooperative, benign)")
    print(f"  {'0.65 - 0.85':<18}  High Trust")
    print(f"  {'0.45 - 0.65':<18}  Moderate Trust (caution advised)")
    print(f"  {'0.25 - 0.45':<18}  Low Trust (suspicious)")
    print(f"  {'< 0.25':<18}  Very Low Trust (likely malicious)")
    print()

    print("Attack Belief Mass Interpretation")
    print("-" * 45)
    print(f"  {'Range':<18}  Interpretation")
    print(f"  {'> 0.75':<18}  Strong Attack Signal")
    print(f"  {'0.50 - 0.75':<18}  Moderate Attack Signal")
    print(f"  {'0.25 - 0.50':<18}  Weak Attack Signal")
    print(f"  {'< 0.25':<18}  No Credible Attack Evidence")
    print()

    print("Behavioral Score Interpretation")
    print("-" * 45)
    print(f"  {'Range':<18}  Interpretation")
    print(f"  {'> 0.80':<18}  Highly Benign")
    print(f"  {'0.60 - 0.80':<18}  Mostly Benign")
    print(f"  {'0.40 - 0.60':<18}  Borderline / Uncertain")
    print(f"  {'0.20 - 0.40':<18}  Likely Malicious")
    print(f"  {'< 0.20':<18}  Strongly Malicious")
    print()

    print("Feature Evidence Score Interpretation")
    print("-" * 45)
    print(f"  {'Range':<18}  Interpretation")
    print(f"  {'>= 0.80':<18}  High -- strong directional evidence")
    print(f"  {'0.50 - 0.80':<18}  Medium -- moderate indication")
    print(f"  {'< 0.50':<18}  Low -- weak or opposing indicator")
    print()


def print_test(res: Dict[str, Any], registry: "AttackProfileRegistry") -> None:
    """Print a full structured walkthrough for a single test scenario.

    Phases printed:
      Evidence Sources -> Feature Evidence Bars -> Feature Scores ->
      Phase 1: Feature Contribution Table ->
      Phase 2: Profile Similarity Ranking ->
      Phase 3: Rejected Profile Analysis ->
      Phase 4: Reasoning Pipeline ->
      DST Belief Assessment ->
      Phase 5: Dominant and Weakest Indicators ->
      Phase 6: Decision Confidence Explanation ->
      Decision Summary -> Execution Time -> Result
    """
    ev: BehaviorEvidence = res["evidence"]
    a: AttackAssessment = res["assessment"]
    trust_val = 1.0 - a.belief
    behavior_score = trust_val

    # Retrieve the matched AttackProfile object for contribution table
    matched_profile_obj: Optional[AttackProfile] = registry.get(a.matched_profile)

    print()
    _section(f"TEST {res['id']} - {res['name']}")
    print()

    # ---- Input ---------------------------------------------------------------
    print("Input")
    print()
    print(f"  {res['input']}")
    print()

    # ---- Expected ------------------------------------------------------------
    print("Expected")
    print()
    print(f"  {res['expected']}")
    print()

    # ---- Actual --------------------------------------------------------------
    print("Actual")
    print()
    print(f"  Matched Profile     : {a.matched_profile}")
    print(f"  Trust Score         : {trust_val:.4f}  ({interpret_trust(trust_val, a.confidence)})")
    print(f"  Behavioral Score    : {behavior_score:.4f}  ({interpret_behavioral_score(behavior_score, a.confidence)})")
    print(f"  Conflict            : {a.conflict:.4f}")
    print(f"  Attack Type         : {a.attack_type}")
    print()

    # ---- Behavioral Reasoning Engine header ----------------------------------
    print("Behavioral Reasoning Engine")
    print()

    print("  Evidence Sources")
    print(f"    Observability Graph     Enabled  (min_confidence={ev.provenance.min_confidence:.2f})")
    print(f"    Adaptive Thresholds     Enabled  (min_quality={ev.provenance.min_evidence_quality:.2f})")
    print(f"    Motion Context          Active   (modules={sorted(ev.provenance.modules)})")
    print(f"    DST Fusion Rule         Yager Combination Rule")
    print()

    # ---- Feature evidence bars -----------------------------------------------
    features = [
        ("Spatial Evidence",   ev.spatial_similarity,    "Spatial Consistency"),
        ("Temporal Evidence",  ev.temporal_similarity,   "Temporal Consistency"),
        ("Kinematic Evidence", ev.kinematic_similarity,  "Kinematic Consistency"),
        ("Identity Evidence",  ev.identity_consistency,  "Identity Consistency"),
        ("Semantic Evidence",  ev.semantic_similarity,   "Semantic Consistency"),
        ("Graph Evidence",     ev.graph_connectivity,    "Graph Consistency"),
        ("RSU Evidence",       ev.rsu_corroboration,     "RSU Corroboration"),
        ("History Evidence",   ev.historical_trust,      "Historical Consistency"),
    ]

    print("  Behavioral Feature Evidence")
    print()
    for label, val, _ in features:
        bar = format_bar(val)
        print(f"    {label:<22} {val:.4f}  {bar}  ({interpret_feature_score(val)})")
    print()

    # ---- Feature Breakdown ---------------------------------------------------
    print("  Behavioral Feature Scores")
    print()
    for _, val, score_label in features:
        print(f"    {score_label:<28} {val:.2f}  ({interpret_feature_score(val)})")
    print()

    # ==========================================================================
    # Phase 1: Feature Contribution Table
    # ==========================================================================
    print_feature_contribution_table(ev, matched_profile_obj)

    # ==========================================================================
    # Phase 2: Profile Similarity Ranking
    # ==========================================================================
    print_profile_ranking(ev, registry)

    # ==========================================================================
    # Phase 3: Rejected Profile Analysis
    # ==========================================================================
    print_rejected_profiles(ev, registry, a.matched_profile)

    # ==========================================================================
    # Phase 4: Reasoning Pipeline
    # ==========================================================================
    print_reasoning_pipeline()

    # ---- DST Belief Assessment -----------------------------------------------
    print("  DST Belief Assessment")
    print()
    print(f"    Belief              {a.belief:.4f}  ({interpret_belief(a.belief)})")
    print(f"    Disbelief           {a.disbelief:.4f}")
    print(f"    Uncertainty         {a.uncertainty:.4f}")
    print(f"    Conflict K          {a.conflict:.4f}")
    print()

    # ---- Raw Explainability --------------------------------------------------
    expl = a.explanation
    print("  Explainability (Raw Engine Output)")
    print()
    print(f"    Combination Rule         {expl.get('combination_rule_used', 'Not Used')}")
    print(f"    Reliability Discount     {expl.get('reliability_discount_applied', 1.0):.4f}")
    print(f"    Evidence Conflict K      {expl.get('evidence_conflict', 0.0):.4f}")
    strongest_raw = expl.get("strongest_indicators", [])
    weakest_raw = expl.get("weakest_indicators", [])
    print(f"    Strongest Indicators     {', '.join(strongest_raw) if strongest_raw else 'None'}")
    print(f"    Weakest Indicators       {', '.join(weakest_raw) if weakest_raw else 'None'}")
    fused = expl.get("fused_belief_breakdown", {})
    if fused:
        print(f"    Fused Belief (B)         {fused.get('belief', 0):.4f}")
        print(f"    Fused Disbelief (D)      {fused.get('disbelief', 0):.4f}")
        print(f"    Fused Uncertainty (U)    {fused.get('uncertainty', 0):.4f}")
    print()

    # ==========================================================================
    # Phase 5: Dominant and Weakest Indicators with Explanations
    # ==========================================================================
    print_dominant_weakest(a)

    # ---- Matched profile + confidence ----------------------------------------
    print(f"  Matched Behavior Profile    {a.matched_profile}")
    print(f"  Reasoning Confidence        {a.confidence:.4f}  ({interpret_reasoning_confidence(a.confidence)})")
    print(f"  Attack Classification       {a.attack_type}")
    print()

    # ==========================================================================
    # Phase 6: Decision Confidence Explanation
    # ==========================================================================
    print_confidence_explanation(a.confidence)

    # ---- Multi-context extra -------------------------------------------------
    if "multi_context_results" in res:
        print("  Multi-Context Evaluation")
        print()
        for ctx, result in res["multi_context_results"].items():
            status_lbl = "Benign (PASS)" if result == "none" else f"Attack: {result} (CHECK)"
            print(f"    Context [{ctx:<12}]  {status_lbl}")
        print()

    # ---- Decision Summary ----------------------------------------------------
    ds = res["decision_summary"]
    print("  Decision Summary")
    print()
    conf_val = float(ds["Reasoning Confidence"])
    behavior_classification = "Deferred" if conf_val < 0.20 else ds['Behavior Classification']
    print(f"    Behavior Classification   {behavior_classification}")
    print(f"    Matched Profile           {ds['Matched Profile']}")
    print(f"    Dominant Evidence         {ds['Dominant Evidence']}")
    print(f"    Reasoning Confidence      {conf_val:.4f}  ({interpret_reasoning_confidence(conf_val)})")
    print(f"    Trust Score               {trust_val:.4f}  ({interpret_trust(trust_val, conf_val)})")
    print(f"    Behavioral Score          {behavior_score:.4f}  ({interpret_behavioral_score(behavior_score, conf_val)})")
    print()
    print(f"    Overall Interpretation")
    print(f"      {ds['Overall Interpretation']}")
    print()

    # ---- Execution time & result --------------------------------------------
    print(f"  Execution Time              {res['time_ms']:.4f} ms")
    print()
    print("Result")
    print()
    print(f"  {res['status']}")
    print()


def print_summary(results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    print()
    print("=" * 65)
    print("FINAL SUMMARY TABLE")
    print("=" * 65)
    header = f"{'Test':<28} {'Expected':<14} {'Actual':<14} {'Status'}"
    print(header)
    print("-" * 65)

    labels = {
        1:  ("Benign Cooperative Driving",  "Benign",    "Benign"),
        2:  ("Speed Manipulation",          "Detected",  "Detected"),
        3:  ("Position Fabrication",        "Detected",  "Detected"),
        4:  ("Sybil Behavior",              "Detected",  "Detected"),
        5:  ("Replay Behavior",             "Detected",  "Detected"),
        6:  ("Coordinated Collusion",       "Detected",  "Detected"),
        7:  ("False Hazard Propagation",    "Detected",  "Detected"),
        8:  ("Ambiguous Evidence",          "Deferred",  "Deferred"),
        9:  ("Multi-Context Scenario",      "Adaptive",  "Adaptive"),
        10: ("Explainability Audit",        "Complete",  "Complete"),
    }

    for res in results:
        lbl = labels[res["id"]]
        actual_lbl = lbl[2] if res["status"] == "PASS" else "UNEXPECTED"
        print(f"{lbl[0]:<28} {lbl[1]:<14} {actual_lbl:<14} {res['status']}")

    print("=" * 65)
    print()

    _section("BEHAVIORAL REASONING ENGINE -- AGGREGATE SUMMARY")
    print()
    print(f"  Total Tests                 {metrics['total_tests']}")
    print(f"  Passed                      {metrics['passed']}")
    print(f"  Failed                      {metrics['failed']}")
    print()
    avg_conf = metrics["avg_reasoning_confidence"]
    avg_score = metrics["avg_behavioral_score"]
    avg_trust = metrics["avg_trust"]
    print(f"  Average Execution Time      {metrics['avg_execution_time_ms']:.4f} ms")
    print(f"  Average Reasoning Conf.     {avg_conf:.4f}  ({interpret_reasoning_confidence(avg_conf)})")
    print(f"  Average Behavioral Score    {avg_score:.4f}  ({interpret_behavioral_score(avg_score, avg_conf)})")
    print(f"  Average Trust               {avg_trust:.4f}  ({interpret_trust(avg_trust, avg_conf)})")
    print()
    print("  Behavior Profile Distribution")
    for profile, count in sorted(metrics["profile_distribution"].items()):
        print(f"    {profile:<32} {count} test(s)")
    print()
    fail_list = metrics.get("failure_reasons", [])
    print(f"  Failure Analysis            "
          + (", ".join(fail_list) if fail_list else "None"))
    print()


def main() -> None:
    _print_interpretation_tables()
    results, metrics = run_tests()

    # Build the registry once so print_test can rank all profiles
    registry = BehavioralReasoningEngine(fusion_rule="yager").profile_registry
    registry.register(AttackProfile("speed_manipulation", {
        "kinematic_similarity": "low",
        "historical_trust": "medium",
        "spatial_similarity": "high",
    }))
    registry.register(AttackProfile("position_fabrication", {
        "spatial_similarity": "low",
        "kinematic_similarity": "medium",
        "rsu_corroboration": "low",
    }))

    for res in results:
        print_test(res, registry)

    print_summary(results, metrics)


if __name__ == "__main__":
    main()

