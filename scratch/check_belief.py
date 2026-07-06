import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from b2_csia.uncertainty import Provenance
from b2_csia.behavior_profile import BehaviorEvidence
from b2_csia.behavior_reasoning import BehavioralReasoningEngine

prov = Provenance(modules=set(["spatial"]), min_evidence_quality=0.9, min_confidence=0.8)

evidence_benign = BehaviorEvidence(
    spatial_similarity=0.2,
    temporal_similarity=0.1,
    kinematic_similarity=0.2,
    semantic_similarity=0.5,
    graph_connectivity=0.1,
    identity_consistency=1.0,
    rsu_corroboration=1.0,
    historical_trust=1.0,
    confidence=0.9,
    provenance=prov
)

for rule in ["yager", "dempster", "murphy"]:
    engine = BehavioralReasoningEngine(fusion_rule=rule)
    assessment = engine.evaluate(evidence_benign, reliability_alpha=0.9)
    print(f"Rule: {rule:10s} -> Match: {assessment.attack_type:10s}, belief: {assessment.belief:.4f}, disbelief: {assessment.disbelief:.4f}, uncertainty: {assessment.uncertainty:.4f}")
