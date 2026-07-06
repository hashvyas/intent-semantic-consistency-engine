import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from b2_csia import CSIA
from b2_csia.experimental import ExperimentConfig, AttackScenarioGenerator
from b2_csia.evidence_extractors import (
    SpatialSimilarityExtractor,
    TemporalSynchronizationExtractor,
    KinematicSimilarityExtractor,
    SemanticSimilarityExtractor,
    IdentityConsistencyExtractor,
    GraphConnectivityExtractor
)
from b2_csia.behavior_profile import BehaviorEvidence
from b2_csia.uncertainty import Provenance

csia = CSIA()
generator = AttackScenarioGenerator(seed=101)
config = ExperimentConfig(
    name="test_val",
    attack_type="sybil",
    vehicle_count=5,
    attacker_count=3,
    traffic_density="dense",
    context="urban"
)
messages = generator.generate_scenario(config)

# Get the attacker window
attackers = [m for m in messages if m.get("is_attacker")]
print(f"Number of attacker messages: {len(attackers)}")

spatial_sim, _, _ = SpatialSimilarityExtractor().extract(attackers)
temporal_sim, _, _ = TemporalSynchronizationExtractor().extract(attackers)
kinematic_sim, _, _ = KinematicSimilarityExtractor().extract(attackers)
semantic_sim, _, _ = SemanticSimilarityExtractor().extract(attackers)
identity_consistency, _, _ = IdentityConsistencyExtractor().extract(attackers)

print(f"spatial_sim: {spatial_sim}")
print(f"temporal_sim: {temporal_sim}")
print(f"kinematic_sim: {kinematic_sim}")
print(f"semantic_sim: {semantic_sim}")
print(f"identity_consistency: {identity_consistency}")
