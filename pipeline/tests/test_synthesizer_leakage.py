"""
pipeline/tests/test_synthesizer_leakage.py
==========================================
Comprehensive unit tests for the A2 requirement: the B2 → B3 synthesizer must
contain no information leakage from B2 (CSIA) reasoning.

All 15 invariants from the A2 specification are covered:

  1.  Trust values never appear.
  2.  Belief values never appear.
  3.  Disbelief values never appear.
  4.  Uncertainty values never appear.
  5.  Confidence values never appear.
  6.  Entropy never appears.
  7.  Replay probability never appears.
  8.  Cluster score never appears.
  9.  Identity consistency never appears.
  10. Objective peer observations remain present.
  11. Objective RSU observations remain present.
  12. Objective local sensor observations remain present.
  13. Contradictory reports remain visible.
  14. Spatial information is preserved.
  15. Temporal information is preserved.
  16. Identical structured inputs produce identical synthesized text.

Each test uses ``SynthesisLeakageValidator.assert_clean()`` so that any
regression immediately produces a descriptive failure message identifying
the exact matched term and character offset.
"""

from __future__ import annotations

import copy
import pytest
from typing import Any, Dict, List, Optional

from pipeline.synthesizer import synthesize_message
from pipeline.leakage_validator import SynthesisLeakageValidator


# ---------------------------------------------------------------------------
# Shared validator instance (stateless, safe to reuse)
# ---------------------------------------------------------------------------
VALIDATOR = SynthesisLeakageValidator()


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_cam_msg(
    station_id: int = 1001,
    station_type: int = 5,
    lat: int = 485512345,
    lon: int = 96123456,
    speed: int = 1500,
    heading: int = 900,
    yaw_rate: int = 0,
    long_accel: int = 0,
    gen_dt: float = 4000.0,
) -> Dict[str, Any]:
    """Construct a minimal well-formed CAM message dictionary."""
    return {
        "header": {
            "station_id": station_id,
            "message_id": 1,
        },
        "cam": {
            "generation_delta_time": gen_dt,
            "cam_parameters": {
                "basic_container": {
                    "station_type": station_type,
                    "reference_position": {
                        "latitude": lat,
                        "longitude": lon,
                    },
                },
                "high_frequency_container": {
                    "basic_vehicle_container_high_frequency": {
                        "speed":   speed,
                        "heading": heading,
                        "yaw_rate": yaw_rate,
                        "longitudinal_acceleration": long_accel,
                    },
                },
            },
        },
    }


def _make_b2_result(
    trust: float = 0.18,
    belief: float = 0.81,
    disbelief: float = 0.12,
    uncertainty: float = 0.07,
    cluster_score: float = 0.42,
    entropy: float = 0.67,
    replay_probability: float = 0.55,
    identity_consistency: float = 0.31,
    confidence: float = 0.90,
) -> Dict[str, Any]:
    """Build a B2 result dict containing all canonical B2-derived fields."""
    return {
        "trust":                trust,
        "belief":               belief,
        "disbelief":            disbelief,
        "uncertainty":          uncertainty,
        "cluster_score":        cluster_score,
        "entropy":              entropy,
        "replay_probability":   replay_probability,
        "identity_consistency": identity_consistency,
        "confidence":           confidence,
        "matched_profile":      "passenger_car",
    }


def _minimal_cluster() -> List[Dict[str, Any]]:
    """Single-message cluster with no peer reports or RSU messages."""
    return [_make_cam_msg()]


def _cluster_with_scene_context(
    peer_reports: Optional[List[Any]] = None,
    rsu_messages: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Single-message cluster with peer reports and RSU messages embedded."""
    msg = _make_cam_msg()
    msg["local_perception"] = {
        "camera": "CLEAR",
        "radar":  "CLEAR",
        "lidar":  "CLEAR",
    }
    msg["scene_context"] = {
        "peer_reports": peer_reports or [],
        "rsu_messages": rsu_messages or [],
    }
    return [msg]


def _cluster_with_peers() -> List[Dict[str, Any]]:
    """Three-message cluster: two peers + one target with scene context."""
    peer1 = _make_cam_msg(station_id=2001, lat=485513000, lon=96124000, speed=1200, gen_dt=3990.0)
    peer2 = _make_cam_msg(station_id=2002, lat=485514000, lon=96125000, speed=800,  gen_dt=3980.0)
    target = _make_cam_msg(station_id=1001)
    target["local_perception"] = {
        "camera": "OBSTACLE_DETECTED",
        "radar":  "OBSTACLE_DETECTED",
        "lidar":  "CLEAR",
    }
    target["scene_context"] = {
        "peer_reports": [
            {
                "station_id": 3001,
                "station_type": 5,
                "event": "no obstacle detected",
                "position": {"latitude": 485512500, "longitude": 96123800},
                "speed": 1400,
            },
            {
                "station_id": 3002,
                "station_type": 5,
                "event": "no obstacle detected",
                "position": {"latitude": 485512600, "longitude": 96124100},
                "speed": 1300,
            },
        ],
        "rsu_messages": [
            {
                "station_id": 9001,
                "event": "no hazard on corridor",
                "position": {"latitude": 485510000, "longitude": 96120000},
            }
        ],
    }
    return [peer1, peer2, target]


# ---------------------------------------------------------------------------
# Invariant 1 — Trust values never appear
# ---------------------------------------------------------------------------

class TestTrustNotLeaked:
    def test_trust_score_absent(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")["text"]
        VALIDATOR.assert_clean(text)

    def test_trust_score_near_zero(self):
        """Even very low trust (would trigger old obstacle_detected=YES) must not leak."""
        b2 = _make_b2_result(trust=0.01)
        text = synthesize_message(_minimal_cluster(), b2, "highway")["text"]
        VALIDATOR.assert_clean(text)

    def test_trust_score_high(self):
        b2 = _make_b2_result(trust=0.99)
        text = synthesize_message(_minimal_cluster(), b2, "urban")["text"]
        VALIDATOR.assert_clean(text)

    def test_no_trust_keyword_in_text(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")["text"]
        assert "trust" not in text.lower(), (
            f"Forbidden word 'trust' found in synthesized text:\n{text}"
        )

    def test_no_trustscore_label(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")["text"]
        assert "TrustScore" not in text
        assert "trust_score" not in text.lower()


# ---------------------------------------------------------------------------
# Invariant 2 — Belief values never appear
# ---------------------------------------------------------------------------

class TestBeliefNotLeaked:
    def test_belief_absent(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(belief=0.999), "urban")["text"]
        VALIDATOR.assert_clean(text)

    def test_no_belief_keyword(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")["text"]
        assert "belief" not in text.lower(), (
            f"Forbidden word 'belief' found in synthesized text:\n{text}"
        )


# ---------------------------------------------------------------------------
# Invariant 3 — Disbelief values never appear
# ---------------------------------------------------------------------------

class TestDisbeliefNotLeaked:
    def test_disbelief_absent(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(disbelief=0.95), "urban")["text"]
        VALIDATOR.assert_clean(text)

    def test_no_disbelief_keyword(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")["text"]
        assert "disbelief" not in text.lower(), (
            f"Forbidden word 'disbelief' found in synthesized text:\n{text}"
        )


# ---------------------------------------------------------------------------
# Invariant 4 — Uncertainty values never appear
# ---------------------------------------------------------------------------

class TestUncertaintyNotLeaked:
    def test_uncertainty_absent(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(uncertainty=0.88), "urban")["text"]
        VALIDATOR.assert_clean(text)

    def test_no_uncertainty_keyword(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")["text"]
        assert "uncertainty" not in text.lower()


# ---------------------------------------------------------------------------
# Invariant 5 — Confidence values never appear
# ---------------------------------------------------------------------------

class TestConfidenceNotLeaked:
    def test_confidence_absent(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(confidence=0.99), "urban")["text"]
        VALIDATOR.assert_clean(text)

    def test_no_confidence_keyword(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")["text"]
        assert "confidence" not in text.lower(), (
            f"Forbidden word 'confidence' found in synthesized text:\n{text}"
        )


# ---------------------------------------------------------------------------
# Invariant 6 — Entropy never appears
# ---------------------------------------------------------------------------

class TestEntropyNotLeaked:
    def test_entropy_absent(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(entropy=0.99), "urban")["text"]
        VALIDATOR.assert_clean(text)

    def test_no_entropy_keyword(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")["text"]
        assert "entropy" not in text.lower()


# ---------------------------------------------------------------------------
# Invariant 7 — Replay probability never appears
# ---------------------------------------------------------------------------

class TestReplayProbabilityNotLeaked:
    def test_replay_probability_absent(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(replay_probability=0.95), "urban")["text"]
        VALIDATOR.assert_clean(text)

    def test_no_replay_keyword(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")["text"]
        assert "replay" not in text.lower()


# ---------------------------------------------------------------------------
# Invariant 8 — Cluster score never appears
# ---------------------------------------------------------------------------

class TestClusterScoreNotLeaked:
    def test_cluster_score_absent(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(cluster_score=0.05), "urban")["text"]
        VALIDATOR.assert_clean(text)

    def test_no_cluster_score_label(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")["text"]
        assert "cluster_score" not in text.lower()
        assert "KinematicSimilarity" not in text


# ---------------------------------------------------------------------------
# Invariant 9 — Identity consistency never appears
# ---------------------------------------------------------------------------

class TestIdentityConsistencyNotLeaked:
    def test_identity_consistency_absent(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(identity_consistency=0.02), "urban")["text"]
        VALIDATOR.assert_clean(text)

    def test_no_identity_consistency_label(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")["text"]
        assert "identity_consistency" not in text.lower()
        assert "IdentityConsistency" not in text


# ---------------------------------------------------------------------------
# Invariant 10 — Objective peer observations remain present
# ---------------------------------------------------------------------------

class TestPeerObservationsPreserved:
    def test_peer_station_id_present(self):
        cluster = _cluster_with_peers()
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        assert "3001" in text, f"Peer station ID 3001 missing from:\n{text}"
        assert "3002" in text, f"Peer station ID 3002 missing from:\n{text}"

    def test_peer_event_present(self):
        cluster = _cluster_with_peers()
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        assert "no obstacle detected" in text.lower(), (
            f"Peer event text missing from:\n{text}"
        )

    def test_peer_position_present(self):
        cluster = _cluster_with_peers()
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        assert "485512500" in text or "485512600" in text, (
            f"Peer position data missing from:\n{text}"
        )

    def test_no_peer_reports_message_when_empty(self):
        cluster = _cluster_with_scene_context(peer_reports=[], rsu_messages=[])
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        assert "No peer reports received" in text

    def test_peer_observations_clean(self):
        cluster = _cluster_with_peers()
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        VALIDATOR.assert_clean(text)

    def test_string_peer_report_included(self):
        cluster = _cluster_with_scene_context(
            peer_reports=["Vehicle OBU-5522 reports debris in lane 2 at 120 m."],
        )
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        assert "OBU-5522" in text, f"String peer report not serialized:\n{text}"
        VALIDATOR.assert_clean(text)


# ---------------------------------------------------------------------------
# Invariant 11 — Objective RSU observations remain present
# ---------------------------------------------------------------------------

class TestRSUObservationsPreserved:
    def test_rsu_station_id_present(self):
        cluster = _cluster_with_peers()
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        assert "9001" in text, f"RSU station ID 9001 missing from:\n{text}"

    def test_rsu_event_present(self):
        cluster = _cluster_with_peers()
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        assert "no hazard on corridor" in text.lower(), (
            f"RSU event text missing from:\n{text}"
        )

    def test_no_rsu_messages_when_empty(self):
        cluster = _cluster_with_scene_context(peer_reports=[], rsu_messages=[])
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        assert "No RSU messages received" in text

    def test_rsu_string_message_included(self):
        cluster = _cluster_with_scene_context(
            rsu_messages=["RSU-7: road surface wet, speed advisory 40 km/h."],
        )
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        assert "RSU-7" in text, f"String RSU message not serialized:\n{text}"
        VALIDATOR.assert_clean(text)

    def test_rsu_observations_clean(self):
        cluster = _cluster_with_peers()
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        VALIDATOR.assert_clean(text)


# ---------------------------------------------------------------------------
# Invariant 12 — Objective local sensor observations remain present
# ---------------------------------------------------------------------------

class TestLocalSensorObservationsPreserved:
    def test_camera_present(self):
        cluster = _cluster_with_peers()
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        assert "camera" in text.lower(), f"Camera observation missing from:\n{text}"
        assert "OBSTACLE_DETECTED" in text

    def test_radar_present(self):
        cluster = _cluster_with_peers()
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        assert "radar" in text.lower(), f"Radar observation missing from:\n{text}"

    def test_lidar_present(self):
        cluster = _cluster_with_peers()
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        assert "lidar" in text.lower(), f"Lidar observation missing from:\n{text}"

    def test_sensor_unknown_when_absent(self):
        cluster = _minimal_cluster()  # no local_perception key
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        assert "UNKNOWN" in text

    def test_sensor_values_clean(self):
        cluster = _cluster_with_peers()
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        VALIDATOR.assert_clean(text)


# ---------------------------------------------------------------------------
# Invariant 13 — Contradictory reports remain visible
# ---------------------------------------------------------------------------

class TestContradictoryReportsVisible:
    """Local sensors report obstacle; peer reports say clear. Both must be present."""

    def _contradiction_cluster(self) -> List[Dict[str, Any]]:
        msg = _make_cam_msg(station_id=1001)
        msg["local_perception"] = {
            "camera": "OBSTACLE_DETECTED",
            "radar":  "OBSTACLE_DETECTED",
            "lidar":  "OBSTACLE_DETECTED",
        }
        msg["scene_context"] = {
            "peer_reports": [
                {"station_id": 4001, "event": "road ahead is clear", "speed": 1500},
                {"station_id": 4002, "event": "no obstacle present", "speed": 1400},
            ],
            "rsu_messages": [
                {"station_id": 8001, "event": "no hazard detected"},
            ],
        }
        return [msg]

    def test_local_detection_visible(self):
        cluster = self._contradiction_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "highway")["text"]
        assert "OBSTACLE_DETECTED" in text, (
            f"Local obstacle detection missing from contradictory scene:\n{text}"
        )

    def test_peer_clear_report_visible(self):
        cluster = self._contradiction_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "highway")["text"]
        assert "road ahead is clear" in text.lower() or "no obstacle present" in text.lower(), (
            f"Peer clear-road report missing from contradictory scene:\n{text}"
        )

    def test_rsu_clear_report_visible(self):
        cluster = self._contradiction_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "highway")["text"]
        assert "no hazard detected" in text.lower(), (
            f"RSU no-hazard report missing from contradictory scene:\n{text}"
        )

    def test_contradiction_scene_is_clean(self):
        cluster = self._contradiction_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "highway")["text"]
        VALIDATOR.assert_clean(text)

    def test_no_explicit_disagreement_count(self):
        """The synthesizer must not emit agreement/disagreement counts."""
        cluster = self._contradiction_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "highway")["text"]
        for forbidden in ("disagree", "agreement", "contradiction", "mismatch", "conflict"):
            assert forbidden not in text.lower(), (
                f"Reasoning word '{forbidden}' found in synthesized text:\n{text}"
            )


# ---------------------------------------------------------------------------
# Invariant 14 — Spatial information is preserved
# ---------------------------------------------------------------------------

class TestSpatialInformationPreserved:
    def test_ego_position_present(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")["text"]
        assert "485512345" in text, f"Ego latitude missing from:\n{text}"
        assert "96123456"  in text, f"Ego longitude missing from:\n{text}"

    def test_peer_position_present(self):
        cluster = _cluster_with_peers()
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        # Cluster peer 1 lat=485513000, peer 2 lat=485514000
        assert "485513000" in text or "485514000" in text, (
            f"Cluster peer position missing from:\n{text}"
        )

    def test_distance_from_ego_present(self):
        """Haversine distances between ego and cluster peers must be present."""
        cluster = _cluster_with_peers()
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        assert "distance=" in text, f"Peer distance field missing from:\n{text}"

    def test_spatial_info_clean(self):
        cluster = _cluster_with_peers()
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        VALIDATOR.assert_clean(text)


# ---------------------------------------------------------------------------
# Invariant 15 — Temporal information is preserved
# ---------------------------------------------------------------------------

class TestTemporalInformationPreserved:
    def test_ego_timestamp_present(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")["text"]
        assert "4000" in text, f"Ego timestamp (gen_dt) missing from:\n{text}"

    def test_cluster_peer_timestamp_present(self):
        cluster = _cluster_with_peers()
        text = synthesize_message(cluster, _make_b2_result(), "urban")["text"]
        # peer1 gen_dt=3990.0, peer2 gen_dt=3980.0
        assert "3990" in text or "3980" in text, (
            f"Cluster peer timestamp missing from:\n{text}"
        )

    def test_temporal_info_clean(self):
        text = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")["text"]
        VALIDATOR.assert_clean(text)


# ---------------------------------------------------------------------------
# Invariant 16 — Identical inputs produce identical outputs (determinism)
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_identical_cluster_produces_identical_text(self):
        cluster = _cluster_with_peers()
        b2 = _make_b2_result()
        result_a = synthesize_message(cluster, b2, "urban")["text"]
        result_b = synthesize_message(cluster, b2, "urban")["text"]
        assert result_a == result_b, "Synthesizer is non-deterministic for identical input."

    def test_different_b2_values_produce_identical_text(self):
        """Changing every B2 value must NOT change the output text."""
        cluster = _cluster_with_peers()
        b2_low  = _make_b2_result(trust=0.01, belief=0.99, disbelief=0.01,
                                   uncertainty=0.00, entropy=0.95, replay_probability=0.90)
        b2_high = _make_b2_result(trust=0.99, belief=0.01, disbelief=0.99,
                                   uncertainty=0.00, entropy=0.01, replay_probability=0.01)
        text_low  = synthesize_message(cluster, b2_low,  "urban")["text"]
        text_high = synthesize_message(cluster, b2_high, "urban")["text"]
        assert text_low == text_high, (
            "Synthesized text differs when only B2 values change — "
            "B2 information is being read by the synthesizer."
        )

    def test_different_context_produces_different_text(self):
        """Context IS a legitimate input; different context should produce different text."""
        cluster = _minimal_cluster()
        b2 = _make_b2_result()
        text_urban   = synthesize_message(cluster, b2, "urban")["text"]
        text_highway = synthesize_message(cluster, b2, "highway")["text"]
        assert text_urban != text_highway, (
            "Context label change did not alter synthesized text."
        )

    def test_repeated_calls_are_stable(self):
        cluster = _minimal_cluster()
        b2 = _make_b2_result()
        texts = [synthesize_message(cluster, b2, "urban")["text"] for _ in range(10)]
        assert len(set(texts)) == 1, "Synthesizer output is not stable across repeated calls."


# ---------------------------------------------------------------------------
# Invariant: empty cluster edge case
# ---------------------------------------------------------------------------

class TestEmptyCluster:
    def test_empty_cluster_returns_no_info_text(self):
        result = synthesize_message([], _make_b2_result(), "urban")
        assert result["text"].startswith("V2X Scene Report: No cooperative")

    def test_empty_cluster_returns_correct_template(self):
        result = synthesize_message([], _make_b2_result(), "urban")
        assert result["template"] == "cooperative_scene_report"

    def test_empty_cluster_sources_empty(self):
        result = synthesize_message([], _make_b2_result(), "urban")
        assert result["sources"] == []

    def test_empty_cluster_clean(self):
        text = synthesize_message([], _make_b2_result(), "urban")["text"]
        VALIDATOR.assert_clean(text)


# ---------------------------------------------------------------------------
# Invariant: return structure
# ---------------------------------------------------------------------------

class TestReturnStructure:
    def test_returns_dict(self):
        result = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")
        assert isinstance(result, dict)

    def test_has_text_key(self):
        result = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")
        assert "text" in result

    def test_has_template_key(self):
        result = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")
        assert result["template"] == "cooperative_scene_report"

    def test_has_sources_key(self):
        result = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")
        assert "sources" in result
        assert isinstance(result["sources"], list)

    def test_text_is_string(self):
        result = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")
        assert isinstance(result["text"], str)

    def test_text_is_non_empty(self):
        result = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban")
        assert len(result["text"]) > 0


# ---------------------------------------------------------------------------
# Validator self-tests — ensure the validator correctly detects known bad text
# ---------------------------------------------------------------------------

class TestValidatorSelfCheck:
    """These tests confirm the validator's own detection logic is correct.
    They are not tests of synthesize_message() — they test SynthesisLeakageValidator.
    """

    def test_clean_text_passes(self):
        result = VALIDATOR.validate(
            "Station 1001 (passengerCar) at position lat=485512345, lon=96123456. "
            "Local sensors: camera=CLEAR, radar=CLEAR, lidar=CLEAR."
        )
        assert result.clean

    def test_trust_score_detected(self):
        result = VALIDATOR.validate("TrustScore=0.92, Belief=0.81")
        assert not result.clean
        categories = {v.category for v in result.violations}
        assert "trust" in categories
        assert "belief" in categories

    def test_disbelief_detected(self):
        result = VALIDATOR.validate("Disbelief=0.12, Uncertainty=0.07")
        assert not result.clean

    def test_entropy_detected(self):
        result = VALIDATOR.validate("TemporalEntropy=0.67")
        assert not result.clean

    def test_replay_probability_detected(self):
        result = VALIDATOR.validate("ReplayProbability=0.55")
        assert not result.clean

    def test_cluster_score_detected(self):
        result = VALIDATOR.validate("KinematicSimilarity=0.42")
        assert not result.clean

    def test_identity_consistency_detected(self):
        result = VALIDATOR.validate("IdentityConsistency=0.31")
        assert not result.clean

    def test_obstacle_alert_detected(self):
        result = VALIDATOR.validate("Obstacle Alert: YES.")
        assert not result.clean

    def test_trust_metadata_section_detected(self):
        result = VALIDATOR.validate("Trust Metadata: TrustScore=0.18.")
        assert not result.clean

    def test_behavioral_evidence_section_detected(self):
        result = VALIDATOR.validate("Behavioral Evidence: KinematicSimilarity=0.42.")
        assert not result.clean

    def test_confidence_keyword_detected(self):
        result = VALIDATOR.validate("confidence=0.90")
        assert not result.clean

    def test_assert_clean_raises_on_violation(self):
        with pytest.raises(AssertionError) as exc_info:
            VALIDATOR.assert_clean("TrustScore=0.91, Belief=0.80")
        assert "LEAKAGE DETECTED" in str(exc_info.value)

    def test_assert_clean_passes_on_clean_text(self):
        VALIDATOR.assert_clean(
            "Station 1001 reports speed=1500, heading=900. "
            "Local camera=CLEAR, radar=CLEAR, lidar=CLEAR."
        )

    def test_violation_has_character_offset(self):
        text = "Station 1001. trust=0.18"
        result = VALIDATOR.validate(text)
        assert not result.clean
        assert any(v.character_offset > 0 for v in result.violations)
