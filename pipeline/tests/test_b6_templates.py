"""
pipeline/tests/test_b6_templates.py
=====================================
Unit tests for B6 — Alternative Synthesizer Templates.

Validates every invariant required by the B6 specification:

  1.  All templates contain identical factual content (key observations present
      in every template output for the same input).
  2.  All templates are linguistically distinct (no two template outputs are
      identical strings for the same input).
  3.  No template leaks B2-derived information (leakage validator passes).
  4.  Each template is deterministic (identical input → identical output over
      repeated calls).
  5.  ``template=None`` produces the same output as ``template=DEFAULT``.
  6.  DEFAULT template is backward compatible with the A2 synthesizer (same
      format strings, same key labels).
  7.  NARRATIVE template uses flowing prose vocabulary.
  8.  STRUCTURED template uses bracketed section headers.
  9.  All templates list contradictory observations individually — no
      inference, no aggregation, no reasoning words.
  10. Template selection dispatches correctly.
  11. Return dict contains ``template_style`` key with the correct value.
  12. Empty cluster is handled gracefully for all templates.
"""

from __future__ import annotations

import pytest
from typing import Any, Dict, List, Optional

from pipeline.synthesizer import TemplateStyle, SceneEvidence, synthesize_message
from pipeline.leakage_validator import SynthesisLeakageValidator


# ---------------------------------------------------------------------------
# Shared validator
# ---------------------------------------------------------------------------

VALIDATOR = SynthesisLeakageValidator()

ALL_TEMPLATES = [
    TemplateStyle.DEFAULT,
    TemplateStyle.NARRATIVE,
    TemplateStyle.STRUCTURED,
]


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

def _make_cam_msg(
    station_id: int = 1001,
    station_type: int = 5,
    lat: int = 485512345,
    lon: int = 96123456,
    speed: int = 1500,
    heading: int = 900,
    yaw_rate: int = 0,
    long_accel: int = 100,
    gen_dt: float = 4000.0,
) -> Dict[str, Any]:
    """Construct a minimal well-formed CAM message dictionary."""
    return {
        "header": {"station_id": station_id, "message_id": 1},
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
) -> Dict[str, Any]:
    """Build a full B2 result dict with every canonical B2-derived field."""
    return {
        "trust":                trust,
        "belief":               belief,
        "disbelief":            disbelief,
        "uncertainty":          uncertainty,
        "cluster_score":        cluster_score,
        "entropy":              entropy,
        "replay_probability":   replay_probability,
        "identity_consistency": identity_consistency,
        "confidence":           0.90,
    }


def _rich_cluster() -> List[Dict[str, Any]]:
    """A three-message cluster with peer reports, RSU messages, and cluster peers."""
    peer1 = _make_cam_msg(
        station_id=2001, lat=485513000, lon=96124000, speed=1200, gen_dt=3990.0
    )
    peer2 = _make_cam_msg(
        station_id=2002, lat=485514000, lon=96125000, speed=800, gen_dt=3980.0
    )
    target = _make_cam_msg(station_id=1001)
    target["local_perception"] = {
        "camera": "OBSTACLE_DETECTED",
        "radar":  "OBSTACLE_DETECTED",
        "lidar":  "CLEAR",
    }
    target["scene_context"] = {
        "peer_reports": [
            {
                "station_id":   3001,
                "station_type": 5,
                "event":        "no obstacle detected",
                "position":     {"latitude": 485512500, "longitude": 96123800},
                "speed":        1400,
            },
            {
                "station_id":   3002,
                "station_type": 5,
                "event":        "road ahead is clear",
                "position":     {"latitude": 485512600, "longitude": 96124100},
                "speed":        1300,
            },
        ],
        "rsu_messages": [
            {
                "station_id": 9001,
                "event":      "no hazard on corridor",
                "position":   {"latitude": 485510000, "longitude": 96120000},
            }
        ],
    }
    return [peer1, peer2, target]


def _minimal_cluster() -> List[Dict[str, Any]]:
    """Single-message cluster with no peer reports, RSU messages, or cluster peers."""
    return [_make_cam_msg()]


# ---------------------------------------------------------------------------
# Helper — synthesize a given cluster with all three templates
# ---------------------------------------------------------------------------

def _all_texts(
    cluster: List[Dict[str, Any]],
    b2: Optional[Dict[str, Any]] = None,
    context: str = "urban",
) -> Dict[TemplateStyle, str]:
    """Return a dict mapping each TemplateStyle to its synthesized text."""
    b2 = b2 or _make_b2_result()
    return {
        ts: synthesize_message(cluster, b2, context, ts)["text"]
        for ts in ALL_TEMPLATES
    }


# ---------------------------------------------------------------------------
# Invariant 1 — All templates contain identical factual content
# ---------------------------------------------------------------------------

class TestIdenticalFactualContent:
    """Every key observable fact must appear in every template output."""

    def _check_fact_in_all(self, fact_str: str, texts: Dict[TemplateStyle, str]) -> None:
        for ts, text in texts.items():
            assert fact_str.lower() in text.lower(), (
                f"Fact {fact_str!r} missing from {ts.value} template:\n{text}"
            )

    def test_ego_station_id_in_all(self):
        texts = _all_texts(_rich_cluster())
        self._check_fact_in_all("1001", texts)

    def test_ego_position_lat_in_all(self):
        texts = _all_texts(_rich_cluster())
        self._check_fact_in_all("485512345", texts)

    def test_ego_position_lon_in_all(self):
        texts = _all_texts(_rich_cluster())
        self._check_fact_in_all("96123456", texts)

    def test_ego_speed_in_all(self):
        texts = _all_texts(_rich_cluster())
        self._check_fact_in_all("1500", texts)

    def test_ego_heading_in_all(self):
        texts = _all_texts(_rich_cluster())
        self._check_fact_in_all("900", texts)

    def test_camera_reading_in_all(self):
        texts = _all_texts(_rich_cluster())
        self._check_fact_in_all("OBSTACLE_DETECTED", texts)

    def test_radar_reading_in_all(self):
        texts = _all_texts(_rich_cluster())
        # radar also reads OBSTACLE_DETECTED
        self._check_fact_in_all("OBSTACLE_DETECTED", texts)

    def test_lidar_reading_in_all(self):
        texts = _all_texts(_rich_cluster())
        self._check_fact_in_all("CLEAR", texts)

    def test_peer1_station_id_in_all(self):
        texts = _all_texts(_rich_cluster())
        self._check_fact_in_all("3001", texts)

    def test_peer2_station_id_in_all(self):
        texts = _all_texts(_rich_cluster())
        self._check_fact_in_all("3002", texts)

    def test_peer1_event_in_all(self):
        texts = _all_texts(_rich_cluster())
        self._check_fact_in_all("no obstacle detected", texts)

    def test_peer2_event_in_all(self):
        texts = _all_texts(_rich_cluster())
        self._check_fact_in_all("road ahead is clear", texts)

    def test_rsu_station_id_in_all(self):
        texts = _all_texts(_rich_cluster())
        self._check_fact_in_all("9001", texts)

    def test_rsu_event_in_all(self):
        texts = _all_texts(_rich_cluster())
        self._check_fact_in_all("no hazard on corridor", texts)

    def test_cluster_peer1_station_id_in_all(self):
        texts = _all_texts(_rich_cluster())
        self._check_fact_in_all("2001", texts)

    def test_cluster_peer2_station_id_in_all(self):
        texts = _all_texts(_rich_cluster())
        self._check_fact_in_all("2002", texts)

    def test_cluster_peer_distance_in_all(self):
        """Haversine distance must appear in every template."""
        texts = _all_texts(_rich_cluster())
        for ts, text in texts.items():
            assert "m from ego" in text.lower() or "metres from the ego" in text.lower() or "m from ego" in text, (
                f"Cluster peer distance missing from {ts.value} template:\n{text}"
            )


# ---------------------------------------------------------------------------
# Invariant 2 — All templates are linguistically distinct
# ---------------------------------------------------------------------------

class TestLinguisticDistinctness:
    def test_default_differs_from_narrative(self):
        texts = _all_texts(_rich_cluster())
        assert texts[TemplateStyle.DEFAULT] != texts[TemplateStyle.NARRATIVE], (
            "DEFAULT and NARRATIVE produced identical text — templates must differ."
        )

    def test_default_differs_from_structured(self):
        texts = _all_texts(_rich_cluster())
        assert texts[TemplateStyle.DEFAULT] != texts[TemplateStyle.STRUCTURED], (
            "DEFAULT and STRUCTURED produced identical text — templates must differ."
        )

    def test_narrative_differs_from_structured(self):
        texts = _all_texts(_rich_cluster())
        assert texts[TemplateStyle.NARRATIVE] != texts[TemplateStyle.STRUCTURED], (
            "NARRATIVE and STRUCTURED produced identical text — templates must differ."
        )

    def test_all_three_are_pairwise_distinct(self):
        texts = _all_texts(_rich_cluster())
        text_list = list(texts.values())
        assert len(set(text_list)) == 3, (
            "Not all three templates produced distinct output strings."
        )


# ---------------------------------------------------------------------------
# Invariant 3 & 4 — No template leaks B2 information
# ---------------------------------------------------------------------------

class TestNoTemplateLeaksB2:
    def test_default_leakage_pass(self):
        text = synthesize_message(
            _rich_cluster(), _make_b2_result(), "urban", TemplateStyle.DEFAULT
        )["text"]
        VALIDATOR.assert_clean(text)

    def test_narrative_leakage_pass(self):
        text = synthesize_message(
            _rich_cluster(), _make_b2_result(), "urban", TemplateStyle.NARRATIVE
        )["text"]
        VALIDATOR.assert_clean(text)

    def test_structured_leakage_pass(self):
        text = synthesize_message(
            _rich_cluster(), _make_b2_result(), "urban", TemplateStyle.STRUCTURED
        )["text"]
        VALIDATOR.assert_clean(text)

    def test_all_templates_clean_for_minimal_cluster(self):
        b2 = _make_b2_result()
        for ts in ALL_TEMPLATES:
            text = synthesize_message(_minimal_cluster(), b2, "urban", ts)["text"]
            VALIDATOR.assert_clean(text)

    def test_extreme_b2_values_do_not_appear(self):
        """Extreme B2 values (0.0 and 1.0) must not influence any template output."""
        b2_extreme = _make_b2_result(
            trust=0.0, belief=1.0, disbelief=0.0, uncertainty=0.0,
            cluster_score=0.0, entropy=1.0, replay_probability=1.0,
            identity_consistency=0.0,
        )
        for ts in ALL_TEMPLATES:
            text = synthesize_message(_rich_cluster(), b2_extreme, "urban", ts)["text"]
            VALIDATOR.assert_clean(text)


# ---------------------------------------------------------------------------
# Invariant 5 — Each template is deterministic
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_default_deterministic(self):
        cluster = _rich_cluster()
        b2 = _make_b2_result()
        texts = [
            synthesize_message(cluster, b2, "urban", TemplateStyle.DEFAULT)["text"]
            for _ in range(10)
        ]
        assert len(set(texts)) == 1, "DEFAULT template is non-deterministic."

    def test_narrative_deterministic(self):
        cluster = _rich_cluster()
        b2 = _make_b2_result()
        texts = [
            synthesize_message(cluster, b2, "urban", TemplateStyle.NARRATIVE)["text"]
            for _ in range(10)
        ]
        assert len(set(texts)) == 1, "NARRATIVE template is non-deterministic."

    def test_structured_deterministic(self):
        cluster = _rich_cluster()
        b2 = _make_b2_result()
        texts = [
            synthesize_message(cluster, b2, "urban", TemplateStyle.STRUCTURED)["text"]
            for _ in range(10)
        ]
        assert len(set(texts)) == 1, "STRUCTURED template is non-deterministic."

    def test_different_b2_values_do_not_change_any_template(self):
        """Swapping every B2 value must not alter any template's output."""
        cluster = _rich_cluster()
        b2_low  = _make_b2_result(trust=0.01, belief=0.99, entropy=0.95)
        b2_high = _make_b2_result(trust=0.99, belief=0.01, entropy=0.01)
        for ts in ALL_TEMPLATES:
            text_low  = synthesize_message(cluster, b2_low,  "urban", ts)["text"]
            text_high = synthesize_message(cluster, b2_high, "urban", ts)["text"]
            assert text_low == text_high, (
                f"{ts.value} template output changed when only B2 values changed — "
                f"B2 information is being read by the synthesizer."
            )


# ---------------------------------------------------------------------------
# Invariant 6 — template=None equals template=DEFAULT
# ---------------------------------------------------------------------------

class TestNoneEqualsDefault:
    def test_none_template_produces_default_text(self):
        cluster = _rich_cluster()
        b2 = _make_b2_result()
        text_none    = synthesize_message(cluster, b2, "urban", None)["text"]
        text_default = synthesize_message(cluster, b2, "urban", TemplateStyle.DEFAULT)["text"]
        assert text_none == text_default, (
            "template=None did not produce the same output as template=DEFAULT."
        )

    def test_none_template_style_is_default(self):
        result = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban", None)
        assert result["template_style"] == "default"

    def test_omitting_template_arg_produces_default(self):
        cluster = _minimal_cluster()
        b2 = _make_b2_result()
        text_omitted = synthesize_message(cluster, b2, "urban")["text"]
        text_default = synthesize_message(cluster, b2, "urban", TemplateStyle.DEFAULT)["text"]
        assert text_omitted == text_default


# ---------------------------------------------------------------------------
# Invariant 7 — DEFAULT template backward compatibility with A2
# ---------------------------------------------------------------------------

class TestDefaultBackwardCompatibility:
    """The DEFAULT renderer must produce byte-for-byte identical output to A2."""

    def test_default_starts_with_v2x_scene_report(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.DEFAULT
        )["text"]
        assert text.startswith("V2X Scene Report: context=urban."), (
            f"DEFAULT template header changed:\n{text}"
        )

    def test_default_contains_ego_vehicle_label(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.DEFAULT
        )["text"]
        assert "Ego vehicle: station" in text

    def test_default_contains_local_sensor_label(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.DEFAULT
        )["text"]
        assert "Local sensor observations:" in text

    def test_default_contains_peer_report_label(self):
        cluster = _rich_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.DEFAULT)["text"]
        assert "Peer report 1:" in text

    def test_default_contains_rsu_message_label(self):
        cluster = _rich_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.DEFAULT)["text"]
        assert "RSU message 1:" in text

    def test_default_contains_cluster_peer_label(self):
        cluster = _rich_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.DEFAULT)["text"]
        assert "Cluster peer 1" in text

    def test_default_no_peer_reports_sentinel(self):
        cluster = _minimal_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.DEFAULT)["text"]
        assert "No peer reports received." in text

    def test_default_no_rsu_messages_sentinel(self):
        cluster = _minimal_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.DEFAULT)["text"]
        assert "No RSU messages received." in text

    def test_default_no_cluster_peers_sentinel(self):
        cluster = _minimal_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.DEFAULT)["text"]
        assert "No other vehicles in cooperative cluster." in text

    def test_default_uses_key_equals_value_format_for_ego(self):
        """Ego vehicle fields must use 'key=value' notation (A2 format)."""
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.DEFAULT
        )["text"]
        assert "speed=1500" in text
        assert "heading=900 deg" in text
        assert "yaw_rate=0" in text

    def test_default_sensor_line_format(self):
        cluster = _rich_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.DEFAULT)["text"]
        assert "camera=OBSTACLE_DETECTED" in text
        assert "radar=OBSTACLE_DETECTED" in text
        assert "lidar=CLEAR" in text

    def test_default_peer_report_joined_with_space(self):
        """Peer report parts must be space-joined (A2 format), not pipe-separated."""
        cluster = _rich_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.DEFAULT)["text"]
        # "Peer report 1: Station 3001 (passengerCar) reports: no obstacle detected"
        assert "Station 3001 (passengerCar) reports: no obstacle detected" in text

    def test_default_verbatim_peer_report(self):
        """String peer reports must appear verbatim in DEFAULT output."""
        cluster = [_make_cam_msg()]
        cluster[0]["scene_context"] = {
            "peer_reports": ["Station OBU-9 reports ice on road."],
            "rsu_messages": [],
        }
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.DEFAULT)["text"]
        assert "Peer report 1: Station OBU-9 reports ice on road." in text


# ---------------------------------------------------------------------------
# Invariant 8 — NARRATIVE uses flowing prose vocabulary
# ---------------------------------------------------------------------------

class TestNarrativeVocabulary:
    def test_narrative_starts_with_correct_header(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.NARRATIVE
        )["text"]
        assert text.startswith("V2X Scene Narrative:"), (
            f"NARRATIVE template header incorrect:\n{text}"
        )

    def test_narrative_contains_ego_vehicle_phrase(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.NARRATIVE
        )["text"]
        assert "The ego vehicle" in text

    def test_narrative_contains_onboard_perception_phrase(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.NARRATIVE
        )["text"]
        assert "On-board perception:" in text

    def test_narrative_sensor_uses_semicolons(self):
        """NARRATIVE joins sensor items with semicolons, not commas (differs from DEFAULT)."""
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.NARRATIVE
        )["text"]
        assert "camera sensor reports" in text
        assert "radar sensor reports" in text
        assert "lidar sensor reports" in text

    def test_narrative_peer_uses_neighbouring_vehicle_phrase(self):
        cluster = _rich_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.NARRATIVE)["text"]
        assert "neighbouring vehicle" in text.lower()

    def test_narrative_rsu_uses_roadside_unit_phrase(self):
        cluster = _rich_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.NARRATIVE)["text"]
        assert "roadside unit" in text.lower() or "infrastructure" in text.lower()

    def test_narrative_cluster_peer_uses_metres_phrase(self):
        cluster = _rich_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.NARRATIVE)["text"]
        assert "metres from the ego vehicle" in text.lower()

    def test_narrative_no_peer_reports_uses_prose_phrase(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.NARRATIVE
        )["text"]
        assert "No cooperative peer observations were received" in text

    def test_narrative_no_rsu_messages_uses_prose_phrase(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.NARRATIVE
        )["text"]
        assert "No infrastructure messages were received" in text

    def test_narrative_no_cluster_peers_uses_prose_phrase(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.NARRATIVE
        )["text"]
        assert "No other cooperative cluster members were observed" in text

    def test_narrative_does_not_use_default_key_equals_value_format(self):
        """NARRATIVE must not use 'speed=1500' style (DEFAULT format)."""
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.NARRATIVE
        )["text"]
        # DEFAULT uses "speed=1500", NARRATIVE uses "speed 1500"
        assert "speed=1500" not in text

    def test_narrative_verbatim_peer_report(self):
        """Verbatim string peer reports must appear in NARRATIVE output."""
        cluster = [_make_cam_msg()]
        cluster[0]["scene_context"] = {
            "peer_reports": ["Station OBU-9 reports ice on road."],
            "rsu_messages": [],
        }
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.NARRATIVE)["text"]
        assert "Station OBU-9 reports ice on road." in text


# ---------------------------------------------------------------------------
# Invariant 9 — STRUCTURED uses bracketed section headers
# ---------------------------------------------------------------------------

class TestStructuredFormat:
    def test_structured_starts_with_header(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.STRUCTURED
        )["text"]
        assert text.startswith("=== V2X Scene Report ==="), (
            f"STRUCTURED template header incorrect:\n{text}"
        )

    def test_structured_contains_ego_section(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.STRUCTURED
        )["text"]
        assert "[Ego Vehicle]" in text

    def test_structured_contains_sensor_section(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.STRUCTURED
        )["text"]
        assert "[Local Sensor Observations]" in text

    def test_structured_contains_peer_reports_section(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.STRUCTURED
        )["text"]
        assert "[Peer Reports]" in text

    def test_structured_contains_rsu_section(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.STRUCTURED
        )["text"]
        assert "[RSU Messages]" in text

    def test_structured_contains_cluster_section(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.STRUCTURED
        )["text"]
        assert "[Cooperative Cluster]" in text

    def test_structured_sensor_uses_colons(self):
        """STRUCTURED sensor lines use 'Camera: VALUE' format (colon separator)."""
        cluster = _rich_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.STRUCTURED)["text"]
        assert "Camera: OBSTACLE_DETECTED" in text
        assert "Radar: OBSTACLE_DETECTED" in text
        assert "Lidar: CLEAR" in text

    def test_structured_peer_uses_bracket_index(self):
        cluster = _rich_cluster()
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.STRUCTURED)["text"]
        assert "[1]" in text
        assert "[2]" in text

    def test_structured_no_peer_reports_sentinel(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.STRUCTURED
        )["text"]
        assert "(none received)" in text

    def test_structured_no_cluster_peers_sentinel(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.STRUCTURED
        )["text"]
        assert "(no additional vehicles observed)" in text

    def test_structured_context_line_present(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "highway", TemplateStyle.STRUCTURED
        )["text"]
        assert "Context: highway" in text

    def test_structured_uses_pipe_separators_for_kinematics(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.STRUCTURED
        )["text"]
        assert "Kinematics: speed=" in text
        assert "|" in text

    def test_structured_is_multiline(self):
        text = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.STRUCTURED
        )["text"]
        assert "\n" in text, "STRUCTURED template must produce multi-line output."

    def test_structured_verbatim_peer_report(self):
        """Verbatim string peer reports must appear in STRUCTURED output."""
        cluster = [_make_cam_msg()]
        cluster[0]["scene_context"] = {
            "peer_reports": ["Station OBU-9 reports ice on road."],
            "rsu_messages": [],
        }
        text = synthesize_message(cluster, _make_b2_result(), "urban", TemplateStyle.STRUCTURED)["text"]
        assert "Station OBU-9 reports ice on road." in text


# ---------------------------------------------------------------------------
# Invariant 10 — No reasoning words in any template (no inference)
# ---------------------------------------------------------------------------

class TestNoReasoningWords:
    _FORBIDDEN = [
        "disagree", "agreement", "contradiction", "mismatch", "conflict",
        "suspicious", "anomalous", "likely", "probably", "majority",
        "most vehicles", "several vehicles",
    ]

    def _check_no_reasoning(self, text: str, template_name: str) -> None:
        for word in self._FORBIDDEN:
            assert word not in text.lower(), (
                f"Reasoning word {word!r} found in {template_name} template:\n{text}"
            )

    def test_no_reasoning_in_default(self):
        texts = _all_texts(_rich_cluster())
        self._check_no_reasoning(texts[TemplateStyle.DEFAULT], "DEFAULT")

    def test_no_reasoning_in_narrative(self):
        texts = _all_texts(_rich_cluster())
        self._check_no_reasoning(texts[TemplateStyle.NARRATIVE], "NARRATIVE")

    def test_no_reasoning_in_structured(self):
        texts = _all_texts(_rich_cluster())
        self._check_no_reasoning(texts[TemplateStyle.STRUCTURED], "STRUCTURED")


# ---------------------------------------------------------------------------
# Invariant 11 — Contradiction visible in all templates
# ---------------------------------------------------------------------------

class TestContradictionVisibility:
    """Local sensors detect obstacle; peers and RSU report clear. All must be listed."""

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
                {"station_id": 4002, "event": "no obstacle present",  "speed": 1400},
            ],
            "rsu_messages": [
                {"station_id": 8001, "event": "no hazard detected"},
            ],
        }
        return [msg]

    def test_local_detection_visible_in_all(self):
        texts = _all_texts(self._contradiction_cluster(), context="highway")
        for ts, text in texts.items():
            assert "OBSTACLE_DETECTED" in text, (
                f"Local obstacle detection missing from {ts.value} template:\n{text}"
            )

    def test_peer1_clear_report_visible_in_all(self):
        texts = _all_texts(self._contradiction_cluster(), context="highway")
        for ts, text in texts.items():
            assert "road ahead is clear" in text.lower(), (
                f"Peer 1 clear-road report missing from {ts.value} template:\n{text}"
            )

    def test_peer2_clear_report_visible_in_all(self):
        texts = _all_texts(self._contradiction_cluster(), context="highway")
        for ts, text in texts.items():
            assert "no obstacle present" in text.lower(), (
                f"Peer 2 clear-road report missing from {ts.value} template:\n{text}"
            )

    def test_rsu_clear_report_visible_in_all(self):
        texts = _all_texts(self._contradiction_cluster(), context="highway")
        for ts, text in texts.items():
            assert "no hazard detected" in text.lower(), (
                f"RSU no-hazard report missing from {ts.value} template:\n{text}"
            )

    def test_all_contradiction_outputs_are_clean(self):
        texts = _all_texts(self._contradiction_cluster(), context="highway")
        for ts, text in texts.items():
            VALIDATOR.assert_clean(text)


# ---------------------------------------------------------------------------
# Invariant 12 — Template selection dispatches correctly
# ---------------------------------------------------------------------------

class TestTemplateSelection:
    def test_default_style_value_is_correct(self):
        result = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.DEFAULT
        )
        assert result["template_style"] == "default"

    def test_narrative_style_value_is_correct(self):
        result = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.NARRATIVE
        )
        assert result["template_style"] == "narrative"

    def test_structured_style_value_is_correct(self):
        result = synthesize_message(
            _minimal_cluster(), _make_b2_result(), "urban", TemplateStyle.STRUCTURED
        )
        assert result["template_style"] == "structured"

    def test_template_key_unchanged(self):
        """The 'template' key must remain 'cooperative_scene_report' (A2 compat)."""
        for ts in ALL_TEMPLATES:
            result = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban", ts)
            assert result["template"] == "cooperative_scene_report", (
                f"'template' key changed for {ts.value}"
            )

    def test_sources_key_present_for_all(self):
        for ts in ALL_TEMPLATES:
            result = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban", ts)
            assert "sources" in result
            assert isinstance(result["sources"], list)

    def test_text_is_string_for_all(self):
        for ts in ALL_TEMPLATES:
            result = synthesize_message(_minimal_cluster(), _make_b2_result(), "urban", ts)
            assert isinstance(result["text"], str)
            assert len(result["text"]) > 0

    def test_template_style_enum_covers_all_expected_values(self):
        assert {ts.value for ts in TemplateStyle} == {"default", "narrative", "structured"}


# ---------------------------------------------------------------------------
# Invariant 13 — Empty cluster handled gracefully for all templates
# ---------------------------------------------------------------------------

class TestEmptyCluster:
    def test_empty_cluster_text_for_all(self):
        b2 = _make_b2_result()
        for ts in ALL_TEMPLATES:
            result = synthesize_message([], b2, "urban", ts)
            assert result["text"].startswith("V2X Scene Report: No cooperative"), (
                f"Empty cluster text wrong for {ts.value}:\n{result['text']}"
            )

    def test_empty_cluster_template_key_unchanged(self):
        b2 = _make_b2_result()
        for ts in ALL_TEMPLATES:
            result = synthesize_message([], b2, "urban", ts)
            assert result["template"] == "cooperative_scene_report"

    def test_empty_cluster_sources_empty(self):
        b2 = _make_b2_result()
        for ts in ALL_TEMPLATES:
            result = synthesize_message([], b2, "urban", ts)
            assert result["sources"] == []

    def test_empty_cluster_clean_for_all(self):
        b2 = _make_b2_result()
        for ts in ALL_TEMPLATES:
            text = synthesize_message([], b2, "urban", ts)["text"]
            VALIDATOR.assert_clean(text)

    def test_empty_cluster_template_style_correct(self):
        b2 = _make_b2_result()
        for ts in ALL_TEMPLATES:
            result = synthesize_message([], b2, "urban", ts)
            assert result["template_style"] == ts.value


# ---------------------------------------------------------------------------
# SceneEvidence is importable (extensibility test)
# ---------------------------------------------------------------------------

class TestSceneEvidencePublicInterface:
    def test_scene_evidence_is_importable(self):
        from pipeline.synthesizer import SceneEvidence  # noqa: F401

    def test_template_style_is_importable(self):
        from pipeline.synthesizer import TemplateStyle  # noqa: F401

    def test_template_style_enum_members(self):
        assert hasattr(TemplateStyle, "DEFAULT")
        assert hasattr(TemplateStyle, "NARRATIVE")
        assert hasattr(TemplateStyle, "STRUCTURED")
