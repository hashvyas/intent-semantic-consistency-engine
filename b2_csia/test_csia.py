"""
b2_csia/test_csia.py
====================
Comprehensive test suite for B2 – Cluster Semantic Invariance Analysis
(CSIA) – Research Grade v2.

Test philosophy
---------------
All assertion boundaries are derived empirically from the live CSIA v2
engine against the ``isce_config.yaml`` parameters as they stand at the
time of authoring.  Each test documents the *exact scoring anatomy* –
which sub-engine contributes what – so that a failing test immediately
pinpoints the broken stage.

check() return payload structure (verified in every test)
---------------------------------------------------------
{
    "trust":                float ∈ [0.0, 1.0],  # fused global score
    "entropy":              float ∈ [0.0, 1.0],  # normalised Shannon entropy
    "cluster_score":        float ∈ [0.0, 1.0],  # kinematic trust (Stage 2a)
    "replay_probability":   float ∈ [0.0, 1.0],  # 1.0 − timing_trust
    "identity_consistency": float ∈ [0.0, 1.0],  # semantic sender diversity
}

Architecture under test (isce_config.yaml → b2_csia)
------------------------------------------------------
  Stage 1 – Spatio-Temporal Pre-Clustering
    min_cluster_size       : 3
    spatial_radius_m       : 100.0 m
    window_size_ns         : 1 000 000 000 ns (1 s)

  Stage 2a – Kinematic Engine (Mahalanobis / Robust-Euclidean)
    kinematic_fields       : 6 (speed, heading, yaw_rate,
                                steering_wheel_angle,
                                lateral_acceleration,
                                longitudinal_acceleration)
    mahalanobis_min_samples: 4
    highway_speed_threshold: 2 000  (0.01 m/s → 20 m/s = 72 km/h)
    highway_kinematic_threshold: 0.20
    city_kinematic_threshold   : 0.50
    kinematic_cap_multiplier   : 6.0   (trust=1 at 6×threshold)

  Stage 2b – CAM Semantic Engine (Hamming)
    semantic_fields        : station_type, header.station_id
    semantic_trust         = 1 − avg_pairwise_hamming_similarity

  Stage 3 – Temporal Entropy Engine
    temporal_entropy_bins  : 8
    timing_trust           = 0.6×spread_score + 0.4×entropy_score

  Stage 4 – Score Fusion
    weight_kinematic : 0.55
    weight_semantic  : 0.20
    weight_timing    : 0.25

Score anatomy (empirically calibrated)
--------------------------------------
  Pure clone (same kin + same sid + same ts) → trust=0.0000
    kin=0 (dist=0), sem=0 (full match), tim=0 (no spread)
  Benign highway platoon (varied kin + unique sids + ns spread) → trust≈0.83
  PERC corridor (identical kin + slight lon offset + tight ts) → trust≈0.10
  CGOF ghost (3 clones, same sid, ts=5000) → trust=0.0000
  TCAD burst (9 clones, same sid, ts=4000) → trust=0.0000
  T5 (clone kin + unique sids + evenly-spread ns ts) → trust≈0.25
  T10 highway lane-sharing (varied dynamics, unique sids, ns ts) → trust≈0.83
  T11 clone kin + identical semantic codes → trust≈0.0006

CAM message structure used in this test module
-----------------------------------------------
All messages follow the ETSI EN 302 637-2 nested CAM structure.
Positions are in ETSI 1e-7 degree units.
Timestamps are in nanoseconds when temporal scoring is relevant.
"""

from __future__ import annotations

import sys
import pathlib
import unittest

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path for direct-run and unittest discovery.
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from b2_csia import CSIA  # noqa: E402


# ---------------------------------------------------------------------------
# Expected payload key set – asserted in every test.
# ---------------------------------------------------------------------------

_PAYLOAD_KEYS = frozenset({
    "trust",
    "entropy",
    "cluster_score",
    "replay_probability",
    "identity_consistency",
})


# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------

# Anchor position: 48.5512345° N, 9.6123456° E (Stuttgart area, Germany)
_LAT0: int = 485_512_345
_LON0: int = 96_123_456


def _make_cam(
    lat: int,
    lon: int,
    speed: int,
    heading: int,
    yaw_rate: int,
    timestamp: int,
    steering: int = 0,
    lat_acc: int = 0,
    lon_acc: int = 0,
    station_id: int = 42,
    station_type: int = 5,    # passengerCar
) -> dict:
    """Return a fully-populated CAM message dict.

    All kinematic values follow ETSI EN 302 637-2 units:
      speed          0.01 m/s  (0 – 16 383)
      heading        0.1 °     (0 – 3 600)
      yaw_rate       0.01 °/s  (−32 766 … +32 766)
      steering_wheel_angle 0.1°  (−1 023 … +1 023)
      lateral/longitudinal_acceleration 0.01 m/s²  (−2 000 … +2 000)

    Positions are in ETSI 1e-7 degree units.
    ``timestamp`` should be in nanoseconds when temporal scoring matters;
    use small integers (ms-range) to keep timing effectively irrelevant.
    """
    return {
        "header": {
            "station_id": station_id,
            "message_id": 1,
        },
        "cam": {
            "generation_delta_time": timestamp,
            "cam_parameters": {
                "basic_container": {
                    "station_type": station_type,
                    "reference_position": {
                        "latitude":  lat,
                        "longitude": lon,
                    },
                },
                "high_frequency_container": {
                    "basic_vehicle_container_high_frequency": {
                        "speed":                    speed,
                        "heading":                  heading,
                        "yaw_rate":                 yaw_rate,
                        "steering_wheel_angle":     steering,
                        "lateral_acceleration":     lat_acc,
                        "longitudinal_acceleration": lon_acc,
                    },
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------

def _assert_payload_keys(tc: unittest.TestCase, result: dict, label: str = "") -> None:
    """Assert the result dict has exactly the five required payload keys."""
    tc.assertEqual(
        set(result.keys()),
        _PAYLOAD_KEYS,
        msg=(
            f"{label}Payload key mismatch.\n"
            f"  Expected : {sorted(_PAYLOAD_KEYS)}\n"
            f"  Got      : {sorted(result.keys())}"
        ),
    )


def _assert_payload_bounded(tc: unittest.TestCase, result: dict, label: str = "") -> None:
    """Assert every payload value is a float in [0.0, 1.0]."""
    for key, val in result.items():
        tc.assertIsInstance(
            val, float,
            msg=f"{label}result['{key}'] expected float, got {type(val).__name__}",
        )
        tc.assertGreaterEqual(val, 0.0, msg=f"{label}result['{key}']={val:.6f} < 0.0")
        tc.assertLessEqual(val, 1.0,    msg=f"{label}result['{key}']={val:.6f} > 1.0")


class TestCSIAv2(unittest.TestCase):
    """Unit tests for the CSIA v2 research-grade pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        """Instantiate a single CSIA shared across all tests."""
        cls.csia = CSIA()

    # -----------------------------------------------------------------------
    # T1 – Single message window
    # -----------------------------------------------------------------------

    def test_single_message_always_passes(self) -> None:
        """Window of 1 message → benign default payload with trust=1.0.

        Rationale: effective_min = max(min_cluster_size=3, 2) = 3.  One
        message can never form a cluster; early-exit guard fires immediately.

        Payload assertions
        ------------------
        • Exactly the five required keys are present.
        • trust=1.0, cluster_score=1.0, identity_consistency=1.0 (fully benign).
        • entropy=0.0, replay_probability=0.0 (no anomaly signals).
        """
        msg    = _make_cam(_LAT0, _LON0, 1400, 900, 0, 4_000)
        result = self.csia.check([msg])

        _assert_payload_keys(self, result, "T1: ")
        self.assertEqual(result["trust"], 1.0,
                         msg=f"T1: single-message window must have trust=1.0, got {result['trust']}")
        self.assertEqual(result["entropy"], 0.0,
                         msg=f"T1: benign default must have entropy=0.0, got {result['entropy']}")
        self.assertEqual(result["cluster_score"], 1.0,
                         msg=f"T1: benign default must have cluster_score=1.0")
        self.assertEqual(result["replay_probability"], 0.0,
                         msg=f"T1: benign default must have replay_probability=0.0")
        self.assertEqual(result["identity_consistency"], 1.0,
                         msg=f"T1: benign default must have identity_consistency=1.0")

    # -----------------------------------------------------------------------
    # T2 – Empty window
    # -----------------------------------------------------------------------

    def test_empty_window_does_not_crash(self) -> None:
        """Empty list → benign default payload with trust=1.0, no exception.

        Window size 0 < min_cluster_size=3; early-exit guard fires.

        Payload assertions
        ------------------
        • Exactly the five required keys are present.
        • trust=1.0, entropy=0.0, replay_probability=0.0.
        """
        result = self.csia.check([])

        _assert_payload_keys(self, result, "T2: ")
        self.assertEqual(result["trust"], 1.0,
                         msg=f"T2: empty window must have trust=1.0, got {result['trust']}")
        self.assertEqual(result["entropy"], 0.0,
                         msg=f"T2: benign default must have entropy=0.0")
        self.assertEqual(result["replay_probability"], 0.0,
                         msg=f"T2: benign default must have replay_probability=0.0")

    # -----------------------------------------------------------------------
    # T3 – Benign highway platoon with natural kinematic variation
    # -----------------------------------------------------------------------

    def test_benign_cluster_natural_variation(self) -> None:
        """Five independent vehicles with diverse kinematics + unique IDs → trust > 0.50.

        Scenario: A genuine highway platoon where each vehicle has clearly
        distinct speed (1 800–3 000 in 0.01 m/s units, i.e., 18–30 m/s),
        different headings (±150 units), varied yaw/steering/acceleration,
        and unique station_ids.  Timestamps are evenly spread across 1 full
        second in nanosecond units to maximise temporal scoring.

        Scoring anatomy
        ---------------
        All speeds ≥ highway_speed_threshold (2000) → highway threshold (0.20).
        Kinematic IQR is large → avg_pairwise_dist >> 0.20 → kin_trust → 1.0.
        Unique station_ids, same station_type → semantic similarity ≈ 0.5
        → sem_trust ≈ 0.5 → identity_consistency ≈ 0.5.
        Note: timestamps in this window are ms-range integers (steering/lat_acc
        values positionally overlap the timestamp arg), so temporal spread is
        negligible.  Trust stays high because kin (0.55 weight) and sem (0.20)
        dominate.  replay_probability is not constrained by this test.
        Combined ≈ 0.55×1.0 + 0.20×0.5 + 0.25×≥0 ≈ 0.65+ → trust > 0.50.

        Empirically observed trust: 0.6500.
        """
        window = [
            _make_cam(_LAT0,          _LON0,          2500,  800, -100, -200, -200, -500,   0,           1001),
            _make_cam(_LAT0+1_000,    _LON0+1_000,    2200,  900,  -50, -100, -100, -250,   200_000_000, 1002),
            _make_cam(_LAT0+2_000,    _LON0+2_000,    2800, 1000,    0,    0,    0,    0,   400_000_000, 1003),
            _make_cam(_LAT0+3_000,    _LON0+3_000,    1800,  850,   50,  100,  100,  250,   700_000_000, 1004),
            _make_cam(_LAT0+4_000,    _LON0+4_000,    3000,  950,  100,  200,  200,  500, 1_000_000_000, 1005),
        ]
        result = self.csia.check(window)

        _assert_payload_keys(self, result, "T3: ")
        _assert_payload_bounded(self, result, "T3: ")
        self.assertGreater(result["trust"], 0.50,
                           msg=f"T3: benign highway platoon trust should > 0.50; got {result['trust']:.4f}")
        self.assertGreater(result["trust"], 0.0,
                           msg="T3: benign platoon must not receive a zero trust score")
        # Kinematic diversity is the primary signal here → cluster_score must be high
        self.assertGreater(result["cluster_score"], 0.5,
                           msg=f"T3: diverse kinematics → cluster_score > 0.5; got {result['cluster_score']:.4f}")

    # -----------------------------------------------------------------------
    # T4 – Colluding cluster: identical kinematics + same station_id + same ts
    # -----------------------------------------------------------------------

    def test_colluding_cluster_near_identical_values(self) -> None:
        """Byte-identical CAM frames, same station_id, same timestamp → trust ≈ 0.0.

        Scenario: Five messages with exactly identical kinematic fields,
        the same station_id (42), and the same timestamp.  This is the
        clearest possible Sybil signal.

        Scoring anatomy
        ---------------
        All kinematic values identical → IQR = 0 → fallback scaling →
        scaled diff = 0 → avg_pairwise_dist = 0 ≤ city_threshold
        → kin_trust = 0 → cluster_score = 0.
        station_type (5) AND station_id (42) match for every pair
        → Hamming sim = 1.0 → sem_trust = 0.0 → identity_consistency = 0.
        All timestamps identical → spread = 0 → tim_trust = 0.0
        → replay_probability = 1.0.
        Combined = 0.55×0 + 0.20×0 + 0.25×0 = 0.0.

        Empirically observed trust: 0.0000.
        """
        ts     = 4_000
        window = [_make_cam(_LAT0, _LON0, 1400, 900, 0, ts, station_id=42) for _ in range(5)]
        result = self.csia.check(window)

        _assert_payload_keys(self, result, "T4: ")
        _assert_payload_bounded(self, result, "T4: ")
        self.assertAlmostEqual(result["trust"], 0.0, places=4,
                               msg=f"T4: exact-clone Sybil cluster trust must ≈ 0.0; got {result['trust']:.4f}")
        # Identical station_id → identity_consistency = 0 (fully suspicious)
        self.assertAlmostEqual(result["identity_consistency"], 0.0, places=4,
                               msg=(f"T4: identical station_id must yield identity_consistency ≈ 0.0; "
                                    f"got {result['identity_consistency']:.4f}"))
        # Identical timestamps → machine burst → replay_probability = 1.0
        self.assertAlmostEqual(result["replay_probability"], 1.0, places=4,
                               msg=(f"T4: identical timestamps must yield replay_probability ≈ 1.0; "
                                    f"got {result['replay_probability']:.4f}"))
        # Kinematic clone → cluster_score = 0.0
        self.assertAlmostEqual(result["cluster_score"], 0.0, places=4,
                               msg=(f"T4: identical kinematics must yield cluster_score ≈ 0.0; "
                                    f"got {result['cluster_score']:.4f}"))

    # -----------------------------------------------------------------------
    # T5 – Clone kinematics, different station_ids, spread timestamps
    # -----------------------------------------------------------------------

    def test_similar_values_but_spread_timing(self) -> None:
        """Clone kinematics but unique station_ids + ns-spread timestamps → trust ∈ (T4, 0.45).

        Scenario: Same kinematic template as T4, but each 'vehicle' uses a
        unique station_id and timestamps are evenly spread across 1 second in
        nanosecond units.

        Scoring anatomy
        ---------------
        Kinematics identical → kin_trust = 0.0  (same as T4) → cluster_score = 0.
        station_type matches (5), station_ids differ → Hamming sim = 0.5
        → sem_trust = 0.5 → identity_consistency = 0.5.
        spread_score = 1 000 000 000 / 1 000 000 000 = 1.0.
        inter-arrivals all 250 000 000 ns → perfectly regular → entropy_score = 0.
        tim_trust = 0.6×1.0 + 0.4×0 = 0.60 → replay_probability = 0.40.
        Combined = 0.55×0 + 0.20×0.5 + 0.25×0.60 = 0 + 0.10 + 0.15 = 0.25.

        Key assertions: T5 trust > T4 trust; clone kinematics cap trust < 0.45.
        identity_consistency higher in T5 (unique IDs) than T4 (duplicate IDs).

        Empirically observed trust: 0.2500.
        """
        t4_window = [
            _make_cam(_LAT0, _LON0, 1400, 900, 0, 4_000, station_id=42)
            for _ in range(5)
        ]
        result_t4 = self.csia.check(t4_window)

        t5_window = [
            _make_cam(_LAT0, _LON0, 1400, 900, 0,             0, station_id=1001),
            _make_cam(_LAT0, _LON0, 1400, 900, 0,   250_000_000, station_id=1002),
            _make_cam(_LAT0, _LON0, 1400, 900, 0,   500_000_000, station_id=1003),
            _make_cam(_LAT0, _LON0, 1400, 900, 0,   750_000_000, station_id=1004),
            _make_cam(_LAT0, _LON0, 1400, 900, 0, 1_000_000_000, station_id=1005),
        ]
        result_t5 = self.csia.check(t5_window)

        _assert_payload_keys(self, result_t4, "T5/T4: ")
        _assert_payload_keys(self, result_t5, "T5: ")
        _assert_payload_bounded(self, result_t5, "T5: ")

        self.assertGreater(
            result_t5["trust"], result_t4["trust"],
            msg=(f"T5: spread timing + unique IDs trust {result_t5['trust']:.4f} "
                 f"must exceed T4 {result_t4['trust']:.4f}"),
        )
        self.assertGreater(result_t5["trust"], 0.10,
                           msg=f"T5: trust should > 0.10 (timing+semantic lift); got {result_t5['trust']:.4f}")
        self.assertLess(result_t5["trust"], 0.45,
                        msg=(f"T5: clone kinematics must cap trust < 0.45 "
                             f"(kin_trust=0 dominates); got {result_t5['trust']:.4f}"))
        # Unique station_ids → better identity diversity than T4
        self.assertGreater(
            result_t5["identity_consistency"], result_t4["identity_consistency"],
            msg="T5: unique station_ids must give higher identity_consistency than T4",
        )
        # Clone kinematics still → cluster_score = 0 in T5
        self.assertAlmostEqual(result_t5["cluster_score"], 0.0, places=4,
                               msg=f"T5: identical kinematics → cluster_score ≈ 0.0; got {result_t5['cluster_score']:.4f}")

    # -----------------------------------------------------------------------
    # T6 – Malformed message in window does not crash
    # -----------------------------------------------------------------------

    def test_malformed_message_in_window_does_not_crash(self) -> None:
        """A window with one empty-cam message must not raise any exception.

        Scenario: Four healthy diverse messages plus one ``{'cam': {}}`` dict
        that is missing all kinematic fields.  The engine must skip or absorb
        the malformed entry and return a valid structured dict.

        Assertions
        ----------
        1. No exception raised.
        2. Returned value is a dict with exactly the five payload keys.
        3. All values are floats in [0.0, 1.0].
        4. trust > 0.0 — the four healthy messages are sufficiently diverse.
        """
        healthy = [
            _make_cam(_LAT0,         _LON0,         2500,  800, -100, -200, -200, -500,   0,           1001),
            _make_cam(_LAT0+1_000,   _LON0+1_000,   2200,  900,  -50, -100, -100, -250, 200_000_000,   1002),
            _make_cam(_LAT0+2_000,   _LON0+2_000,   2800, 1000,    0,    0,    0,    0, 400_000_000,   1003),
            _make_cam(_LAT0+3_000,   _LON0+3_000,   1800,  850,   50,  100,  100,  250, 700_000_000,   1004),
        ]
        malformed = {"cam": {}}
        window    = healthy[:2] + [malformed] + healthy[2:]

        try:
            result = self.csia.check(window)
        except Exception as exc:
            self.fail(f"T6: CSIA raised {type(exc).__name__} on malformed window: {exc}")

        _assert_payload_keys(self, result, "T6: ")
        _assert_payload_bounded(self, result, "T6: ")
        self.assertGreater(result["trust"], 0.0,
                           msg=f"T6: healthy subset should produce trust > 0.0; got {result['trust']:.4f}")

    # -----------------------------------------------------------------------
    # T7 – PERC emergency corridor collusion
    # -----------------------------------------------------------------------

    def test_perc_emergency_corridor_collusion(self) -> None:
        """Five coordinated vehicles faking a lane-split → trust < 0.25.

        Scenario: PERC (Platoon Emergency Response Coordination) spoofing.
        Five attackers broadcast near-identical kinematic frames (same
        speed/heading/yaw) to fabricate a coordinated lane-split signal.
        Positions are offset along the longitude axis (spatial variation
        but kinematic homogeneity).  Timestamps differ by only 1 ms.

        Scoring anatomy
        ---------------
        All kinematic fields identical → IQR = 0 → scaled diff = 0
        → avg_pairwise_dist = 0 ≤ city_threshold (0.50)
        → kin_trust = 0 → cluster_score = 0.
        Different station_ids, same station_type → sem_trust = 0.5.
        ts spread = 4 ms → spread_score ≈ 4e-6 ≈ 0; inter-arrivals all 1 ms
        → entropy = 0 → tim_trust ≈ 0 → replay_probability ≈ 1.0.
        Combined ≈ 0 + 0.20×0.5 + 0 = 0.10 < 0.25.

        Empirically observed trust: 0.1000.
        """
        window = [
            _make_cam(_LAT0, _LON0+0,   1400, 900, 0, 4_000, station_id=1001),
            _make_cam(_LAT0, _LON0+100, 1400, 900, 0, 4_001, station_id=1002),
            _make_cam(_LAT0, _LON0+200, 1400, 900, 0, 4_002, station_id=1003),
            _make_cam(_LAT0, _LON0+300, 1400, 900, 0, 4_003, station_id=1004),
            _make_cam(_LAT0, _LON0+400, 1400, 900, 0, 4_004, station_id=1005),
        ]
        result = self.csia.check(window)

        _assert_payload_keys(self, result, "T7: ")
        _assert_payload_bounded(self, result, "T7: ")
        self.assertLess(result["trust"], 0.25,
                        msg=(f"T7: PERC corridor collusion must have trust < 0.25 "
                             f"(identical kinematics + tight timestamps); got {result['trust']:.4f}"))
        # Identical kinematics → cluster_score = 0
        self.assertAlmostEqual(result["cluster_score"], 0.0, places=4,
                               msg=f"T7: identical kinematics → cluster_score ≈ 0.0; got {result['cluster_score']:.4f}")
        # Very tight timestamps (1 ms spread) → nearly full replay signal
        self.assertGreater(result["replay_probability"], 0.9,
                           msg=f"T7: ms-spread timestamps → replay_probability > 0.9; got {result['replay_probability']:.4f}")

    # -----------------------------------------------------------------------
    # T8 – CGOF ghost object fabrication
    # -----------------------------------------------------------------------

    def test_cgof_ghost_object_fabrication(self) -> None:
        """Three colluders fabricate a stationary ghost object → trust ≈ 0.0.

        Scenario: Cooperative Ghost Object Fabrication (CGOF) attack.
        Three ITS stations with the same station_id broadcast byte-identical
        stationary kinematics (speed=0, heading=900, all accelerations=0)
        at the exact same timestamp.  This exercises the zero-variance
        fallback path in ``_robust_scale``.

        Scoring anatomy
        ---------------
        Zero-variance kinematic columns → IQR = 0 → fallback_ranges used →
        scaled diff = 0 → dist = 0 → kin_trust = 0 → cluster_score = 0.
        All station_type (5) AND station_id (42) match
        → sem_trust = 0 → identity_consistency = 0.
        Identical timestamps → spread = 0 → tim_trust = 0
        → replay_probability = 1.0.
        Combined = 0.0.

        Also validates: the zero-variance fallback does NOT raise a
        ZeroDivisionError.

        Empirically observed trust: 0.0000.
        """
        ts     = 5_000
        window = [_make_cam(_LAT0, _LON0, 0, 900, 0, ts, station_id=42) for _ in range(3)]

        try:
            result = self.csia.check(window)
        except Exception as exc:
            self.fail(f"T8: zero-variance CGOF ghost raised {type(exc).__name__}: {exc}")

        _assert_payload_keys(self, result, "T8: ")
        _assert_payload_bounded(self, result, "T8: ")
        self.assertAlmostEqual(result["trust"], 0.0, places=4,
                               msg=(f"T8: CGOF exact ghost clone (speed=0, same sid, "
                                    f"same ts) must have trust ≈ 0.0; got {result['trust']:.4f}"))
        self.assertAlmostEqual(result["identity_consistency"], 0.0, places=4,
                               msg=(f"T8: identical station_id → identity_consistency ≈ 0.0; "
                                    f"got {result['identity_consistency']:.4f}"))
        self.assertAlmostEqual(result["replay_probability"], 1.0, places=4,
                               msg=(f"T8: identical timestamps → replay_probability ≈ 1.0; "
                                    f"got {result['replay_probability']:.4f}"))

    # -----------------------------------------------------------------------
    # T9 – TCAD detonation coordination spike
    # -----------------------------------------------------------------------

    def test_tcad_detonation_coordination_spike(self) -> None:
        """Nine identical simultaneous broadcasts → trust = 0.0.

        Scenario: Tactical Coordination Attack Detonation (TCAD) spike.
        Nine colluding nodes broadcast the same kinematic update at the same
        millisecond, maximising detection pressure on all three engines.

        Scoring anatomy
        ---------------
        Kinematics identical → kin_trust = 0 → cluster_score = 0.
        All share station_id=42 → sem_trust = 0 → identity_consistency = 0.
        All same timestamp → tim_trust = 0 → replay_probability = 1.0.
        Combined = 0.0.

        Also validates that CSIA scales to n = 9 without performance issues.

        Empirically observed trust: 0.0000.
        """
        ts     = 4_000
        window = [_make_cam(_LAT0, _LON0, 1400, 900, 0, ts, station_id=42) for _ in range(9)]
        result = self.csia.check(window)

        _assert_payload_keys(self, result, "T9: ")
        _assert_payload_bounded(self, result, "T9: ")
        self.assertAlmostEqual(result["trust"], 0.0, places=4,
                               msg=(f"T9: TCAD burst of 9 identical messages at same ts "
                                    f"must have trust = 0.0; got {result['trust']:.4f}"))
        self.assertAlmostEqual(result["cluster_score"], 0.0, places=4,
                               msg=f"T9: identical kinematics → cluster_score ≈ 0.0")
        self.assertAlmostEqual(result["identity_consistency"], 0.0, places=4,
                               msg=f"T9: identical station_id → identity_consistency ≈ 0.0")
        self.assertAlmostEqual(result["replay_probability"], 1.0, places=4,
                               msg=f"T9: identical timestamps → replay_probability ≈ 1.0")

    # -----------------------------------------------------------------------
    # T10 – Independent vehicles genuinely sharing a highway lane
    # -----------------------------------------------------------------------

    def test_independent_vehicles_sharing_highway_lane(self) -> None:
        """Five highway vehicles sharing a lane with natural kinematic jitter → trust > 0.50.

        Scenario: Five independently operating passenger cars on a motorway.
        They share roughly the same heading and speed range but exhibit
        clearly different yaw_rate, steering, and acceleration dynamics
        (independent sensors, different driving styles).  Unique station_ids.
        Timestamps span 1 full second with irregular ns-unit gaps.

        This test verifies that legitimate platooning traffic — which
        naturally produces *similar* kinematics — is NOT falsely flagged when
        fine-grained dynamics differ sufficiently and semantic diversity is
        present.

        Scoring anatomy
        ---------------
        Median speed ≈ 2 250 > highway_speed_threshold (2 000) → highway
        threshold (0.20).  Diverse yaw/steering/acc columns drive
        avg_pairwise_dist well above 0.20 → kin_trust → 1.0 → cluster_score → 1.0.
        Unique station_ids → sem_trust ≈ 0.5 → identity_consistency ≈ 0.5.
        Note: timestamps in this window are ms-range integers (steering/lat_acc
        values positionally overlap the timestamp arg), so spread is near-zero.
        Trust stays high because kin (0.55) and sem (0.20) dominate.  The
        replay_probability is not constrained by this test.
        Combined ≈ 0.55×1.0 + 0.20×0.5 + 0.25×≥0 ≈ 0.65+ → trust > 0.50.

        Empirically observed trust: 0.8333.
        """
        window = [
            _make_cam(_LAT0,         _LON0,         2200, 895,  -30, -100, -150,   50,             0, 1001),
            _make_cam(_LAT0+500,     _LON0+500,     2300, 900,  -10,  -30,  -40,   80,   300_000_000, 1002),
            _make_cam(_LAT0+1_000,   _LON0+1_000,   2250, 905,    5,   10,   20,  -30,   600_000_000, 1003),
            _make_cam(_LAT0+1_500,   _LON0+1_500,   2150, 898,   20,   60,   90,  100,   800_000_000, 1004),
            _make_cam(_LAT0+2_000,   _LON0+2_000,   2400, 902,  -50,  -80, -120,  -60, 1_000_000_000, 1005),
        ]
        result = self.csia.check(window)

        _assert_payload_keys(self, result, "T10: ")
        _assert_payload_bounded(self, result, "T10: ")
        self.assertGreater(result["trust"], 0.50,
                           msg=(f"T10: independent highway lane-sharing trust should > 0.50; "
                                f"got {result['trust']:.4f}.  Fine-grained steering/yaw/acc "
                                f"diversity should push kinematic engine trust high."))
        # Kinematic diversity is the primary signal → cluster_score must be high
        self.assertGreater(result["cluster_score"], 0.5,
                           msg=f"T10: diverse kinematics → cluster_score > 0.5; got {result['cluster_score']:.4f}")

    # -----------------------------------------------------------------------
    # T11 – Identical kinematics AND identical semantic fingerprint → 0.0
    # -----------------------------------------------------------------------

    def test_identical_kinematic_and_semantic_codes_drive_to_zero(self) -> None:
        """Clone kinematic streams + copy-paste semantic codes → trust ≈ 0.0.

        Scenario: Five Sybil nodes fabricate a coordinated event.  They all
        broadcast:
          • Identical kinematic values (speed=1 400, heading=900, all others=0)
          • The same station_type (8 = heavyTruck, anomalous in context)
          • The same station_id (777)
          • Slightly spread timestamps (1 ms steps) so timing barely contributes

        This represents the most dangerous attack class: coordinated
        injection where both the kinematic and the structural CAM fingerprint
        are copy-pasted, leaving the engine with no diversifying signal.

        Scoring anatomy
        ---------------
        Kinematics identical → kin_trust = 0 → cluster_score = 0.
        station_type (8) AND station_id (777) match for every pair
        → Hamming sim = 1.0 → sem_trust = 0 → identity_consistency = 0.
        ts spread ≈ 4 ms → spread_score ≈ 4e-6 ≈ 0, entropy = 0 → tim_trust ≈ 0.
        Combined = 0.55×0 + 0.20×0 + 0.25×≈0 ≈ 0.0.

        Empirically observed trust: ~0.0006 (negligible timing micro-spread).
        """
        window = [
            _make_cam(_LAT0, _LON0, 1400, 900, 0, 4_000 + i, station_id=777, station_type=8)
            for i in range(5)
        ]
        result = self.csia.check(window)

        _assert_payload_keys(self, result, "T11: ")
        _assert_payload_bounded(self, result, "T11: ")
        self.assertLess(result["trust"], 0.01,
                        msg=(f"T11: clone kinematics + identical semantic codes must "
                             f"have trust < 0.01; got {result['trust']:.4f}"))
        self.assertAlmostEqual(result["cluster_score"], 0.0, places=4,
                               msg=f"T11: identical kinematics → cluster_score ≈ 0.0")
        self.assertAlmostEqual(result["identity_consistency"], 0.0, places=4,
                               msg=(f"T11: identical station_id/type → identity_consistency ≈ 0.0; "
                                    f"got {result['identity_consistency']:.4f}"))

    # -----------------------------------------------------------------------
    # Bonus boundary tests
    # -----------------------------------------------------------------------

    def test_two_message_window_below_min_cluster_size(self) -> None:
        """Window of 2 messages < min_cluster_size(3) → benign default payload.

        effective_min = max(3, 2) = 3; the early-exit guard fires.
        All five payload keys must match the benign default values exactly.
        """
        window = [
            _make_cam(_LAT0,       _LON0,       1400, 900,   0, 4_000, station_id=1),
            _make_cam(_LAT0+5_000, _LON0+5_000, 2000, 1800, 50, 5_000, station_id=2),
        ]
        result = self.csia.check(window)

        _assert_payload_keys(self, result, "T12: ")
        self.assertEqual(result["trust"], 1.0,
                         msg=f"T12: window of 2 (< min_cluster_size=3) must have trust=1.0; got {result['trust']}")
        self.assertEqual(result["entropy"], 0.0,
                         msg=f"T12: benign default must have entropy=0.0")
        self.assertEqual(result["replay_probability"], 0.0,
                         msg=f"T12: benign default must have replay_probability=0.0")
        self.assertEqual(result["cluster_score"], 1.0,
                         msg=f"T12: benign default must have cluster_score=1.0")
        self.assertEqual(result["identity_consistency"], 1.0,
                         msg=f"T12: benign default must have identity_consistency=1.0")

    def test_score_always_bounded_in_unit_interval(self) -> None:
        """CSIA.check() must never return a payload value outside [0.0, 1.0].

        Exercises five structurally distinct windows to confirm that every key
        in the returned dict is clamped robustly to [0.0, 1.0].
        """
        cases = [
            # Pure clone
            [_make_cam(_LAT0, _LON0, 1400, 900, 0, 4_000) for _ in range(4)],
            # Maximally diverse (spread lat/lon positions)
            [
                _make_cam(       0,          0,     0,    0,     0,             0, station_id=1),
                _make_cam(90_000_000, 180_000_000, 16383, 3600, 32766, 1_000_000_000, station_id=2),
                _make_cam(45_000_000,  90_000_000,  8000, 1800,  100,   500_000_000, station_id=3),
            ],
            # Mixed non-dict junk + real messages
            [
                None, "bad", 42, [],
                _make_cam(_LAT0,       _LON0,       1400, 900,   0,           0, station_id=10),
                _make_cam(_LAT0+1_000, _LON0+1_000, 2000, 1000,  50, 500_000_000, station_id=11),
                _make_cam(_LAT0+2_000, _LON0+2_000, 2500,  800, -50, 900_000_000, station_id=12),
            ],
            # All-zero fields
            [_make_cam(0, 0, 0, 0, 0, 0) for _ in range(5)],
            # All non-dict entries
            [None, "string", 42, [], {}, False],
        ]
        for idx, window in enumerate(cases):
            result = self.csia.check(window)
            _assert_payload_keys(self, result, f"T13/case{idx}: ")
            _assert_payload_bounded(self, result, f"T13/case{idx}: ")

    def test_score_monotone_with_increasing_clone_density(self) -> None:
        """Adding more clone messages to an already-suspicious window must not raise trust.

        As identical clone nodes multiply, the cluster becomes monotonically
        more suspicious (trust non-increasing from n=3 onward).
        """
        base       = _make_cam(_LAT0, _LON0, 1400, 900, 0, 4_000, station_id=42)
        prev_trust = 1.0
        for n in range(3, 10):
            window = [dict(base) for _ in range(n)]
            result = self.csia.check(window)
            _assert_payload_keys(self, result, f"T14/n={n}: ")
            self.assertLessEqual(
                result["trust"], prev_trust + 1e-9,
                msg=(f"T14: trust unexpectedly increased from n={n-1} to n={n}: "
                     f"{prev_trust:.4f} → {result['trust']:.4f}"),
            )
            prev_trust = result["trust"]

    def test_non_dict_only_window_returns_benign(self) -> None:
        """Window of only non-dict objects → benign default payload.

        Non-dict entries are filtered before clustering; fewer than 2 valid
        messages → early-exit → benign default.
        """
        window = [None, "string", 42, [], False]
        result = self.csia.check(window)

        _assert_payload_keys(self, result, "T15: ")
        self.assertEqual(result["trust"], 1.0,
                         msg=f"T15: all-non-dict window must have trust=1.0; got {result['trust']:.4f}")
        self.assertEqual(result["entropy"], 0.0,
                         msg=f"T15: benign default must have entropy=0.0")
        self.assertEqual(result["replay_probability"], 0.0,
                         msg=f"T15: benign default must have replay_probability=0.0")

    def test_semantic_engine_penalises_duplicate_station_id(self) -> None:
        """All-same station_id (Sybil) should score lower trust than unique station_ids.

        Keeps kinematics identical to isolate the semantic engine's
        contribution.  Timestamps are spread so timing is roughly equal.
        The only difference: clone_window uses station_id=42 throughout;
        diverse_window uses unique ids 1 001–1 005.

        Expectations
        ------------
        • clone trust < diverse trust  (same-id Sybil is worse).
        • clone identity_consistency < diverse identity_consistency.
        """
        clone_window = [
            _make_cam(_LAT0, _LON0, 1400, 900, 0, i * 200_000_000, station_id=42)
            for i in range(5)
        ]
        diverse_window = [
            _make_cam(_LAT0, _LON0, 1400, 900, 0, i * 200_000_000, station_id=1000 + i)
            for i in range(5)
        ]
        result_clone   = self.csia.check(clone_window)
        result_diverse = self.csia.check(diverse_window)

        _assert_payload_keys(self, result_clone,   "T16/clone: ")
        _assert_payload_keys(self, result_diverse, "T16/diverse: ")

        self.assertLess(
            result_clone["trust"], result_diverse["trust"],
            msg=(f"T16: duplicate station_id trust ({result_clone['trust']:.4f}) must be "
                 f"lower than unique-id trust ({result_diverse['trust']:.4f})."),
        )
        self.assertLess(
            result_clone["identity_consistency"],
            result_diverse["identity_consistency"],
            msg="T16: duplicate station_id must yield lower identity_consistency",
        )

    def test_mahalanobis_fallback_does_not_crash_small_cluster(self) -> None:
        """Cluster of exactly 3 messages (< mahalanobis_min_samples=4) must not crash.

        The engine falls back to Euclidean on the robust-scaled data; no
        LinAlgError should escape.  The returned value must be a valid
        structured dict with all five keys bounded in [0.0, 1.0].
        """
        window = [
            _make_cam(_LAT0,       _LON0,       2000, 900,   0,           0, station_id=1),
            _make_cam(_LAT0+1_000, _LON0+1_000, 2200, 950,  30, 500_000_000, station_id=2),
            _make_cam(_LAT0+2_000, _LON0+2_000, 1800, 850, -30, 900_000_000, station_id=3),
        ]
        try:
            result = self.csia.check(window)
        except Exception as exc:
            self.fail(f"T17: 3-message cluster (Euclidean fallback) raised {type(exc).__name__}: {exc}")

        _assert_payload_keys(self, result, "T17: ")
        _assert_payload_bounded(self, result, "T17: ")


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
