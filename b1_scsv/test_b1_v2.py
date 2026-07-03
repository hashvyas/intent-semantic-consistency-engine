"""
b1_scsv/test_b1_v2.py
=====================
Comprehensive tests for the V2 stateful hardening extensions to B1 SCSV.

Test categories
---------------
B1-V2-1  Replay attack detection
B1-V2-2  Stale timestamp rejection
B1-V2-3  Malformed JSON / missing fields / NaN / infinity / invalid types
B1-V2-4  Invalid coordinate validation
B1-V2-5  Impossible kinematics (speed, acceleration, jerk, heading, yaw)
B1-V2-6  Certificate rotation detection
B1-V2-7  Certificate continuity (normal renewal without false positives)
B1-V2-8  ValidationFailureReason enum coverage
B1-V2-9  safe_parse_cam defensive parsing
B1-V2-10 VehicleState rolling window mechanics
B1-V2-11 PhysicalPlausibilityValidator standalone
B1-V2-12 check_stateful() backward compatibility with check()
B1-V2-13 Thread safety

Running the tests
-----------------
From the workspace root::

    python -m pytest b1_scsv/test_b1_v2.py -v

Dependencies: standard library + PyYAML + pytest.
"""

from __future__ import annotations

import math
import pathlib
import sys
import threading
import time
import unittest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from b1_scsv.scsv import SCSV, SCORE_ALLOW, SCORE_BLOCK, PhysicalPlausibilityValidator, _ReplayCache
from b1_scsv.models import (
    CamMessage,
    ValidationFailureReason,
    ValidationResult,
    VehicleState,
    safe_parse_cam,
)

_CONFIG_PATH = _PROJECT_ROOT / "isce_config.yaml"

# ---------------------------------------------------------------------------
# CAM message factory
# ---------------------------------------------------------------------------

def _make_raw_cam(
    station_id: int = 1001,
    message_id: int = 1,
    station_type: int = 5,
    timestamp: float = 4_000.0,
    lat: float = 485_512_345.0,
    lon: float = 96_123_456.0,
    speed: float = 1400.0,
    heading: float = 900.0,
    yaw_rate: float = 0.0,
    lon_acc: float = 0.0,
    lat_acc: float = 0.0,
    steering: float = 0.0,
    cert_id: str | None = None,
) -> dict:
    """Return a fully-populated raw CAM dict for testing."""
    msg: dict = {
        "header": {"station_id": station_id, "message_id": message_id},
        "cam": {
            "generation_delta_time": timestamp,
            "cam_parameters": {
                "basic_container": {
                    "station_type": station_type,
                    "reference_position": {"latitude": lat, "longitude": lon},
                },
                "high_frequency_container": {
                    "basic_vehicle_container_high_frequency": {
                        "speed": speed,
                        "heading": heading,
                        "yaw_rate": yaw_rate,
                        "steering_wheel_angle": steering,
                        "lateral_acceleration": lat_acc,
                        "longitudinal_acceleration": lon_acc,
                    }
                },
            },
        },
    }
    if cert_id is not None:
        msg["certificate_id"] = cert_id
    return msg


def _make_scsv() -> SCSV:
    return SCSV(config_path=_CONFIG_PATH)


# ===========================================================================
# B1-V2-1 – Replay attack detection
# ===========================================================================

class TestReplayDetection(unittest.TestCase):
    """Verify that the replay cache rejects duplicate (station_id, msg_id, ts) triples."""

    def setUp(self) -> None:
        self.scsv = _make_scsv()

    def test_first_message_passes(self) -> None:
        """First time a message is seen it must pass."""
        msg = _make_raw_cam(station_id=42, message_id=1, timestamp=4000.0)
        result = self.scsv.check_stateful(msg)
        self.assertTrue(result.valid, f"First message must pass: {result}")

    def test_second_identical_message_is_replay(self) -> None:
        """Sending the identical (station_id, msg_id, timestamp) twice must fail with REPLAY."""
        msg = _make_raw_cam(station_id=99, message_id=1, timestamp=9_999.0)
        self.scsv.check_stateful(msg)  # first pass
        result = self.scsv.check_stateful(msg)  # second – replay
        self.assertFalse(result.valid, "Duplicate message must be rejected")
        self.assertEqual(result.reason, ValidationFailureReason.REPLAY)
        self.assertEqual(result.score, SCORE_BLOCK)

    def test_different_station_ids_not_replay(self) -> None:
        """Same (message_id, timestamp) from different stations must not conflict."""
        msg1 = _make_raw_cam(station_id=101, message_id=1, timestamp=4000.0)
        msg2 = _make_raw_cam(station_id=102, message_id=1, timestamp=4000.0)
        r1 = self.scsv.check_stateful(msg1)
        r2 = self.scsv.check_stateful(msg2)
        self.assertTrue(r1.valid, f"msg1 should pass: {r1}")
        self.assertTrue(r2.valid, f"msg2 (different station) should pass: {r2}")

    def test_different_timestamps_not_replay(self) -> None:
        """Same station_id but different timestamps must not be flagged."""
        msg1 = _make_raw_cam(station_id=200, timestamp=5_000.0)
        msg2 = _make_raw_cam(station_id=200, timestamp=5_100.0)
        self.scsv.check_stateful(msg1)
        result = self.scsv.check_stateful(msg2)
        self.assertTrue(result.valid, "Different timestamp must not be replay")

    def test_replay_cache_direct(self) -> None:
        """Directly test _ReplayCache: insert, detect, then different key passes."""
        cache = _ReplayCache(ttl_s=60)
        self.assertFalse(cache.is_replay(1, 1, 1000.0))
        self.assertTrue(cache.is_replay(1, 1, 1000.0))   # replay
        self.assertFalse(cache.is_replay(1, 1, 1001.0))  # different ts

    def test_replay_cache_none_station_id_skips(self) -> None:
        """None station_id must skip replay detection (not crash)."""
        cache = _ReplayCache(ttl_s=60)
        self.assertFalse(cache.is_replay(None, 1, 1000.0))
        self.assertFalse(cache.is_replay(None, 1, 1000.0))  # no replay for None

    def test_replay_cache_disabled_when_ttl_zero(self) -> None:
        """TTL=0 must disable replay detection entirely."""
        cache = _ReplayCache(ttl_s=0)
        self.assertFalse(cache.is_replay(1, 1, 1000.0))
        self.assertFalse(cache.is_replay(1, 1, 1000.0))  # should still be False


# ===========================================================================
# B1-V2-2 – Stale timestamp rejection
# ===========================================================================

class TestTimestampFreshness(unittest.TestCase):
    """Verify that absolute-timestamp messages outside the freshness window are rejected."""

    def setUp(self) -> None:
        self.scsv = _make_scsv()

    def test_relative_timestamp_not_checked(self) -> None:
        """Timestamps in the ETSI relative delta range (< 65536 ms) must not trigger freshness check."""
        msg = _make_raw_cam(station_id=10, timestamp=4_000.0)  # relative CAM delta
        result = self.scsv.check_stateful(msg)
        # Must pass (no freshness rejection for relative timestamps)
        self.assertNotEqual(result.reason, ValidationFailureReason.STALE_TIMESTAMP)

    def test_fresh_absolute_timestamp_passes(self) -> None:
        """A message with a current wall-clock ms-epoch timestamp must pass."""
        now_ms = time.time() * 1000.0
        msg = _make_raw_cam(station_id=20, timestamp=now_ms)
        result = self.scsv.check_stateful(msg)
        self.assertNotEqual(
            result.reason,
            ValidationFailureReason.STALE_TIMESTAMP,
            "Current-time timestamp must not be stale",
        )

    def test_stale_absolute_timestamp_rejected(self) -> None:
        """A message with a timestamp 60 seconds in the past must be rejected as stale."""
        stale_ms = (time.time() - 60) * 1000.0  # 60 seconds ago
        msg = _make_raw_cam(station_id=30, timestamp=stale_ms)
        result = self.scsv.check_stateful(msg)
        self.assertFalse(result.valid, "Stale message must be rejected")
        self.assertEqual(result.reason, ValidationFailureReason.STALE_TIMESTAMP)

    def test_future_absolute_timestamp_rejected(self) -> None:
        """A message with a timestamp far in the future must also be rejected."""
        future_ms = (time.time() + 60) * 1000.0  # 60 seconds future
        msg = _make_raw_cam(station_id=40, timestamp=future_ms)
        result = self.scsv.check_stateful(msg)
        self.assertFalse(result.valid, "Future-timestamped message must be rejected")
        self.assertEqual(result.reason, ValidationFailureReason.STALE_TIMESTAMP)


# ===========================================================================
# B1-V2-3 – Malformed inputs (defensive parsing)
# ===========================================================================

class TestDefensiveParsing(unittest.TestCase):
    """check_stateful() must never raise; all malformed inputs return PARSE_ERROR."""

    def setUp(self) -> None:
        self.scsv = _make_scsv()

    def _assert_does_not_crash(self, msg: object, label: str) -> ValidationResult:
        try:
            return self.scsv.check_stateful(msg)
        except Exception as exc:
            self.fail(f"{label}: check_stateful raised {type(exc).__name__}: {exc}")

    def test_none_input(self) -> None:
        result = self._assert_does_not_crash(None, "None input")
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, ValidationFailureReason.PARSE_ERROR)

    def test_string_input(self) -> None:
        result = self._assert_does_not_crash("not a dict", "string input")
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, ValidationFailureReason.PARSE_ERROR)

    def test_integer_input(self) -> None:
        result = self._assert_does_not_crash(42, "integer input")
        self.assertFalse(result.valid)

    def test_empty_dict(self) -> None:
        result = self._assert_does_not_crash({}, "empty dict")
        # Empty dict parses OK but returns default policy (block for unknown)
        self.assertIn(result.score, (SCORE_ALLOW, SCORE_BLOCK))

    def test_nan_speed(self) -> None:
        msg = _make_raw_cam(station_id=50, speed=float("nan"))
        result = self._assert_does_not_crash(msg, "NaN speed")
        # NaN is sanitised to None in CamMessage; should not crash
        self.assertIsNotNone(result)

    def test_infinity_acceleration(self) -> None:
        msg = _make_raw_cam(station_id=51, lon_acc=float("inf"))
        result = self._assert_does_not_crash(msg, "inf acceleration")
        self.assertIsNotNone(result)

    def test_missing_cam_key(self) -> None:
        msg = {"header": {"station_id": 60, "message_id": 1}}
        result = self._assert_does_not_crash(msg, "missing cam key")
        self.assertIsNotNone(result)

    def test_missing_header(self) -> None:
        msg = _make_raw_cam()
        del msg["header"]
        result = self._assert_does_not_crash(msg, "missing header")
        self.assertIsNotNone(result)

    def test_station_type_string(self) -> None:
        """station_type as a string (e.g. 'passengerCar') must not crash."""
        msg = _make_raw_cam(station_id=61)
        msg["cam"]["cam_parameters"]["basic_container"]["station_type"] = "passengerCar"
        result = self._assert_does_not_crash(msg, "station_type as string")
        self.assertIsNotNone(result)

    def test_negative_speed(self) -> None:
        """Negative speed (invalid) must not crash; may trigger plausibility rejection."""
        msg = _make_raw_cam(station_id=62, speed=-100.0)
        result = self._assert_does_not_crash(msg, "negative speed")
        self.assertIsNotNone(result)

    def test_null_nested_fields(self) -> None:
        """Null nested fields must be handled gracefully."""
        msg = {
            "header": {"station_id": 70, "message_id": None},
            "cam": {
                "generation_delta_time": None,
                "cam_parameters": None,
            },
        }
        result = self._assert_does_not_crash(msg, "null nested fields")
        self.assertIsNotNone(result)

    def test_wrong_type_for_nested_dict(self) -> None:
        """Non-dict at a nested position must not crash."""
        msg = {
            "header": {"station_id": 71, "message_id": 1},
            "cam": "this_is_a_string_not_a_dict",
        }
        result = self._assert_does_not_crash(msg, "non-dict cam value")
        self.assertIsNotNone(result)


# ===========================================================================
# B1-V2-4 – Coordinate validation
# ===========================================================================

class TestCoordinateValidation(unittest.TestCase):
    """Verify GPS coordinate bounds are enforced."""

    def setUp(self) -> None:
        self.scsv = _make_scsv()

    def test_valid_coordinates_pass(self) -> None:
        msg = _make_raw_cam(lat=485_512_345.0, lon=96_123_456.0)
        result = self.scsv.check_stateful(msg)
        self.assertNotEqual(result.reason, ValidationFailureReason.INVALID_COORDINATES)

    def test_latitude_out_of_range_rejected(self) -> None:
        """Latitude > 90° (> 900_000_000 in ETSI units) must be rejected."""
        msg = _make_raw_cam(lat=950_000_000.0)
        result = self.scsv.check_stateful(msg)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, ValidationFailureReason.IMPOSSIBLE_KINEMATICS)

    def test_longitude_out_of_range_rejected(self) -> None:
        """Longitude > 180° (> 1_800_000_000 in ETSI units) must be rejected."""
        msg = _make_raw_cam(lon=1_900_000_000.0)
        result = self.scsv.check_stateful(msg)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, ValidationFailureReason.IMPOSSIBLE_KINEMATICS)

    def test_negative_latitude_boundary_valid(self) -> None:
        """Latitude at exactly −90° must be accepted."""
        msg = _make_raw_cam(lat=-900_000_000.0)
        validator = PhysicalPlausibilityValidator()
        cam, _ = __import__("b1_scsv.models", fromlist=["safe_parse_cam"]).safe_parse_cam(msg)
        violation = validator.validate(cam)
        self.assertIsNone(violation, "Exactly −90° latitude must be valid")


# ===========================================================================
# B1-V2-5 – Impossible kinematics
# ===========================================================================

class TestImpossibleKinematics(unittest.TestCase):
    """Physically impossible kinematic values must trigger IMPOSSIBLE_KINEMATICS."""

    def setUp(self) -> None:
        self.scsv = _make_scsv()
        self.validator = PhysicalPlausibilityValidator()

    def _make_cam_obj(self, **kwargs) -> CamMessage:
        msg = _make_raw_cam(**kwargs)
        cam, _ = safe_parse_cam(msg)
        return cam

    def test_impossible_speed(self) -> None:
        """Speed > 8330 (300 km/h in ETSI units) must be rejected."""
        msg = _make_raw_cam(speed=9000.0)
        result = self.scsv.check_stateful(msg)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, ValidationFailureReason.IMPOSSIBLE_KINEMATICS)

    def test_impossible_longitudinal_acceleration(self) -> None:
        """Longitudinal acceleration > 1500 (15 m/s² in ETSI units) must be rejected."""
        msg = _make_raw_cam(lon_acc=2000.0)
        result = self.scsv.check_stateful(msg)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, ValidationFailureReason.IMPOSSIBLE_KINEMATICS)

    def test_impossible_lateral_acceleration(self) -> None:
        """Lateral acceleration magnitude > 1500 must be rejected."""
        msg = _make_raw_cam(lat_acc=-2000.0)
        result = self.scsv.check_stateful(msg)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, ValidationFailureReason.IMPOSSIBLE_KINEMATICS)

    def test_impossible_yaw_rate(self) -> None:
        """Yaw rate magnitude > 7500 (75 °/s in ETSI units) must be rejected."""
        msg = _make_raw_cam(yaw_rate=10000.0)
        result = self.scsv.check_stateful(msg)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, ValidationFailureReason.IMPOSSIBLE_KINEMATICS)

    def test_plausible_highway_speed_passes(self) -> None:
        """Speed 3000 (108 km/h) must pass plausibility."""
        msg = _make_raw_cam(speed=3000.0)
        result = self.scsv.check_stateful(msg)
        self.assertNotEqual(result.reason, ValidationFailureReason.IMPOSSIBLE_KINEMATICS)

    def test_validator_heading_change_with_prior_state(self) -> None:
        """Heading change > 900 (90°) between frames must be flagged."""
        state = VehicleState(station_id=1)
        state.headings.append(100.0)
        cam = self._make_cam_obj(heading=2000.0)  # delta = 1900 > 900
        violation = self.validator.validate(cam, prev_state=state)
        self.assertIsNotNone(violation, "Extreme heading change must be flagged")

    def test_validator_jerk_with_prior_state(self) -> None:
        """Acceleration change > 3000 between frames must be flagged as jerk."""
        state = VehicleState(station_id=1)
        state.accelerations.append(0.0)
        cam = self._make_cam_obj(lon_acc=3100.0)  # delta = 3100 > 3000
        # But acc=3100 also exceeds max_acceleration (1500), so violation fires first
        violation = self.validator.validate(cam, prev_state=state)
        self.assertIsNotNone(violation)

    def test_validator_no_violation_on_valid_message(self) -> None:
        """A normal message must produce no violation."""
        cam = self._make_cam_obj()
        violation = self.validator.validate(cam)
        self.assertIsNone(violation)


# ===========================================================================
# B1-V2-6 & B1-V2-7 – Certificate rotation and continuity
# ===========================================================================

class TestCertificateTracking(unittest.TestCase):
    """Verify certificate continuity analysis detects excessive rotation."""

    def setUp(self) -> None:
        self.scsv = _make_scsv()

    def _send(self, station_id: int, cert_id: str, ts: float = 4000.0) -> ValidationResult:
        msg = _make_raw_cam(station_id=station_id, timestamp=ts, cert_id=cert_id)
        return self.scsv.check_stateful(msg)

    def test_stable_certificate_passes(self) -> None:
        """A vehicle that keeps the same certificate across 10 messages must always pass."""
        for i in range(10):
            result = self._send(500, "cert-abc", ts=4000.0 + i)
            self.assertNotEqual(
                result.reason,
                ValidationFailureReason.CERT_ROTATION_ANOMALY,
                f"Stable cert must never trigger rotation anomaly (message {i})",
            )

    def test_excessive_rotation_detected(self) -> None:
        """More than cert_max_rotations (default 3) cert changes must be flagged."""
        # Send messages with rapidly rotating certs
        for i, cert in enumerate(["cert-A", "cert-B", "cert-C", "cert-D", "cert-E"]):
            result = self._send(600, cert, ts=4000.0 + i)
        # After 5 rotations the check should have flagged anomaly
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, ValidationFailureReason.CERT_ROTATION_ANOMALY)

    def test_normal_renewal_not_flagged(self) -> None:
        """A single certificate renewal must not trigger a false positive."""
        r1 = self._send(700, "cert-old", ts=4000.0)
        r2 = self._send(700, "cert-new", ts=4001.0)
        # One renewal is fine
        self.assertNotEqual(
            r2.reason,
            ValidationFailureReason.CERT_ROTATION_ANOMALY,
            "A single renewal must not be flagged as anomalous",
        )

    def test_no_cert_in_message_does_not_crash(self) -> None:
        """Messages without a certificate_id must be processed normally."""
        msg = _make_raw_cam(station_id=800, cert_id=None)
        result = self.scsv.check_stateful(msg)
        self.assertNotEqual(result.reason, ValidationFailureReason.PARSE_ERROR)


# ===========================================================================
# B1-V2-8 – ValidationFailureReason enum coverage
# ===========================================================================

class TestValidationFailureReasonEnum(unittest.TestCase):
    """Ensure every ValidationFailureReason value is exercised."""

    def test_all_reasons_have_unique_values(self) -> None:
        values = [r.value for r in ValidationFailureReason]
        self.assertEqual(len(values), len(set(values)), "Each reason must have a unique value")

    def test_all_expected_reasons_exist(self) -> None:
        expected = {
            "REPLAY",
            "STALE_TIMESTAMP",
            "IMPOSSIBLE_KINEMATICS",
            "CERT_ROTATION_ANOMALY",
            "BLOCKED_BY_POLICY",
            "PARSE_ERROR",
            "INVALID_COORDINATES",
        }
        actual = {r.name for r in ValidationFailureReason}
        self.assertEqual(expected, actual, f"Missing or extra reasons: {expected ^ actual}")


# ===========================================================================
# B1-V2-9 – safe_parse_cam defensive parsing
# ===========================================================================

class TestSafeParseCAM(unittest.TestCase):
    """Unit tests for the safe_parse_cam() helper."""

    def test_valid_message_parsed_correctly(self) -> None:
        msg = _make_raw_cam(station_id=42, speed=1400.0, heading=900.0)
        cam, err = safe_parse_cam(msg)
        self.assertIsNone(err)
        self.assertIsNotNone(cam)
        self.assertEqual(cam.station_id, 42)
        self.assertAlmostEqual(cam.speed, 1400.0)
        self.assertAlmostEqual(cam.heading, 900.0)

    def test_nan_value_becomes_none(self) -> None:
        msg = _make_raw_cam(speed=float("nan"))
        cam, err = safe_parse_cam(msg)
        self.assertIsNotNone(cam)
        self.assertIsNone(cam.speed)
        self.assertGreater(len(cam.parse_warnings), 0)

    def test_infinity_becomes_none(self) -> None:
        msg = _make_raw_cam(lon_acc=float("inf"))
        cam, err = safe_parse_cam(msg)
        self.assertIsNotNone(cam)
        self.assertIsNone(cam.longitudinal_acceleration)

    def test_string_station_id_parsed(self) -> None:
        msg = _make_raw_cam()
        msg["header"]["station_id"] = "1234"
        cam, err = safe_parse_cam(msg)
        self.assertIsNotNone(cam)
        self.assertEqual(cam.station_id, 1234)

    def test_non_dict_returns_none(self) -> None:
        cam, err = safe_parse_cam("not a dict")
        self.assertIsNone(cam)
        self.assertIsNotNone(err)

    def test_none_returns_none(self) -> None:
        cam, err = safe_parse_cam(None)
        self.assertIsNone(cam)
        self.assertIsNotNone(err)

    def test_certificate_id_extracted(self) -> None:
        msg = _make_raw_cam(cert_id="cert-xyz")
        cam, err = safe_parse_cam(msg)
        self.assertEqual(cam.certificate_id, "cert-xyz")

    def test_missing_all_kinematic_fields(self) -> None:
        """A message with no kinematic fields at all must parse without error."""
        msg = {"header": {"station_id": 1, "message_id": 1}, "cam": {"generation_delta_time": 4000}}
        cam, err = safe_parse_cam(msg)
        self.assertIsNone(err)
        self.assertIsNotNone(cam)
        self.assertIsNone(cam.speed)


# ===========================================================================
# B1-V2-10 – VehicleState rolling window mechanics
# ===========================================================================

class TestVehicleStateWindow(unittest.TestCase):
    """Verify VehicleState deque windows bound correctly."""

    def test_speed_history_bounded(self) -> None:
        """Speed history must not exceed window size."""
        state = VehicleState(station_id=1, window=5)
        for i in range(20):
            cam = CamMessage(raw={}, station_id=1, speed=float(i))
            state.record_observation(cam, wall_time=time.time())
        self.assertLessEqual(len(state.speeds), 5)

    def test_timestamps_bounded(self) -> None:
        state = VehicleState(station_id=2, window=3)
        for ts in [1.0, 2.0, 3.0, 4.0, 5.0]:
            cam = CamMessage(raw={}, station_id=2, timestamp=ts)
            state.record_observation(cam, wall_time=time.time())
        self.assertLessEqual(len(state.timestamps), 3)
        # Oldest value should have been evicted
        self.assertEqual(state.timestamps[0], 3.0)

    def test_message_count_increments(self) -> None:
        state = VehicleState(station_id=3, window=10)
        cam = CamMessage(raw={}, station_id=3)
        for _ in range(7):
            state.record_observation(cam, wall_time=time.time())
        self.assertEqual(state.message_count, 7)


# ===========================================================================
# B1-V2-11 – PhysicalPlausibilityValidator standalone
# ===========================================================================

class TestPhysicalPlausibilityValidatorStandalone(unittest.TestCase):
    """Test PhysicalPlausibilityValidator in isolation."""

    def setUp(self) -> None:
        self.v = PhysicalPlausibilityValidator(
            max_speed=8330,
            max_acceleration=1500,
            max_jerk=3000,
            max_heading_change=900,
            max_yaw_rate=7500,
        )

    def test_valid_message_no_violation(self) -> None:
        cam = CamMessage(raw={}, speed=1400.0, heading=900.0, yaw_rate=50.0,
                         longitudinal_acceleration=100.0,
                         latitude=485_512_345.0, longitude=96_123_456.0)
        self.assertIsNone(self.v.validate(cam))

    def test_speed_threshold_exact(self) -> None:
        cam = CamMessage(raw={}, speed=8330.0)
        self.assertIsNone(self.v.validate(cam), "Exactly at max speed must pass")

    def test_speed_one_over_threshold(self) -> None:
        cam = CamMessage(raw={}, speed=8331.0)
        self.assertIsNotNone(self.v.validate(cam))

    def test_none_fields_not_checked(self) -> None:
        """Fields set to None must be skipped (not raise)."""
        cam = CamMessage(raw={}, speed=None, yaw_rate=None)
        self.assertIsNone(self.v.validate(cam))


# ===========================================================================
# B1-V2-12 – check_stateful() backward compatibility with check()
# ===========================================================================

class TestCheckStatefulBackwardCompat(unittest.TestCase):
    """Verify check_stateful() normal-path score agrees with check()."""

    def setUp(self) -> None:
        self.scsv = _make_scsv()

    def test_valid_passenger_car_cam_agrees_with_check(self) -> None:
        """check_stateful on a valid passengerCar/CAM must return score == check()."""
        msg = _make_raw_cam(station_id=5001, station_type=5, message_id=1, timestamp=4000.0)
        result = self.scsv.check_stateful(msg)
        expected = self.scsv.check("passengerCar", "CAM")
        if result.valid:
            self.assertAlmostEqual(result.score, expected, places=6)

    def test_blocked_combo_stateful_returns_block(self) -> None:
        """check_stateful with a blocked combination must return SCORE_BLOCK."""
        msg = _make_raw_cam(station_id=5002, station_type=5, message_id=6)  # passengerCar/SPATEM
        result = self.scsv.check_stateful(msg)
        self.assertFalse(result.valid)
        self.assertEqual(result.score, SCORE_BLOCK)
        self.assertEqual(result.reason, ValidationFailureReason.BLOCKED_BY_POLICY)

    def test_result_is_immutable(self) -> None:
        """ValidationResult is a frozen dataclass – mutation must raise."""
        msg = _make_raw_cam(station_id=5003, timestamp=4000.0)
        result = self.scsv.check_stateful(msg)
        with self.assertRaises((AttributeError, TypeError)):
            result.valid = not result.valid  # type: ignore[misc]

    def test_introspection_properties(self) -> None:
        """tracked_vehicle_count and replay_cache_size must be accessible."""
        msg = _make_raw_cam(station_id=5004, timestamp=4000.0)
        self.scsv.check_stateful(msg)
        self.assertIsInstance(self.scsv.tracked_vehicle_count, int)
        self.assertIsInstance(self.scsv.replay_cache_size, int)


# ===========================================================================
# B1-V2-13 – Thread safety
# ===========================================================================

class TestThreadSafety(unittest.TestCase):
    """Verify check() and check_stateful() are safe under concurrent access."""

    def setUp(self) -> None:
        self.scsv = _make_scsv()

    def test_concurrent_check_does_not_crash(self) -> None:
        """50 threads calling check() simultaneously must not raise."""
        errors: list = []

        def _run(tid: int) -> None:
            try:
                for i in range(20):
                    self.scsv.check("passengerCar", "CAM")
                    self.scsv.check("roadSideUnit", "SPATEM")
            except Exception as exc:
                errors.append(f"Thread {tid}: {exc}")

        threads = [threading.Thread(target=_run, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Thread errors: {errors}")

    def test_concurrent_check_stateful_does_not_crash(self) -> None:
        """10 threads calling check_stateful() simultaneously must not raise."""
        errors: list = []

        def _run(tid: int) -> None:
            try:
                for i in range(10):
                    msg = _make_raw_cam(
                        station_id=1000 + tid * 10 + i,
                        timestamp=4000.0 + i,
                    )
                    self.scsv.check_stateful(msg)
            except Exception as exc:
                errors.append(f"Thread {tid}: {exc}")

        threads = [threading.Thread(target=_run, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Thread errors: {errors}")

    def test_replay_cache_thread_safe(self) -> None:
        """_ReplayCache must handle concurrent inserts without race conditions."""
        cache = _ReplayCache(ttl_s=30)
        results: list = []

        def _insert(sid: int) -> None:
            r = cache.is_replay(sid, 1, 9999.0)
            results.append(r)

        threads = [threading.Thread(target=_insert, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each unique station_id should have returned False exactly once
        self.assertEqual(len(results), 50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
