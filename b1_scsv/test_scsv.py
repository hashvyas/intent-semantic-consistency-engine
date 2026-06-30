"""
b1_scsv/test_scsv.py
====================
Test suite for B1 – Sender Certificate Semantic Validation (SCSV).

Test categories
---------------
1. **Synthetic / hand-crafted tests** – cover every rule branch, the
   wildcard path, the default-policy fallback, and graceful handling of
   malformed inputs.  Adversarial cases (block-expected) are generated from
   the ETSI station_type enumeration because no real attack samples exist in
   the current V2AIX dataset.

2. **Real decoded CAM integration tests** – load a JSON file of genuine
   ETSI CAM beacons (passengerCar traffic captured in the V2AIX dataset) and
   assert that SCSV.check() returns 1.0 for every one of them, confirming
   the validator does not false-flag legitimate traffic.

Running the tests
-----------------
From the workspace root::

    python -m pytest b1_scsv/test_scsv.py -v

Or without pytest::

    python b1_scsv/test_scsv.py

Dependencies: standard library + PyYAML (already required by scsv.py).
"""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when run directly
# ---------------------------------------------------------------------------
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from b1_scsv.scsv import SCSV, SCORE_ALLOW, SCORE_BLOCK  # noqa: E402

# Path to the shared config (auto-discovered by SCSV, but made explicit here
# so tests fail loudly if the config is missing rather than with a confusing
# AttributeError).
_CONFIG_PATH = _PROJECT_ROOT / "isce_config.yaml"

# Path where the user will drop real decoded CAM samples.
_REAL_SAMPLES_PATH = _PROJECT_ROOT / "data" / "real_cam_samples.json"


# ===========================================================================
# Helper
# ===========================================================================

def _make_scsv() -> SCSV:
    """Instantiate a fresh SCSV validator pointing at the project config."""
    return SCSV(config_path=_CONFIG_PATH)


# ===========================================================================
# Synthetic / hand-crafted tests
# ===========================================================================

class TestSCSVSyntheticAllow(unittest.TestCase):
    """Tests for combinations that must return SCORE_ALLOW (1.0).

    These use well-formed, known-good (station_type, message_type) pairs
    drawn from ETSI ITS standards.  All of them represent legitimate
    on-road or infrastructure behaviour.
    """

    def setUp(self) -> None:
        self.scsv = _make_scsv()

    # -----------------------------------------------------------------------
    # Required test cases from the specification
    # -----------------------------------------------------------------------

    def test_passenger_car_cam_allowed(self) -> None:
        """passengerCar sending CAM must be allowed (score 1.0).

        This is the most common message type in the V2AIX dataset.
        An SCSV that blocks this combination would generate enormous
        false-positive rates in any deployment.
        """
        score = self.scsv.check("passengerCar", "CAM")
        self.assertEqual(score, SCORE_ALLOW, "passengerCar/CAM must score 1.0")

    def test_rsu_cam_allowed(self) -> None:
        """roadSideUnit sending CAM must be allowed (wildcard rule).

        RSUs are trusted infrastructure and are permitted to originate
        any message type; the wildcard rule in the config covers this.
        """
        score = self.scsv.check("roadSideUnit", "CAM")
        self.assertEqual(score, SCORE_ALLOW)

    def test_rsu_spatem_allowed(self) -> None:
        """roadSideUnit sending SPATEM must be allowed (wildcard rule).

        SPATEM (Signal Phase And Timing) is a primary RSU message type.
        """
        score = self.scsv.check("roadSideUnit", "SPATEM")
        self.assertEqual(score, SCORE_ALLOW)

    def test_rsu_mapem_allowed(self) -> None:
        """roadSideUnit sending MAPEM must be allowed (wildcard rule).

        MAPEM (MAP Extended Message) describes intersection topology;
        only RSUs originate it legitimately.
        """
        score = self.scsv.check("roadSideUnit", "MAPEM")
        self.assertEqual(score, SCORE_ALLOW)

    def test_rsu_denm_allowed(self) -> None:
        """roadSideUnit sending DENM must be allowed (wildcard rule)."""
        score = self.scsv.check("roadSideUnit", "DENM")
        self.assertEqual(score, SCORE_ALLOW)

    def test_bus_cam_allowed(self) -> None:
        """bus sending CAM must be allowed.

        Buses participate in C-ITS with CAM beacons just like passenger cars.
        """
        score = self.scsv.check("bus", "CAM")
        self.assertEqual(score, SCORE_ALLOW)

    def test_motorcycle_cam_allowed(self) -> None:
        """motorcycle sending CAM must be allowed."""
        score = self.scsv.check("motorcycle", "CAM")
        self.assertEqual(score, SCORE_ALLOW)

    def test_pedestrian_cam_allowed(self) -> None:
        """pedestrian VRU device sending CAM must be allowed."""
        score = self.scsv.check("pedestrian", "CAM")
        self.assertEqual(score, SCORE_ALLOW)

    def test_heavy_truck_cam_allowed(self) -> None:
        """heavyTruck sending CAM must be allowed."""
        score = self.scsv.check("heavyTruck", "CAM")
        self.assertEqual(score, SCORE_ALLOW)

    def test_passenger_car_denm_allowed(self) -> None:
        """passengerCar sending DENM (hazard warning) must be allowed."""
        score = self.scsv.check("passengerCar", "DENM")
        self.assertEqual(score, SCORE_ALLOW)

    # -----------------------------------------------------------------------
    # Integer-valued inputs (message decoded as numeric IDs)
    # -----------------------------------------------------------------------

    def test_integer_station_type_passenger_car(self) -> None:
        """Integer station_type=5 (passengerCar) with string 'CAM' must be allowed.

        Decoders may emit the raw integer instead of the enum name;
        SCSV must resolve it transparently.
        """
        score = self.scsv.check(5, "CAM")   # 5 = passengerCar per ETSI
        self.assertEqual(score, SCORE_ALLOW)

    def test_integer_message_type_cam(self) -> None:
        """Integer message_type=1 (CAM) with string 'passengerCar' must be allowed."""
        score = self.scsv.check("passengerCar", 1)  # 1 = CAM per ETSI
        self.assertEqual(score, SCORE_ALLOW)

    def test_both_integer_passenger_car_cam(self) -> None:
        """Both station_type=5 and message_type=1 as integers must be allowed."""
        score = self.scsv.check(5, 1)
        self.assertEqual(score, SCORE_ALLOW)

    def test_integer_rsu_spatem(self) -> None:
        """Integer station_type=15 (roadSideUnit) with integer message_type=6 (SPATEM)."""
        score = self.scsv.check(15, 6)
        self.assertEqual(score, SCORE_ALLOW)


class TestSCSVSyntheticBlock(unittest.TestCase):
    """Tests for combinations that must return SCORE_BLOCK (0.0).

    These are **adversarial / synthetic** cases.  No real attack samples
    exist in the current dataset; these test cases are constructed directly
    from the ETSI ITS station_type enumeration and ETSI message-type
    registry to probe the blocking logic.

    The intuition behind each case is documented in the test docstring.
    """

    def setUp(self) -> None:
        self.scsv = _make_scsv()

    def test_passenger_car_spatem_blocked(self) -> None:
        """passengerCar originating SPATEM must be blocked (score 0.0).

        SPATEM (Signal Phase And Timing) is an RSU-only infrastructure
        message.  A passenger car claiming to originate it is either
        misconfigured or spoofed.  Synthetic adversarial case: attacker
        attempts to inject false signal-phase data from a vehicle-type
        certificate.
        """
        score = self.scsv.check("passengerCar", "SPATEM")
        self.assertEqual(score, SCORE_BLOCK)

    def test_passenger_car_mapem_blocked(self) -> None:
        """passengerCar originating MAPEM must be blocked (score 0.0).

        MAPEM describes road-topology; only RSUs/infrastructure publish it.
        Synthetic adversarial case: attacker with a vehicle certificate
        attempts to re-define an intersection map.
        """
        score = self.scsv.check("passengerCar", "MAPEM")
        self.assertEqual(score, SCORE_BLOCK)

    def test_passenger_car_srem_blocked(self) -> None:
        """passengerCar originating SREM (Signal Request) must be blocked.

        SREM (Signal Request Extended Message) is sent by authorised
        fleets (e.g. emergency vehicles) to request green-light priority.
        A plain passenger car must not originate it; doing so is a
        traffic-signal abuse vector.
        """
        score = self.scsv.check("passengerCar", "SREM")
        self.assertEqual(score, SCORE_BLOCK)

    def test_bus_spatem_blocked(self) -> None:
        """bus originating SPATEM must be blocked (score 0.0).

        Buses may legitimately send CAM and DENM but must not claim to
        be a traffic-signal infrastructure node.  Synthetic adversarial
        case: a compromised bus OBU injects false signal-phase data.
        """
        score = self.scsv.check("bus", "SPATEM")
        self.assertEqual(score, SCORE_BLOCK)

    def test_pedestrian_spatem_blocked(self) -> None:
        """pedestrian VRU device originating SPATEM must be blocked.

        A pedestrian personal ITS device has no business emitting
        SPATEM messages.  Synthetic adversarial case: rogue app on
        a smartphone tries to inject signal-phase data.
        """
        score = self.scsv.check("pedestrian", "SPATEM")
        self.assertEqual(score, SCORE_BLOCK)

    def test_unknown_station_type_blocked(self) -> None:
        """Station type 'unknown' (ETSI code 0) sending anything must be blocked.

        An ITS-S that cannot identify itself is untrustworthy by definition.
        Synthetic adversarial case: attacker replays a message with the
        station_type field zeroed out to avoid attribution.
        """
        score_cam = self.scsv.check("unknown", "CAM")
        score_denm = self.scsv.check("unknown", "DENM")
        self.assertEqual(score_cam, SCORE_BLOCK, "unknown/CAM must be blocked")
        self.assertEqual(score_denm, SCORE_BLOCK, "unknown/DENM must be blocked")

    def test_integer_unknown_station_type_blocked(self) -> None:
        """Integer station_type=0 (unknown) must resolve and be blocked.

        Same as above but tests the integer-to-string resolution path.
        """
        score = self.scsv.check(0, "CAM")
        self.assertEqual(score, SCORE_BLOCK)


class TestSCSVDefaultPolicy(unittest.TestCase):
    """Tests for the default_policy fallback path.

    Verifies that a (station_type, message_type) combination not covered by
    any explicit rule triggers the configured default_policy.  The default
    policy in isce_config.yaml is 'allow', so unrecognised combinations
    should return 1.0.
    """

    def setUp(self) -> None:
        self.scsv = _make_scsv()

    def test_unknown_combination_uses_default_policy(self) -> None:
        """An unrecognised (station_type, message_type) pair must use default_policy.

        lightTruck / RTCMEM has no explicit rule in the rule table.  The
        method must fall through to whatever default_policy is configured;
        this test honours that value so it stays valid regardless of whether
        the policy is 'allow' or 'block'.
        """
        expected = self.scsv.default_score   # honours whatever policy is configured
        score = self.scsv.check("lightTruck", "RTCMEM")
        self.assertEqual(score, expected)

    def test_totally_unrecognised_type_uses_default_policy(self) -> None:
        """A string not in any ETSI enum falls through to default_policy.

        Both inputs are syntactically valid strings but unknown to the
        config; the validator must neither crash nor silently accept them
        without consulting the policy.
        """
        expected = self.scsv.default_score
        score = self.scsv.check("alienVehicle", "UBER_MSG")
        self.assertEqual(score, expected)


class TestSCSVMalformedInput(unittest.TestCase):
    """Tests for robustness against malformed or unexpected inputs.

    The pipeline may deliver partially-decoded or corrupt messages.
    SCSV must never raise an exception regardless of input type.
    """

    def setUp(self) -> None:
        self.scsv = _make_scsv()

    def test_malformed_station_type_does_not_crash(self) -> None:
        """None as station_type must not raise; falls back to default_policy.

        Real decoders can emit None for missing fields.
        """
        try:
            score = self.scsv.check(None, "CAM")
        except Exception as exc:
            self.fail(f"SCSV.check raised unexpectedly: {exc}")
        self.assertIn(score, (SCORE_ALLOW, SCORE_BLOCK))

    def test_malformed_message_type_does_not_crash(self) -> None:
        """None as message_type must not raise; falls back to default_policy."""
        try:
            score = self.scsv.check("passengerCar", None)
        except Exception as exc:
            self.fail(f"SCSV.check raised unexpectedly: {exc}")
        self.assertIn(score, (SCORE_ALLOW, SCORE_BLOCK))

    def test_both_none_does_not_crash(self) -> None:
        """Both inputs None must not raise."""
        try:
            score = self.scsv.check(None, None)
        except Exception as exc:
            self.fail(f"SCSV.check raised unexpectedly: {exc}")
        self.assertIn(score, (SCORE_ALLOW, SCORE_BLOCK))

    def test_empty_string_station_type_does_not_crash(self) -> None:
        """Empty string station_type must not raise."""
        try:
            score = self.scsv.check("", "CAM")
        except Exception as exc:
            self.fail(f"SCSV.check raised unexpectedly: {exc}")
        self.assertIn(score, (SCORE_ALLOW, SCORE_BLOCK))

    def test_whitespace_only_station_type_does_not_crash(self) -> None:
        """Whitespace-only station_type must not raise (stripped to empty string)."""
        try:
            score = self.scsv.check("   ", "CAM")
        except Exception as exc:
            self.fail(f"SCSV.check raised unexpectedly: {exc}")
        self.assertIn(score, (SCORE_ALLOW, SCORE_BLOCK))

    def test_integer_out_of_enum_range_does_not_crash(self) -> None:
        """An integer station_type with no enum entry (e.g. 999) must not crash."""
        try:
            score = self.scsv.check(999, "CAM")
        except Exception as exc:
            self.fail(f"SCSV.check raised unexpectedly: {exc}")
        self.assertIn(score, (SCORE_ALLOW, SCORE_BLOCK))

    def test_float_station_type_does_not_crash(self) -> None:
        """A float station_type (e.g. 5.0) must not raise; treated as string."""
        try:
            score = self.scsv.check(5.0, "CAM")
        except Exception as exc:
            self.fail(f"SCSV.check raised unexpectedly: {exc}")
        self.assertIn(score, (SCORE_ALLOW, SCORE_BLOCK))

    def test_dict_station_type_does_not_crash(self) -> None:
        """A dict station_type must not raise; converted to string, no rule match."""
        try:
            score = self.scsv.check({"type": "passengerCar"}, "CAM")
        except Exception as exc:
            self.fail(f"SCSV.check raised unexpectedly: {exc}")
        self.assertIn(score, (SCORE_ALLOW, SCORE_BLOCK))

    def test_list_station_type_does_not_crash(self) -> None:
        """A list station_type must not raise."""
        try:
            score = self.scsv.check(["passengerCar"], "CAM")
        except Exception as exc:
            self.fail(f"SCSV.check raised unexpectedly: {exc}")
        self.assertIn(score, (SCORE_ALLOW, SCORE_BLOCK))


# ===========================================================================
# Fix 1 regression tests: deny-by-default policy
# ===========================================================================

class TestSCSVDenyByDefault(unittest.TestCase):
    """Tests that verify the deny-by-default policy (Fix 1).

    These tests confirm that:
      * The configured default_policy is now 'block' (score 0.0).
      * Any (station_type, message_type) pair not covered by an explicit
        allow rule falls through and is blocked.
      * Every station_type expected to legitimately send CAM still has an
        explicit allow rule and is NOT accidentally blocked by the policy
        flip.
    """

    # Station types that are expected to have an explicit CAM allow rule.
    # This list is the ground-truth: every type that should be able to
    # broadcast position via CAM in any ETSI ITS deployment.
    _CAM_ALLOWED_STATION_TYPES = [
        "passengerCar",
        "bus",
        "lightTruck",
        "heavyTruck",
        "motorcycle",
        "moped",
        "cyclist",
        "pedestrian",
        "tram",
        "specialVehicle",
        "agricultural",
        "trailer",
        "lightVruVehicle",
        "animal",
        "roadSideUnit",   # covered by the wildcard allow rule
    ]

    def setUp(self) -> None:
        self.scsv = _make_scsv()

    def test_default_policy_is_block(self) -> None:
        """SCSV.default_score must be 0.0 (deny-by-default, fail-closed).

        Verifies that the isce_config.yaml default_policy key has been
        changed from 'allow' to 'block' and that SCSV loaded it correctly.
        An allow-by-default policy would silently pass any combination not
        in the rule table; deny-by-default ensures unknown combinations are
        rejected until explicitly whitelisted.
        """
        self.assertEqual(
            self.scsv.default_score,
            SCORE_BLOCK,
            "default_policy must be 'block' (SCORE_BLOCK = 0.0)",
        )

    def test_unhandled_combination_blocks_by_default(self) -> None:
        """A (station_type, message_type) pair with no matching rule must score 0.0.

        trailer / SSEM has no explicit rule in isce_config.yaml.  Under the
        deny-by-default policy it must be blocked.  This is a concrete
        assertion of 0.0, not a dynamic lookup, so the test will fail loudly
        if a rule is added for this combination without updating the test.
        """
        score = self.scsv.check("trailer", "SSEM")
        self.assertEqual(
            score,
            SCORE_BLOCK,
            "trailer/SSEM has no rule and must be blocked by default_policy=block",
        )

    def test_all_known_cam_senders_still_allowed(self) -> None:
        """Every station_type expected to send CAM must still score 1.0 after the
        default_policy flip from 'allow' to 'block'.

        This is the primary regression guard for Fix 1: changing the default
        policy would silently block legitimate CAM traffic from any station
        type that lost its explicit allow rule.  All station types in
        _CAM_ALLOWED_STATION_TYPES must have an explicit allow rule for CAM.
        Failures are collected and reported in bulk.
        """
        failures = []
        for st in self._CAM_ALLOWED_STATION_TYPES:
            score = self.scsv.check(st, "CAM")
            if score != SCORE_ALLOW:
                failures.append(f"  {st}/CAM → {score:.2f} (expected 1.0)")
        if failures:
            self.fail(
                f"CAM allow regression after default_policy flip "
                f"({len(failures)} station types affected):\n" + "\n".join(failures)
            )


# ===========================================================================
# Fix 1: new explicit block rules for infrastructure-only message types
# ===========================================================================

class TestSCSVNewBlockRules(unittest.TestCase):
    """Tests for the eight new block rules added in Fix 1.

    These rules cover infrastructure-only message types that road-vehicle
    station types must never originate.  Each test is a synthetic adversarial
    case: no real attack samples exist in the V2AIX dataset.
    """

    def setUp(self) -> None:
        self.scsv = _make_scsv()

    def test_passenger_car_rtcmem_blocked(self) -> None:
        """passengerCar originating RTCMEM (GNSS correction data) must be blocked.

        RTCMEM carries differential GNSS corrections and is originated by
        infrastructure reference stations.  A passenger car broadcasting
        RTCMEM could attempt to poison vehicle positioning.
        """
        score = self.scsv.check("passengerCar", "RTCMEM")
        self.assertEqual(score, SCORE_BLOCK)

    def test_bus_rtcmem_blocked(self) -> None:
        """bus originating RTCMEM must be blocked.

        Same rationale as passengerCar: RTCMEM is infrastructure-only.
        """
        score = self.scsv.check("bus", "RTCMEM")
        self.assertEqual(score, SCORE_BLOCK)

    def test_passenger_car_ivim_blocked(self) -> None:
        """passengerCar originating IVIM (Infrastructure to Vehicle Information) must be blocked.

        IVIM conveys road-operator information (speed limits, road works, etc.).
        A vehicle originating it could inject false regulatory notices.
        """
        score = self.scsv.check("passengerCar", "IVIM")
        self.assertEqual(score, SCORE_BLOCK)

    def test_bus_ivim_blocked(self) -> None:
        """bus originating IVIM must be blocked."""
        score = self.scsv.check("bus", "IVIM")
        self.assertEqual(score, SCORE_BLOCK)

    def test_passenger_car_evcsn_blocked(self) -> None:
        """passengerCar originating EVCSN (EV Charging Spot Notification) must be blocked.

        EVCSN is published by charging-network operators.  A passenger car
        originating it could advertise fake charging points.
        """
        score = self.scsv.check("passengerCar", "EVCSN")
        self.assertEqual(score, SCORE_BLOCK)

    def test_bus_evcsn_blocked(self) -> None:
        """bus originating EVCSN must be blocked."""
        score = self.scsv.check("bus", "EVCSN")
        self.assertEqual(score, SCORE_BLOCK)

    def test_passenger_car_saem_blocked(self) -> None:
        """passengerCar originating SAEM (Service Announcement Extended Message) must be blocked.

        SAEM is sent by infrastructure/operator nodes to advertise ITS
        services.  A vehicle claiming to be a service provider is anomalous.
        """
        score = self.scsv.check("passengerCar", "SAEM")
        self.assertEqual(score, SCORE_BLOCK)

    def test_bus_saem_blocked(self) -> None:
        """bus originating SAEM must be blocked."""
        score = self.scsv.check("bus", "SAEM")
        self.assertEqual(score, SCORE_BLOCK)


# ===========================================================================
# Fix 2: numeric string resolution
# ===========================================================================

class TestSCSVNumericStringResolution(unittest.TestCase):
    """Tests for Fix 2: digit-only strings as station_type / message_type.

    Some decoders serialise ETSI enum values as strings containing only
    digits (e.g. '5' instead of the integer 5 or the name 'passengerCar').
    The resolver must handle all three forms identically.
    """

    def setUp(self) -> None:
        self.scsv = _make_scsv()

    def test_station_type_as_numeric_string_resolves(self) -> None:
        """check('5', 'CAM') must behave identically to check(5, 'CAM').

        station_type '5' is a digit-only string that must resolve to
        'passengerCar' (ETSI code 5) before rule matching, producing 1.0.
        """
        score_int = self.scsv.check(5, "CAM")
        score_str = self.scsv.check("5", "CAM")
        self.assertEqual(
            score_int,
            score_str,
            f"check(5, 'CAM')={score_int} != check('5', 'CAM')={score_str}: "
            "numeric string station_type must resolve identically to its integer form",
        )
        self.assertEqual(score_str, SCORE_ALLOW)

    def test_message_type_as_numeric_string_resolves(self) -> None:
        """check('passengerCar', '1') must behave identically to check('passengerCar', 1).

        message_type '1' is a digit-only string that must resolve to 'CAM'
        (ETSI message_id 1) before rule matching, producing 1.0.
        """
        score_int = self.scsv.check("passengerCar", 1)
        score_str = self.scsv.check("passengerCar", "1")
        self.assertEqual(
            score_int,
            score_str,
            f"check('passengerCar', 1)={score_int} != check('passengerCar', '1')={score_str}: "
            "numeric string message_type must resolve identically to its integer form",
        )
        self.assertEqual(score_str, SCORE_ALLOW)

    def test_both_numeric_strings_resolve(self) -> None:
        """check('5', '1') must produce 1.0 (passengerCar / CAM, both as digit strings)."""
        score = self.scsv.check("5", "1")
        self.assertEqual(score, SCORE_ALLOW)

    def test_out_of_range_numeric_string_falls_to_default(self) -> None:
        """A digit string with no enum entry ('999') must fall through to default_policy.

        The resolver must not crash; it returns '' for unrecognised integers,
        which triggers the default (now block) policy.
        """
        score = self.scsv.check("999", "CAM")
        self.assertEqual(
            score,
            SCORE_BLOCK,
            "'999' has no enum entry; must fall through to default_policy=block",
        )

    def test_numeric_string_rsu_spatem_allowed(self) -> None:
        """check('15', '6') must produce 1.0 (roadSideUnit / SPATEM, digit strings).

        Verifies that the wildcard RSU rule fires correctly when both inputs
        arrive as digit-only strings.
        """
        score = self.scsv.check("15", "6")
        self.assertEqual(score, SCORE_ALLOW)



class TestSCSVRealCAMSamples(unittest.TestCase):
    """Integration test: validate real decoded CAM messages from V2AIX captures.

    This test class loads a JSON file of genuine ETSI CAM beacons decoded
    from the V2AIX dataset.  Every sample in the file is confirmed-legitimate
    passenger vehicle traffic, so SCSV.check() must return 1.0 for all of
    them (no false positives).

    The JSON file is expected at::

        <project_root>/data/real_cam_samples.json

    Format (list of message dicts)::

        [
          {
            "header": {
              "station_id": 1234567,
              "message_id": 1
            },
            "cam": {
              "generation_delta_time": 4000,
              "cam_parameters": {
                "basic_container": {
                  "station_type": 5,
                  "reference_position": {
                    "latitude": 485500000,
                    "longitude": 96100000
                  }
                },
                "high_frequency_container": {
                  "basic_vehicle_container_high_frequency": {
                    "heading": 900,
                    "speed": 1400,
                    "yaw_rate": 0
                  }
                }
              }
            }
          },
          ...
        ]

    If the file is absent the test is skipped with a clear message,
    allowing the suite to pass in CI environments that do not have
    real sample data.
    """

    _SCSV_INSTANCE: SCSV | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._SCSV_INSTANCE = _make_scsv()

    def _extract_fields(self, msg: dict) -> tuple[str | int, str | int]:
        """Extract (station_type, message_type) from a decoded CAM message dict.

        Follows the V2AIX nested structure documented in the module header.
        The message_id integer is resolved to a string by SCSV internally,
        so we can pass it either way.

        Parameters
        ----------
        msg:
            A decoded CAM message as a Python dict.

        Returns
        -------
        tuple
            (station_type, message_type) ready for SCSV.check().
        """
        station_type = (
            msg.get("cam", {})
               .get("cam_parameters", {})
               .get("basic_container", {})
               .get("station_type")
        )
        message_type = (
            msg.get("header", {})
               .get("message_id")
        )
        return station_type, message_type

    @unittest.skipUnless(
        _REAL_SAMPLES_PATH.exists(),
        f"Real CAM samples file not found at {_REAL_SAMPLES_PATH} – skipping integration test. "
        "Drop real_cam_samples.json in the data/ directory to enable this test.",
    )
    def test_all_real_cam_samples_pass(self) -> None:
        """Every real CAM sample must receive a score of 1.0 (no false positives).

        This test is the primary regression guard against over-blocking:
        if any rule change causes SCSV to flag legitimate passengerCar CAM
        traffic, this test will catch it immediately.

        The test iterates each message in the JSON file, calls
        SCSV.check() with the extracted (station_type, message_type),
        and asserts a score of 1.0.  Failures are collected and reported
        in bulk so that a single noisy rule does not obscure the full scope
        of impact.
        """
        assert self._SCSV_INSTANCE is not None

        with _REAL_SAMPLES_PATH.open("r", encoding="utf-8") as fh:
            samples: list[dict] = json.load(fh)

        self.assertIsInstance(samples, list, "real_cam_samples.json must be a JSON array")
        self.assertGreater(len(samples), 0, "real_cam_samples.json must not be empty")

        failures: list[str] = []

        for idx, msg in enumerate(samples):
            station_type, message_type = self._extract_fields(msg)
            score = self._SCSV_INSTANCE.check(station_type, message_type)
            if score != SCORE_ALLOW:
                station_id = msg.get("header", {}).get("station_id", "?")
                failures.append(
                    f"  Sample #{idx} (station_id={station_id}): "
                    f"station_type={station_type!r}, message_type={message_type!r} "
                    f"→ score={score:.2f} (expected 1.0)"
                )

        if failures:
            self.fail(
                f"SCSV false-positives detected in {len(failures)}/{len(samples)} "
                f"real CAM samples:\n" + "\n".join(failures)
            )


# ===========================================================================
# Entry point (run without pytest)
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
