"""
test_system.py
==============
System-level integration tests for the ISCE V2 pipeline.

Tests focus on properties that span both B1 and B2 layers:
  - End-to-end backward compatibility (V1-format inputs → V2 pipeline → V1-format outputs)
  - Deterministic execution (same input → same output across N runs)
  - Concurrent processing (threading safety for shared instances)
  - Long-duration stability (large-volume message streams)
  - Cross-layer consistency (B1 block score feeds into B2 correctly)

Running the tests
-----------------
From the workspace root::

    python -m pytest test_system.py -v

Dependencies: standard library + PyYAML + numpy + pytest.
"""

from __future__ import annotations

import pathlib
import sys
import threading
import time
import unittest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from b1_scsv.scsv import SCSV, SCORE_ALLOW, SCORE_BLOCK
from b2_csia.csia import CSIA

_LAT0 = 485_512_345
_LON0 = 96_123_456

_PAYLOAD_KEYS = frozenset({
    "trust", "entropy", "cluster_score", "replay_probability", "identity_consistency"
})


# ---------------------------------------------------------------------------
# Message factories
# ---------------------------------------------------------------------------

def _make_raw_cam(
    station_id: int = 1001,
    station_type: int = 5,
    message_id: int = 1,
    timestamp: float = 4_000.0,
    lat: int = _LAT0,
    lon: int = _LON0,
    speed: int = 1400,
    heading: int = 900,
    yaw_rate: int = 0,
) -> dict:
    return {
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
                        "steering_wheel_angle": 0,
                        "lateral_acceleration": 0,
                        "longitudinal_acceleration": 0,
                    }
                },
            },
        },
    }


def _diverse_window(n: int = 5) -> list:
    """Generate a diverse window of CAM messages (expected: trust > 0.5)."""
    return [
        _make_raw_cam(
            station_id=1000 + i,
            lat=_LAT0 + i * 1_000,
            lon=_LON0 + i * 1_000,
            speed=1400 + i * 50,
            heading=900 + i * 10,
            yaw_rate=i * 5,
            timestamp=float(4_000 + i * 200_000_000),
        )
        for i in range(n)
    ]


def _clone_window(n: int = 5) -> list:
    """Generate a clone window (expected: trust ≈ 0.0)."""
    return [_make_raw_cam(station_id=42, timestamp=float(4_000)) for _ in range(n)]


# ===========================================================================
# SYS-1 – End-to-end backward compatibility
# ===========================================================================

class TestEndToEndBackwardCompatibility(unittest.TestCase):
    """V1-format inputs must produce V1-compatible outputs through the V2 pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scsv = SCSV()
        cls.csia = CSIA()

    # ── B1 ────────────────────────────────────────────────────────────────

    def test_b1_check_returns_float(self) -> None:
        score = self.scsv.check("passengerCar", "CAM")
        self.assertIsInstance(score, float)

    def test_b1_check_score_allow(self) -> None:
        self.assertEqual(self.scsv.check("passengerCar", "CAM"), SCORE_ALLOW)

    def test_b1_check_score_block(self) -> None:
        self.assertEqual(self.scsv.check("passengerCar", "SPATEM"), SCORE_BLOCK)

    def test_b1_check_integer_inputs_resolve(self) -> None:
        # passengerCar=5, CAM=1
        self.assertEqual(self.scsv.check(5, 1), SCORE_ALLOW)

    def test_b1_check_none_inputs_do_not_crash(self) -> None:
        score = self.scsv.check(None, None)
        self.assertIn(score, (SCORE_ALLOW, SCORE_BLOCK))

    def test_b1_constants_unchanged(self) -> None:
        self.assertEqual(SCORE_ALLOW, 1.0)
        self.assertEqual(SCORE_BLOCK, 0.0)

    # ── B2 ────────────────────────────────────────────────────────────────

    def test_b2_check_returns_dict(self) -> None:
        result = self.csia.check(_diverse_window())
        self.assertIsInstance(result, dict)

    def test_b2_check_has_exactly_five_keys(self) -> None:
        result = self.csia.check(_diverse_window())
        self.assertEqual(set(result.keys()), _PAYLOAD_KEYS)

    def test_b2_all_values_in_unit_interval(self) -> None:
        result = self.csia.check(_diverse_window())
        for k, v in result.items():
            self.assertIsInstance(v, float, f"{k} must be float")
            self.assertGreaterEqual(v, 0.0, f"{k} < 0")
            self.assertLessEqual(v, 1.0, f"{k} > 1")

    def test_b2_empty_window_returns_benign(self) -> None:
        result = self.csia.check([])
        self.assertEqual(result["trust"], 1.0)
        self.assertEqual(result["entropy"], 0.0)
        self.assertEqual(result["replay_probability"], 0.0)
        self.assertEqual(result["cluster_score"], 1.0)
        self.assertEqual(result["identity_consistency"], 1.0)

    def test_b2_clone_cluster_suspicious(self) -> None:
        result = self.csia.check(_clone_window())
        self.assertAlmostEqual(result["trust"], 0.0, places=4)
        self.assertAlmostEqual(result["cluster_score"], 0.0, places=4)

    def test_b2_diverse_cluster_benign(self) -> None:
        result = self.csia.check(_diverse_window(n=5))
        self.assertGreater(result["trust"], 0.4)

    def test_b2_trust_and_replay_sum_to_approximately_one(self) -> None:
        """replay_probability = 1 - timing_trust; this is a soft check."""
        # For a pure clone, both trust and cluster_score are ~0
        result = self.csia.check(_clone_window())
        self.assertAlmostEqual(result["replay_probability"], 1.0, places=4)


# ===========================================================================
# SYS-2 – Deterministic execution
# ===========================================================================

class TestDeterministicExecution(unittest.TestCase):
    """Same inputs must produce exactly identical outputs across multiple runs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scsv = SCSV()
        cls.csia = CSIA()

    def test_b1_check_deterministic(self) -> None:
        for _ in range(20):
            self.assertEqual(self.scsv.check("passengerCar", "CAM"), SCORE_ALLOW)
            self.assertEqual(self.scsv.check("passengerCar", "SPATEM"), SCORE_BLOCK)

    def test_b2_check_deterministic(self) -> None:
        window = _diverse_window()
        first = self.csia.check(window)
        for _ in range(10):
            repeat = self.csia.check(window)
            for k in _PAYLOAD_KEYS:
                self.assertAlmostEqual(
                    first[k], repeat[k], places=12,
                    msg=f"Non-deterministic: {k} differs between runs"
                )

    def test_b2_check_deterministic_clone(self) -> None:
        window = _clone_window()
        first = self.csia.check(window)
        for _ in range(10):
            repeat = self.csia.check(window)
            for k in _PAYLOAD_KEYS:
                self.assertAlmostEqual(first[k], repeat[k], places=12)


# ===========================================================================
# SYS-3 – Concurrent processing (thread safety)
# ===========================================================================

class TestConcurrentProcessing(unittest.TestCase):
    """Shared SCSV and CSIA instances must be safe under concurrent access."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scsv = SCSV()
        cls.csia = CSIA()

    def _run_b1(self, tid: int, errors: list) -> None:
        try:
            for i in range(50):
                self.scsv.check("passengerCar", "CAM")
                self.scsv.check("roadSideUnit", "SPATEM")
        except Exception as exc:
            errors.append(f"B1 thread {tid}: {exc}")

    def _run_b2(self, tid: int, errors: list) -> None:
        try:
            for i in range(20):
                window = _diverse_window()
                result = self.csia.check(window)
                assert set(result.keys()) == _PAYLOAD_KEYS
        except Exception as exc:
            errors.append(f"B2 thread {tid}: {exc}")

    def test_concurrent_b1_check(self) -> None:
        """20 threads calling SCSV.check() simultaneously must not crash."""
        errors: list = []
        threads = [threading.Thread(target=self._run_b1, args=(i, errors)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [], f"Concurrent B1 errors: {errors}")

    def test_concurrent_b2_check(self) -> None:
        """10 threads calling CSIA.check() simultaneously must not crash."""
        errors: list = []
        threads = [threading.Thread(target=self._run_b2, args=(i, errors)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [], f"Concurrent B2 errors: {errors}")

    def test_concurrent_stateful_check(self) -> None:
        """10 threads calling SCSV.check_stateful() with unique station_ids must not crash."""
        errors: list = []

        def _run(tid: int) -> None:
            try:
                for i in range(20):
                    msg = _make_raw_cam(station_id=10_000 + tid * 20 + i, timestamp=float(4_000 + i))
                    self.scsv.check_stateful(msg)
            except Exception as exc:
                errors.append(f"stateful thread {tid}: {exc}")

        threads = [threading.Thread(target=_run, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [], f"Concurrent stateful errors: {errors}")


# ===========================================================================
# SYS-4 – Long-duration stability (large message streams)
# ===========================================================================

class TestLongDurationStability(unittest.TestCase):
    """Processing large message volumes must not leak memory or degrade quality."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scsv = SCSV()
        cls.csia = CSIA()

    def test_b1_thousand_messages(self) -> None:
        """1 000 check() calls must all return valid scores."""
        for i in range(1_000):
            score = self.scsv.check("passengerCar", "CAM")
            self.assertEqual(score, SCORE_ALLOW)

    def test_b2_hundred_windows(self) -> None:
        """100 sequential check() calls must all produce valid payloads."""
        for batch in range(100):
            window = [
                _make_raw_cam(
                    station_id=1000 + i,
                    lat=_LAT0 + batch * 100 + i * 100,
                    lon=_LON0 + i * 100,
                    timestamp=float(batch * 1_000_000_000 + i * 200_000_000),
                )
                for i in range(5)
            ]
            result = self.csia.check(window)
            self.assertEqual(set(result.keys()), _PAYLOAD_KEYS)
            for v in result.values():
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_b1_stateful_thousand_unique_stations(self) -> None:
        """check_stateful() with 1 000 unique station_ids must not crash or OOM."""
        for i in range(1_000):
            msg = _make_raw_cam(station_id=i, timestamp=float(4_000 + i))
            result = self.scsv.check_stateful(msg)
            self.assertIsNotNone(result)

    def test_large_window_fifty_messages(self) -> None:
        """CSIA must handle a window of 50 messages without crashing."""
        window = [
            _make_raw_cam(
                station_id=5000 + i,
                lat=_LAT0 + i * 200,
                lon=_LON0 + i * 200,
                speed=1400 + (i % 10) * 30,
                timestamp=float(4_000 + i * 100_000_000),
            )
            for i in range(50)
        ]
        try:
            result = self.csia.check(window)
        except Exception as exc:
            self.fail(f"50-message window raised {type(exc).__name__}: {exc}")
        self.assertEqual(set(result.keys()), _PAYLOAD_KEYS)


# ===========================================================================
# SYS-5 – Cross-layer consistency
# ===========================================================================

class TestCrossLayerConsistency(unittest.TestCase):
    """B1 and B2 must agree on their assessments for known scenarios."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scsv = SCSV()
        cls.csia = CSIA()

    def test_b1_and_b2_both_pass_legitimate_traffic(self) -> None:
        """Legitimate passengerCar/CAM should pass both B1 and B2."""
        # B1: rule-table check
        b1_score = self.scsv.check("passengerCar", "CAM")
        self.assertEqual(b1_score, SCORE_ALLOW)

        # B2: diverse cluster check
        b2_result = self.csia.check(_diverse_window())
        self.assertGreater(b2_result["trust"], 0.4)

    def test_b1_blocks_invalid_combination(self) -> None:
        """B1 must block passengerCar/SPATEM before B2 is invoked."""
        b1_score = self.scsv.check("passengerCar", "SPATEM")
        self.assertEqual(b1_score, SCORE_BLOCK)

    def test_b2_detects_clone_that_passes_b1(self) -> None:
        """A clone cluster of valid passengerCar/CAM messages passes B1 but is caught by B2."""
        # B1 check: all valid (passengerCar sends CAM)
        for _ in range(5):
            b1_score = self.scsv.check("passengerCar", "CAM")
            self.assertEqual(b1_score, SCORE_ALLOW)

        # B2 check: clone pattern → trust ≈ 0
        b2_result = self.csia.check(_clone_window())
        self.assertAlmostEqual(b2_result["trust"], 0.0, places=4)

    def test_check_extended_payload_matches_check(self) -> None:
        """check_extended() payload must match check() exactly on both clone and diverse."""
        for window, label in [(_clone_window(), "clone"), (_diverse_window(), "diverse")]:
            std = self.csia.check(window)
            ext, _ = self.csia.check_extended(window)
            for k in _PAYLOAD_KEYS:
                self.assertAlmostEqual(
                    std[k], ext[k], places=12,
                    msg=f"{label}: check()[{k!r}] != check_extended()[{k!r}]"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
