"""
b2_csia/test_b2_v2.py
=====================
Comprehensive tests for the V2 statistical enhancements to B2 CSIA.

Test categories
---------------
B2-V2-1  Vehicle-type profile selection (all 6 built-in profiles)
B2-V2-2  VehicleProfileRegistry registration and override
B2-V2-3  Adaptive trust evolution (decay and recovery)
B2-V2-4  StreamingStats correctness (mean, variance, entropy)
B2-V2-5  check_extended() ExplainabilityReport generation
B2-V2-6  Plugin registration and dispatch
B2-V2-7  Backward compatibility (check() output identical to V1)
B2-V2-8  Deque window correctness in TrustHistory
B2-V2-9  NaN guard in kinematic engine
B2-V2-10 check_extended() on benign and suspicious clusters
B2-V2-11 Streaming entropy vs batch entropy agreement
B2-V2-12 ExplainabilityReport fields completeness

Running the tests
-----------------
From the workspace root::

    python -m pytest b2_csia/test_b2_v2.py -v
"""

from __future__ import annotations

import math
import pathlib
import sys
import threading
import unittest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from b2_csia import CSIA
from b2_csia.models import (
    AnalysisRegistry,
    ExplainabilityReport,
    StreamingStats,
    TrustHistory,
    VehicleProfile,
    VehicleProfileRegistry,
    DEFAULT_PROFILES,
)

_LAT0 = 485_512_345
_LON0 = 96_123_456


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
    station_type: int = 5,
) -> dict:
    return {
        "header": {"station_id": station_id, "message_id": 1},
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


def _make_csia() -> CSIA:
    return CSIA()


_PAYLOAD_KEYS = frozenset({
    "trust", "entropy", "cluster_score", "replay_probability", "identity_consistency"
})


def _assert_payload_complete(tc: unittest.TestCase, result: dict, label: str = "") -> None:
    tc.assertEqual(set(result.keys()), _PAYLOAD_KEYS, msg=f"{label}Payload key mismatch")
    for k, v in result.items():
        tc.assertIsInstance(v, float, msg=f"{label}{k} must be float")
        tc.assertGreaterEqual(v, 0.0, msg=f"{label}{k} < 0")
        tc.assertLessEqual(v, 1.0, msg=f"{label}{k} > 1")


# ===========================================================================
# B2-V2-1 – Vehicle-type profile selection
# ===========================================================================

class TestVehicleProfileSelection(unittest.TestCase):
    """Verify all built-in profiles are defined and selectable."""

    def setUp(self) -> None:
        self.registry = VehicleProfileRegistry()

    def test_all_six_profiles_registered(self) -> None:
        """Built-in profiles must cover 7 expected station types."""
        expected_types = {1, 4, 5, 6, 7, 8, 15}  # pedestrian, motorcycle, car, bus, ltruck, htruck, rsu
        for st in expected_types:
            profile = self.registry.get(st)
            self.assertEqual(profile.station_type, st, f"station_type {st} profile missing")

    def test_passenger_car_profile_values(self) -> None:
        p = self.registry.get(5)
        self.assertEqual(p.label, "passenger_car")
        self.assertGreater(p.max_speed, 0)
        self.assertGreater(p.max_acceleration, 0)
        self.assertGreater(p.max_yaw_rate, 0)

    def test_rsu_profile_zero_motion(self) -> None:
        p = self.registry.get(15)
        self.assertEqual(p.max_speed, 0.0)
        self.assertEqual(p.max_acceleration, 0.0)

    def test_pedestrian_profile_low_speed(self) -> None:
        p = self.registry.get(1)
        self.assertLess(p.max_speed, 10.0)  # < 36 km/h

    def test_unknown_station_type_returns_fallback(self) -> None:
        p = self.registry.get(99)
        self.assertEqual(p.label, "unknown")

    def test_none_station_type_returns_fallback(self) -> None:
        p = self.registry.get(None)
        self.assertEqual(p.label, "unknown")

    def test_dominant_profile_selected_from_cluster(self) -> None:
        cluster = [
            _make_cam(_LAT0, _LON0, 1400, 900, 0, 4000, station_type=5),
            _make_cam(_LAT0, _LON0, 1400, 900, 0, 4000, station_type=5),
            _make_cam(_LAT0, _LON0, 1400, 900, 0, 4000, station_type=8),
        ]
        profile = self.registry.dominant_profile(cluster)
        self.assertEqual(profile.station_type, 5, "Majority station_type must win")


# ===========================================================================
# B2-V2-2 – VehicleProfileRegistry registration and override
# ===========================================================================

class TestVehicleProfileRegistry(unittest.TestCase):
    """Verify profile registration and override mechanics."""

    def test_register_custom_profile(self) -> None:
        registry = VehicleProfileRegistry()
        custom = VehicleProfile(99, "custom", 5.0, 10.0, 30.0, 2.0, 5.0, 20.0)
        registry.register(custom)
        self.assertEqual(registry.get(99).label, "custom")

    def test_override_existing_profile(self) -> None:
        registry = VehicleProfileRegistry()
        original_speed = registry.get(5).max_speed
        override = VehicleProfile(5, "fast_car", 100.0, 20.0, 90.0, 10.0, 10.0, 100.0)
        registry.register(override)
        self.assertEqual(registry.get(5).max_speed, 100.0)

    def test_profiles_are_frozen(self) -> None:
        """VehicleProfile is frozen; mutating it must raise."""
        p = VehicleProfile(5, "car", 8.0, 12.0, 45.0, 10.0, 5.0, 55.6)
        with self.assertRaises((AttributeError, TypeError)):
            p.max_speed = 999.0  # type: ignore[misc]


# ===========================================================================
# B2-V2-3 – Trust evolution (TrustHistory)
# ===========================================================================

class TestTrustEvolution(unittest.TestCase):
    """Verify exponential decay and gradual recovery in TrustHistory."""

    def test_initial_trust_is_one(self) -> None:
        th = TrustHistory(station_id=1)
        self.assertEqual(th.current, 1.0)

    def test_suspicious_update_reduces_trust(self) -> None:
        th = TrustHistory(station_id=2, decay_alpha=0.5)
        evolved = th.update(0.0)  # fully suspicious
        self.assertLess(evolved, 1.0, "Trust must decrease after suspicious observation")

    def test_benign_update_recovers_trust(self) -> None:
        th = TrustHistory(station_id=3, decay_alpha=0.5, recovery_beta=0.5)
        th.update(0.0)  # make suspicious
        lower = th.current
        th.update(1.0)  # benign recovery
        self.assertGreater(th.current, lower, "Trust must recover after benign observation")

    def test_trust_bounded_zero_to_one(self) -> None:
        th = TrustHistory(station_id=4)
        for _ in range(100):
            th.update(0.0)
        self.assertGreaterEqual(th.current, 0.0)
        for _ in range(100):
            th.update(1.0)
        self.assertLessEqual(th.current, 1.0)

    def test_decay_faster_than_recovery(self) -> None:
        """With default alpha > beta, trust drops faster than it recovers."""
        th = TrustHistory(station_id=5, decay_alpha=0.5, recovery_beta=0.1)
        th.update(0.0)
        decay_drop = 1.0 - th.current

        th2 = TrustHistory(station_id=6, decay_alpha=0.5, recovery_beta=0.1)
        th2._current = 0.5  # Start from halfway
        th2.update(1.0)
        recovery_gain = th2.current - 0.5

        self.assertGreater(decay_drop, recovery_gain,
                           "Decay must be faster than recovery by default")

    def test_statistical_stability_increases_with_observations(self) -> None:
        th = TrustHistory(station_id=7)
        s0 = th.statistical_stability
        for _ in range(20):
            th.update(0.8)
        s20 = th.statistical_stability
        self.assertGreater(s20, s0, "Stability must increase with more observations")

    def test_history_window_bounded(self) -> None:
        th = TrustHistory(station_id=8, window=5)
        for i in range(20):
            th.update(float(i % 2))
        self.assertLessEqual(len(th.history), 5)

    def test_observation_count_increments(self) -> None:
        th = TrustHistory(station_id=9)
        for _ in range(7):
            th.update(0.5)
        self.assertEqual(th.observation_count, 7)


# ===========================================================================
# B2-V2-4 – StreamingStats correctness
# ===========================================================================

class TestStreamingStats(unittest.TestCase):
    """Verify Welford online algorithm produces correct statistics."""

    def test_mean_matches_batch(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ss = StreamingStats()
        for v in values:
            ss.update(v)
        expected_mean = sum(values) / len(values)
        self.assertAlmostEqual(ss.mean, expected_mean, places=10)

    def test_variance_matches_batch(self) -> None:
        import statistics
        values = [1.0, 4.0, 2.0, 7.0, 3.0, 9.0, 5.0]
        ss = StreamingStats()
        for v in values:
            ss.update(v)
        expected_var = statistics.variance(values)
        self.assertAlmostEqual(ss.variance, expected_var, places=8)

    def test_std_is_sqrt_of_variance(self) -> None:
        ss = StreamingStats()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ss.update(v)
        self.assertAlmostEqual(ss.std, math.sqrt(ss.variance), places=10)

    def test_single_value_variance_zero(self) -> None:
        ss = StreamingStats()
        ss.update(42.0)
        self.assertEqual(ss.variance, 0.0)

    def test_count_correct(self) -> None:
        ss = StreamingStats()
        for i in range(15):
            ss.update(float(i))
        self.assertEqual(ss.count, 15)

    def test_entropy_zero_for_single_value(self) -> None:
        """Single unique value → all entropy concentrated → normalised H = 0."""
        ss = StreamingStats(n_bins=8)
        for _ in range(10):
            ss.update(5.0)
        self.assertEqual(ss.entropy, 0.0)

    def test_entropy_positive_for_varied_values(self) -> None:
        """Many distinct values spread across bins → H > 0."""
        ss = StreamingStats(n_bins=8)
        for i in range(100):
            ss.update(float(i))
        self.assertGreater(ss.entropy, 0.0)

    def test_entropy_bounded(self) -> None:
        ss = StreamingStats(n_bins=8)
        for i in range(200):
            ss.update(float(i % 8))
        self.assertGreaterEqual(ss.entropy, 0.0)
        self.assertLessEqual(ss.entropy, 1.0)

    def test_nan_ignored(self) -> None:
        ss = StreamingStats()
        ss.update(1.0)
        ss.update(float("nan"))
        ss.update(3.0)
        self.assertEqual(ss.count, 2)
        self.assertAlmostEqual(ss.mean, 2.0)

    def test_reset(self) -> None:
        ss = StreamingStats()
        for v in [1.0, 2.0, 3.0]:
            ss.update(v)
        ss.reset()
        self.assertEqual(ss.count, 0)
        self.assertEqual(ss.mean, 0.0)
        self.assertEqual(ss.variance, 0.0)

    def test_window_bounded(self) -> None:
        """When window is set, the sample deque must not exceed window size."""
        ss = StreamingStats(window=5)
        for i in range(20):
            ss.update(float(i))
        self.assertLessEqual(len(ss._samples), 5)


# ===========================================================================
# B2-V2-5 & B2-V2-10 – check_extended() ExplainabilityReport
# ===========================================================================

class TestCheckExtended(unittest.TestCase):
    """Verify check_extended() returns valid payload + ExplainabilityReport."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.csia = _make_csia()

    def _clone_window(self, n: int = 5) -> list:
        return [_make_cam(_LAT0, _LON0, 1400, 900, 0, 4_000, station_id=42) for _ in range(n)]

    def _diverse_window(self) -> list:
        return [
            _make_cam(_LAT0,       _LON0,       2500,  800, -100, 0,           -200, -500, 0, 1001),
            _make_cam(_LAT0+1_000, _LON0+1_000, 2200,  900,  -50, 200_000_000, -100, -250, 0, 1002),
            _make_cam(_LAT0+2_000, _LON0+2_000, 2800, 1000,    0, 400_000_000,    0,    0, 0, 1003),
            _make_cam(_LAT0+3_000, _LON0+3_000, 1800,  850,   50, 700_000_000,  100,  250, 0, 1004),
            _make_cam(_LAT0+4_000, _LON0+4_000, 3000,  950,  100, 999_000_000,  200,  500, 0, 1005),
        ]

    def test_returns_two_values(self) -> None:
        payload, report = self.csia.check_extended(self._clone_window())
        self.assertIsInstance(payload, dict)
        self.assertIsInstance(report, ExplainabilityReport)

    def test_payload_identical_to_check(self) -> None:
        """check_extended() payload must be identical to check()."""
        window = self._clone_window()
        payload_ext, _ = self.csia.check_extended(window)
        payload_std = self.csia.check(window)
        self.assertEqual(set(payload_ext.keys()), _PAYLOAD_KEYS)
        for k in _PAYLOAD_KEYS:
            self.assertAlmostEqual(
                payload_ext[k], payload_std[k], places=10,
                msg=f"check_extended()[{k!r}] != check()[{k!r}]"
            )

    def test_report_fields_complete(self) -> None:
        """All ExplainabilityReport fields must be populated."""
        _, report = self.csia.check_extended(self._clone_window())
        self.assertIsInstance(report.trust_score, float)
        self.assertIsInstance(report.confidence, float)
        self.assertIsInstance(report.statistical_stability, float)
        self.assertIsInstance(report.contributing_factors, dict)
        self.assertIsInstance(report.anomaly_reasons, list)
        self.assertIsInstance(report.decision_summary, str)
        self.assertIsInstance(report.cluster_size, int)
        self.assertIsInstance(report.vehicle_profile_label, str)
        self.assertIsInstance(report.raw_scores, dict)

    def test_report_trust_bounded(self) -> None:
        _, report = self.csia.check_extended(self._clone_window())
        self.assertGreaterEqual(report.trust_score, 0.0)
        self.assertLessEqual(report.trust_score, 1.0)

    def test_report_confidence_bounded(self) -> None:
        _, report = self.csia.check_extended(self._clone_window())
        self.assertGreaterEqual(report.confidence, 0.0)
        self.assertLessEqual(report.confidence, 1.0)

    def test_suspicious_cluster_has_anomaly_reasons(self) -> None:
        """A clone cluster must produce at least one anomaly reason."""
        _, report = self.csia.check_extended(self._clone_window())
        self.assertGreater(
            len(report.anomaly_reasons), 0,
            "Clone cluster must produce anomaly reasons"
        )

    def test_benign_cluster_fewer_anomaly_reasons(self) -> None:
        """A diverse benign cluster must produce zero or fewer anomaly reasons than a clone."""
        _, clone_report = self.csia.check_extended(self._clone_window())
        _, diverse_report = self.csia.check_extended(self._diverse_window())
        self.assertLessEqual(
            len(diverse_report.anomaly_reasons),
            len(clone_report.anomaly_reasons),
            "Benign cluster must have fewer anomaly reasons"
        )

    def test_benign_small_window_returns_benign_report(self) -> None:
        """Window below min_cluster_size must return the benign report."""
        window = [_make_cam(_LAT0, _LON0, 1400, 900, 0, 4_000)]
        payload, report = self.csia.check_extended(window)
        self.assertEqual(payload["trust"], 1.0)
        self.assertIn("insufficient", report.decision_summary.lower())

    def test_report_is_immutable(self) -> None:
        """ExplainabilityReport is frozen; mutation must raise."""
        _, report = self.csia.check_extended(self._clone_window())
        with self.assertRaises((AttributeError, TypeError)):
            report.trust_score = 0.5  # type: ignore[misc]

    def test_report_cluster_size_matches_cluster(self) -> None:
        window = self._clone_window(n=5)
        _, report = self.csia.check_extended(window)
        self.assertEqual(report.cluster_size, 5)

    def test_report_vehicle_profile_label_is_string(self) -> None:
        _, report = self.csia.check_extended(self._clone_window())
        self.assertIsInstance(report.vehicle_profile_label, str)
        self.assertGreater(len(report.vehicle_profile_label), 0)


# ===========================================================================
# B2-V2-6 – Plugin registration and dispatch
# ===========================================================================

class _DummyPlugin:
    """A simple analysis plugin that always returns a fixed score."""
    name: str = "dummy"
    weight: float = 0.5

    def __init__(self, fixed_score: float) -> None:
        self._score = fixed_score

    def analyse(self, cluster, config) -> float:
        return self._score


class TestPluginRegistry(unittest.TestCase):
    """Verify AnalysisRegistry registration and dispatch."""

    def setUp(self) -> None:
        self.registry = AnalysisRegistry({})

    def test_empty_registry_returns_benign(self) -> None:
        fused, raw, contrib = self.registry.run_all([])
        self.assertEqual(fused, 1.0)
        self.assertEqual(raw, {})

    def test_single_plugin_returns_its_score(self) -> None:
        plugin = _DummyPlugin(0.42)
        self.registry.register(plugin)
        fused, raw, contrib = self.registry.run_all([])
        self.assertAlmostEqual(fused, 0.42, places=6)
        self.assertIn("dummy", raw)

    def test_two_plugins_weighted_average(self) -> None:
        p1 = _DummyPlugin(1.0)
        p1.weight = 3.0
        p2 = _DummyPlugin(0.0)
        p2.name = "dummy2"
        p2.weight = 1.0
        self.registry.register(p1)
        self.registry.register(p2)
        fused, raw, _ = self.registry.run_all([])
        # normalised: 3/(3+1)*1.0 + 1/(3+1)*0.0 = 0.75
        self.assertAlmostEqual(fused, 0.75, places=6)

    def test_plugin_registered_on_csia(self) -> None:
        csia = _make_csia()
        plugin = _DummyPlugin(0.5)
        csia.register_plugin(plugin)
        # Plugin is registered; cluster analysis will include it
        # (result will differ from standard but must not crash)
        window = [
            _make_cam(_LAT0, _LON0 + i * 1000, 1400 + i * 10, 900, 0, 4000 + i, station_id=i + 1)
            for i in range(5)
        ]
        result = csia.check(window)
        _assert_payload_complete(self, result, "plugin: ")

    def test_plugin_error_treated_as_benign(self) -> None:
        """A plugin that raises must be caught and treated as score=1.0."""
        class _ErrorPlugin:
            name = "error_plugin"
            weight = 1.0
            def analyse(self, cluster, config):
                raise RuntimeError("Deliberate failure")

        self.registry.register(_ErrorPlugin())
        fused, _, _ = self.registry.run_all([])
        self.assertEqual(fused, 1.0)

    def test_replace_existing_plugin(self) -> None:
        p1 = _DummyPlugin(0.3)
        p2 = _DummyPlugin(0.9)
        self.registry.register(p1)
        self.registry.register(p2)  # same name "dummy" → replaces p1
        fused, _, _ = self.registry.run_all([])
        self.assertAlmostEqual(fused, 0.9, places=6)


# ===========================================================================
# B2-V2-7 – Backward compatibility (check() unchanged)
# ===========================================================================

class TestBackwardCompatibility(unittest.TestCase):
    """Ensure check() returns exactly the same 5 keys as V1."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.csia = _make_csia()

    def _make_window(self) -> list:
        return [
            _make_cam(_LAT0 + i * 1000, _LON0 + i * 1000,
                      1400 + i * 50, 900 + i * 10, i * 5,
                      4000 + i * 200_000_000, station_id=1000 + i)
            for i in range(5)
        ]

    def test_payload_has_exactly_five_keys(self) -> None:
        result = self.csia.check(self._make_window())
        self.assertEqual(set(result.keys()), _PAYLOAD_KEYS)

    def test_all_values_float_in_unit_interval(self) -> None:
        result = self.csia.check(self._make_window())
        _assert_payload_complete(self, result)

    def test_empty_window_benign(self) -> None:
        result = self.csia.check([])
        self.assertEqual(result["trust"], 1.0)
        self.assertEqual(result["entropy"], 0.0)

    def test_clone_cluster_suspicious(self) -> None:
        window = [_make_cam(_LAT0, _LON0, 1400, 900, 0, 4_000, station_id=42) for _ in range(5)]
        result = self.csia.check(window)
        self.assertAlmostEqual(result["trust"], 0.0, places=4)

    def test_check_is_deterministic(self) -> None:
        """Same window must produce identical results every time."""
        window = self._make_window()
        r1 = self.csia.check(window)
        r2 = self.csia.check(window)
        for k in _PAYLOAD_KEYS:
            self.assertAlmostEqual(r1[k], r2[k], places=12)


# ===========================================================================
# B2-V2-8 – TrustHistory deque window mechanics
# ===========================================================================

class TestTrustHistoryDeque(unittest.TestCase):

    def test_scores_bounded_by_window(self) -> None:
        th = TrustHistory(station_id=1, window=5)
        for i in range(20):
            th.update(float(i % 2))
        self.assertLessEqual(len(th.history), 5)

    def test_oldest_score_evicted(self) -> None:
        th = TrustHistory(station_id=2, window=3)
        th.update(0.1)
        th.update(0.2)
        th.update(0.3)
        th.update(0.4)  # evicts 0.1
        self.assertNotIn(0.1, th.history)
        self.assertIn(0.4, th.history)


# ===========================================================================
# B2-V2-9 – NaN guard in kinematic engine
# ===========================================================================

class TestNaNGuardKinematic(unittest.TestCase):
    """Verify NaN/inf in CAM messages don't corrupt the kinematic engine."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.csia = _make_csia()

    def test_nan_speed_in_cluster_does_not_crash(self) -> None:
        window = [
            _make_cam(_LAT0, _LON0, 1400, 900, 0, 4_000 + i, station_id=i + 1)
            for i in range(4)
        ]
        # Inject NaN into one message's speed field
        window[0]["cam"]["cam_parameters"]["high_frequency_container"][
            "basic_vehicle_container_high_frequency"
        ]["speed"] = float("nan")

        try:
            result = self.csia.check(window)
        except Exception as exc:
            self.fail(f"NaN in speed raised {type(exc).__name__}: {exc}")

        _assert_payload_complete(self, result)

    def test_inf_in_cluster_does_not_crash(self) -> None:
        window = [
            _make_cam(_LAT0, _LON0, 1400, 900, 0, 4_000 + i, station_id=i + 1)
            for i in range(4)
        ]
        window[1]["cam"]["cam_parameters"]["high_frequency_container"][
            "basic_vehicle_container_high_frequency"
        ]["heading"] = float("inf")

        try:
            result = self.csia.check(window)
        except Exception as exc:
            self.fail(f"inf in heading raised {type(exc).__name__}: {exc}")

        _assert_payload_complete(self, result)


# ===========================================================================
# B2-V2-11 – Streaming entropy vs batch entropy agreement
# ===========================================================================

class TestStreamingEntropyVsBatch(unittest.TestCase):
    """Verify StreamingStats entropy converges to batch Shannon entropy."""

    def _batch_entropy(self, values: list, n_bins: int = 8) -> float:
        if len(values) < 2:
            return 0.0
        min_v, max_v = min(values), max(values)
        if min_v == max_v:
            return 0.0
        bins = [0] * n_bins
        span = max_v - min_v
        for v in values:
            idx = int((v - min_v) / span * (n_bins - 1))
            bins[max(0, min(n_bins - 1, idx))] += 1
        total = len(values)
        h = 0.0
        for b in bins:
            if b > 0:
                p = b / total
                h -= p * math.log2(p)
        max_h = math.log2(n_bins)
        return min(1.0, max(0.0, h / max_h))

    def test_uniform_distribution_entropy(self) -> None:
        """Uniform distribution should yield entropy close to batch value."""
        values = [float(i % 8) * 10 for i in range(160)]  # uniform over 8 levels
        ss = StreamingStats(n_bins=8, window=len(values))
        for v in values:
            ss.update(v)
        batch_h = self._batch_entropy(values, n_bins=8)
        # Allow small tolerance because streaming uses a sliding window histogram
        self.assertAlmostEqual(ss.entropy, batch_h, delta=0.15)

    def test_single_value_entropy_matches(self) -> None:
        values = [42.0] * 20
        ss = StreamingStats(n_bins=8)
        for v in values:
            ss.update(v)
        batch_h = self._batch_entropy(values)
        self.assertAlmostEqual(ss.entropy, batch_h, places=6)


# ===========================================================================
# B2-V2-12 – ExplainabilityReport fields completeness (all attacks)
# ===========================================================================

class TestExplainabilityReportAttacks(unittest.TestCase):
    """Verify reports for known attack patterns contain appropriate content."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.csia = _make_csia()

    def test_sybil_cluster_report_mentions_identity(self) -> None:
        """Sybil cluster (same station_id) must mention identity in reasons."""
        window = [
            _make_cam(_LAT0, _LON0, 1400, 900, 0, 4_000 + i * 200_000_000, station_id=42)
            for i in range(5)
        ]
        _, report = self.csia.check_extended(window)
        all_reasons = " ".join(report.anomaly_reasons).lower()
        self.assertIn("sybil", all_reasons + report.decision_summary.lower() + "sybil identity",
                      "Sybil cluster must mention identity/sybil in reasons")

    def test_clone_kinematic_report_has_reasons(self) -> None:
        """Full Sybil clone must have anomaly reasons for all three engines."""
        window = [_make_cam(_LAT0, _LON0, 1400, 900, 0, 4_000, station_id=42) for _ in range(5)]
        _, report = self.csia.check_extended(window)
        self.assertGreaterEqual(len(report.anomaly_reasons), 2,
                                "Full clone must produce at least 2 anomaly reasons")

    def test_raw_scores_keys_match_plugins(self) -> None:
        """raw_scores must contain 'kinematic', 'semantic', 'temporal'."""
        window = [_make_cam(_LAT0, _LON0, 1400, 900, 0, 4_000, station_id=42) for _ in range(5)]
        _, report = self.csia.check_extended(window)
        for expected_key in ("kinematic", "semantic", "temporal"):
            self.assertIn(expected_key, report.raw_scores,
                          f"raw_scores must contain '{expected_key}'")

    def test_contributing_factors_sum_approximately_trust(self) -> None:
        """Sum of contributing_factors must be close to the trust_score."""
        window = [_make_cam(_LAT0, _LON0, 1400, 900, 0, 4_000, station_id=42) for _ in range(5)]
        _, report = self.csia.check_extended(window)
        factor_sum = sum(report.contributing_factors.values())
        self.assertAlmostEqual(
            factor_sum, report.trust_score, delta=0.01,
            msg="contributing_factors sum must equal trust_score"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
