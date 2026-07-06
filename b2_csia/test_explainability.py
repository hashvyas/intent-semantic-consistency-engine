import unittest
from typing import Dict, Any, List
from b2_csia.csia import CSIA
from b2_csia.models import ExplainabilityReport
from b2_csia.uncertainty import MassFunction

def _make_cam_message(station_id: int, lat: float, lon: float, speed: float, heading: float, acc: float, timestamp: float) -> Dict[str, Any]:
    return {
        "header": {
            "station_id": station_id,
        },
        "cam": {
            "generation_delta_time": timestamp,
            "cam_parameters": {
                "basic_container": {
                    "station_type": 5,
                    "reference_position": {
                        "latitude": int(lat * 1e7),
                        "longitude": int(lon * 1e7),
                    }
                },
                "high_frequency_container": {
                    "basic_vehicle_container_high_frequency": {
                        "speed": int(speed * 100.0),
                        "heading": int(heading * 10.0),
                        "longitudinal_acceleration": int(acc * 100.0),
                        "yaw_rate": 0,
                    }
                }
            }
        }
    }

class TestExplainabilityAndConfidence(unittest.TestCase):
    def setUp(self) -> None:
        # Construct CSIA with research extensions enabled
        overrides = {
            "research_extensions": {
                "enabled": True,
            }
        }
        self.csia = CSIA(config_overrides=overrides)

    def test_dst_reporting_consistency(self) -> None:
        """Verify that ExplainabilityReport's belief/disbelief/uncertainty match the final propagated values."""
        # 5 identical messages (classic kinematic clone)
        window = [
            _make_cam_message(station_id=1001 + i, lat=48.5512345, lon=9.6123456, speed=15.0, heading=90.0, acc=0.0, timestamp=1000.0 + i * 100.0)
            for i in range(5)
        ]
        
        payload, report = self.csia.check_extended(window)
        
        self.assertAlmostEqual(report.belief, payload.get("belief", 0.0), places=4)
        self.assertAlmostEqual(report.disbelief, payload.get("disbelief", 0.0), places=4)
        self.assertAlmostEqual(report.uncertainty, payload.get("uncertainty", 0.0), places=4)
        self.assertAlmostEqual(report.belief + report.disbelief + report.uncertainty, 1.0, places=4)

    def test_explainability_evidence_specific(self) -> None:
        """Verify that the generated reasons and evidence summary refer to actual checks and parameters."""
        # Benign single message
        window = [_make_cam_message(station_id=1001, lat=48.5512345, lon=9.6123456, speed=15.0, heading=90.0, acc=0.0, timestamp=1000.0)]
        _, report = self.csia.check_extended(window)
        
        # Check evidence reasons are populated
        self.assertGreater(len(report.evidence_reasons), 0)
        
        # Check reasons refer to specific validation details
        reasons_text = " ".join(report.evidence_reasons).lower()
        self.assertIn("structural validation", reasons_text)
        self.assertIn("replay", reasons_text)
        self.assertIn("certificate", reasons_text)
        self.assertIn("adaptive threshold", reasons_text)
        self.assertIn("isolated", reasons_text)
        
        # Check evidence summary keys
        summary = report.evidence_summary
        self.assertEqual(summary.get("Validation"), "PASS")
        self.assertEqual(summary.get("Replay"), "Not Detected")
        self.assertEqual(summary.get("Certificate"), "Consistent")
        self.assertEqual(summary.get("Neighbors"), "0")

    def test_confidence_calibration_bounds(self) -> None:
        """Verify that confidence remains bounded and correctly reflects cluster size / neighbor presence."""
        # Single isolated vehicle (should have low confidence)
        window_isolated = [_make_cam_message(station_id=1001, lat=48.5512345, lon=9.6123456, speed=15.0, heading=90.0, acc=0.0, timestamp=1000.0)]
        _, report_isolated = self.csia.check_extended(window_isolated)
        
        # 1 node -> cooperative confidence = 1 / (1 + 5) = 0.1667. Overall confidence should be low.
        self.assertLess(report_isolated.confidence, 0.25)
        
        # Multi-node cluster (should have higher confidence)
        window_cluster = [
            _make_cam_message(station_id=1001 + i, lat=48.5512345 + i*0.0001, lon=9.6123456 + i*0.0001, speed=15.0 + i, heading=90.0, acc=0.0, timestamp=1000.0)
            for i in range(5)
        ]
        _, report_cluster = self.csia.check_extended(window_cluster)
        
        self.assertGreater(report_cluster.confidence, report_isolated.confidence)
        self.assertLessEqual(report_cluster.confidence, 1.0)

    def test_no_contradictions(self) -> None:
        """Verify that belief states sum to 1.0 and do not contain contradictory values."""
        window = [
            _make_cam_message(station_id=1001 + i, lat=48.5512345, lon=9.6123456, speed=15.0, heading=90.0, acc=0.0, timestamp=1000.0)
            for i in range(5)
        ]
        _, report = self.csia.check_extended(window)
        
        # Basic sum validation
        total_mass = report.belief + report.disbelief + report.uncertainty
        self.assertAlmostEqual(total_mass, 1.0, places=5)
        
        # Ensure values are mathematically coherent
        self.assertTrue(0.0 <= report.belief <= 1.0)
        self.assertTrue(0.0 <= report.disbelief <= 1.0)
        self.assertTrue(0.0 <= report.uncertainty <= 1.0)
