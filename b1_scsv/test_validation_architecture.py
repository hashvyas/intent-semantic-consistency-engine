import pytest
import time
from b1_scsv.scsv import SCSV
from b1_scsv.models import ValidationFailureReason, ValidationAssessment, CamMessage
from b2_csia.csia import CSIA
from b2_csia.models import ExplainabilityReport

@pytest.fixture
def scsv():
    return SCSV()

@pytest.fixture
def csia():
    config_overrides = {
        "research_extensions": {
            "enabled": True
        }
    }
    return CSIA(config_overrides=config_overrides)

def test_fatal_malformed_json(scsv):
    # Completely invalid structure (non-dict)
    res = scsv.check_stateful("not a dict")
    assert res.fatal is True
    assert res.validation_score == 0.0
    assert not res.valid
    assert res.reason == ValidationFailureReason.PARSE_ERROR
    assert "structure" in res.checks
    assert not res.checks["structure"]

def test_fatal_missing_mandatory_fields(scsv):
    # Dict with missing station_id
    msg = {
        "header": {
            "message_id": 1
        },
        "cam": {
            "generation_delta_time": 123456,
            "cam_parameters": {
                "basic_container": {
                    "station_type": 5,
                    "reference_position": {
                        "latitude": 48123456,
                        "longitude": 9123456
                    }
                }
            }
        }
    }
    # station_id missing
    res = scsv.check_stateful(msg)
    assert res.fatal is True
    assert res.validation_score == 0.0
    assert not res.checks["structure"]
    assert "Missing mandatory fields" in res.reasons[0]

def test_recoverable_stale_timestamp(scsv):
    # Msg with old timestamp
    now = time.time()
    stale_ts = (now - 10.0) * 1000.0 # 10s old, freshness is 5s
    msg = {
        "header": {
            "station_id": 1001,
            "message_id": 1
        },
        "cam": {
            "generation_delta_time": stale_ts,
            "cam_parameters": {
                "basic_container": {
                    "station_type": 5,
                    "reference_position": {
                        "latitude": 48123456,
                        "longitude": 9123456
                    }
                },
                "high_frequency_container": {
                    "basic_vehicle_container_high_frequency": {
                        "speed": 1500,
                        "heading": 900,
                        "longitudinal_acceleration": 100,
                        "yaw_rate": 0
                    }
                }
            }
        }
    }
    
    res = scsv.check_stateful(msg)
    assert res.fatal is False
    assert res.checks["structure"] is True
    assert res.checks["timestamp"] is False
    # Score should have penalty deducted
    assert res.validation_score == 1.0 - 0.20 # stale timestamp penalty is 0.20
    assert "Timestamp stale" in res.reasons[0]

def test_recoverable_replay(scsv):
    now = time.time()
    ts = now * 1000.0
    msg = {
        "header": {
            "station_id": 1002,
            "message_id": 1
        },
        "cam": {
            "generation_delta_time": ts,
            "cam_parameters": {
                "basic_container": {
                    "station_type": 5,
                    "reference_position": {
                        "latitude": 48123456,
                        "longitude": 9123456
                    }
                },
                "high_frequency_container": {
                    "basic_vehicle_container_high_frequency": {
                        "speed": 1500,
                        "heading": 900,
                        "longitudinal_acceleration": 100,
                        "yaw_rate": 0
                    }
                }
            }
        }
    }
    
    # First check: passes
    res1 = scsv.check_stateful(msg)
    assert res1.fatal is False
    assert res1.checks["replay"] is True
    assert res1.validation_score == 1.0
    
    # Second check (identical message): triggers replay
    res2 = scsv.check_stateful(msg)
    assert res2.fatal is False
    assert res2.checks["replay"] is False
    assert res2.validation_score == 1.0 - 0.30 # replay penalty is 0.30
    assert "Replay detected" in res2.reasons[0]

def test_explainability_integration(scsv, csia):
    # Run a complete sequence of messages through pipeline and verify ExplainabilityReport
    now = time.time()
    
    # Configure SCSV freshness tolerance to 2 seconds for this test
    scsv._freshness_ms = 2000.0
    
    # Configure CSIA parameters for clustering 2 messages
    csia._min_cluster_size = 2
    csia._window_size_ns = 5_000_000_000.0
    
    # Create a cluster with 2 messages from vehicle 1003
    msg1 = {
        "header": {"station_id": 1003, "message_id": 1},
        "cam": {
            "generation_delta_time": now * 1000.0,
            "cam_parameters": {
                "basic_container": {
                    "station_type": 5,
                    "reference_position": {"latitude": 48123456, "longitude": 9123456}
                },
                "high_frequency_container": {
                    "basic_vehicle_container_high_frequency": {
                        "speed": 1500,
                        "heading": 900,
                        "longitudinal_acceleration": 100,
                        "yaw_rate": 0
                    }
                }
            }
        }
    }
    
    # Stale timestamp recoverable msg from same vehicle (3s old, triggers > 2s freshness limit)
    stale_ts = (now - 3.0) * 1000.0
    msg2 = {
        "header": {"station_id": 1003, "message_id": 1},
        "cam": {
            "generation_delta_time": stale_ts,
            "cam_parameters": {
                "basic_container": {
                    "station_type": 5,
                    "reference_position": {"latitude": 48123480, "longitude": 9123480}
                },
                "high_frequency_container": {
                    "basic_vehicle_container_high_frequency": {
                        "speed": 1600,
                        "heading": 910,
                        "longitudinal_acceleration": 120,
                        "yaw_rate": 0
                    }
                }
            }
        }
    }
    
    # Execute B1 and attach assessments
    res1 = scsv.check_stateful(msg1)
    msg1["_validation_assessment"] = res1
    
    res2 = scsv.check_stateful(msg2)
    msg2["_validation_assessment"] = res2
    
    assert res2.validation_score == 1.0 - 0.20 # stale timestamp penalty
    
    # Execute B2 pipeline
    payload, report = csia.check_extended([msg1, msg2])
    
    # Verify validation fields inside ExplainabilityReport
    assert report.validation_score == 0.80
    assert res1.confidence < 1.0
    assert report.validation_confidence == res2.confidence
    assert report.fatal is False
    assert "stale_timestamp" in report.applied_penalties
    assert report.applied_penalties["stale_timestamp"] == 0.20
