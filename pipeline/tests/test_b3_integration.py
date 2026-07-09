from __future__ import annotations
import os
import pathlib
import pytest
from unittest.mock import MagicMock, patch
from pipeline.b3_bridge import (
    SemanticGateClassifier, 
    classify_text, 
    resolve_model_path, 
    _load_b3_config
)
from b3.solution_stb.b3_semantic_gate.inference import SemanticGateResult, get_predictor

def test_resolve_model_path():
    # Test absolute path resolving
    cwd = os.getcwd()
    assert resolve_model_path(cwd) == os.path.abspath(cwd)
    
    # Test non-existent path
    fake_path = "non_existent_model_dir"
    assert resolve_model_path(fake_path) == os.path.abspath(fake_path)

def test_load_b3_config(tmp_path):
    # Test custom yaml config path
    config_file = tmp_path / "test_config.yaml"
    config_content = """
b3_semantic_gate:
  model_path: "mock/path"
  max_length: 128
  batch_size: 16
  device: "cpu"
"""
    config_file.write_text(config_content)
    config = _load_b3_config(config_file)
    assert config.get("model_path") == "mock/path"
    assert config.get("max_length") == 128
    assert config.get("batch_size") == 16
    assert config.get("device") == "cpu"

def test_classifier_graceful_fallback(tmp_path):
    # Test fallback with a config pointing to a non-existent model path
    config_file = tmp_path / "non_existent_model_config.yaml"
    config_content = """
b3_semantic_gate:
  model_path: "non_existent_model_dir"
"""
    config_file.write_text(config_content)
    classifier = SemanticGateClassifier(config_path=config_file)
    res = classifier.classify("some message")
    assert not res["available"]
    assert res["label"] is None
    assert "not found" in res["status"]

@patch("b3.solution_stb.b3_semantic_gate.inference.get_predictor")
def test_classifier_happy_path(mock_get_predictor, tmp_path):
    # Create a mock model dir to bypass existence check
    mock_model_dir = tmp_path / "mock_model"
    mock_model_dir.mkdir()
    
    config_file = tmp_path / "test_config.yaml"
    config_content = f"""
b3_semantic_gate:
  model_path: "{mock_model_dir.as_posix()}"
  max_length: 256
"""
    config_file.write_text(config_content)
    
    # Configure mock predictor
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = [
        SemanticGateResult(label="MALICIOUS_SEMANTIC_MANIPULATION", label_id=1, confidence=0.95)
    ]
    mock_get_predictor.return_value = mock_predictor
    
    classifier = SemanticGateClassifier(config_path=config_file)
    assert classifier.error_status is None
    
    res = classifier.classify("suspicious message")
    assert res["available"]
    assert res["label"] == "MALICIOUS"
    assert res["confidence"] == 0.95
    assert res["status"] == "ok"
    
    # Verify mock predictor was called correctly
    mock_predictor.predict.assert_called_once_with(["suspicious message"])
