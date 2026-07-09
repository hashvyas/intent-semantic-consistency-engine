"""
pipeline/b3_bridge.py
======================
Stable adapter bridge module for B3 semantic classification.
Provides integration contract and default stub implementation.
"""

from __future__ import annotations
import os
import pathlib
import sys
from typing import Any, Dict, Optional
import yaml

_DEFAULT_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "isce_config.yaml"

def resolve_model_path(model_path: str) -> str:
    """Resolve model path against absolute path and the workspace root."""
    if os.path.exists(model_path):
        return os.path.abspath(model_path)
    # Try relative to workspace root (parent of pipeline dir)
    workspace_root = pathlib.Path(__file__).resolve().parent.parent
    candidate = workspace_root / model_path
    if candidate.exists():
        return os.path.abspath(candidate)
    return os.path.abspath(model_path)

def _load_b3_config(config_path: Optional[str | os.PathLike] = None) -> Dict[str, Any]:
    """Load configuration from isce_config.yaml."""
    path = pathlib.Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data.get("b3_semantic_gate", {})
    except Exception:
        return {}

class StubSemanticClassifier:
    """Default stub classifier returning 'unavailable' state.
    Allows testing/running the pipeline without an actual B3 model dependency.
    """
    def classify(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "available": False,
            "label": None,
            "confidence": None,
            "status": "B3 integration unavailable"
        }

class SemanticGateClassifier:
    """Real B3 classifier wrapper loading the trained DeBERTa model and performing inference."""
    def __init__(self, config_path: Optional[str | os.PathLike] = None) -> None:
        self.config = _load_b3_config(config_path)
        self.model_path = self.config.get("model_path", "b3/solution_stb/b3_semantic_gate/model/semantic_gate_v3")
        self.max_length = self.config.get("max_length", 256)
        self.device = self.config.get("device", None)
        self.predictor = None
        self.error_status = None
        
        # Verify model directory exists
        resolved_path = resolve_model_path(self.model_path)
        if not os.path.exists(resolved_path):
            self.error_status = f"B3 model checkpoint not found at {resolved_path}"
            return

        try:
            # Ensure workspace is in sys.path so b3 package is importable
            workspace_root = str(pathlib.Path(__file__).resolve().parent.parent)
            if workspace_root not in sys.path:
                sys.path.insert(0, workspace_root)
                
            from b3.solution_stb.b3_semantic_gate.inference import get_predictor
            self.predictor = get_predictor(self.model_path, max_length=self.max_length, device=self.device)
        except Exception as e:
            self.error_status = f"Failed to initialize B3 predictor: {str(e)}"

    def classify(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.error_status or self.predictor is None:
            return {
                "available": False,
                "label": None,
                "confidence": None,
                "status": self.error_status or "B3 predictor uninitialized"
            }
        try:
            results = self.predictor.predict([message])
            if not results:
                return {
                    "available": False,
                    "label": None,
                    "confidence": None,
                    "status": "Inference returned empty results"
                }
            res = results[0]
            # Standardize label mapping (map MALICIOUS_SEMANTIC_MANIPULATION to MALICIOUS)
            label_name = "MALICIOUS" if res.label == "MALICIOUS_SEMANTIC_MANIPULATION" else res.label
            return {
                "available": True,
                "label": label_name,
                "confidence": res.confidence,
                "status": "ok"
            }
        except Exception as e:
            return {
                "available": False,
                "label": None,
                "confidence": None,
                "status": f"Inference execution error: {str(e)}"
            }

_CLASSIFIER_INSTANCE: Optional[SemanticGateClassifier] = None

def classify_text(message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Interfacing function for B3 semantic classifier.
    """
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is None:
        _CLASSIFIER_INSTANCE = SemanticGateClassifier()
    return _CLASSIFIER_INSTANCE.classify(message, metadata)
