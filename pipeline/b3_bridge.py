"""
pipeline/b3_bridge.py
======================
Stable adapter bridge module for B3 semantic classification.
Provides integration contract and default stub implementation.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

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

def classify_text(message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Interfacing function for B3 semantic classifier.
    By default, uses StubSemanticClassifier.
    Replace this with real B3 model logic when integrated.
    """
    classifier = StubSemanticClassifier()
    return classifier.classify(message, metadata)
