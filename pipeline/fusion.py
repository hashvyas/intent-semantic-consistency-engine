"""
pipeline/fusion.py
==================
Fuses validation results (B1), trust reasoning results (B2), and semantic results (B3).
Forwards legacy decisions and exposes the extension point for future B3 semantic influence.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

def fuse_results(
    b1_result: Dict[str, Any],
    b2_result: Dict[str, Any],
    b3_result: Dict[str, Any]
) -> Tuple[str, str, Dict[str, Any]]:
    """Fuse B1, B2, and B3 outputs into a final decision, reasoning, and metadata.
    
    Directly forwards the legacy decision and reasoning when B3 is unavailable.
    Provides the single location where B3 semantic results can influence decisions in the future.
    """
    # 1. Forward the legacy decision logic
    # B1 SCSV validation issues or B2 trust anomalies determine the legacy decision
    b1_fatal = b1_result.get("fatal", False)
    b2_trust = b2_result.get("trust", 1.0)
    
    # Check if a custom classification decision is stored on B2
    matched_profile = b2_result.get("matched_profile", "none")
    
    if b1_fatal:
        decision = "REJECT"
        reasons = b1_result.get("reasons", [])
        reason = f"B1 validation fatal failure: {', '.join(reasons) if reasons else 'structural or physical anomaly'}."
    elif b2_trust < 0.4:
        decision = "REJECT"
        reason = f"B2 trust score ({b2_trust:.2f}) is below the critical threshold (0.40) indicating anomalous behavior."
    elif b2_trust < 0.7:
        decision = "CAUTION"
        reason = f"B2 trust score ({b2_trust:.2f}) indicates mild or suspicious anomalies, caution recommended."
    else:
        decision = "ACCEPT"
        reason = f"Both B1 validation and B2 trust score ({b2_trust:.2f}) indicate benign cooperative behavior."

    contributors = ["B1"]
    if not b1_fatal:
        contributors.append("B2")
        
    b3_available = b3_result.get("available", False)
    if b3_available:
        contributors.append("B3")

    # 2. B3 Semantic Influence Extension Point
    if b3_available:
        # Future B3 integration logic can be implemented here.
        # e.g., if b3_result.get("label") == "spoofing" and b3_result.get("confidence") > 0.8:
        #           decision = "REJECT"
        #           reason += " Confirmed by B3 semantic classifier."
        pass

    fusion_details = {
        "contributors": contributors,
        "b3_used": b3_available,
        "confidence": float(b2_result.get("confidence", b1_result.get("confidence", 1.0)))
    }

    return decision, reason, fusion_details
