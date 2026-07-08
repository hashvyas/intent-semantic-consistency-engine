"""
pipeline/orchestrator.py
========================
ISCEPipeline orchestrator. Integrates B1, B2, Message Synthesizer, B3 Bridge, and Fusion.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional

from b1_scsv.scsv import SCSV
from b2_csia.csia import CSIA
from pipeline.synthesizer import synthesize_message
from pipeline.b3_bridge import classify_text
from pipeline.fusion import fuse_results

class ISCEPipeline:
    """Orchestrator managing SCSV, CSIA, Message Synthesizer, B3 Adapter, and Fusion."""
    
    def __init__(
        self,
        scsv: Optional[SCSV] = None,
        csia: Optional[CSIA] = None
    ) -> None:
        self.scsv = scsv or SCSV()
        self.csia = csia or CSIA()

    def run(
        self,
        messages: List[Dict[str, Any]],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the pipeline from B1 to Fusion.
        
        Parameters
        ----------
        messages: List[dict]
            Window of messages, with target message as messages[-1].
        context: Optional[str]
            Operational context candidate (e.g. 'urban', 'rural', 'highway').
            
        Returns
        -------
        dict
            Pipeline result dictionary.
        """
        if not messages:
            raise ValueError("ISCEPipeline.run: empty messages window supplied.")

        t_total_start = time.perf_counter()
        target_msg = messages[-1]

        # 1. Run B1 (SCSV)
        t_b1_start = time.perf_counter()
        if target_msg.get("_validation_assessment") is not None:
            b1_res = target_msg["_validation_assessment"]
        else:
            b1_res = self.scsv.check_stateful(target_msg)
        b1_ms = (time.perf_counter() - t_b1_start) * 1000.0

        if hasattr(b1_res, "valid"):
            b1_dict = {
                "valid": b1_res.valid,
                "fatal": getattr(b1_res, "fatal", False),
                "score": getattr(b1_res, "validation_score", 1.0),
                "confidence": getattr(b1_res, "confidence", 1.0),
                "reasons": getattr(b1_res, "reasons", []),
                "checks": getattr(b1_res, "checks", {}),
                "details": getattr(b1_res, "details", {})
            }
            b1_fatal = getattr(b1_res, "fatal", False)
        elif isinstance(b1_res, dict):
            b1_dict = b1_res
            b1_fatal = b1_res.get("fatal", False)
        else:
            b1_dict = {
                "valid": True,
                "fatal": False,
                "score": 1.0,
                "confidence": 1.0,
                "reasons": [],
                "checks": {},
                "details": {}
            }
            b1_fatal = False

        # If B1 validation is fatal, bypass B2 and go straight to fusion/rejection
        if b1_fatal:
            b2_dict = {
                "trust": 0.0,
                "entropy": 0.0,
                "cluster_score": 0.0,
                "replay_probability": 0.0,
                "identity_consistency": 0.0,
                "belief": 0.0,
                "disbelief": 1.0,
                "uncertainty": 0.0,
                "confidence": 1.0,
                "matched_profile": "none"
            }
            
            t_synt_start = time.perf_counter()
            synthesized_message = synthesize_message(messages, b2_dict, context)
            synt_ms = (time.perf_counter() - t_synt_start) * 1000.0
            
            t_bridge_start = time.perf_counter()
            b3_result = classify_text(synthesized_message["text"], synthesized_message)
            bridge_ms = (time.perf_counter() - t_bridge_start) * 1000.0
            
            t_fuse_start = time.perf_counter()
            decision, reason, fusion_details = fuse_results(b1_dict, b2_dict, b3_result)
            fuse_ms = (time.perf_counter() - t_fuse_start) * 1000.0
            
            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            
            return {
                "b1": b1_dict,
                "b2": b2_dict,
                "synthesized_message": synthesized_message,
                "b3": b3_result,
                "decision": decision,
                "reason": reason,
                "fusion": fusion_details,
                "latencies": {
                    "b1_ms": b1_ms,
                    "b2_ms": 0.0,
                    "synthesizer_ms": synt_ms,
                    "bridge_ms": bridge_ms,
                    "fusion_ms": fuse_ms,
                    "total_ms": total_ms
                }
            }

        # 2. Run B2 (CSIA)
        # Store validation assessment on target msg for B2 consumption
        target_msg["_validation_assessment"] = b1_res
        
        t_b2_start = time.perf_counter()
        b2_payload, b2_report = self.csia.check_extended(messages)
        b2_ms = (time.perf_counter() - t_b2_start) * 1000.0

        b2_dict = dict(b2_payload)
        if b2_report:
            b2_dict.update({
                "confidence": b2_report.confidence,
                "belief": b2_report.belief,
                "disbelief": b2_report.disbelief,
                "uncertainty": b2_report.uncertainty,
                "matched_profile": b2_report.vehicle_profile_label,
                "anomaly_reasons": b2_report.anomaly_reasons,
                "decision_summary": b2_report.decision_summary,
                "evidence_summary": b2_report.evidence_summary,
                "evidence_reasons": b2_report.evidence_reasons
            })
        else:
            b2_dict.update({
                "confidence": 1.0,
                "belief": b2_payload.get("belief", b2_payload.get("trust", 1.0)),
                "disbelief": b2_payload.get("disbelief", 0.0),
                "uncertainty": b2_payload.get("uncertainty", 0.0),
                "matched_profile": "unknown"
            })

        # 3. Run Message Synthesizer
        t_synt_start = time.perf_counter()
        synthesized_message = synthesize_message(messages, b2_dict, context)
        synt_ms = (time.perf_counter() - t_synt_start) * 1000.0

        # 4. Run B3 Adapter Bridge
        t_bridge_start = time.perf_counter()
        b3_result = classify_text(synthesized_message["text"], synthesized_message)
        bridge_ms = (time.perf_counter() - t_bridge_start) * 1000.0

        # 5. Run Pipeline Fusion
        t_fuse_start = time.perf_counter()
        decision, reason, fusion_details = fuse_results(b1_dict, b2_dict, b3_result)
        fuse_ms = (time.perf_counter() - t_fuse_start) * 1000.0

        total_ms = (time.perf_counter() - t_total_start) * 1000.0

        return {
            "b1": b1_dict,
            "b2": b2_dict,
            "synthesized_message": synthesized_message,
            "b3": b3_result,
            "decision": decision,
            "reason": reason,
            "fusion": fusion_details,
            "latencies": {
                "b1_ms": b1_ms,
                "b2_ms": b2_ms,
                "synthesizer_ms": synt_ms,
                "bridge_ms": bridge_ms,
                "fusion_ms": fuse_ms,
                "total_ms": total_ms
            }
        }
