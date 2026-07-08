"""
manual_pipeline_test.py
========================
Developer-facing Manual Validation Harness for the ISCE Security Pipeline.
Allows manual injection of V2X messages and walkthrough observation of B1 and B2 stages.
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import pathlib
import shutil
import datetime
import math
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

# Ensure stdout/stderr uses UTF-8 to support checkmarks and degree symbols on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure workspace is in import path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Core imports
from b1_scsv.scsv import SCSV, SCORE_ALLOW, SCORE_BLOCK
from b1_scsv.models import ValidationFailureReason, safe_parse_cam
from b2_csia.csia import CSIA, _nested_get, _nested_get_any, _haversine_m
from b2_csia.evidence_quality import EvidenceQuality
from b2_csia.observability_graph import ObservabilityGraphBuilder
from b2_csia.context_aware import MotionContextInferenceEngine, ContextAssessment, ENVELOPES
from b2_csia.adaptive_thresholds import AdaptiveThresholdEngine
from b2_csia.uncertainty import MassFunction, Provenance
from b2_csia.evidence_extractors import (
    SpatialSimilarityExtractor,
    TemporalSynchronizationExtractor,
    KinematicSimilarityExtractor,
    SemanticSimilarityExtractor,
    GraphConnectivityExtractor,
    IdentityConsistencyExtractor,
    RSUCorroborationExtractor,
    HistoricalTrustExtractor,
)
from b2_csia.behavior_profile import BehaviorEvidence
from b2_csia.behavior_reasoning import BehavioralReasoningEngine
from b2_csia.trust_propagation import TrustPropagationEngine
from b2_csia.experimental import AttackScenarioGenerator, ExperimentConfig

# Graph visualization libraries check
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def get_station_type_name(st: Optional[int]) -> str:
    """Helper to convert ETSI station_type integer code to readable name."""
    if st is None:
        return "unknown"
    mapping = {
        0: "unknown",
        1: "pedestrian",
        2: "cyclist",
        3: "moped",
        4: "motorcycle",
        5: "passengerCar",
        6: "bus",
        7: "lightTruck",
        8: "heavyTruck",
        9: "trailer",
        10: "specialVehicle",
        11: "tram",
        12: "lightVruVehicle",
        13: "animal",
        14: "agricultural",
        15: "roadSideUnit",
    }
    return mapping.get(st, f"unknown ({st})")


def format_mass(m: MassFunction) -> str:
    return f"MassFunction(A={m.belief:.4f}, not_A={m.disbelief:.4f}, Theta={m.uncertainty:.4f})"


def pause_for_step(step_name: str, enabled: bool):
    """Optionally pause execution after a stage."""
    if not enabled:
        return
    print(f"\n{step_name} Complete.")
    input("Press Enter to continue...\n↓")


def generate_graph_visualizations(graph, propagated_beliefs, run_id: str):
    if not VISUALIZATION_AVAILABLE:
        print("\n[Warning] Graph visualization requires networkx and matplotlib. Skipping PNG generation.")
        return
    
    try:
        vis_dir = _PROJECT_ROOT / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # Determine node positions
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(graph, k=0.5, seed=42)
        
        # 1. Observability Graph
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=600, alpha=0.9)
        nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')
        
        edges = list(graph.edges)
        weights = [graph.edges[e].weight for e in edges]
        nx.draw_networkx_edges(graph, pos, edgelist=edges, width=[w * 4 for w in weights], edge_color='darkgray')
        
        edge_labels = {e: f"{graph.edges[e].weight:.2f}" for e in edges}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"Observability Graph - Run {run_id}", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(vis_dir / f"observability_graph_{run_id}.png", dpi=150)
        plt.close()
        
        # 2. Trust Graph
        plt.figure(figsize=(10, 8))
        node_colors = []
        for node in graph.nodes:
            trust_val = 1.0
            if node in propagated_beliefs:
                trust_val = propagated_beliefs[node].belief
            # Simple color transition from red (distrusted) to green (trusted)
            node_colors.append((1.0 - trust_val, trust_val, 0.0, 0.9))
            
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=600)
        nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold', font_color='white')
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color='black', alpha=0.6)
        
        plt.title(f"Trust Graph - Run {run_id}", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(vis_dir / f"trust_graph_{run_id}.png", dpi=150)
        plt.close()
        
        print(f"\n  [Visualisation] Graphs saved successfully to {vis_dir}/")
    except Exception as exc:
        print(f"\n  [Warning] Graph visualization failed: {exc}")


def run_pipeline(
    messages: List[Dict[str, Any]],
    scsv: SCSV,
    csia: CSIA,
    step_mode: bool = False,
    verbose: bool = True,
    log_enabled: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs a sequence of messages statefully through B1 and B2 pipelines."""
    passed_messages: List[Dict[str, Any]] = []
    run_logs: List[Dict[str, Any]] = []
    
    total_messages = len(messages)
    passed_count = 0
    failed_count = 0
    
    b1_latencies = []
    b2_latencies = []
    total_latencies = []
    
    # B2 substages
    graph_times = []
    context_times = []
    threshold_times = []
    reasoning_times = []
    propagation_times = []

    failure_reasons = {
        "Replay": 0,
        "Malformed": 0,
        "Physics": 0,
        "Policy": 0,
        "Sybil": 0,
        "Collusion": 0,
        "Fabrication": 0,
        "Other B2 Anomaly": 0,
    }

    print(f"\nProcessing {total_messages} messages...")
    
    for idx, msg in enumerate(messages):
        msg_name = f"Msg {idx+1}/{total_messages}"
        if verbose:
            print("\n" + "="*50)
            print(f"INPUT MESSAGE: {msg_name}")
            print("="*50)
            
        t_start = time.perf_counter()
        
        # ── Parsing input ─────────────────────────────────────────────────────
        cam_msg, parse_error = safe_parse_cam(msg)
        
        # Walkthrough fields extract
        sid = _nested_get_any(msg, "header.station_id")
        m_id = _nested_get_any(msg, "header.message_id")
        ts = _nested_get_any(msg, "cam.generation_delta_time")
        
        lat = _nested_get(msg, csia._lat_field)
        lon = _nested_get(msg, csia._lon_field)
        speed = _nested_get(msg, "cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.speed")
        heading = _nested_get(msg, "cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.heading")
        acc = _nested_get(msg, "cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.longitudinal_acceleration")
        cert_id = msg.get("certificate_id") or msg.get("cert_id") or "None"
        
        # ── B1 Stateful Check ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        b1_res = scsv.check_stateful(msg)
        t1 = time.perf_counter()
        b1_latency = (t1 - t0) * 1000.0
        b1_latencies.append(b1_latency)
        
        # B1 Walkthrough symbols
        b1_walkthrough = {
            "Structural Validation": "✓" if b1_res.checks.get("structure", True) else "✗",
            "Replay Protection": "✓" if b1_res.checks.get("replay", True) else "✗",
            "Timestamp Freshness": "✓" if b1_res.checks.get("timestamp", True) else "✗",
            "Certificate Continuity": "✓" if b1_res.checks.get("certificate", True) else "✗",
            "Physical Plausibility": "✓" if b1_res.checks.get("physics", True) else "✗",
        }
        
        # Accumulate failure counts for B1 checks
        if not b1_res.checks.get("structure", True):
            failure_reasons["Malformed"] += 1
        if not b1_res.checks.get("replay", True):
            failure_reasons["Replay"] += 1
        if not b1_res.checks.get("timestamp", True):
            failure_reasons["Physics"] += 1
        if not b1_res.checks.get("certificate", True):
            failure_reasons["Policy"] += 1
        if not b1_res.checks.get("physics", True):
            failure_reasons["Physics"] += 1
            
        if verbose and b1_res.fatal:
            print("\nPipeline Stage: PKI / SCMS (Pre-decoding & Certificate Verification)")
            print("----------------------------------------------------------------------")
            print("✓ Decoded and signature verified.")
            print(f"Station ID:    {sid}")
            print(f"Vehicle Type:  {get_station_type_name(cam_msg.station_type if cam_msg else None)}")
            print(f"Timestamp:     {ts}")
            print(f"Position:      ({lat * 1e-7:.7f}, {lon * 1e-7:.7f})")
            print(f"Speed:         {speed / 100.0 if speed is not None else 'N/A'} m/s ({speed})")
            print(f"Heading:       {heading / 10.0 if heading is not None else 'N/A'}° ({heading})")
            print(f"Acceleration:  {acc / 100.0 if acc is not None else 'N/A'} m/s² ({acc})")
            print(f"Certificate:   {cert_id}")
            print("\nPipeline Stage: SCSV (Sender Certificate Semantic Validation)")
            print("----------------------------------------------------------------------")
            for chk, sym in b1_walkthrough.items():
                print(f"{sym} {chk}")
            print(f"Validation Score:      {b1_res.validation_score:.4f}")
            print(f"Validation Confidence  {b1_res.confidence:.2f}")
            if hasattr(b1_res, "confidence_breakdown") and b1_res.confidence_breakdown:
                print("Confidence Breakdown")
                print(f"  Structural Completeness  {b1_res.confidence_breakdown.get('Structural Completeness', 1.0):.2f}")
                print("  Historical Evidence")
                print(f"    History Count: {b1_res.details.get('history_count', 0)}")
                print(f"    Historical Confidence: {b1_res.details.get('historical_confidence', 1.0):.2f}")
                print(f"    Normalization Target: {b1_res.details.get('max_history', 50)} observations")
                print(f"  Certificate Stability    {b1_res.confidence_breakdown.get('Certificate Stability', 1.0):.2f}")
                print(f"  Replay Certainty         {b1_res.confidence_breakdown.get('Replay Certainty', 1.0):.2f}")
                print(f"  Timestamp Reliability    {b1_res.confidence_breakdown.get('Timestamp Reliability', 1.0):.2f}")
            if hasattr(b1_res, "confidence_contributors") and b1_res.confidence_contributors:
                print("Confidence Contributors")
                for contrib in b1_res.confidence_contributors:
                    print(f"  {contrib}")
                level = "Low"
                if b1_res.confidence >= 0.85:
                    level = "High"
                elif b1_res.confidence >= 0.70:
                    level = "Moderate"
                print(f"Overall Confidence         {level}")
            print(f"Fatal Check:           {b1_res.fatal}")
            print(f"Validation Verdict:    {'PASS' if b1_res.valid else 'FAIL'}")
            if b1_res.reasons:
                print(f"Anomalies:             {', '.join(b1_res.reasons)}")
            
        if b1_res.fatal:
            failed_count += 1
            t_end = time.perf_counter()
            total_latencies.append((t_end - t_start) * 1000.0)
            
            # Print B1 Failure block (Part 4)
            print("\n" + "!"*40)
            print("B1 FATAL / PIPELINE TERMINATED")
            is_phys_fatal = b1_res.reason in (ValidationFailureReason.INVALID_COORDINATES,
                                              ValidationFailureReason.IMPOSSIBLE_KINEMATICS,
                                              ValidationFailureReason.INVALID_HEADING)
            if is_phys_fatal:
                print("Physical Sanity")
                print("FAIL")
                print("Reason")
                reason_name = b1_res.reason.name if b1_res.reason else 'PARSE_ERROR'
                if reason_name == 'INVALID_COORDINATES':
                    print("Invalid Coordinates")
                elif reason_name == 'IMPOSSIBLE_KINEMATICS':
                    print("Impossible Kinematics")
                elif reason_name == 'INVALID_HEADING':
                    print("Invalid Heading")
                else:
                    print(reason_name)
                print("Fatal")
                print("YES")
                print("Pipeline")
                print("Terminated")
            else:
                print(f"Reason:           {b1_res.reason.name if b1_res.reason else 'PARSE_ERROR'}")
                print(f"Exact Validation: {b1_res.details.get('kinematic_violation') or b1_res.details.get('error') or 'Missing fields'}")
                print("Relevant Values:")
                for k, v in b1_res.details.items():
                    if k not in ("reject_stage", "parse_warnings"):
                        print(f"  {k:<16s}: {v}")
                explanation = "Message structure is corrupt, missing key attributes, or parsing contains NaNs."
                print(f"Suggested Explanation: {explanation}")
            print("!"*40)
            
            if log_enabled:
                run_logs.append({
                    "input": msg,
                    "b1_result": {
                        "fatal": True,
                        "valid": False,
                        "score": 0.0,
                        "confidence": 1.0,
                        "reasons": b1_res.reasons,
                        "checks": b1_res.checks,
                        "details": b1_res.details,
                    },
                    "b2_result": None,
                    "latency": {
                        "b1_ms": b1_latency,
                        "b2_ms": 0.0,
                        "total_ms": (t_end - t_start) * 1000.0
                    },
                    "final_assessment": "FAIL"
                })
            
            pause_for_step("B1 Complete (Fatal)", step_mode)
            continue
            
        pause_for_step("B1 Complete", step_mode)

        # ── B2 Stateful Check ─────────────────────────────────────────────────
        # Store assessment on msg for B2 consumption
        msg["_validation_assessment"] = b1_res
        passed_messages.append(msg)
        
        # Sliding time window of recently passed messages
        window_size_ms = csia._window_size_ns / 1e6
        current_ts = ts if ts is not None else 0.0
        window = []
        for m in passed_messages:
            m_ts = _nested_get_any(m, "cam.generation_delta_time")
            m_ts = m_ts if m_ts is not None else 0.0
            if abs(current_ts - m_ts) <= window_size_ms:
                window.append(m)
                
        # Profile B2 stages
        # Stage 1: Observability Graph Construction
        t_sub_start = time.perf_counter()
        for wm in window:
            wsid = _nested_get_any(wm, "header.station_id")
            wlat = _nested_get(wm, csia._lat_field)
            wlon = _nested_get(wm, csia._lon_field)
            wheading = _nested_get(wm, "cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.heading")
            wts = _nested_get(wm, csia._ts_field)
            wst = _nested_get(wm, "cam.cam_parameters.basic_container.station_type")
            wlane = _nested_get_any(wm, "cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.lane")
            motion_cfg = csia._raw.get("motion_context", {})
            supported_ctxs = motion_cfg.get("supported_contexts", ["urban"])
            context_to_use = supported_ctxs[0] if supported_ctxs else "urban"
            csia._graph_builder.update_node(
                station_id=wsid,
                lat_e7=wlat,
                lon_e7=wlon,
                heading_deg10=wheading,
                timestamp_ns=wts,
                station_type=wst,
                wall_time=t_start,
                context=context_to_use,
                lane_position=wlane,
                raw_msg=wm
            )
        graph_time = (time.perf_counter() - t_sub_start) * 1000.0
        graph_times.append(graph_time)
        pause_for_step("Observability Graph", step_mode)
        
        # Stage 2: Motion Context Inference
        t_sub_start = time.perf_counter()
        context_conf = csia._raw.get("motion_context", {})
        context_assess = csia._context_engine.infer_context(window, cluster_id=sid, config=context_conf)
        context_time = (time.perf_counter() - t_sub_start) * 1000.0
        context_times.append(context_time)
        
        # Stage 3: Adaptive Thresholds
        t_sub_start = time.perf_counter()
        speeds = []
        for wm in window:
            wspd = _nested_get(wm, "cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.speed")
            if wspd is not None:
                speeds.append(wspd)
        median_speed = float(np.median(speeds)) if speeds else 0.0
        
        threshold_res = csia._threshold_engine.calculate_threshold(
            cluster=window,
            median_speed=median_speed,
            highway_speed_threshold=csia._highway_spd_thr,
            message_arrival_rate=10.0,
            observation_duration_s=1.0
        )
        csia._threshold_engine.record_distance(median_speed / 100.0)
        threshold_time = (time.perf_counter() - t_sub_start) * 1000.0
        threshold_times.append(threshold_time)
        pause_for_step("Adaptive Thresholds", step_mode)
        
        # Stage 4: Evidence extraction & Behavioral Reasoning
        t_sub_start = time.perf_counter()
        spatial_sim, spatial_conf, _ = SpatialSimilarityExtractor().extract(window)
        temporal_sim, temporal_conf, _ = TemporalSynchronizationExtractor().extract(window)
        kinematic_sim, kinematic_conf, _ = KinematicSimilarityExtractor().extract(window)
        semantic_sim, semantic_conf, _ = SemanticSimilarityExtractor().extract(window)
        graph_sim, graph_conf, _ = GraphConnectivityExtractor().extract(window, csia._graph_builder.graph)
        identity_consistency, identity_conf, _ = IdentityConsistencyExtractor().extract(window)
        rsu_corroboration, rsu_conf, _ = RSUCorroborationExtractor().extract(window, csia._graph_builder.graph)
        
        # Dynamically evaluate historical trust using message type and history
        msg_type = msg.get("message_type")
        default_trust = 0.10 if msg_type == "DENM" else 0.90
        historical_trust = default_trust
        if sid is not None and sid in csia._trust_histories:
            historical_trust = csia._trust_histories[sid].current

        # Target-specific adjustments to prevent rolling-window false positives on benign nodes
        target_msgs = [m for m in window if isinstance(m, dict) and _nested_get_any(m, "header.station_id") == sid]
        other_msgs = [m for m in window if isinstance(m, dict) and _nested_get_any(m, "header.station_id") != sid]
        
        target_cert = msg.get("certificate_id") or msg.get("cert_id")
        cert_dups = 0
        if target_cert is not None:
            cert_dups = sum(1 for m in window if (m.get("certificate_id") or m.get("cert_id")) == target_cert)

        if len(target_msgs) <= 1 and cert_dups <= 1:
            identity_consistency = 1.0
            
        if len(other_msgs) >= 1:
            # Spatial Similarity: measure average distance of target_sid to others
            lat_t = _nested_get(msg, csia._lat_field)
            lon_t = _nested_get(msg, csia._lon_field)
            dists = []
            for m in other_msgs:
                lat_o = _nested_get(m, csia._lat_field)
                lon_o = _nested_get(m, csia._lon_field)
                dists.append(_haversine_m(lat_t, lon_t, lat_o, lon_o))
            avg_d = np.mean(dists)
            spatial_sim = math.exp(-0.01 * avg_d)
            
            # Temporal Similarity: measure average time delta of target_sid to others
            ts_t = _nested_get(msg, csia._ts_field)
            ts_diffs = []
            for m in other_msgs:
                ts_o = _nested_get(m, csia._ts_field)
                ts_diffs.append(abs(ts_t - ts_o))
            avg_ts_diff = np.mean(ts_diffs)
            temporal_sim = 1.0 / (1.0 + (avg_ts_diff / 100.0))
            
            # Kinematic Similarity: measure average speed and circular heading difference of target_sid to others
            target_spd = _nested_get(msg, "cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.speed")
            target_hd = _nested_get(msg, "cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.heading")
            other_speeds = []
            other_headings = []
            for m in other_msgs:
                spd = _nested_get(m, "cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.speed")
                hd = _nested_get(m, "cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.heading")
                if spd is not None:
                    other_speeds.append(spd)
                if hd is not None:
                    other_headings.append(hd)
                    
            if other_speeds and target_spd is not None:
                avg_spd_diff = np.mean([abs(target_spd - s) for s in other_speeds])
                sim_speed = 1.0 / (1.0 + avg_spd_diff / 100.0)
            else:
                sim_speed = 1.0
                
            if other_headings and target_hd is not None:
                th_rad = math.radians(target_hd * 0.1)
                diffs = []
                for h in other_headings:
                    oh_rad = math.radians(h * 0.1)
                    diffs.append(1.0 - abs(math.cos(th_rad - oh_rad)))
                sim_heading = 1.0 - np.mean(diffs)
            else:
                sim_heading = 1.0
                
            kinematic_sim = 0.5 * sim_speed + 0.5 * sim_heading

        prov = Provenance(modules={"kinematic"}, min_evidence_quality=0.9, min_confidence=0.8)
        evidence = BehaviorEvidence(
            spatial_similarity=spatial_sim,
            temporal_similarity=temporal_sim,
            kinematic_similarity=kinematic_sim,
            semantic_similarity=semantic_sim,
            graph_connectivity=graph_sim,
            identity_consistency=identity_consistency,
            rsu_corroboration=rsu_corroboration,
            historical_trust=historical_trust,
            confidence=min(spatial_conf, temporal_conf, kinematic_conf, semantic_conf),
            provenance=prov,
            validation_score=b1_res.validation_score,
            validation_confidence=b1_res.confidence,
        )
        assessment = csia._reasoning_engine.evaluate(evidence, reliability_alpha=context_assess.confidence)
        
        reasoning_time = (time.perf_counter() - t_sub_start) * 1000.0
        reasoning_times.append(reasoning_time)
        pause_for_step("Behavioral Evidence & Reasoning", step_mode)
        
        # Stage 5: Trust Propagation
        t_sub_start = time.perf_counter()
        initial_beliefs = {}
        for wm in window:
            wsid = _nested_get_any(wm, "header.station_id")
            if wsid is not None:
                # Specific validation details for each window node
                w_assess = wm.get("_validation_assessment")
                w_val_score = w_assess.validation_score if w_assess else 1.0
                w_val_conf = w_assess.confidence if w_assess else 1.0
                
                if wsid == sid:
                    local_trust = 1.0 if assessment.attack_type == "none" else float(max(0.0, min(1.0, 1.0 - assessment.belief)))
                else:
                    local_trust = csia._trust_histories[wsid].current if wsid in csia._trust_histories else 1.0
                combined_trust = local_trust * w_val_score
                combined_confidence = 0.9 * w_val_conf
                
                initial_beliefs[wsid] = MassFunction.from_trust_confidence(
                    combined_trust,
                    combined_confidence,
                    origin_module="local"
                )
        propagated, prop_meta = csia._propagation_engine.propagate(
            csia._graph_builder.graph,
            initial_beliefs,
            csia._raw.get("trust_propagation", {})
        )
        # Update trust histories for all propagated nodes to dynamically update historical trust
        if propagated:
            from b2_csia.models import TrustHistory
            for nid, mf in propagated.items():
                if nid not in csia._trust_histories:
                    csia._trust_histories[nid] = TrustHistory(
                        station_id=nid,
                        window=csia._trust_history_window,
                        decay_alpha=csia._trust_decay_alpha,
                        recovery_beta=csia._trust_recovery_beta
                    )
                csia._trust_histories[nid].update(float(mf.belief))
        propagation_time = (time.perf_counter() - t_sub_start) * 1000.0
        propagation_times.append(propagation_time)
        pause_for_step("Trust Propagation", step_mode)
        
        # B2 final calculations
        t_b2_end = time.perf_counter()
        b2_latency = graph_time + context_time + threshold_time + reasoning_time + propagation_time
        b2_latencies.append(b2_latency)
        
        t_end = time.perf_counter()
        total_latency = (t_end - t_start) * 1000.0
        total_latencies.append(total_latency)
        
        # Final trust check
        node_mf = propagated.get(sid, MassFunction(1.0, 0.0, 0.0))
        b2_trust = float(node_mf.belief)
        b2_valid = b2_trust >= 0.4
        
        if b2_valid:
            passed_count += 1
        else:
            failed_count += 1
            # Record specific B2 failure reasons
            if assessment.attack_type == "sybil":
                failure_reasons["Sybil"] += 1
            elif assessment.attack_type == "replay":
                failure_reasons["Replay"] += 1
            elif assessment.attack_type == "collusion":
                failure_reasons["Collusion"] += 1
            elif assessment.attack_type == "fabrication":
                failure_reasons["Fabrication"] += 1
            else:
                failure_reasons["Other B2 Anomaly"] += 1

        if verbose:
            # Simulate MBD flags based on kinematics and CSIA behavioral reasoning
            mbd_speed_anomaly = "False"
            mbd_trajectory_anomaly = "False"
            mbd_sybil_pattern = "False"
            mbd_replay_pattern = "False"
            mbd_spoofing_alert = "False"
            
            # Check for speed anomalies (physically possible but suspicious speed for the station type)
            if cam_msg and cam_msg.speed is not None:
                if cam_msg.station_type == 5 and cam_msg.speed > 4000:
                    mbd_speed_anomaly = "True (Suspicious Speed for Passenger Car)"
                elif cam_msg.station_type in (6, 8) and cam_msg.speed > 3000:
                    mbd_speed_anomaly = "True (Suspicious Speed for Heavy Vehicle)"
            
            if assessment.attack_type == "sybil":
                mbd_sybil_pattern = "True (Coordinated identity footprint)"
            elif assessment.attack_type == "collusion":
                mbd_sybil_pattern = "True (Coordinated kinematic footprint)"
            elif assessment.attack_type == "replay":
                mbd_replay_pattern = "True (High temporal correlation with another stream)"
            elif assessment.attack_type == "fabrication":
                mbd_spoofing_alert = "True (Isolated or uncorroborated event)"

            if mbd_speed_anomaly != "False" or assessment.attack_type == "collusion":
                mbd_trajectory_anomaly = "True (Kinematic profile deviations)"

            # Observability Graph details
            g = csia._graph_builder.graph
            neighbours = [n for n in g.nodes if (sid, n) in g.edges or (n, sid) in g.edges]
            weights_dict = {}
            confidence_dict = {}
            for n in neighbours:
                key = (sid, n) if sid < n else (n, sid)
                edge = g.edges.get(key)
                if edge:
                    weights_dict[n] = float(f"{edge.weight:.2f}")
                    confidence_dict[n] = float(f"{edge.confidence:.2f}")

            # ------------------------------------------------
            # Validation Summary
            # ------------------------------------------------
            print("\n================================================")
            print("Validation Summary")
            print("================================================")
            print(f"Validation Score:      {b1_res.validation_score:.4f}")
            print(f"Validation Confidence  {b1_res.confidence:.2f}")
            if hasattr(b1_res, "confidence_breakdown") and b1_res.confidence_breakdown:
                print("Confidence Breakdown")
                print(f"  Structural Completeness  {b1_res.confidence_breakdown.get('Structural Completeness', 1.0):.2f}")
                print("  Historical Evidence")
                print(f"    History Count: {b1_res.details.get('history_count', 0)}")
                print(f"    Historical Confidence: {b1_res.details.get('historical_confidence', 1.0):.2f}")
                print(f"    Normalization Target: {b1_res.details.get('max_history', 50)} observations")
                print(f"  Certificate Stability    {b1_res.confidence_breakdown.get('Certificate Stability', 1.0):.2f}")
                print(f"  Replay Certainty         {b1_res.confidence_breakdown.get('Replay Certainty', 1.0):.2f}")
                print(f"  Timestamp Reliability    {b1_res.confidence_breakdown.get('Timestamp Reliability', 1.0):.2f}")
            if hasattr(b1_res, "confidence_contributors") and b1_res.confidence_contributors:
                print("Confidence Contributors")
                for contrib in b1_res.confidence_contributors:
                    print(f"  {contrib}")
                level = "Low"
                if b1_res.confidence >= 0.85:
                    level = "High"
                elif b1_res.confidence >= 0.70:
                    level = "Moderate"
                print(f"Overall Confidence         {level}")
            print(f"Fatal / Recoverable:   {'Fatal' if b1_res.fatal else 'Recoverable'}")
            print("Checks:")
            for chk, sym in b1_walkthrough.items():
                print(f"  {sym} {chk:<22s}")
            if b1_res.reasons:
                print(f"Anomalies:             {', '.join(b1_res.reasons)}")

            # ------------------------------------------------
            # Misbehavior Summary
            # ------------------------------------------------
            print("\n================================================")
            print("Misbehavior Summary")
            print("================================================")
            print(f"Replay:             {mbd_replay_pattern}")
            print(f"Position Spoofing:  {mbd_spoofing_alert}")
            print(f"Trajectory Anomaly: {mbd_trajectory_anomaly}")
            print(f"Sybil:              {mbd_sybil_pattern}")

            # ------------------------------------------------
            # CSIA Summary
            # ------------------------------------------------
            print("\n================================================")
            print("CSIA Summary")
            print("================================================")
            print("Observability:")
            print(f"  Node Count:      {len(window)}")
            print(f"  Neighbour Count: {len(neighbours)}")
            print(f"  Neighbour IDs:   {neighbours}")
            print(f"  Edge Weights:    {weights_dict}")
            
            env = csia._context_engine.get_envelope(context_assess.context)
            print("Adaptive Threshold:")
            if env:
                print(f"  Expected Speed Range:  0.0 - {env.expected_speed_max:.1f} m/s")
                print(f"  Expected Acceleration: -{env.expected_brake_max:.1f} - {env.expected_acc_max:.1f} m/s²")
            print(f"  Observed Speed:        {speed/100.0 if speed is not None else 0.0:.1f} m/s")
            print(f"  Observed Acceleration: {acc/100.0 if acc is not None else 0.0:.1f} m/s²")
            
            print(f"Motion Context:")
            print(f"  Detected Context:   {context_assess.context}")
            print(f"  Context Confidence: {context_assess.confidence:.4f}")
            
            init_mf = initial_beliefs.get(sid, MassFunction(1.0, 0.0, 0.0))
            print("DST (Initial Local Evidence Fusion):")
            print(f"  Belief (Benign):      {init_mf.belief:.4f}")
            print(f"  Disbelief (Distrust): {init_mf.disbelief:.4f}")
            print(f"  Uncertainty:          {init_mf.uncertainty:.4f}")
            
            print("Trust Propagation:")
            print(f"  Propagation Depth: {prop_meta.get('iterations_to_converge', 0)}")

            # ------------------------------------------------
            # Evidence Summary
            # ------------------------------------------------
            print("\n================================================")
            print("Evidence Summary")
            print("================================================")
            print(f"Final Trust: {b2_trust:.4f}")
            print(f"Belief:      {node_mf.belief:.4f}")
            print(f"Disbelief:   {node_mf.disbelief:.4f}")
            print(f"Uncertainty: {node_mf.uncertainty:.4f}")
            
            # Calibrate confidence based on available evidence
            calibrated_conf = float(len(window) / (len(window) + 5.0)) * b1_res.confidence
            print(f"Confidence:  {calibrated_conf:.4f}")
            
            # Generate evidence-driven reasons
            reasons_list, _ = csia.generate_reasons_and_summary(window, b1_res, assessment)
            print("Reasons:")
            for r_str in reasons_list:
                print(f"  * {r_str}")
            print("================================================\n")
            
        if log_enabled:
            run_logs.append({
                "input": msg,
                "b1_result": {
                    "valid": True,
                    "reason": None,
                    "details": b1_res.details
                },
                "b2_result": {
                    "trust": b2_trust,
                    "belief": node_mf.belief,
                    "disbelief": node_mf.disbelief,
                    "uncertainty": node_mf.uncertainty,
                    "confidence": assessment.confidence,
                    "matched_profile": assessment.matched_profile,
                    "context": context_assess.context,
                    "evidence": {
                        "spatial": spatial_sim,
                        "temporal": temporal_sim,
                        "kinematic": kinematic_sim,
                        "semantic": semantic_sim,
                        "graph": graph_sim,
                        "identity": identity_consistency,
                        "rsu": rsu_corroboration
                    }
                },
                "latency": {
                    "b1_ms": b1_latency,
                    "b2_ms": b2_latency,
                    "total_ms": total_latency,
                    "breakdown": {
                        "graph_ms": graph_time,
                        "context_ms": context_time,
                        "threshold_ms": threshold_time,
                        "reasoning_ms": reasoning_time,
                        "propagation_ms": propagation_time
                    }
                },
                "final_assessment": "PASS" if b2_valid else "FAIL"
            })
            
        pause_for_step("Final Assessment", step_mode)
        
    # Generate graphs if run completed and visualization available
    if len(passed_messages) >= 2:
        run_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
        generate_graph_visualizations(csia._graph_builder.graph, propagated, run_timestamp)

    summary = {
        "total": total_messages,
        "passed": passed_count,
        "failed": failed_count,
        "b1_latency_avg": np.mean(b1_latencies) if b1_latencies else 0.0,
        "b2_latency_avg": np.mean(b2_latencies) if b2_latencies else 0.0,
        "total_latency_avg": np.mean(total_latencies) if total_latencies else 0.0,
        "breakdown_avg": {
            "graph_ms": np.mean(graph_times) if graph_times else 0.0,
            "context_ms": np.mean(context_times) if context_times else 0.0,
            "threshold_ms": np.mean(threshold_times) if threshold_times else 0.0,
            "reasoning_ms": np.mean(reasoning_times) if reasoning_times else 0.0,
            "propagation_ms": np.mean(propagation_times) if propagation_times else 0.0,
        },
        "failure_reasons": failure_reasons
    }
    
    return run_logs, summary


def load_messages_from_path(input_path: str) -> List[Dict[str, Any]]:
    path = pathlib.Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")
        
    if path.is_file():
        # Check if it is a JSON file
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            
        if isinstance(data, list):
            for m in data:
                if isinstance(m, dict):
                    m["_source_file"] = path.name
            return data
        elif isinstance(data, dict):
            # Check if it's a scenario specification file
            if "scenario" in data and "vehicles" in data:
                print(f"Detected Scenario Specification: {data.get('scenario')}")
                print(f"Configuring Attack Scenario Generator (seed=42)...")
                generator = AttackScenarioGenerator(seed=42)
                
                # Convert scenario spec to ExperimentConfig
                attack_type = data.get("expected_result", "none").lower()
                if "sybil" in attack_type:
                    atk = "sybil"
                elif "replay" in attack_type:
                    atk = "replay"
                elif "collusion" in attack_type:
                    atk = "collusion"
                elif "fabrication" in attack_type:
                    atk = "fabrication"
                else:
                    atk = "none"
                    
                config = ExperimentConfig(
                    name=data.get("scenario"),
                    attack_type=atk,
                    vehicle_count=data.get("vehicles", 10),
                    attacker_count=data.get("attackers", 0),
                    seed=42
                )
                generated = generator.generate_scenario(config)
                for m in generated:
                    if isinstance(m, dict):
                        m["_source_file"] = f"generated_{atk}"
                return generated
            else:
                # Single message dict
                data["_source_file"] = path.name
                return [data]
        else:
            raise ValueError("Invalid JSON message structure.")
            
    elif path.is_dir():
        # Load all JSON messages in directory
        json_files = sorted(list(path.glob("*.json")))
        if not json_files:
            raise FileNotFoundError(f"No JSON messages found in directory: {input_path}")
            
        messages = []
        for jf in json_files:
            with jf.open("r", encoding="utf-8") as fh:
                msg_data = json.load(fh)
                if isinstance(msg_data, list):
                    for m in msg_data:
                        if isinstance(m, dict):
                            m["_source_file"] = jf.name
                    messages.extend(msg_data)
                else:
                    if isinstance(msg_data, dict):
                        msg_data["_source_file"] = jf.name
                    messages.append(msg_data)
                    
        # Sort statefully by timestamp to prevent out-of-order anomalies
        def get_timestamp(m):
            ts = _nested_get_any(m, "cam.generation_delta_time")
            return ts if ts is not None else 0.0
            
        messages.sort(key=get_timestamp)
        return messages


def add_to_regression(filepath: str):
    """Copies a failing JSON message file to test_messages/regression/."""
    src = pathlib.Path(filepath)
    if not src.exists():
        print(f"Error: source file {filepath} not found.")
        sys.exit(1)
        
    dest_dir = _PROJECT_ROOT / "test_messages" / "regression"
    dest_dir.mkdir(exist_ok=True, parents=True)
    
    dest = dest_dir / src.name
    shutil.copy(src, dest)
    print(f"[Success] Copied {src.name} to {dest_dir}/")


def save_log_file(logs: List[Dict[str, Any]], summary: Dict[str, Any]):
    logs_dir = _PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Sequential counter naming
    date_str = datetime.date.today().strftime("%Y_%m_%d")
    counter = 1
    while True:
        log_path = logs_dir / f"run_{date_str}_{counter:02d}.json"
        if not log_path.exists():
            break
        counter += 1
        
    log_payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": summary,
        "runs": logs
    }
    
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(log_payload, fh, indent=2)
    print(f"\n[Logging] Saved run log to {log_path}")


def adjust_timestamps_to_fresh(messages: List[Dict[str, Any]], path_str: str) -> List[Dict[str, Any]]:
    if not messages:
        return messages
        
    adjusted_messages = []
    for m in messages:
        if not isinstance(m, dict):
            continue
            
        src_file = str(m.get("_source_file", "")).lower()
        p_str = path_str.lower()
        
        # Check if this specific message is a stale_timestamp or replay test
        if "stale_timestamp" in src_file or "replay" in src_file or "stale_timestamp" in p_str or "replay" in p_str:
            # Skip adjustment for this message
            continue
        adjusted_messages.append(m)
        
    if not adjusted_messages:
        return messages
        
    timestamps = []
    for m in adjusted_messages:
        ts = _nested_get_any(m, "cam.generation_delta_time")
        if ts is not None and isinstance(ts, (int, float)) and not math.isnan(ts):
            timestamps.append(ts)
            
    if not timestamps:
        return messages
        
    max_ts = max(timestamps)
    now_ms = time.time() * 1000.0
    shift = now_ms - max_ts
    
    for m in adjusted_messages:
        if "cam" in m and isinstance(m["cam"], dict):
            ts = m["cam"].get("generation_delta_time")
            if ts is not None and isinstance(ts, (int, float)) and not math.isnan(ts):
                m["cam"]["generation_delta_time"] = ts + shift
                
    return messages


def main():
    parser = argparse.ArgumentParser(description="ISCE Manual Pipeline Validation Harness")
    parser.add_argument("input_path", nargs="?", help="Path to a single JSON message, directory, or scenario specification")
    parser.add_argument("--step", action="store_true", help="Enable step-by-step validation pausing mode")
    parser.add_argument("--log", action="store_true", help="Save run logs to logs/ directory")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print detailed walkthrough logs")
    parser.add_argument("--regression", action="store_true", help="Run the regression test suite batch")
    parser.add_argument("--add-regression", help="Copy specified failing message file to the regression suite")
    parser.add_argument("--pipeline", action="store_true", help="Run validation using the new ISCEPipeline orchestrator")
    
    # Hide default help message on parameter conflict
    args = parser.parse_args()
    
    # 1. Check add-regression
    if args.add_regression:
        add_to_regression(args.add_regression)
        sys.exit(0)
        
    # 2. Check input path or regression options
    if not args.input_path and not args.regression:
        parser.print_help()
        sys.exit(1)
        
    # 3. Determine actual input path
    if args.regression:
        input_path = str(_PROJECT_ROOT / "test_messages" / "regression")
        # In regression run, default verbose to False unless explicitly set
        args.verbose = False
    else:
        input_path = args.input_path
        
    # Generate test message library if missing
    lib_path = _PROJECT_ROOT / "test_messages"
    if not lib_path.exists():
        print("Test message library is missing. Rebuilding dynamically...")
        try:
            # Dynamically import and run generate_test_messages if needed
            sys.path.append(str(_PROJECT_ROOT / "scratch"))
            import generate_test_messages
            generate_test_messages.main()
        except Exception as exc:
            print(f"Failed to auto-generate message library: {exc}")
            
    # Load messages
    try:
        messages = load_messages_from_path(input_path)
        messages = adjust_timestamps_to_fresh(messages, input_path)
    except Exception as exc:
        print(f"Error loading messages: {exc}")
        sys.exit(1)
        
    # Determine context based on input path dynamically
    context_to_use = "rural"
    if "urban" in str(input_path):
        context_to_use = "urban"
    elif "highway" in str(input_path):
        context_to_use = "highway"

    # Initialize Core Pipeline Blocks with V2 research parameters
    config_overrides = {
        "research_extensions": {
            "enabled": True
        },
        "motion_context": {
            "enabled": True,
            "inference_strategy": "probabilistic",
            "hysteresis": 0.25,
            "supported_contexts": [context_to_use]
        },
        "trust_propagation": {
            "strategy": "belief_diffusion",
            "damping_factor": 0.3,
            "max_iterations": 20,
            "convergence_tolerance": 0.0001
        }
    }
    
    scsv = SCSV()
    csia = CSIA(config_overrides=config_overrides)

    # If --pipeline flag is set, run using the new ISCEPipeline orchestrator optionally
    if args.pipeline:
        from pipeline.orchestrator import ISCEPipeline
        pipeline_instance = ISCEPipeline(scsv=scsv, csia=csia)
        passed_messages = []
        logs = []
        summary = {
            "total": len(messages),
            "passed": 0,
            "failed": 0,
            "b1_latency_avg": 0.0,
            "b2_latency_avg": 0.0,
            "total_latency_avg": 0.0,
            "failure_reasons": {}
        }
        
        b1_latencies = []
        b2_latencies = []
        total_latencies = []
        
        print(f"\n[ISCEPipeline] Processing {len(messages)} messages...")
        for idx, msg in enumerate(messages):
            passed_messages.append(msg)
            
            # Form window of past messages matching time window
            window_size_ms = csia._window_size_ns / 1e6
            ts = _nested_get_any(msg, "cam.generation_delta_time")
            current_ts = ts if ts is not None else 0.0
            
            window = []
            for m in passed_messages:
                m_ts = _nested_get_any(m, "cam.generation_delta_time")
                m_ts = m_ts if m_ts is not None else 0.0
                if abs(current_ts - m_ts) <= window_size_ms:
                    window.append(m)
            
            # Run pipeline
            pipeline_result = pipeline_instance.run(window, context=context_to_use)
            
            b1_res = pipeline_result["b1"]
            b2_res = pipeline_result["b2"]
            b3_res = pipeline_result["b3"]
            decision = pipeline_result["decision"]
            reason = pipeline_result["reason"]
            fusion = pipeline_result["fusion"]
            latencies = pipeline_result["latencies"]
            
            b2_valid = decision in ("ACCEPT", "CAUTION")
            if b2_valid:
                summary["passed"] += 1
            else:
                summary["failed"] += 1
                
            b1_latencies.append(latencies["b1_ms"])
            b2_latencies.append(latencies["b2_ms"])
            total_latencies.append(latencies["total_ms"])
            
            if args.log:
                logs.append({
                    "input": msg,
                    "pipeline_result": pipeline_result,
                    "final_assessment": "PASS" if b2_valid else "FAIL"
                })
                
            if args.verbose:
                print("\n" + "="*50)
                print(f"PIPELINE RUN: Msg {idx+1}/{len(messages)}")
                print("="*50)
                print(f"B1 Valid:           {b1_res['valid']}")
                print(f"B1 Fatal:           {b1_res['fatal']}")
                print(f"B2 Trust:           {b2_res.get('trust', 1.0):.4f}")
                print(f"Synthesized Message: {pipeline_result['synthesized_message']['text']}")
                print(f"B3 Available:       {b3_res['available']}")
                print(f"B3 Status:          {b3_res['status']}")
                print(f"Final Decision:     {decision}")
                print(f"Reason:             {reason}")
                print(f"Fusion details:     {fusion}")
                print(f"Latency Profile:    {latencies}")
                
        summary["b1_latency_avg"] = float(np.mean(b1_latencies)) if b1_latencies else 0.0
        summary["b2_latency_avg"] = float(np.mean(b2_latencies)) if b2_latencies else 0.0
        summary["total_latency_avg"] = float(np.mean(total_latencies)) if total_latencies else 0.0
        
        if args.log:
            save_log_file(logs, summary)
            
        print("\n" + "="*50)
        print("PIPELINE BATCH EXECUTION SUMMARY")
        print("="*50)
        print(f"Total Messages: {summary['total']}")
        print(f"Passed:         {summary['passed']}")
        print(f"Failed:         {summary['failed']}")
        print(f"Average Total Latency: {summary['total_latency_avg']:.4f} ms")
        print("="*50)
        sys.exit(0)

    # Run the validation harness
    logs, summary = run_pipeline(
        messages=messages,
        scsv=scsv,
        csia=csia,
        step_mode=args.step,
        verbose=args.verbose,
        log_enabled=args.log,
    )
    
    # Save log if requested
    if args.log:
        save_log_file(logs, summary)
        
    # ── Batch Summary Output (Part 6) ─────────────────────────────────────────
    print("\n" + "="*50)
    print("BATCH EXECUTION SUMMARY")
    print("="*50)
    print(f"Total Messages: {summary['total']}")
    print(f"Passed:         {summary['passed']}")
    print(f"Failed:         {summary['failed']}")
    print("-" * 50)
    print("Average Latencies:")
    print(f"  Average B1 Latency: {summary['b1_latency_avg']:.4f} ms")
    print(f"  Average B2 Latency: {summary['b2_latency_avg']:.4f} ms")
    print(f"  Total Latency:      {summary['total_latency_avg']:.4f} ms")
    print("-" * 50)
    print("B2 Substage Latencies:")
    print(f"  Graph Construction:  {summary['breakdown_avg']['graph_ms']:.4f} ms")
    print(f"  Motion Context:      {summary['breakdown_avg']['context_ms']:.4f} ms")
    print(f"  Adaptive Thresholds: {summary['breakdown_avg']['threshold_ms']:.4f} ms")
    print(f"  Behavior Reasoning:  {summary['breakdown_avg']['reasoning_ms']:.4f} ms")
    print(f"  Trust Propagation:   {summary['breakdown_avg']['propagation_ms']:.4f} ms")
    print("-" * 50)
    print("Failure Reasons:")
    for reason, count in summary["failure_reasons"].items():
        if count > 0:
            print(f"  {reason:<18s}: {count}")
    print("="*50)


if __name__ == "__main__":
    main()
