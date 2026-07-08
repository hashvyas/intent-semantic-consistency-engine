"""
scenario_generation/generator.py
================================
Framework for generating deterministic, parameterized, held-out evaluation scenarios
with realistic distribution shifts.
"""

from __future__ import annotations

import os
import json
import math
import shutil
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ScenarioConfig:
    """Configuration parameter set for a single evaluation scenario.

    Supports generating diverse distribution shifts across multiple dimensions.
    """
    scenario_id: str
    scenario_family: str      # "benign" | "replay" | "sybil" | "collusion" | "fabrication" | "manipulation"
    attack_type: str          # "none" | "replay" | "sybil" | "collusion" | "fabrication" | "position_manipulation" | "speed_manipulation" | "certificate_switching"
    traffic_density: str      # "sparse" | "moderate" | "dense"
    road_context: str         # "urban" | "highway" | "rural"
    vehicle_count: int
    attacker_count: int
    seed: int
    expected_label: str       # "BENIGN" | "MALICIOUS"
    message_count: int = 20   # total messages in the scenario sequence


class HeldOutScenarioGenerator:
    """Generates physically plausible, deterministic, schema-compatible V2X evaluation scenarios

    incorporating realistic distribution shifts.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def _get_context_parameters(self, context: str) -> Dict[str, Any]:
        """Return base physical speed and path variation bounds for a context.

        Parameters
        ----------
        context : str
            "urban", "highway", or "rural".
        """
        if context == "highway":
            return {
                "base_speed_mps": 28.0,      # ~100 km/h
                "speed_jitter_mps": 2.0,
                "base_heading_deg10": 1800,  # Heading South-ish (180 deg)
                "heading_jitter_deg10": 20,
                "yaw_rate_jitter": 5,        # stable yaw
                "spacing_m": 60.0,
            }
        elif context == "rural":
            return {
                "base_speed_mps": 20.0,      # ~72 km/h
                "speed_jitter_mps": 3.0,
                "base_heading_deg10": 2700,  # Heading West-ish (270 deg)
                "heading_jitter_deg10": 50,
                "yaw_rate_jitter": 20,
                "spacing_m": 40.0,
            }
        else:  # "urban"
            return {
                "base_speed_mps": 11.0,      # ~40 km/h
                "speed_jitter_mps": 2.5,
                "base_heading_deg10": 900,   # Heading East-ish (90 deg)
                "heading_jitter_deg10": 100,
                "yaw_rate_jitter": 50,
                "spacing_m": 20.0,
            }

    def _get_vehicle_types(self, density: str) -> List[int]:
        """Return ETSI station_type codes distributed for composition shifts.

        Compositions:
          - 5: passengerCar (majority)
          - 8: heavyTruck
          - 4: motorcycle
          - 15: roadSideUnit
        """
        # Seeded selection to maintain deterministic layout
        types = [5] * 10
        if density == "dense":
            types.extend([8, 8, 4, 4, 15, 15])
        elif density == "moderate":
            types.extend([8, 4, 15])
        else:  # sparse
            types.extend([8, 15])
        return types

    def generate_scenario(self, config: ScenarioConfig) -> List[Dict[str, Any]]:
        """Generate a sequential list of V2X message dictionaries matching config."""
        self.rng.seed(config.seed)

        # Baseline physical anchor (ETSI 1e-7 degree units)
        # Lat/Lon for central Germany / central Europe area
        base_lat = 485512345
        base_lon = 96123456

        ctx_params = self._get_context_parameters(config.road_context)
        composition_types = self._get_vehicle_types(config.traffic_density)

        # Spacing conversions from meters to E7 coordinates
        lat_per_m = 1e7 / 111132.9
        lon_per_m = 1e7 / (111319.9 * math.cos(math.radians(base_lat * 1e-7)))

        # Define time window and update rate
        # dense -> 10Hz (100ms updates), moderate -> 5Hz (200ms), sparse -> 2Hz (500ms)
        if config.traffic_density == "dense":
            hz = 10.0
        elif config.traffic_density == "moderate":
            hz = 5.0
        else:
            hz = 2.0
        ts_step_ms = int(1000.0 / hz)

        # Allocate vehicle structures
        vehicles: List[Dict[str, Any]] = []
        for i in range(config.vehicle_count):
            station_id = 1001 + i
            st_type = self.rng.choice(composition_types)

            # Distribute starting position along a path
            spacing = ctx_params["spacing_m"]
            lat_offset = i * spacing * lat_per_m
            lon_offset = i * spacing * lon_per_m

            # Base speed, heading, yaw
            speed_mps = ctx_params["base_speed_mps"] + self.rng.uniform(
                -ctx_params["speed_jitter_mps"], ctx_params["speed_jitter_mps"]
            )
            heading_deg10 = ctx_params["base_heading_deg10"] + self.rng.randint(
                -ctx_params["heading_jitter_deg10"], ctx_params["heading_jitter_deg10"]
            )
            yaw_rate = self.rng.randint(-ctx_params["yaw_rate_jitter"], ctx_params["yaw_rate_jitter"])

            # Zero kinematics for RSUs
            if st_type == 15:
                speed_mps = 0.0
                heading_deg10 = 0
                yaw_rate = 0

            vehicles.append({
                "station_id": station_id,
                "station_type": st_type,
                "lat": int(base_lat + lat_offset),
                "lon": int(base_lon + lon_offset),
                "speed_mps": speed_mps,
                "heading_deg10": heading_deg10 % 3600,
                "yaw_rate": yaw_rate,
                "is_attacker": False,
                "certificate_id": f"CERT_{st_type}_{station_id}",
                "msg_type": "CAM",
            })

        # Identify attacker indices if configuration requires attackers
        attacker_indices = []
        if config.attacker_count > 0:
            # Reserve vehicles as attackers (prefer non-RSU vehicles)
            potential = [idx for idx, v in enumerate(vehicles) if v["station_type"] != 15]
            if not potential:
                potential = list(range(len(vehicles)))
            attacker_indices = self.rng.sample(potential, min(config.attacker_count, len(potential)))
            for idx in attacker_indices:
                vehicles[idx]["is_attacker"] = True

        messages: List[Dict[str, Any]] = []
        base_ts_ms = 4000.0 + self.rng.randint(100, 100000)

        # Track rolling updates per vehicle to compute smooth trajectories
        # and support stateful SCSV kinematics tracking
        vehicle_states = [dict(v) for v in vehicles]

        for step in range(config.message_count):
            # Each step, each active vehicle broadcasts
            # To add variation, some packets may arrive slightly delayed or be dropped
            for idx, state in enumerate(vehicle_states):
                # Update position based on kinematics if not RSU
                if state["station_type"] != 15:
                    # Trajectory calculation: update lat/lon using heading and speed
                    h_rad = math.radians(state["heading_deg10"] * 0.1)
                    dist_moved = state["speed_mps"] * (ts_step_ms / 1000.0)
                    
                    dx = dist_moved * math.sin(h_rad)
                    dy = dist_moved * math.cos(h_rad)
                    
                    state["lat"] += int(dy * lat_per_m)
                    state["lon"] += int(dx * lon_per_m)

                    # Dynamic smooth changes to simulate real movement (motion variation)
                    # speed sinusoid variation
                    state["speed_mps"] += self.rng.uniform(-0.3, 0.3)
                    # Yaw rate updates heading
                    state["heading_deg10"] = int(state["heading_deg10"] + state["yaw_rate"] * (ts_step_ms / 1000.0)) % 3600
                    state["yaw_rate"] += self.rng.randint(-5, 5)

                # Base timestamp for this transmission
                ts_ms = base_ts_ms + step * ts_step_ms
                # Add tiny communication jitter (temporal variation)
                ts_ms += self.rng.uniform(-2.0, 2.0)

                # Construct default benign CAM message
                msg = self._build_raw_message(state, ts_ms, expected_label=config.expected_label)

                # ── Apply Attack Logic if this node is a configured attacker ──
                if state["is_attacker"]:
                    attack = config.attack_type.lower().strip()

                    if attack == "replay":
                        # Delay target's kinematics from previous messages
                        # Look back in generated messages from a benign vehicle
                        benign_msgs = [m for m in messages if not m.get("is_attacker", False)]
                        if benign_msgs:
                            target_ref = self.rng.choice(benign_msgs)
                            # Clone target's kinematics
                            ref_cam = target_ref["cam"]["cam_parameters"]
                            msg["cam"]["cam_parameters"]["basic_container"]["reference_position"] = dict(
                                ref_cam["basic_container"]["reference_position"]
                            )
                            msg["cam"]["cam_parameters"]["high_frequency_container"]["basic_vehicle_container_high_frequency"] = dict(
                                ref_cam["high_frequency_container"]["basic_vehicle_container_high_frequency"]
                            )
                        # Retain replayer's ID but timestamp is replayed/jittered
                        msg["is_attacker"] = True
                        msg["expected_label"] = "MALICIOUS"

                    elif attack == "sybil":
                        # Shared certificate under multiple station IDs
                        # Broadcast cloned observations under multiple random IDs
                        msg["certificate_id"] = "CERT_SYBIL_SHARED"
                        msg["cert_id"] = "CERT_SYBIL_SHARED"
                        msg["is_attacker"] = True
                        msg["expected_label"] = "MALICIOUS"

                    elif attack == "collusion":
                        # Coordinate close trajectories to fake identical footprints
                        # Modify position/speed to match attacker 0's footprint
                        primary_attacker_idx = attacker_indices[0]
                        primary_state = vehicle_states[primary_attacker_idx]
                        
                        msg["cam"]["cam_parameters"]["high_frequency_container"]["basic_vehicle_container_high_frequency"]["speed"] = int(
                            primary_state["speed_mps"] * 100
                        )
                        msg["cam"]["cam_parameters"]["high_frequency_container"]["basic_vehicle_container_high_frequency"]["heading"] = int(
                            primary_state["heading_deg10"]
                        )
                        # Overlap coordinates (very close)
                        msg["cam"]["cam_parameters"]["basic_container"]["reference_position"]["latitude"] = int(
                            primary_state["lat"] + self.rng.randint(-5, 5)
                        )
                        msg["cam"]["cam_parameters"]["basic_container"]["reference_position"]["longitude"] = int(
                            primary_state["lon"] + self.rng.randint(-5, 5)
                        )
                        msg["is_attacker"] = True
                        msg["expected_label"] = "MALICIOUS"

                    elif attack == "fabrication":
                        # Isolated warning alert (DENM) from far away
                        msg["message_type"] = "DENM"
                        msg["cam"]["cam_parameters"]["basic_container"]["reference_position"]["latitude"] = int(
                            state["lat"] + 50000 * lat_per_m # 50km away
                        )
                        msg["cam"]["cam_parameters"]["basic_container"]["reference_position"]["longitude"] = int(
                            state["lon"] + 50000 * lon_per_m
                        )
                        # Add a warning cause directly inside the DENM representation
                        msg["scene_context"] = {
                            "peer_reports": [],
                            "rsu_messages": []
                        }
                        # Add raw warning trigger details
                        msg["event"] = "obstacle"
                        msg["is_attacker"] = True
                        msg["expected_label"] = "MALICIOUS"

                    elif attack == "position_manipulation":
                        # GPS Spoofing / position jump
                        # Add huge coordinate offset that violates physics constraints
                        msg["cam"]["cam_parameters"]["basic_container"]["reference_position"]["latitude"] = int(
                            state["lat"] + 1500 * lat_per_m # abrupt 1.5 km jump
                        )
                        msg["is_attacker"] = True
                        msg["expected_label"] = "MALICIOUS"

                    elif attack == "speed_manipulation":
                        # Impossible speed reading (exceeds ETSI limits)
                        msg["cam"]["cam_parameters"]["high_frequency_container"]["basic_vehicle_container_high_frequency"]["speed"] = 99999
                        msg["is_attacker"] = True
                        msg["expected_label"] = "MALICIOUS"

                    elif attack == "certificate_switching":
                        # Rotate certificate rapidly on every transmission
                        msg["certificate_id"] = f"CERT_ROT_TEMP_{step}"
                        msg["cert_id"] = f"CERT_ROT_TEMP_{step}"
                        msg["is_attacker"] = True
                        msg["expected_label"] = "MALICIOUS"

                messages.append(msg)

        # Enforce chronological ordering
        messages.sort(key=lambda m: m["cam"]["generation_delta_time"])
        return messages

    def _build_raw_message(self, state: Dict[str, Any], ts_ms: float, expected_label: str) -> Dict[str, Any]:
        """Map vehicle state properties into a standard V2X JSON dictionary format."""
        msg = {
            "header": {
                "station_id": state["station_id"],
                "message_id": 1 if state["msg_type"] == "CAM" else 2,
            },
            "cam": {
                "generation_delta_time": round(ts_ms, 2),
                "cam_parameters": {
                    "basic_container": {
                        "station_type": state["station_type"],
                        "reference_position": {
                            "latitude": state["lat"],
                            "longitude": state["lon"],
                        }
                    },
                    "high_frequency_container": {
                        "basic_vehicle_container_high_frequency": {
                            "speed": int(state["speed_mps"] * 100),
                            "heading": int(state["heading_deg10"]),
                            "yaw_rate": int(state["yaw_rate"]),
                            "steering_wheel_angle": 0,
                            "lateral_acceleration": 0,
                            "longitudinal_acceleration": 0,
                        }
                    }
                }
            },
            "certificate_id": state["certificate_id"],
            "cert_id": state["certificate_id"],
            "is_attacker": state["is_attacker"],
            "expected_label": expected_label,
        }

        # Add empty local perception / context wrappers for B2 parser stability
        msg["local_perception"] = {
            "camera": "CLEAR",
            "radar": "CLEAR",
            "lidar": "CLEAR"
        }
        msg["scene_context"] = {
            "peer_reports": [],
            "rsu_messages": []
        }

        # Add RSU corroborated hazards if an RSU is present
        if state["station_type"] == 15:
            msg["cam"]["cam_parameters"]["high_frequency_container"]["basic_vehicle_container_high_frequency"]["speed"] = 0
            msg["cam"]["cam_parameters"]["high_frequency_container"]["basic_vehicle_container_high_frequency"]["heading"] = 0
            msg["cam"]["cam_parameters"]["high_frequency_container"]["basic_vehicle_container_high_frequency"]["yaw_rate"] = 0

        return msg


# ---------------------------------------------------------------------------
# Held-Out Scenario Generation Orchestrator
# ---------------------------------------------------------------------------

def generate_held_out_suite(
    output_dir: str,
    seed: int = 42,
    configs_limit: Optional[int] = None,
    message_count_override: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate a suite of 30 distinct scenarios spanning all 6 families.

    Writes to separate subfolders in output_dir, and writes a master metadata.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    generator = HeldOutScenarioGenerator(seed)

    configs = [
        # --- 1. Benign Scenarios (no attackers) ---
        ScenarioConfig("benign_sparse_urban", "benign", "none", "sparse", "urban", 4, 0, seed + 1, "BENIGN"),
        ScenarioConfig("benign_mod_rural", "benign", "none", "moderate", "rural", 12, 0, seed + 2, "BENIGN"),
        ScenarioConfig("benign_dense_highway", "benign", "none", "dense", "highway", 35, 0, seed + 3, "BENIGN"),
        ScenarioConfig("benign_dense_urban", "benign", "none", "dense", "urban", 32, 0, seed + 4, "BENIGN"),
        ScenarioConfig("benign_sparse_highway", "benign", "none", "sparse", "highway", 5, 0, seed + 5, "BENIGN"),

        # --- 2. Replay Scenarios ---
        ScenarioConfig("replay_sparse_urban", "replay", "replay", "sparse", "urban", 5, 1, seed + 6, "MALICIOUS"),
        ScenarioConfig("replay_mod_rural", "replay", "replay", "moderate", "rural", 14, 2, seed + 7, "MALICIOUS"),
        ScenarioConfig("replay_dense_highway", "replay", "replay", "dense", "highway", 30, 4, seed + 8, "MALICIOUS"),
        ScenarioConfig("replay_dense_urban", "replay", "replay", "dense", "urban", 32, 3, seed + 9, "MALICIOUS"),
        ScenarioConfig("replay_mod_highway", "replay", "replay", "moderate", "highway", 12, 2, seed + 10, "MALICIOUS"),

        # --- 3. Sybil Scenarios ---
        ScenarioConfig("sybil_sparse_rural", "sybil", "sybil", "sparse", "rural", 4, 1, seed + 11, "MALICIOUS"),
        ScenarioConfig("sybil_mod_urban", "sybil", "sybil", "moderate", "urban", 12, 2, seed + 12, "MALICIOUS"),
        ScenarioConfig("sybil_dense_highway", "sybil", "sybil", "dense", "highway", 35, 5, seed + 13, "MALICIOUS"),
        ScenarioConfig("sybil_dense_urban", "sybil", "sybil", "dense", "urban", 30, 4, seed + 14, "MALICIOUS"),
        ScenarioConfig("sybil_mod_rural", "sybil", "sybil", "moderate", "rural", 14, 2, seed + 15, "MALICIOUS"),

        # --- 4. Collusion Scenarios ---
        ScenarioConfig("collusion_sparse_highway", "collusion", "collusion", "sparse", "highway", 5, 2, seed + 16, "MALICIOUS"),
        ScenarioConfig("collusion_mod_urban", "collusion", "collusion", "moderate", "urban", 12, 3, seed + 17, "MALICIOUS"),
        ScenarioConfig("collusion_dense_rural", "collusion", "collusion", "dense", "rural", 32, 4, seed + 18, "MALICIOUS"),
        ScenarioConfig("collusion_dense_highway", "collusion", "collusion", "dense", "highway", 30, 5, seed + 19, "MALICIOUS"),
        ScenarioConfig("collusion_mod_rural", "collusion", "collusion", "moderate", "rural", 14, 3, seed + 20, "MALICIOUS"),

        # --- 5. Fabrication Scenarios ---
        ScenarioConfig("fabrication_sparse_urban", "fabrication", "fabrication", "sparse", "urban", 4, 1, seed + 21, "MALICIOUS"),
        ScenarioConfig("fabrication_mod_rural", "fabrication", "fabrication", "moderate", "rural", 12, 2, seed + 22, "MALICIOUS"),
        ScenarioConfig("fabrication_dense_highway", "fabrication", "fabrication", "dense", "highway", 35, 4, seed + 23, "MALICIOUS"),
        ScenarioConfig("fabrication_dense_urban", "fabrication", "fabrication", "dense", "urban", 30, 3, seed + 24, "MALICIOUS"),
        ScenarioConfig("fabrication_mod_highway", "fabrication", "fabrication", "moderate", "highway", 14, 2, seed + 25, "MALICIOUS"),

        # --- 6. Manipulation Scenarios ---
        ScenarioConfig("position_manip_sparse_highway", "manipulation", "position_manipulation", "sparse", "highway", 4, 1, seed + 26, "MALICIOUS"),
        ScenarioConfig("speed_manip_mod_urban", "manipulation", "speed_manipulation", "moderate", "urban", 12, 2, seed + 27, "MALICIOUS"),
        ScenarioConfig("cert_switch_dense_rural", "manipulation", "certificate_switching", "dense", "rural", 35, 4, seed + 28, "MALICIOUS"),
        ScenarioConfig("position_manip_dense_urban", "manipulation", "position_manipulation", "dense", "urban", 30, 3, seed + 29, "MALICIOUS"),
        ScenarioConfig("cert_switch_mod_highway", "manipulation", "certificate_switching", "moderate", "highway", 14, 2, seed + 30, "MALICIOUS"),
    ]

    if configs_limit is not None:
        configs = configs[:configs_limit]

    suite_metadata = []

    for cfg in configs:
        if message_count_override is not None:
            cfg.message_count = message_count_override

        scenario_dir = os.path.join(output_dir, cfg.scenario_id)
        if os.path.exists(scenario_dir):
            shutil.rmtree(scenario_dir)
        os.makedirs(scenario_dir, exist_ok=True)

        messages = generator.generate_scenario(cfg)

        # Write messages as sorted files inside the scenario directory
        for idx, msg in enumerate(messages):
            filename = f"msg_{idx:03d}.json"
            with open(os.path.join(scenario_dir, filename), "w", encoding="utf-8") as f:
                json.dump(msg, f, indent=2)

        # Save individual metadata.json in the scenario folder
        meta = asdict(cfg)
        # Add generated counts
        meta["generated_message_count"] = len(messages)
        meta["generated_attacker_messages"] = sum(1 for m in messages if m.get("is_attacker"))
        
        with open(os.path.join(scenario_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        suite_metadata.append(meta)

    # Save master metadata.json
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(suite_metadata, f, indent=2)

    return suite_metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Held-Out Scenario Generator for B5")
    parser.add_argument("--output-dir", default="test_messages/held_out", help="Output directory path")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_parser_args() if hasattr(parser, "parse_parser_args") else parser.parse_args()
    generate_held_out_suite(args.output_dir, args.seed)
    print(f"Generated held-out scenario suite at {args.output_dir}")
