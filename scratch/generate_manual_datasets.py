import os
import json
import math
import random
import shutil

# Root directory of the project
PROJECT_ROOT = "c:\\isce"
TEST_MESSAGES_ROOT = os.path.join(PROJECT_ROOT, "test_messages", "b2")

# Set random seed for reproducibility
rng = random.Random(42)

def create_cam_message(station_id, station_type, lat, lon, speed, heading, yaw, ts, is_attacker, cert_id=None, msg_type="CAM"):
    msg = {
        "header": {
            "station_id": station_id,
            "message_id": 1
        },
        "cam": {
            "generation_delta_time": round(float(ts), 2),
            "cam_parameters": {
                "basic_container": {
                    "station_type": station_type,
                    "reference_position": {
                        "latitude": int(lat),
                        "longitude": int(lon)
                    }
                },
                "high_frequency_container": {
                    "basic_vehicle_container_high_frequency": {
                        "speed": int(speed),
                        "heading": int(heading),
                        "yaw_rate": int(yaw),
                        "steering_wheel_angle": 0,
                        "lateral_acceleration": 0,
                        "longitudinal_acceleration": 0
                    }
                }
            }
        },
        "is_attacker": is_attacker
    }
    if cert_id is not None:
        msg["certificate_id"] = cert_id
        msg["cert_id"] = cert_id
    if msg_type != "CAM":
        msg["message_type"] = msg_type
    return msg

def save_dataset(name, messages):
    dir_path = os.path.join(TEST_MESSAGES_ROOT, name)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    
    # Sort messages by timestamp to enforce stateful ordering
    messages.sort(key=lambda m: m["cam"]["generation_delta_time"])
    
    for idx, msg in enumerate(messages):
        filename = f"msg_{idx:03d}.json"
        with open(os.path.join(dir_path, filename), "w", encoding="utf-8") as f:
            json.dump(msg, f, indent=2)
    print(f"Generated {len(messages)} messages for {name}")

def generate_all():
    # -------------------------------------------------------------
    # 1. Generate Benign Background Nodes (used across datasets)
    # -------------------------------------------------------------
    def get_benign_messages(count=15, start_ts=1000.0, spacing_ts=2000.0):
        msgs = []
        base_lat = 485512345
        base_lon = 96123456
        for i in range(count):
            station_id = 1001 + i
            # Move along a realistic path with ~30m spacing (3000 E7 units)
            lat = base_lat + i * 3000
            lon = base_lon + i * 3000
            speed = 1100 + rng.randint(-100, 100) # ~11 m/s
            heading = 900 + rng.randint(-50, 50)
            yaw = rng.randint(-20, 20)
            ts = start_ts + i * spacing_ts
            # Unique cert for each benign vehicle
            msgs.append(create_cam_message(station_id, 5, lat, lon, speed, heading, yaw, ts, False, cert_id=station_id))
        return msgs

    # -------------------------------------------------------------
    # 2. Sybil Dataset
    # -------------------------------------------------------------
    # 15 benign nodes + 5 Sybil clones
    sybil_msgs = get_benign_messages(count=15)
    
    # Base parameters for the physical vehicle being spoofed
    base_lat = 485512000
    base_lon = 96123000
    base_speed = 1500
    base_heading = 900
    base_yaw = 0
    base_ts = 40000.0
    
    for a in range(5):
        # 5 fake identities
        station_id = 9001 + a
        
        # Introduce variations:
        # ±1–2 m GPS noise
        mag = rng.uniform(1.0, 2.0)
        sign_x = rng.choice([-1, 1])
        sign_y = rng.choice([-1, 1])
        dx = mag * sign_x
        dy = mag * sign_y
        lat_noise = dy * (1e7 / 111132.9)
        lon_noise = dx * (1e7 / (111319.9 * math.cos(math.radians(base_lat * 1e-7))))
        
        # ±0.1 m/s speed noise (±10 ETSI)
        speed_noise = rng.uniform(-0.1, 0.1) * 100
        
        # ±1° heading noise (±10 ETSI)
        heading_noise = rng.uniform(-1.0, 1.0) * 10
        
        # ±2–5 ms timestamp jitter
        jitter = rng.uniform(2.0, 5.0) * rng.choice([-1, 1])
        
        msg = create_cam_message(
            station_id=station_id,
            station_type=5,
            lat=base_lat + lat_noise,
            lon=base_lon + lon_noise,
            speed=base_speed + speed_noise,
            heading=base_heading + heading_noise,
            yaw=base_yaw,
            ts=base_ts + jitter,
            is_attacker=True,
            cert_id=9999 # Shared certificate ID
        )
        sybil_msgs.append(msg)
        
    save_dataset("sybil", sybil_msgs)

    # -------------------------------------------------------------
    # 3. Replay Dataset
    # -------------------------------------------------------------
    # 15 benign nodes + 5 replay clone messages
    replay_msgs = get_benign_messages(count=15)
    
    # Replay target is benign vehicle 1001 (first message) kinematics
    target_msg = replay_msgs[0]
    target_lat = target_msg["cam"]["cam_parameters"]["basic_container"]["reference_position"]["latitude"]
    target_lon = target_msg["cam"]["cam_parameters"]["basic_container"]["reference_position"]["longitude"]
    target_speed = target_msg["cam"]["cam_parameters"]["high_frequency_container"]["basic_vehicle_container_high_frequency"]["speed"]
    target_heading = target_msg["cam"]["cam_parameters"]["high_frequency_container"]["basic_vehicle_container_high_frequency"]["heading"]
    target_yaw = target_msg["cam"]["cam_parameters"]["high_frequency_container"]["basic_vehicle_container_high_frequency"]["yaw_rate"]
    
    for a in range(5):
        station_id = 8001 + a
        # Slight delays: 25ms spacing starting from base_ts
        ts = 40000.0 + a * 25.0
        msg = create_cam_message(
            station_id=station_id,
            station_type=5,
            lat=target_lat,
            lon=target_lon,
            speed=target_speed,
            heading=target_heading,
            yaw=target_yaw,
            ts=ts,
            is_attacker=True,
            cert_id=8888 # Shared certificate
        )
        replay_msgs.append(msg)
        
    save_dataset("replay", replay_msgs)

    # -------------------------------------------------------------
    # 4. Collusion Dataset
    # -------------------------------------------------------------
    # 15 benign nodes + 5 colluding nodes
    collusion_msgs = get_benign_messages(count=15)
    
    base_lat = 485512000
    base_lon = 96123000
    
    for a in range(5):
        station_id = 7001 + a
        # 1ms intervals
        ts = 40000.0 + a * 1.0
        msg = create_cam_message(
            station_id=station_id,
            station_type=5,
            lat=base_lat + a * 10,
            lon=base_lon + a * 10,
            speed=1500,
            heading=900,
            yaw=0,
            ts=ts,
            is_attacker=True,
            cert_id=station_id, # Unique certificates
        )
        collusion_msgs.append(msg)
        
    save_dataset("collusion", collusion_msgs)

    # -------------------------------------------------------------
    # 5. Fabrication Dataset
    # -------------------------------------------------------------
    # 15 benign nodes + 5 fabricating nodes
    fabrication_msgs = get_benign_messages(count=15)
    
    base_lat = 485512000
    base_lon = 96123000
    
    for a in range(5):
        station_id = 6001 + a
        # 10ms intervals, larger spatial offsets (far away)
        ts = 40000.0 + a * 200.0
        msg = create_cam_message(
            station_id=station_id,
            station_type=5,
            lat=base_lat + a * 200,
            lon=base_lon + a * 200,
            speed=1500,
            heading=900,
            yaw=0,
            ts=ts,
            is_attacker=True,
            cert_id=station_id, # Unique certificates
            msg_type="DENM"
        )
        fabrication_msgs.append(msg)
        
    save_dataset("fabrication", fabrication_msgs)

    # -------------------------------------------------------------
    # 6. Mixed Dataset
    # -------------------------------------------------------------
    # Reuse the Sybil setup as it represents a mix of normal traffic and attackers
    save_dataset("mixed", sybil_msgs)

if __name__ == "__main__":
    generate_all()
