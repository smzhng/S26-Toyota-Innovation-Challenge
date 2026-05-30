"""
vision.py — Black pipe detection for the maze mission.

How it works:
  1. Captures frames from a camera (USB webcam or Pi camera)
  2. Shows live HSV tuning sliders so you can isolate the black pipe color on the day
  3. When a large enough blob matching the color is detected for N consecutive frames,
     it calculates the pipe's arena coordinates using the robot's current odometry
  4. Sends {"type": "target_located", "x_cm": ..., "y_cm": ...} to the arbiter over TCP
  5. The arbiter plans a maze-avoiding path and sends robot_A to the pipe

Run on your laptop or Pi:
    python vision.py --env ../robot_a.env

Controls:
    q — quit
    s — save current HSV values to hsv_config.json
    l — load HSV values from hsv_config.json
    t — manually trigger target_located with current detection (for testing)
    r — reset sent flag so detection fires again
"""

import argparse
import json
import math
import os
import socket
import threading
import time

import cv2
import numpy as np
from dotenv import load_dotenv

# ----------------------------
# Configuration
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="../robot_a.env", help="Path to robot env file")
parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
parser.add_argument("--robot-id", default="robot_A", help="Which robot this camera is on")
parser.add_argument("--no-send", action="store_true", help="Detect only, don't send to arbiter")
args = parser.parse_args()

load_dotenv(args.env)

SERVER_HOST = os.getenv("SERVER_HOST_IP_ADDRESS", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "9000"))
ROBOT_ID = args.robot_id

# How many consecutive frames we must see the blob before firing target_located
CONFIRM_FRAMES = 8

# Minimum blob area in pixels to count as a real detection (filters noise)
MIN_BLOB_AREA = 2000

# Camera field of view — used to convert pixel offset to angle
# Astra Pro Plus: ~60 degrees horizontal FOV
CAMERA_HFOV_DEG = 60.0

# Estimated camera height above the ground in cm
# Tune this to match where the camera is mounted on your robot
CAMERA_HEIGHT_CM = 20.0

# Config file for saving/loading tuned HSV values
HSV_CONFIG_FILE = "hsv_config.json"

# Default HSV range — will be overwritten by sliders or loaded config
# These defaults target a black pipe: low saturation, low value.
# Fine-tune with the sliders on the day (press 's' to save).
DEFAULT_HSV = {
    "h_min": 0,  "h_max": 180,
    "s_min": 0,  "s_max": 80,
    "v_min": 0,  "v_max": 60,
}

# ----------------------------
# Arbiter communication
# ----------------------------
arbiter_sock = None
arbiter_lock = threading.Lock()
target_sent = False   # Only send once per run unless reset


def connect_to_arbiter() -> bool:
    global arbiter_sock
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect((SERVER_HOST, SERVER_PORT))
        s.settimeout(None)

        # Register as a vision client
        hello = json.dumps({
            "type": "hello",
            "robot_id": ROBOT_ID,
            "client_name": f"{ROBOT_ID}_vision",
            "state": "vision_ready",
        }) + "\n"
        s.sendall(hello.encode("utf-8"))

        with arbiter_lock:
            arbiter_sock = s

        print(f"[VISION] Connected to arbiter at {SERVER_HOST}:{SERVER_PORT}")
        return True

    except Exception as e:
        print(f"[VISION] Could not connect to arbiter: {e}")
        print("[VISION] Running in detection-only mode (--no-send)")
        return False


def send_target_located(x_cm: float, y_cm: float) -> None:
    global target_sent

    if target_sent:
        return

    msg = json.dumps({
        "type": "target_located",
        "x_cm": round(x_cm, 1),
        "y_cm": round(y_cm, 1),
        "detected_by": ROBOT_ID,
        "t_ms": int(time.time() * 1000),
    }) + "\n"

    with arbiter_lock:
        if arbiter_sock is not None:
            try:
                arbiter_sock.sendall(msg.encode("utf-8"))
                target_sent = True
                print(f"[VISION] Sent target_located: ({x_cm:.1f}, {y_cm:.1f}) cm")
            except Exception as e:
                print(f"[VISION] Failed to send target_located: {e}")
        else:
            print(f"[VISION] (no-send mode) Would send target_located: ({x_cm:.1f}, {y_cm:.1f}) cm")
            target_sent = True


# ----------------------------
# Telemetry reader
# ----------------------------
# Reads the robot's latest pose from the arbiter so we can convert
# camera-relative coordinates into arena coordinates.
latest_pose = {"x_cm": 200.0, "y_cm": 200.0, "theta_deg": 90.0}
pose_lock = threading.Lock()


def telemetry_listener() -> None:
    """Background thread: reads telemetry updates from the arbiter."""
    buffer = ""
    while True:
        try:
            with arbiter_lock:
                sock = arbiter_sock
            if sock is None:
                time.sleep(0.5)
                continue

            data = sock.recv(4096)
            if not data:
                break

            buffer += data.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    if msg.get("type") == "telemetry" and msg.get("robot_id") == ROBOT_ID:
                        with pose_lock:
                            latest_pose["x_cm"] = float(msg.get("x_cm", latest_pose["x_cm"]))
                            latest_pose["y_cm"] = float(msg.get("y_cm", latest_pose["y_cm"]))
                            latest_pose["theta_deg"] = float(msg.get("theta_deg", latest_pose["theta_deg"]))
                except Exception:
                    pass

        except Exception as e:
            print(f"[VISION] Telemetry listener error: {e}")
            time.sleep(1.0)


# ----------------------------
# Coordinate conversion
# ----------------------------
def camera_to_arena(
    pixel_x: int,
    pixel_y: int,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float]:
    """
    Convert a detected blob's pixel position to arena coordinates in cm.

    The camera is mounted on the robot facing forward.
    - Horizontal pixel offset from centre → horizontal angle → lateral offset at distance
    - Vertical pixel position → tilt angle → forward distance via trig

    Returns (arena_x_cm, arena_y_cm).
    """
    with pose_lock:
        robot_x = latest_pose["x_cm"]
        robot_y = latest_pose["y_cm"]
        robot_theta_deg = latest_pose["theta_deg"]

    # Horizontal angle from camera centre
    h_fov_rad = math.radians(CAMERA_HFOV_DEG)
    pixel_offset_x = pixel_x - frame_w / 2.0
    angle_h_rad = (pixel_offset_x / (frame_w / 2.0)) * (h_fov_rad / 2.0)

    # Vertical pixel position gives elevation angle, which gives forward distance
    # Using pinhole model: tan(elevation) = camera_height / forward_distance
    # Assume vertical FOV ≈ HFOV * (frame_h / frame_w)
    v_fov_rad = h_fov_rad * (frame_h / frame_w)
    pixel_offset_y = pixel_y - frame_h / 2.0
    # Pixels below centre = looking down = closer object
    elevation_rad = -(pixel_offset_y / (frame_h / 2.0)) * (v_fov_rad / 2.0)

    if abs(elevation_rad) < 0.01:
        # Object near horizon — cap at max range
        forward_dist_cm = 300.0
    else:
        forward_dist_cm = CAMERA_HEIGHT_CM / math.tan(elevation_rad)
        forward_dist_cm = max(5.0, min(400.0, forward_dist_cm))

    # Lateral offset at that distance
    lateral_dist_cm = forward_dist_cm * math.tan(angle_h_rad)

    # Transform from robot frame to arena frame
    theta_rad = math.radians(robot_theta_deg)

    # Robot frame: forward = +X_robot, left = +Y_robot
    # Arena frame: rotated by robot heading
    arena_x = robot_x + forward_dist_cm * math.cos(theta_rad) - lateral_dist_cm * math.sin(theta_rad)
    arena_y = robot_y + forward_dist_cm * math.sin(theta_rad) + lateral_dist_cm * math.cos(theta_rad)

    # Clamp to arena bounds (400 cm x 400 cm)
    arena_x = max(5.0, min(395.0, arena_x))
    arena_y = max(5.0, min(395.0, arena_y))

    return arena_x, arena_y


# ----------------------------
# HSV config save / load
# ----------------------------
def save_hsv(values: dict) -> None:
    with open(HSV_CONFIG_FILE, "w") as f:
        json.dump(values, f, indent=2)
    print(f"[VISION] HSV config saved to {HSV_CONFIG_FILE}")


def load_hsv() -> dict | None:
    if not os.path.exists(HSV_CONFIG_FILE):
        return None
    try:
        with open(HSV_CONFIG_FILE) as f:
            values = json.load(f)
        print(f"[VISION] Loaded HSV config from {HSV_CONFIG_FILE}: {values}")
        return values
    except Exception as e:
        print(f"[VISION] Could not load HSV config: {e}")
        return None


# ----------------------------
# Main detection loop
# ----------------------------
def main() -> None:
    global target_sent

    # Connect to arbiter (non-blocking — detection works even without connection)
    if not args.no_send:
        connected = connect_to_arbiter()
        if connected:
            threading.Thread(target=telemetry_listener, daemon=True).start()
    else:
        print("[VISION] --no-send mode: detection only, nothing will be sent to arbiter")

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[VISION] Could not open camera {args.camera}")
        return

    ret, frame = cap.read()
    if not ret:
        print("[VISION] Could not read from camera")
        cap.release()
        return

    frame_h, frame_w = frame.shape[:2]
    print(f"[VISION] Camera opened: {frame_w}x{frame_h}")

    # Create tuning window with trackbars
    cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Tuner", 600, 300)

    # Load saved config or use defaults
    hsv = load_hsv() or DEFAULT_HSV.copy()

    cv2.createTrackbar("H min", "HSV Tuner", hsv["h_min"], 180, lambda x: None)
    cv2.createTrackbar("H max", "HSV Tuner", hsv["h_max"], 180, lambda x: None)
    cv2.createTrackbar("S min", "HSV Tuner", hsv["s_min"], 255, lambda x: None)
    cv2.createTrackbar("S max", "HSV Tuner", hsv["s_max"], 255, lambda x: None)
    cv2.createTrackbar("V min", "HSV Tuner", hsv["v_min"], 255, lambda x: None)
    cv2.createTrackbar("V max", "HSV Tuner", hsv["v_max"], 255, lambda x: None)

    confirm_count = 0
    last_arena_pos = None

    print("\n[VISION] Controls: q=quit  s=save HSV  l=load HSV  t=manual trigger  r=reset sent flag")
    print("[VISION] Tune the sliders until only the black pipe shows as white in the mask window\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[VISION] Lost camera feed")
            break

        # Read current slider values
        h_min = cv2.getTrackbarPos("H min", "HSV Tuner")
        h_max = cv2.getTrackbarPos("H max", "HSV Tuner")
        s_min = cv2.getTrackbarPos("S min", "HSV Tuner")
        s_max = cv2.getTrackbarPos("S max", "HSV Tuner")
        v_min = cv2.getTrackbarPos("V min", "HSV Tuner")
        v_max = cv2.getTrackbarPos("V max", "HSV Tuner")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Convert to HSV and apply mask
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower, upper)

        # Morphological cleanup — removes small noise blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        display = frame.copy()
        detected = False
        best_cx, best_cy, best_area = 0, 0, 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_BLOB_AREA:
                continue

            # Bounding box and centroid
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Keep largest blob only
            if area > best_area:
                best_area = area
                best_cx, best_cy = cx, cy
                detected = True

            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(display, [cnt], -1, (0, 200, 0), 1)

        if detected:
            arena_x, arena_y = camera_to_arena(best_cx, best_cy, frame_w, frame_h)
            last_arena_pos = (arena_x, arena_y)
            confirm_count += 1

            # Draw centroid and info
            cv2.circle(display, (best_cx, best_cy), 8, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"Arena: ({arena_x:.0f}, {arena_y:.0f}) cm  confirm: {confirm_count}/{CONFIRM_FRAMES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

            if confirm_count >= CONFIRM_FRAMES and not target_sent:
                print(f"[VISION] Pipe confirmed at arena ({arena_x:.1f}, {arena_y:.1f}) cm")
                send_target_located(arena_x, arena_y)
                cv2.putText(
                    display, "PIPE LOCATED — SENT",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                )
        else:
            confirm_count = max(0, confirm_count - 1)

        # Status overlay
        sent_color = (0, 255, 0) if target_sent else (0, 165, 255)
        sent_text = "PIPE SENT — ROBOT NAVIGATING" if target_sent else "searching for black pipe..."
        cv2.putText(display, sent_text, (10, frame_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, sent_color, 2)

        cv2.putText(display, f"blob area: {best_area:.0f}  min: {MIN_BLOB_AREA}",
                    (10, frame_h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Detection", display)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            save_hsv({
                "h_min": h_min, "h_max": h_max,
                "s_min": s_min, "s_max": s_max,
                "v_min": v_min, "v_max": v_max,
            })

        elif key == ord("l"):
            loaded = load_hsv()
            if loaded:
                cv2.setTrackbarPos("H min", "HSV Tuner", loaded["h_min"])
                cv2.setTrackbarPos("H max", "HSV Tuner", loaded["h_max"])
                cv2.setTrackbarPos("S min", "HSV Tuner", loaded["s_min"])
                cv2.setTrackbarPos("S max", "HSV Tuner", loaded["s_max"])
                cv2.setTrackbarPos("V min", "HSV Tuner", loaded["v_min"])
                cv2.setTrackbarPos("V max", "HSV Tuner", loaded["v_max"])

        elif key == ord("t"):
            if last_arena_pos:
                print(f"[VISION] Manual trigger at ({last_arena_pos[0]:.1f}, {last_arena_pos[1]:.1f}) cm")
                target_sent = False   # allow re-send on manual trigger
                send_target_located(*last_arena_pos)
            else:
                print("[VISION] No detection yet — point camera at target first")

        elif key == ord("r"):
            target_sent = False
            confirm_count = 0
            print("[VISION] Reset — will re-detect and re-send on next confirmation")

    cap.release()
    cv2.destroyAllWindows()
    with arbiter_lock:
        if arbiter_sock:
            arbiter_sock.close()


if __name__ == "__main__":
    main()
