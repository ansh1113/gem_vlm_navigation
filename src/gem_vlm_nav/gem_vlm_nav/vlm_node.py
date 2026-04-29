#!/usr/bin/env python3
"""
vlm_node.py — Closed-Loop VLM Receding-Horizon Navigation Planner
==================================================================
Runs GPT-4o (vision) in a background thread at ~0.2Hz.

Each cycle the VLM receives:
  [1] Front RGB camera image
  [2] LiDAR BEV top-down image (from lidar_bev_node)
  [3] Current vehicle position + heading (text)
  [4] Original user command (cached from first call)
  [5] Current status + last reasoning (continuity between cycles)

The VLM outputs a JSON plan:
  {
    "status":    "searching" | "navigating" | "arrived",
    "reasoning": "<brief explanation of what the VLM sees>",
    "waypoints": [{"x": float, "y": float}, ...]   // 2-4 ENU map coords
  }

Waypoints are passed through a kinematic feasibility filter
(min turning radius, min spacing) before publishing.

The plan is published to /vlm_waypoints where planner_node
immediately preempts its current path and replans toward
waypoint[0] using A* on the live costmap.

DRY_RUN mode: set DRY_RUN=True to bypass API calls during
development. Returns a hardcoded JSON so you can test the
full pipeline without spending API credits.

Topics (in):
  /oak/rgb/image_raw   — Front RGB camera
  /lidar_bev_image     — BEV image from lidar_bev_node
  /navsatfix           — GNSS position
  /insnavgeod          — INS heading
  /vlm_command         — User command string (String)

Topics (out):
  /vlm_waypoints       — WaypointArray (geometry_msgs/PoseArray)
                         Each pose.position.x/y is an ENU waypoint
  /vlm_status          — String: searching|navigating|arrived|waiting
"""

import math
import base64
import json
import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose
from septentrio_gnss_driver.msg import INSNavGeod
from cv_bridge import CvBridge
import cv2
import pymap3d as pm
from openai import OpenAI

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

ORIGIN_LAT = 40.0927422
ORIGIN_LON = -88.2359639

# Operating area bounds (ENU meters)
X_MIN, X_MAX = -25.0, 75.0
Y_MIN, Y_MAX =  -5.0, 20.0

# GEM e4 kinematic limits
WHEELBASE    = 2.57   # m
MAX_STEER    = 0.6    # rad
SPEED        = 2.0    # m/s

# Minimum turning radius = L / tan(max_steer) ≈ 3.76m
MIN_TURN_R   = WHEELBASE / math.tan(MAX_STEER)   # ~3.76m

# Waypoint constraints for kinematic feasibility
MIN_WP_DIST  = MIN_TURN_R + 0.5   # min distance to each waypoint (m)
MAX_WP_DIST  = 25.0               # max lookahead per waypoint (m)
MIN_WP_SEP   = 4.0                # min separation between consecutive waypoints (m)
MAX_WAYPOINTS = 4                 # max waypoints per VLM cycle

# VLM cycle rate
VLM_HZ = 0.2          # ~1 call per 5 seconds

# GPT model
GPT_MODEL    = "gpt-4o"
GPT_MAX_TOK  = 300    # keep output short — JSON only
IMAGE_DETAIL = "low"  # low detail = cheaper (~85 tokens/image vs ~1700)

# Set True during development to skip API calls
DRY_RUN = True

# Dry run returns this hardcoded response for pipeline testing
DRY_RUN_RESPONSE = {
    "status": "navigating",
    "reasoning": "DRY RUN: cone assumed at (20, 8). Moving toward it.",
    "waypoints": [
        {"x": 10.0, "y": 5.0},
        {"x": 20.0, "y": 8.0}
    ]
}

# System prompt — tells GPT its role and output format
SYSTEM_PROMPT = """You are the high-level navigation planner for an autonomous vehicle \
(GEM e4 electric car) operating in a university parking lot.

You receive two images every cycle:
  IMAGE 1: Front RGB camera — what the vehicle sees ahead
  IMAGE 2: LiDAR Bird's-Eye-View — top-down map, vehicle at center pointing UP, \
obstacles shown as bright points, distance rings at 5m and 10m

Your job: given the user's navigation command, decide where the vehicle should \
drive next and output a short sequence of waypoints in the map's ENU coordinate \
frame (x=East meters, y=North meters from origin).

Rules:
- Output ONLY valid JSON, no explanation outside the JSON
- Waypoints must be reachable given a car turning radius of 3.8m
- Space waypoints at least 4m apart
- Maximum 4 waypoints per response
- If the goal object is not visible, output waypoints to explore or reposition \
  the vehicle to improve visibility (e.g. turn around, move to open area)
- If the vehicle has arrived within 2m of the goal, set status to "arrived"
- Keep reasoning under 2 sentences

Output format:
{
  "status": "searching" | "navigating" | "arrived",
  "reasoning": "<what you see and why you chose these waypoints>",
  "waypoints": [{"x": <float>, "y": <float>}, ...]
}"""


# ═══════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════

def ins_heading_to_yaw(h: float) -> float:
    return math.radians(90.0 - h) if h < 270.0 else math.radians(450.0 - h)

def img_to_base64(cv_img: np.ndarray) -> str:
    """Encode OpenCV BGR image to base64 JPEG string for GPT-4o."""
    _, buf = cv2.imencode('.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode('utf-8')


# ═══════════════════════════════════════════════════════════════
#  KINEMATIC FEASIBILITY FILTER
# ═══════════════════════════════════════════════════════════════

def filter_waypoints(raw_waypoints: list,
                     car_x: float, car_y: float,
                     car_yaw: float) -> list:
    """
    Filter VLM-output waypoints for kinematic feasibility.

    Checks:
      1. Each waypoint is within lot bounds
      2. Distance from current position >= MIN_WP_DIST (turning radius)
      3. Consecutive waypoints are spaced >= MIN_WP_SEP apart
      4. No more than MAX_WAYPOINTS returned

    Waypoints that fail are dropped (not corrected) — the VLM will
    regenerate better ones on the next cycle.
    """
    filtered = []
    prev_x, prev_y = car_x, car_y

    for wp in raw_waypoints:
        wx = float(wp.get('x', 0.0))
        wy = float(wp.get('y', 0.0))

        # Bounds check
        if not (X_MIN + 2 < wx < X_MAX - 2 and Y_MIN + 2 < wy < Y_MAX - 2):
            continue

        # Distance from previous point (car pos for first wp)
        sep = math.hypot(wx - prev_x, wy - prev_y)

        if len(filtered) == 0:
            # First waypoint: must be reachable given turning radius
            if sep < MIN_WP_DIST:
                continue
        else:
            # Subsequent waypoints: must be spaced reasonably
            if sep < MIN_WP_SEP:
                continue

        # Max distance cap — don't send car across the entire lot in one step
        if sep > MAX_WP_DIST:
            # Clip to MAX_WP_DIST along same direction
            angle = math.atan2(wy - prev_y, wx - prev_x)
            wx = prev_x + MAX_WP_DIST * math.cos(angle)
            wy = prev_y + MAX_WP_DIST * math.sin(angle)

        filtered.append({'x': wx, 'y': wy})
        prev_x, prev_y = wx, wy

        if len(filtered) >= MAX_WAYPOINTS:
            break

    return filtered


# ═══════════════════════════════════════════════════════════════
#  NODE
# ═══════════════════════════════════════════════════════════════

class VLMNode(Node):
    def __init__(self):
        super().__init__('vlm_node')

        # Sensor state
        self.bridge        = CvBridge()
        self.latest_rgb    = None
        self.latest_bev    = None
        self.car_x         = None
        self.car_y         = None
        self.car_yaw       = 0.0
        self.car_heading   = 0.0   # raw degrees

        # Mission state
        self.command       = None   # cached from /vlm_command
        self.last_status   = "waiting"
        self.last_reasoning = ""
        self._lock         = threading.Lock()

        # GPT client
        if not DRY_RUN:
            self.client = OpenAI()   # reads OPENAI_API_KEY from env
        else:
            self.client = None
            self.get_logger().warn(
                "DRY_RUN mode enabled — GPT-4o calls are skipped.")

        # ── Subscriptions ──────────────────────────────────────
        self.create_subscription(
            Image, '/oak/rgb/image_raw', self.rgb_cb, 10)
        # NOTE: try /oak/color/image_raw if above is missing on live vehicle

        self.create_subscription(
            Image, '/lidar_bev_image', self.bev_cb, 10)

        self.create_subscription(
            NavSatFix, '/navsatfix', self.gps_cb, 10)

        self.create_subscription(
            INSNavGeod, '/insnavgeod', self.ins_cb, 10)

        # Command — TRANSIENT_LOCAL so we don't miss it if published before node starts
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            String, '/vlm_command', self.command_cb, qos)

        # ── Publishers ─────────────────────────────────────────
        wp_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.wp_pub     = self.create_publisher(PoseArray, '/vlm_waypoints', wp_qos)
        self.status_pub = self.create_publisher(String,    '/vlm_status',    10)

        # ── VLM timer ──────────────────────────────────────────
        # Timer triggers the VLM cycle — actual inference runs in a thread
        # so we never block the ROS executor
        self._vlm_running = False
        self.create_timer(1.0 / VLM_HZ, self._vlm_timer_cb)

        self.get_logger().info(
            f"VLM node ready. DRY_RUN={DRY_RUN}  "
            f"Model={GPT_MODEL}  Rate={VLM_HZ}Hz  "
            f"MIN_TURN_R={MIN_TURN_R:.2f}m  "
            "Publish /vlm_command to start.")

    # ── Sensor callbacks ──────────────────────────────────────

    def rgb_cb(self, msg: Image):
        self.latest_rgb = msg

    def bev_cb(self, msg: Image):
        self.latest_bev = msg

    def gps_cb(self, msg: NavSatFix):
        try:
            e, n, _ = pm.geodetic2enu(
                msg.latitude, msg.longitude, 0,
                ORIGIN_LAT, ORIGIN_LON, 0)
            self.car_x, self.car_y = float(e), float(n)
        except Exception:
            pass

    def ins_cb(self, msg: INSNavGeod):
        if msg.heading is not None and not math.isnan(msg.heading):
            self.car_heading = msg.heading
            self.car_yaw     = ins_heading_to_yaw(msg.heading)

    def command_cb(self, msg: String):
        cmd = msg.data.strip()
        if cmd:
            self.command      = cmd
            self.last_status  = "searching"
            self.last_reasoning = ""
            self.get_logger().info(f"Command received: '{cmd}'")

    # ── VLM timer ─────────────────────────────────────────────

    def _vlm_timer_cb(self):
        """
        Called at VLM_HZ. Launches VLM inference in a background thread
        so the ROS executor is never blocked.
        """
        if self.command is None:
            return
        if self.last_status == "arrived":
            return
        if self._vlm_running:
            # Previous call still in flight — skip this cycle
            self.get_logger().debug("VLM: previous call still running, skipping.")
            return
        if self.car_x is None:
            self.get_logger().warn("VLM: waiting for GPS lock.")
            return
        if self.latest_rgb is None or self.latest_bev is None:
            self.get_logger().warn("VLM: waiting for camera/BEV images.")
            return

        # Snapshot current sensor state (thread-safe copies)
        rgb_msg    = self.latest_rgb
        bev_msg    = self.latest_bev
        car_x      = self.car_x
        car_y      = self.car_y
        car_yaw    = self.car_yaw
        car_heading = self.car_heading
        command    = self.command
        last_status    = self.last_status
        last_reasoning = self.last_reasoning

        self._vlm_running = True
        t = threading.Thread(
            target=self._vlm_inference,
            args=(rgb_msg, bev_msg, car_x, car_y,
                  car_yaw, car_heading, command,
                  last_status, last_reasoning),
            daemon=True)
        t.start()

    # ── VLM inference (background thread) ────────────────────

    def _vlm_inference(self, rgb_msg, bev_msg,
                       car_x, car_y, car_yaw, car_heading,
                       command, last_status, last_reasoning):
        t0 = time.time()
        try:
            # Convert ROS images to OpenCV
            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            bev_cv = self.bridge.imgmsg_to_cv2(bev_msg, 'bgr8')

            # Encode to base64
            rgb_b64 = img_to_base64(rgb_cv)
            bev_b64 = img_to_base64(bev_cv)

            # Build user message
            user_text = (
                f"Command: \"{command}\"\n"
                f"Vehicle position: x={car_x:.1f}m, y={car_y:.1f}m, "
                f"heading={car_heading:.1f}° ({math.degrees(car_yaw):.1f}° ENU yaw)\n"
                f"Current status: {last_status}\n"
                f"Last reasoning: {last_reasoning if last_reasoning else 'none yet'}\n\n"
                f"Image 1 (front camera) and Image 2 (LiDAR BEV, vehicle at centre "
                f"pointing UP, rings at 5m and 10m) are attached.\n"
                f"Output your JSON plan now."
            )

            if DRY_RUN:
                result = DRY_RUN_RESPONSE
                self.get_logger().info(
                    f"[DRY RUN] Skipping GPT call. "
                    f"Returning hardcoded response.")
            else:
                response = self.client.chat.completions.create(
                    model=GPT_MODEL,
                    max_tokens=GPT_MAX_TOK,
                    temperature=0.2,   # low temp = consistent spatial reasoning
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{rgb_b64}",
                                "detail": IMAGE_DETAIL}},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{bev_b64}",
                                "detail": IMAGE_DETAIL}},
                        ]}
                    ]
                )
                raw_text = response.choices[0].message.content.strip()
                # Strip markdown fences if GPT wraps in ```json ... ```
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                result = json.loads(raw_text)

            elapsed = time.time() - t0
            self.get_logger().info(
                f"VLM [{elapsed:.1f}s] "
                f"status={result.get('status')}  "
                f"reasoning='{result.get('reasoning', '')[:80]}'  "
                f"waypoints={len(result.get('waypoints', []))}")

            # Update status
            self.last_status    = result.get('status', 'searching')
            self.last_reasoning = result.get('reasoning', '')

            # Publish status string
            status_msg = String()
            status_msg.data = self.last_status
            self.status_pub.publish(status_msg)

            if self.last_status == "arrived":
                self.get_logger().info("VLM: ARRIVED at goal!")
                return

            # Kinematic feasibility filter
            raw_wps   = result.get('waypoints', [])
            valid_wps = filter_waypoints(raw_wps, car_x, car_y, car_yaw)

            if not valid_wps:
                self.get_logger().warn(
                    f"VLM returned {len(raw_wps)} waypoints but "
                    f"0 passed kinematic filter. Skipping publish.")
                return

            # Publish as PoseArray — planner reads pose.position.x/y
            pa = PoseArray()
            pa.header.stamp    = self.get_clock().now().to_msg()
            pa.header.frame_id = 'map'
            for wp in valid_wps:
                p = Pose()
                p.position.x = float(wp['x'])
                p.position.y = float(wp['y'])
                p.position.z = 0.0
                p.orientation.w = 1.0
                pa.poses.append(p)
            self.wp_pub.publish(pa)

            self.get_logger().info(
                f"Published {len(valid_wps)} waypoints "
                f"(filtered from {len(raw_wps)}): "
                + "  ".join(f"({w['x']:.1f},{w['y']:.1f})" for w in valid_wps))

        except json.JSONDecodeError as e:
            self.get_logger().error(f"VLM JSON parse error: {e}")
        except Exception as e:
            self.get_logger().error(f"VLM inference error: {e}")
        finally:
            self._vlm_running = False


# ═══════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = VLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()