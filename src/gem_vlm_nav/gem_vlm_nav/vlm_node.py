#!/usr/bin/env python3
"""
vlm_node.py — One-Shot VLM High-Level Waypoint Generator
========================================================

Behavior:
  - User sends /vlm_command.
  - VLM runs ONCE for that command.
  - VLM outputs high-level intent:
      local_goal / map_goal / world_goal / arrived
  - Node converts that intent into fixed ENU world-frame waypoints.
  - Waypoints are LOCKED and repeatedly republished.
  - The VLM does NOT update the waypoint as the car moves.
  - A new /vlm_command clears the old goal and runs the VLM once again.

This prevents the "moving carrot" problem where local goals keep shifting with
the car's updated pose.
"""

import math
import base64
import json
import threading
import time
import re

import rospy
import numpy as np
import cv2
import pymap3d as pm

from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose
from septentrio_gnss_driver.msg import INSNavGeod
from cv_bridge import CvBridge


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

ORIGIN_LAT = 40.0928381
ORIGIN_LON = -88.2356367

X_MIN = -50
X_MAX =  40
Y_MIN = -12
Y_MAX =  5

WHEELBASE = 2.57
MAX_STEER = 0.6
MIN_TURN_R = WHEELBASE / math.tan(MAX_STEER)

MIN_WP_DIST = MIN_TURN_R + 0.5
MAX_WP_DIST = 18.0
MIN_WP_SEP = 3.0
MAX_WAYPOINTS = 5

REPUBLISH_HZ = 10.0

GPT_MODEL = "gpt-4o"
GPT_MAX_TOK = 500
RGB_DETAIL = "high"
BEV_DETAIL = "high"

DRY_RUN = False

MAP_MARGIN = 5.0
ARRIVAL_RADIUS_M = 2.0


SYSTEM_PROMPT = f"""
You are a high-level VLM navigation controller for a GEM e4 autonomous vehicle.

You receive:
1. Front RGB camera image.
2. LiDAR BEV image.
3. Vehicle pose in ENU world coordinates.
4. User command.
5. Map bounds.

Coordinate facts:
- ENU world frame: x = East, y = North.
- Vehicle yaw is given in ENU.
- Vehicle forward and left vectors are explicitly given.
- The LiDAR BEV image is vehicle-centered.
- In the BEV image, the vehicle is at the center and vehicle-forward is UP.
- The BEV image is NOT north-up.
- Map bounds:
  x in [{X_MIN:.2f}, {X_MAX:.2f}]
  y in [{Y_MIN:.2f}, {Y_MAX:.2f}]

You are a HIGH-LEVEL controller. Choose the next navigation intent.

You MUST output only valid JSON.

Allowed schemas:

1. local_goal
Use for perception-heavy commands:
"drive up to this cone", "go next to that object", "move toward the opening".

{{
  "status": "searching" | "navigating",
  "plan_type": "local_goal",
  "reasoning": "...",
  "local_goal": {{
    "forward_m": number,
    "left_m": number,
    "stop_distance_m": number
  }}
}}

Meaning:
- forward_m > 0 is ahead of the vehicle.
- left_m > 0 is left of the vehicle.
- Choose a single fixed local target relative to the CURRENT vehicle pose.
- Keep forward_m between 4 and 18.
- Keep left_m between -10 and 10.

2. map_goal
Use for global map commands:
"top-left corner", "upper right", "center", "north side", etc.

{{
  "status": "navigating",
  "plan_type": "map_goal",
  "reasoning": "...",
  "map_goal": "top_left" | "top_right" | "bottom_left" | "bottom_right" |
              "top" | "bottom" | "left" | "right" | "center"
}}

Interpretation:
- top = high y / north.
- bottom = low y / south.
- left = low x / west.
- right = high x / east.

3. world_goal
Use only if confident about a specific ENU coordinate.

{{
  "status": "navigating",
  "plan_type": "world_goal",
  "reasoning": "...",
  "world_goal": {{
    "x": number,
    "y": number
  }}
}}

4. arrived

{{
  "status": "arrived",
  "plan_type": "arrived",
  "reasoning": "...",
  "waypoints": []
}}

Rules:
- Prefer local_goal for visible-object tasks.
- Prefer map_goal for map/corner/direction tasks.
- Do not output moving/tracking behavior.
- This VLM call happens once; choose a fixed goal.
- Keep reasoning under 2 sentences.
"""


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def ins_heading_to_yaw(h: float) -> float:
    return math.radians(90.0 - h) if h < 270.0 else math.radians(450.0 - h)


def img_to_base64(cv_img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp_to_bounds(x, y, margin=2.0):
    return (
        clamp(x, X_MIN + margin, X_MAX - margin),
        clamp(y, Y_MIN + margin, Y_MAX - margin),
    )


def local_to_world(car_x, car_y, car_yaw, forward_m, left_m):
    fwd_x = math.cos(car_yaw)
    fwd_y = math.sin(car_yaw)

    left_x = -math.sin(car_yaw)
    left_y = math.cos(car_yaw)

    wx = car_x + forward_m * fwd_x + left_m * left_x
    wy = car_y + forward_m * fwd_y + left_m * left_y

    return wx, wy


def map_goal_to_world(goal_name):
    gx_mid = 0.5 * (X_MIN + X_MAX)
    gy_mid = 0.5 * (Y_MIN + Y_MAX)

    west = X_MIN + MAP_MARGIN
    east = X_MAX - MAP_MARGIN
    south = Y_MIN + MAP_MARGIN
    north = Y_MAX - MAP_MARGIN

    table = {
        "top_left":     (west, north),
        "top_right":    (east, north),
        "bottom_left":  (west, south),
        "bottom_right": (east, south),
        "top":          (gx_mid, north),
        "bottom":       (gx_mid, south),
        "left":         (west, gy_mid),
        "right":        (east, gy_mid),
        "center":       (gx_mid, gy_mid),
    }

    return table.get(goal_name, (gx_mid, gy_mid))


def extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    return json.loads(m.group(0))


def clamp_local_goal_to_feasible(forward_m, left_m):
    forward_m = clamp(forward_m, MIN_WP_DIST, MAX_WP_DIST)

    max_left = 0.45 * forward_m
    left_m = clamp(left_m, -max_left, max_left)

    return forward_m, left_m


def build_waypoints_to_goal(car_x, car_y, car_yaw, goal_x, goal_y):
    goal_x, goal_y = clamp_to_bounds(goal_x, goal_y)

    dx = goal_x - car_x
    dy = goal_y - car_y
    dist = math.hypot(dx, dy)

    if dist < ARRIVAL_RADIUS_M:
        return []

    heading_to_goal = math.atan2(dy, dx)
    heading_err = normalize_angle(heading_to_goal - car_yaw)

    waypoints = []

    if abs(heading_err) > math.radians(100):
        turn_sign = 1.0 if heading_err > 0 else -1.0
        wx1, wy1 = local_to_world(car_x, car_y, car_yaw, 7.0, turn_sign * 4.0)
        wx1, wy1 = clamp_to_bounds(wx1, wy1)
        waypoints.append({"x": wx1, "y": wy1})
        prev_x, prev_y = wx1, wy1
    else:
        prev_x, prev_y = car_x, car_y

    remaining = math.hypot(goal_x - prev_x, goal_y - prev_y)
    if remaining < 1e-6:
        return waypoints

    steps = int(math.ceil(remaining / MAX_WP_DIST))
    steps = max(1, min(steps, MAX_WAYPOINTS - len(waypoints)))

    for i in range(1, steps + 1):
        t = float(i) / float(steps)
        wx = prev_x + t * (goal_x - prev_x)
        wy = prev_y + t * (goal_y - prev_y)
        wx, wy = clamp_to_bounds(wx, wy)

        if waypoints:
            sep = math.hypot(wx - waypoints[-1]["x"], wy - waypoints[-1]["y"])
        else:
            sep = math.hypot(wx - car_x, wy - car_y)

        if sep >= MIN_WP_SEP or i == steps:
            waypoints.append({"x": wx, "y": wy})

        if len(waypoints) >= MAX_WAYPOINTS:
            break

    return waypoints


def filter_waypoints(raw_waypoints, car_x, car_y):
    filtered = []
    prev_x, prev_y = car_x, car_y

    for wp in raw_waypoints:
        try:
            wx = float(wp["x"])
            wy = float(wp["y"])
        except Exception:
            continue

        wx, wy = clamp_to_bounds(wx, wy)
        sep = math.hypot(wx - prev_x, wy - prev_y)

        if len(filtered) == 0 and sep < MIN_WP_DIST:
            if sep < 1e-3:
                continue
            scale = MIN_WP_DIST / sep
            wx = prev_x + (wx - prev_x) * scale
            wy = prev_y + (wy - prev_y) * scale
            wx, wy = clamp_to_bounds(wx, wy)
            sep = math.hypot(wx - prev_x, wy - prev_y)

        if len(filtered) > 0 and sep < MIN_WP_SEP:
            continue

        if sep > MAX_WP_DIST:
            ang = math.atan2(wy - prev_y, wx - prev_x)
            wx = prev_x + MAX_WP_DIST * math.cos(ang)
            wy = prev_y + MAX_WP_DIST * math.sin(ang)
            wx, wy = clamp_to_bounds(wx, wy)

        filtered.append({"x": wx, "y": wy})
        prev_x, prev_y = wx, wy

        if len(filtered) >= MAX_WAYPOINTS:
            break

    return filtered


# ═══════════════════════════════════════════════════════════════
# NODE
# ═══════════════════════════════════════════════════════════════

class VLMNode:
    def __init__(self):
        rospy.init_node("vlm_node", anonymous=False)

        self.bridge = CvBridge()

        self.latest_rgb = None
        self.latest_bev = None

        self.car_x = None
        self.car_y = None
        self.car_yaw = 0.0
        self.car_heading = 0.0

        self.command = None
        self.last_status = "waiting"
        self.last_reasoning = ""
        self.last_plan_type = "none"

        self.goal_locked = False
        self.locked_waypoints = []
        self.latest_valid_wps = []

        self._wp_lock = threading.Lock()
        self._vlm_running = False

        if not DRY_RUN:
            from openai import OpenAI
            self.client = OpenAI()
        else:
            self.client = None
            rospy.logwarn("[vlm] DRY_RUN=True")

        rospy.Subscriber("/oak/rgb/image_raw", Image, self.rgb_cb, queue_size=1)
        rospy.Subscriber("/lidar_bev_image", Image, self.bev_cb, queue_size=1)
        rospy.Subscriber("/gps/fix", NavSatFix, self.gps_cb, queue_size=1)
        rospy.Subscriber("/septentrio_gnss/insnavgeod", INSNavGeod, self.ins_cb, queue_size=1)
        rospy.Subscriber("/vlm_command", String, self.command_cb, queue_size=1)

        self.wp_pub = rospy.Publisher("/vlm_waypoints", PoseArray, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher("/vlm_status", String, queue_size=1, latch=True)

        rospy.Timer(rospy.Duration(1.0 / REPUBLISH_HZ), self._republish_cb)

        rospy.loginfo("[vlm] Ready. One-shot VLM mode. New command = new fixed waypoint set.")

    def rgb_cb(self, msg):
        self.latest_rgb = msg

    def bev_cb(self, msg):
        self.latest_bev = msg

    def gps_cb(self, msg):
        try:
            e, n, _ = pm.geodetic2enu(
                msg.latitude,
                msg.longitude,
                0,
                ORIGIN_LAT,
                ORIGIN_LON,
                0
            )
            self.car_x = float(e)
            self.car_y = float(n)
        except Exception as ex:
            rospy.logwarn_throttle(5.0, "[vlm] GPS conversion failed: %s", str(ex))

    def ins_cb(self, msg):
        if msg.heading is not None and not math.isnan(msg.heading):
            self.car_heading = float(msg.heading)
            self.car_yaw = ins_heading_to_yaw(self.car_heading)

    def command_cb(self, msg):
        cmd = msg.data.strip()
        if not cmd:
            return

        rospy.loginfo("[vlm] New command received: '%s'. Clearing old locked goal.", cmd)

        self.command = cmd
        self.last_status = "searching"
        self.last_reasoning = ""
        self.last_plan_type = "none"

        with self._wp_lock:
            self.goal_locked = False
            self.locked_waypoints = []
            self.latest_valid_wps = []

        self._publish_status("searching")
        self._start_vlm_thread_once()

    def _publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def _republish_cb(self, _event):
        with self._wp_lock:
            wps = list(self.latest_valid_wps)

        pa = PoseArray()
        pa.header.stamp = rospy.Time.now()
        pa.header.frame_id = "world"

        for wp in wps:
            p = Pose()
            p.position.x = float(wp["x"])
            p.position.y = float(wp["y"])
            p.position.z = 0.0
            p.orientation.w = 1.0
            pa.poses.append(p)

        self.wp_pub.publish(pa)

    def _start_vlm_thread_once(self):
        if self._vlm_running:
            rospy.logwarn("[vlm] VLM already running; ignoring duplicate trigger.")
            return

        with self._wp_lock:
            if self.goal_locked:
                rospy.loginfo("[vlm] Goal already locked; not running VLM again.")
                return

        if self.car_x is None or self.car_y is None:
            rospy.logwarn("[vlm] Waiting for GPS/ENU pose before VLM call.")
            return

        if self.latest_rgb is None:
            rospy.logwarn("[vlm] Waiting for RGB image before VLM call.")
            return

        if self.latest_bev is None:
            rospy.logwarn("[vlm] Waiting for BEV image before VLM call.")
            return

        self._vlm_running = True

        args = (
            self.latest_rgb,
            self.latest_bev,
            self.car_x,
            self.car_y,
            self.car_yaw,
            self.car_heading,
            self.command,
        )

        threading.Thread(target=self._vlm_inference, args=args, daemon=True).start()

    def _make_user_text(self, car_x, car_y, car_yaw, car_heading, command):
        fwd_x = math.cos(car_yaw)
        fwd_y = math.sin(car_yaw)
        left_x = -math.sin(car_yaw)
        left_y = math.cos(car_yaw)

        return f"""
Command: "{command}"

Current vehicle state:
- ENU position: x={car_x:.2f}, y={car_y:.2f}
- INS heading: {car_heading:.2f} deg
- ENU yaw: {math.degrees(car_yaw):.2f} deg
- Forward vector: dx={fwd_x:.4f}, dy={fwd_y:.4f}
- Left vector: dx={left_x:.4f}, dy={left_y:.4f}

Formula:
A local goal F meters forward and L meters left becomes:
x = {car_x:.2f} + F*{fwd_x:.4f} + L*{left_x:.4f}
y = {car_y:.2f} + F*{fwd_y:.4f} + L*{left_y:.4f}

Map:
- x_min={X_MIN:.2f}, x_max={X_MAX:.2f}
- y_min={Y_MIN:.2f}, y_max={Y_MAX:.2f}
- top/north = high y
- bottom/south = low y
- left/west = low x
- right/east = high x

Images:
- Image 1 is front RGB.
- Image 2 is LiDAR BEV.
- The BEV is vehicle-centered: car at center, forward is up.
- The BEV is not north-up.

Important:
This is a ONE-SHOT call. Choose a fixed goal from the current pose.
The waypoint will not update as the vehicle moves.

Output only valid JSON.
"""

    def _call_vlm(self, rgb_msg, bev_msg, car_x, car_y, car_yaw, car_heading, command):
        rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        bev_cv = self.bridge.imgmsg_to_cv2(bev_msg, "bgr8")

        rgb_b64 = img_to_base64(rgb_cv)
        bev_b64 = img_to_base64(bev_cv)

        user_text = self._make_user_text(
            car_x,
            car_y,
            car_yaw,
            car_heading,
            command,
        )

        response = self.client.chat.completions.create(
            model=GPT_MODEL,
            max_tokens=GPT_MAX_TOK,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{rgb_b64}",
                                "detail": RGB_DETAIL,
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{bev_b64}",
                                "detail": BEV_DETAIL,
                            },
                        },
                    ],
                },
            ],
        )

        raw = response.choices[0].message.content.strip()
        result = extract_json(raw)
        return result, raw

    def _interpret_vlm_result(self, result, car_x, car_y, car_yaw):
        status = result.get("status", "searching")
        plan_type = result.get("plan_type", "local_goal")
        reasoning = result.get("reasoning", "")

        if status == "arrived" or plan_type == "arrived":
            return "arrived", "arrived", reasoning, []

        if plan_type == "map_goal":
            goal_name = str(result.get("map_goal", "center")).strip().lower()
            goal_x, goal_y = map_goal_to_world(goal_name)

            rospy.loginfo(
                "[vlm] map_goal=%s -> fixed world goal (%.2f, %.2f)",
                goal_name,
                goal_x,
                goal_y,
            )

            wps = build_waypoints_to_goal(car_x, car_y, car_yaw, goal_x, goal_y)
            return status, plan_type, reasoning, wps

        if plan_type == "world_goal":
            wg = result.get("world_goal", {})
            goal_x = float(wg.get("x", car_x))
            goal_y = float(wg.get("y", car_y))
            goal_x, goal_y = clamp_to_bounds(goal_x, goal_y)

            rospy.loginfo(
                "[vlm] world_goal -> fixed clipped world goal (%.2f, %.2f)",
                goal_x,
                goal_y,
            )

            wps = build_waypoints_to_goal(car_x, car_y, car_yaw, goal_x, goal_y)
            return status, plan_type, reasoning, wps

        lg = result.get("local_goal", {})

        forward_m = float(lg.get("forward_m", 8.0))
        left_m = float(lg.get("left_m", 0.0))

        forward_m, left_m = clamp_local_goal_to_feasible(forward_m, left_m)

        goal_x, goal_y = local_to_world(car_x, car_y, car_yaw, forward_m, left_m)
        goal_x, goal_y = clamp_to_bounds(goal_x, goal_y)

        rospy.loginfo(
            "[vlm] local_goal F=%.2f L=%.2f -> fixed world goal (%.2f, %.2f)",
            forward_m,
            left_m,
            goal_x,
            goal_y,
        )

        wps = build_waypoints_to_goal(car_x, car_y, car_yaw, goal_x, goal_y)
        return status, "local_goal", reasoning, wps

    def _vlm_inference(self, rgb_msg, bev_msg, car_x, car_y, car_yaw, car_heading, command):
        t0 = time.time()

        try:
            if DRY_RUN:
                result = {
                    "status": "navigating",
                    "plan_type": "local_goal",
                    "reasoning": "DRY RUN fixed local goal.",
                    "local_goal": {
                        "forward_m": 10.0,
                        "left_m": 0.0,
                        "stop_distance_m": 0.0,
                    },
                }
                raw = json.dumps(result)
            else:
                result, raw = self._call_vlm(
                    rgb_msg,
                    bev_msg,
                    car_x,
                    car_y,
                    car_yaw,
                    car_heading,
                    command,
                )

            rospy.loginfo("[vlm] Raw VLM JSON: %s", json.dumps(result))

            status, plan_type, reasoning, raw_wps = self._interpret_vlm_result(
                result,
                car_x,
                car_y,
                car_yaw,
            )

            valid_wps = filter_waypoints(raw_wps, car_x, car_y)

            self.last_status = status
            self.last_plan_type = plan_type
            self.last_reasoning = reasoning

            self._publish_status(status)

            if status == "arrived":
                with self._wp_lock:
                    self.goal_locked = True
                    self.locked_waypoints = []
                    self.latest_valid_wps = []
                rospy.loginfo("[vlm] ARRIVED: %s", reasoning)
                return

            if not valid_wps:
                rospy.logwarn(
                    "[vlm] No valid waypoints from plan_type=%s. Goal not locked.",
                    plan_type,
                )
                self._publish_status("searching")
                return

            with self._wp_lock:
                self.locked_waypoints = list(valid_wps)
                self.latest_valid_wps = list(valid_wps)
                self.goal_locked = True

            elapsed = time.time() - t0

            rospy.loginfo(
                "[vlm] LOCKED fixed waypoints in %.2fs. status=%s plan_type=%s reasoning='%s'",
                elapsed,
                status,
                plan_type,
                reasoning[:120],
            )

            rospy.loginfo(
                "[vlm] Locked waypoints: %s",
                " ".join(f"({w['x']:.1f},{w['y']:.1f})" for w in valid_wps),
            )

        except json.JSONDecodeError as e:
            rospy.logerr("[vlm] JSON parse error: %s", str(e))
            self._publish_status("searching")
        except Exception as e:
            rospy.logerr("[vlm] Inference error: %s", str(e))
            self._publish_status("searching")
        finally:
            self._vlm_running = False


def main():
    VLMNode()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
