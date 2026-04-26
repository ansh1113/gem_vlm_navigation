#!/usr/bin/env python3
"""
perception_node.py — VLM Perception Node for GEM e4 Language-Conditioned Navigation
=====================================================================================
Pipeline:
  1. Subscribes to /vlm_command (String) — natural language goal from operator
  2. LLM (GPT-4o-mini) classifies intent: visual_search vs fixed_landmark
  3. OWL-ViT detects target object in camera frame
  4. LiDAR angular slice under bounding box gives accurate depth Z
  5. Pinhole projection → local (x, y) in vehicle frame
  6. INSNavGeod heading + GNSS → ENU map frame goal
  7. Publishes PoseStamped to /vlm_goal for planner_node

Topics (in):
  /oak/rgb/image_raw      — OAK-D LR front stereo camera (RGB)
                            NOTE: On live vehicle verify with `ros2 topic list`.
                            May be /oak/color/image_raw on newer driver versions.
  /navsatfix              — Septentrio GNSS (lat/lon)
  /insnavgeod             — Septentrio INS heading (degrees, same as pure pursuit)
  /ouster/points          — Ouster OS1-128 LiDAR pointcloud

Topics (out):
  /vlm_goal               — PoseStamped goal in ENU map frame (TRANSIENT_LOCAL QoS)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, DurabilityPolicy
from septentrio_gnss_driver.msg import INSNavGeod

import json
import time
import math
import numpy as np
import cv2
import torch
from PIL import Image as PILImage
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from openai import OpenAI
import pymap3d as pm

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

# ENU origin — same as pure pursuit controller
ORIGIN_LAT = 40.0927422
ORIGIN_LON = -88.2359639

# Safety offset applied when modifier says "near", "behind" etc.
SAFETY_OFFSET_M = 3.0

# OAK-D LR front stereo camera intrinsics (1280x800 @ 23fps)
# Calibrate these properly from the camera info topic if available:
#   ros2 topic echo /oak/rgb/camera_info
CAMERA_INTRINSICS = {
    'fx': 800.0,   # focal length x (pixels)
    'fy': 800.0,   # focal length y (pixels)
    'cx': 640.0,   # principal point x (pixels) — half of 1280
    'cy': 400.0,   # principal point y (pixels) — half of 800
}

# OWL-ViT detection confidence threshold — lower = more detections but noisier
OWLVIT_THRESHOLD = 0.01

# Minimum number of LiDAR points in the angular slice to trust depth estimate
MIN_LIDAR_POINTS = 5

# Fallback depth if LiDAR gives no points under the bounding box (meters)
FALLBACK_DEPTH_M = 10.0

# ─────────────────────────────────────────────────────────────
#  HEADING CONVERSION  (matches pure_pursuit_ros2.py exactly)
# ─────────────────────────────────────────────────────────────

def ins_heading_to_yaw(heading_deg: float) -> float:
    """
    Convert Septentrio INSNavGeod heading (degrees, 0=North, CW) to
    ROS ENU yaw (radians, 0=East, CCW). Same formula used in pure pursuit.
    """
    if heading_deg < 270.0:
        return math.radians(90.0 - heading_deg)
    else:
        return math.radians(450.0 - heading_deg)


# ─────────────────────────────────────────────────────────────
#  NODE
# ─────────────────────────────────────────────────────────────

class VLMPerceptionNode(Node):
    def __init__(self):
        super().__init__('vlm_perception_node')

        # Sensor state
        self.bridge       = CvBridge()
        self.latest_image = None
        self.latest_lat   = None
        self.latest_lon   = None
        self.latest_yaw   = None   # None until first INS message
        self.latest_pc    = None

        # ── Subscriptions ──────────────────────────────────────
        self.create_subscription(
            Image, '/oak/rgb/image_raw', self.image_cb, 10)
        # NOTE: If running on live vehicle and image is missing, try:
        #   /oak/color/image_raw

        self.create_subscription(
            NavSatFix, '/navsatfix', self.gps_cb, 10)

        # INSNavGeod for heading — same source as pure pursuit controller
        self.create_subscription(
            INSNavGeod, '/insnavgeod', self.ins_cb, 10)

        self.create_subscription(
            PointCloud2, '/ouster/points', self.lidar_cb, 10)

        self.create_subscription(
            String, '/vlm_command', self.command_cb, 10)

        # ── Publisher ──────────────────────────────────────────
        # TRANSIENT_LOCAL so planner receives goal even if it starts late
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.goal_pub = self.create_publisher(PoseStamped, '/vlm_goal', qos)

        # ── Load models ────────────────────────────────────────
        self.get_logger().info("Loading LLM client and OWL-ViT model...")
        self.llm_client   = OpenAI()   # reads OPENAI_API_KEY from environment
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"  OWL-ViT running on: {self.device}")

        self.vlm_processor = OwlViTProcessor.from_pretrained(
            "google/owlvit-base-patch32")
        self.vlm_model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32").to(self.device).eval()

        self.get_logger().info("Perception node ready — waiting for /vlm_command")

    # ── Sensor callbacks ──────────────────────────────────────

    def image_cb(self, msg):
        self.latest_image = msg

    def gps_cb(self, msg):
        self.latest_lat = msg.latitude
        self.latest_lon = msg.longitude

    def ins_cb(self, msg):
        """
        Use INSNavGeod heading (degrees) from Septentrio.
        This matches how heading is used in pure_pursuit_ros2.py.
        """
        if msg.heading is not None and not math.isnan(msg.heading):
            self.latest_yaw = ins_heading_to_yaw(msg.heading)

    def lidar_cb(self, msg):
        # Store raw message — processed on demand in _get_depth_from_lidar
        self.latest_pc = msg

    # ── Command handler ───────────────────────────────────────

    def command_cb(self, msg):
        print(f"\n[COMMAND] '{msg.data}'", flush=True)

        # Sensor readiness check
        image_ok = self.latest_image is not None
        gps_ok   = self.latest_lat is not None
        lidar_ok = self.latest_pc is not None
        ins_ok   = self.latest_yaw is not None

        if not (image_ok and gps_ok and lidar_ok):
            print(f">>> WAITING FOR SENSORS: "
                  f"[Image:{image_ok}] [GPS:{gps_ok}] "
                  f"[LiDAR:{lidar_ok}] [INS:{ins_ok}]", flush=True)
            return

        if not ins_ok:
            print(">>> WARNING: INS heading not yet received. "
                  "Using fallback yaw=0.0 (East).", flush=True)

        # ── Step 1: LLM intent parsing ─────────────────────────
        print(">>> Step 1: Querying LLM for intent...", flush=True)
        intent = self._parse_intent(msg.data)
        print(f">>> LLM decision: type={intent['type']}  "
              f"target='{intent['target']}'  "
              f"modifier={intent.get('modifier')}", flush=True)

        # ── Step 2: Route by intent type ──────────────────────
        if intent['type'] == 'fixed_landmark':
            self._handle_fixed_landmark(intent['target'])
            return

        # visual_search — run OWL-ViT
        print(f">>> Step 2: Running OWL-ViT for '{intent['target']}'...", flush=True)
        t0 = time.time()
        cv_img  = self.bridge.imgmsg_to_cv2(
            self.latest_image, desired_encoding='bgr8')
        pil_img = PILImage.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        bbox    = self._get_bbox(pil_img, intent['target'])
        print(f">>> OWL-ViT inference: {time.time()-t0:.2f}s", flush=True)

        if bbox is None:
            print(f">>> VISION FAIL: '{intent['target']}' not found in frame.",
                  flush=True)
            return

        # ── Step 3: LiDAR depth under bounding box ─────────────
        print(">>> Step 3: Estimating depth from LiDAR...", flush=True)
        lx, ly = self._get_local_goal_from_lidar(
            bbox, (pil_img.height, pil_img.width), intent.get('modifier'))
        print(f">>> Local goal (vehicle frame): x={lx:.2f}m  y={ly:.2f}m",
              flush=True)

        # ── Step 4: Vehicle frame → ENU map frame ─────────────
        gx, gy = self._to_global(lx, ly)
        if math.isnan(gx) or math.isnan(gy):
            return   # error already logged inside _to_global

        # ── Step 5: Publish ────────────────────────────────────
        self._publish_goal(gx, gy)

    # ── Fixed landmark handler ────────────────────────────────

    def _handle_fixed_landmark(self, target: str):
        """
        Map known named locations to pre-surveyed ENU coordinates.
        Add your actual surveyed points here before deployment.
        These are placeholders — measure them with RTK GPS on-site.
        """
        LANDMARKS = {
            'start':         (-20.0,  5.0),
            'end':           ( 68.0,  5.0),
            'center':        ( 25.0,  7.5),
            'charging':      ( 65.0, 15.0),
            'highbay door':  (  0.0, 18.0),
        }
        key = target.lower().strip()
        if key in LANDMARKS:
            gx, gy = LANDMARKS[key]
            print(f">>> Fixed landmark '{key}' → ENU ({gx:.1f}, {gy:.1f})",
                  flush=True)
            self._publish_goal(gx, gy)
        else:
            print(f">>> Unknown landmark '{target}'. "
                  f"Known: {list(LANDMARKS.keys())}", flush=True)

    # ── LLM intent parsing ────────────────────────────────────

    def _parse_intent(self, cmd: str) -> dict:
        prompt = (
            f"Analyze the robotics navigation command: '{cmd}'\n\n"
            "Classify it into ONE of two types:\n"
            "1. 'visual_search' — physical object to find with camera "
            "(cone, person, box, chair, barrier, car, etc.)\n"
            "2. 'fixed_landmark' — named static location "
            "(start, end, charging, highbay door (names like 'highbay door' are fixed, but just 'door' is a visual_search), etc.)\n\n"
            "Return ONLY a JSON object with:\n"
            "  'type':     'visual_search' or 'fixed_landmark'\n"
            "  'target':   the specific noun (e.g. 'cone', 'start')\n"
            "  'modifier': 'near', 'front', 'behind', 'left', 'right', or null\n\n"
            "Examples:\n"
            "  'go to the orange cone'          → "
            "{\"type\":\"visual_search\",\"target\":\"cone\",\"modifier\":null}\n"
            "  'park near the cone'             → "
            "{\"type\":\"visual_search\",\"target\":\"cone\",\"modifier\":\"near\"}\n"
            "  'drive to the charging station'  → "
            "{\"type\":\"fixed_landmark\",\"target\":\"charging\",\"modifier\":null}\n"
            "  'go to start'                    → "
            "{\"type\":\"fixed_landmark\",\"target\":\"start\",\"modifier\":null}"
        )
        try:
            res = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a robotics intent parser. "
                                "Return only valid JSON, no explanation."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,   # deterministic
            )
            return json.loads(res.choices[0].message.content)
        except Exception as e:
            print(f">>> LLM ERROR: {e}", flush=True)
            # Safe fallback: do nothing
            return {'type': 'fixed_landmark', 'target': 'none', 'modifier': None}

    # ── OWL-ViT object detection ──────────────────────────────

    def _get_bbox(self, img: PILImage.Image, target: str):
        """
        Run OWL-ViT open-vocabulary detection.
        Returns [xmin, ymin, xmax, ymax] in pixels, or None.
        NOTE: On CPU this takes ~3-5s. On vehicle GPU it will be faster.
        """
        search_prompt = f"a photo of a {target}"

        inputs = self.vlm_processor(
            text=[[target]], images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.vlm_model(**inputs)

        results = self.vlm_processor.post_process_grounded_object_detection(
            outputs,
            threshold=OWLVIT_THRESHOLD,
            target_sizes=torch.tensor([img.size[::-1]]).to(self.device)
        )[0]

        if len(results["scores"]) == 0:
            return None

        top_idx = results["scores"].argmax().item()
        score   = results["scores"][top_idx].item()
        print(f">>> OWL-ViT: confidence={score:.3f}", flush=True)

        if score < 0.04: 
            print(f">>> VISION FAIL: Confidence ({score:.3f}) is too low to trust.", flush=True)
            return None

        return results["boxes"][top_idx].tolist()

    # ── LiDAR depth estimation ────────────────────────────────

    def _get_depth_from_lidar(self, bbox, img_shape) -> float:
        """
        Estimate distance to the detected object by finding LiDAR points
        that fall within the horizontal angular range of the bounding box.

        This is much more accurate than a naive "front points" median because
        it only uses points that are actually behind the bounding box pixels.

        Coordinate frame: Ouster x=forward, y=left, z=up (sensor frame).
        The camera and LiDAR are assumed to be co-boresighted (close enough
        for this application given the vehicle size). Fine-tune by measuring
        the Ouster→camera extrinsic offset on-site.
        """
        xmin, ymin, xmax, ymax = bbox
        fx = CAMERA_INTRINSICS['fx']
        cx = CAMERA_INTRINSICS['cx']

        # Convert pixel x-range of bbox to horizontal angle range in LiDAR frame
        # Positive angle = left of center (LiDAR y-axis), negative = right
        angle_left  = math.atan2(cx - xmin, fx)   # left edge of bbox
        angle_right = math.atan2(cx - xmax, fx)   # right edge of bbox
        if angle_left < angle_right:
            angle_left, angle_right = angle_right, angle_left

        pts_list = list(pc2.read_points(
            self.latest_pc, field_names=("x", "y", "z"), skip_nans=True))

        if not pts_list:
            self.get_logger().warn("LiDAR pointcloud is empty. Using fallback depth.")
            return FALLBACK_DEPTH_M

        points = np.array([[p[0], p[1], p[2]] for p in pts_list], dtype=np.float32)
        points = points[~np.isnan(points).any(axis=1)]

        if len(points) == 0:
            return FALLBACK_DEPTH_M

        # Only forward-facing points (x > 0.5m to avoid ego-returns)
        fwd = points[points[:, 0] > 0.5]
        if len(fwd) == 0:
            return FALLBACK_DEPTH_M

        # Angular filter: keep points whose azimuth falls under the bbox
        azimuth = np.arctan2(fwd[:, 1], fwd[:, 0])   # atan2(y, x) in LiDAR frame
        in_slice = (azimuth >= angle_right) & (azimuth <= angle_left)

        # Optional: also filter by vertical angle using bbox ymin/ymax
        # (skipped here — good enough for flat parking lot objects)

        slice_pts = fwd[in_slice]
        if len(slice_pts) < MIN_LIDAR_POINTS:
            self.get_logger().warn(
                f"Only {len(slice_pts)} LiDAR points in bbox slice "
                f"(need {MIN_LIDAR_POINTS}). Using fallback depth.")
            return FALLBACK_DEPTH_M

        # Use 20th percentile of forward distance (x) — closer than median,
        # avoids picking up far background behind the object
        Z = float(np.percentile(slice_pts[:, 0], 20))
        print(f">>> LiDAR depth: {Z:.2f}m  ({len(slice_pts)} pts in slice)",
              flush=True)
        return Z

    def _get_local_goal_from_lidar(self, bbox, img_shape, modifier) -> tuple:
        """
        Project bounding box center + LiDAR depth to local (x, y) in
        vehicle ENU frame (x=forward, y=left).
        """
        xmin, ymin, xmax, ymax = bbox
        u = (xmin + xmax) / 2.0   # pixel column of bbox center

        Z = self._get_depth_from_lidar(bbox, img_shape)

        # Pinhole back-projection
        X_cam = (u - CAMERA_INTRINSICS['cx']) * Z / CAMERA_INTRINSICS['fx']
        # Camera frame: x=right, z=forward → vehicle frame: x=forward, y=left
        x_loc = Z          # forward distance
        y_loc = -X_cam     # lateral (negative because camera x is right)

        # Apply spatial modifier
        if modifier in ('near', 'front'):
            x_loc -= SAFETY_OFFSET_M          # stop short of object
        elif modifier == 'behind':
            x_loc += SAFETY_OFFSET_M          # go past object
        elif modifier == 'left':
            y_loc += SAFETY_OFFSET_M          # offset left
        elif modifier == 'right':
            y_loc -= SAFETY_OFFSET_M          # offset right

        return x_loc, y_loc

    # ── Coordinate transform ──────────────────────────────────

    def _to_global(self, lx: float, ly: float) -> tuple:
        """
        Transform local vehicle-frame (x=forward, y=left) offset to
        absolute ENU map coordinates using GNSS + INS heading.
        """
        # Heading
        if self.latest_yaw is None or math.isnan(self.latest_yaw):
            current_yaw = 0.0
            print(">>> WARNING: INS yaw invalid. Fallback yaw=0.0 (East).",
                  flush=True)
        else:
            current_yaw = self.latest_yaw

        # GNSS validity
        if (self.latest_lat is None
                or math.isnan(self.latest_lat)
                or self.latest_lat == 0.0):
            print(">>> ERROR: GPS latitude invalid!", flush=True)
            return float('nan'), float('nan')

        print(f">>> GPS: lat={self.latest_lat:.7f}  lon={self.latest_lon:.7f}  "
              f"yaw={math.degrees(current_yaw):.1f}°", flush=True)

        try:
            # GPS position in ENU map frame
            ce, cn, _ = pm.geodetic2enu(
                self.latest_lat, self.latest_lon, 0,
                ORIGIN_LAT, ORIGIN_LON, 0)

            # Rotate vehicle-frame offset by heading to get ENU offset
            gx = ce + lx * math.cos(current_yaw) - ly * math.sin(current_yaw)
            gy = cn + lx * math.sin(current_yaw) + ly * math.cos(current_yaw)
            return gx, gy

        except Exception as e:
            print(f">>> TRANSFORM ERROR: {e}", flush=True)
            return float('nan'), float('nan')

    # ── Goal publisher ────────────────────────────────────────

    def _publish_goal(self, x: float, y: float):
        if math.isnan(x) or math.isnan(y):
            print(">>> ERROR: Goal is NaN — not publishing.", flush=True)
            return

        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        # Orientation left as identity — planner derives heading from path tangent
        self.goal_pub.publish(msg)

        print(f"\n[SUCCESS] Goal published → ENU East={x:.2f}m  North={y:.2f}m\n",
              flush=True)
        self.get_logger().info(
            f"Goal published: ({x:.2f}, {y:.2f}) in map frame")


# ─────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = VLMPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()