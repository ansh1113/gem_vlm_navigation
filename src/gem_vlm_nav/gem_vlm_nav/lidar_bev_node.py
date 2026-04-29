#!/usr/bin/env python3
"""
lidar_bev_node.py — LiDAR Bird's-Eye-View Image + Costmap Publisher
=====================================================================
Runs at 5Hz. Produces two outputs every cycle:

1. /lidar_bev_image  (sensor_msgs/Image)
   A top-down grayscale image of the parking lot rendered from the
   Ouster LiDAR pointcloud. Passed directly to the VLM as a second
   camera input giving 360° spatial context.

   Rendering:
     - Slice pointcloud at obstacle height (0.1–2.0m above sensor)
     - Project each point onto 2D grid from above
     - Intensity = point density (brighter = more returns = solid object)
     - Vehicle footprint drawn at center as white rectangle
     - North arrow + scale bar overlaid for VLM spatial orientation

2. /vlm_costmap  (nav_msgs/OccupancyGrid)
   Live binary occupancy grid consumed by planner_node for A* and
   obstacle avoidance. Separate from the BEV image — finer resolution.

Topics (in):
  /ouster/points   — Ouster OS1-128 LiDAR pointcloud
  /navsatfix       — Septentrio GNSS (for ENU position)
  /insnavgeod      — Septentrio INS heading

Topics (out):
  /lidar_bev_image — Bird's-eye-view image for VLM
  /vlm_costmap     — OccupancyGrid for path planner
"""

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, NavSatFix, Image
from nav_msgs.msg import OccupancyGrid
from septentrio_gnss_driver.msg import INSNavGeod
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import pymap3d as pm
import scipy.ndimage as ndimage

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

ORIGIN_LAT = 40.0927422
ORIGIN_LON = -88.2359639

# Operating area
X_MIN, X_MAX = -25.0, 75.0
Y_MIN, Y_MAX =  -5.0, 20.0

# BEV image — vehicle centred, shows SENSOR_RANGE meters around car
# Smaller range = more detail around vehicle for VLM
BEV_RANGE_M   = 20.0   # meters in each direction from vehicle
BEV_RES_M     = 0.15   # meters per pixel — finer than costmap
BEV_SIZE_PX   = int(2 * BEV_RANGE_M / BEV_RES_M)   # square image

# Costmap grid — full lot coverage, coarser for A*
GRID_RES_M    = 0.5
GRID_NC       = int(np.ceil((X_MAX - X_MIN) / GRID_RES_M))
GRID_NR       = int(np.ceil((Y_MAX - Y_MIN) / GRID_RES_M))

# LiDAR height filter
Z_MIN =  0.10   # m — ignore ground returns
Z_MAX =  2.50   # m — ignore ceiling / overhead structure

# GEM e4 footprint for BEV overlay
CAR_LENGTH = 2.9
CAR_WIDTH  = 1.4

# BEV publish rate
BEV_HZ = 5


# ═══════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════

def ins_heading_to_yaw(h: float) -> float:
    return math.radians(90.0 - h) if h < 270.0 else math.radians(450.0 - h)

def world_to_cell(x, y):
    c = int((x - X_MIN) / GRID_RES_M)
    r = int((y - Y_MIN) / GRID_RES_M)
    return c, r

def cell_to_world(c, r):
    return X_MIN + (c + 0.5) * GRID_RES_M, Y_MIN + (r + 0.5) * GRID_RES_M

def in_bounds_grid(c, r):
    return 0 <= r < GRID_NR and 0 <= c < GRID_NC


# ═══════════════════════════════════════════════════════════════
#  NODE
# ═══════════════════════════════════════════════════════════════

class LidarBEVNode(Node):
    def __init__(self):
        super().__init__('lidar_bev_node')

        self.bridge     = CvBridge()
        self.latest_pc  = None
        self.car_x      = None
        self.car_y      = None
        self.car_yaw    = 0.0

        # Occupancy grid — pre-fill boundary walls
        self.grid = self._build_base_grid()

        # Subscriptions
        self.create_subscription(
            PointCloud2, '/ouster/points', self.lidar_cb, 10)
        self.create_subscription(
            NavSatFix, '/navsatfix', self.gps_cb, 10)
        self.create_subscription(
            INSNavGeod, '/insnavgeod', self.ins_cb, 10)

        # Publishers
        self.bev_pub  = self.create_publisher(Image,         '/lidar_bev_image', 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/vlm_costmap',     10)

        # Timer
        self.create_timer(1.0 / BEV_HZ, self.publish_cycle)

        self.get_logger().info(
            f"LiDAR BEV node ready. "
            f"BEV: {BEV_SIZE_PX}x{BEV_SIZE_PX}px @ {BEV_RES_M}m/px  "
            f"Range: ±{BEV_RANGE_M}m around vehicle")

    # ── Callbacks ─────────────────────────────────────────────

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
            self.car_yaw = ins_heading_to_yaw(msg.heading)

    def lidar_cb(self, msg: PointCloud2):
        # Store latest — processed on publish cycle
        self.latest_pc = msg

    # ── Main publish cycle ────────────────────────────────────

    def publish_cycle(self):
        if self.latest_pc is None or self.car_x is None:
            return

        points = self._parse_pointcloud(self.latest_pc)
        if points is None:
            return

        # Height filter
        obs_mask = (points[:, 2] > Z_MIN) & (points[:, 2] < Z_MAX)
        obs_pts  = points[obs_mask]

        # Transform sensor-frame points → ENU map frame
        map_pts = self._sensor_to_map(obs_pts)

        # Update costmap grid
        self._update_grid(map_pts)

        # Generate and publish BEV image
        bev_img = self._render_bev(map_pts)
        self._publish_bev(bev_img)

        # Publish costmap
        self._publish_costmap()

    # ── Pointcloud parsing ────────────────────────────────────

    def _parse_pointcloud(self, msg: PointCloud2):
        pts_list = list(pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True))
        if not pts_list:
            return None
        pts = np.array([[p[0], p[1], p[2]] for p in pts_list], dtype=np.float32)
        pts = pts[~np.isnan(pts).any(axis=1)]
        return pts if len(pts) > 0 else None

    # ── Coordinate transform ──────────────────────────────────

    def _sensor_to_map(self, pts: np.ndarray) -> np.ndarray:
        """Transform Ouster sensor-frame points to ENU map frame."""
        cos_y = math.cos(self.car_yaw)
        sin_y = math.sin(self.car_yaw)
        
        # --- FIX: 180-Degree LiDAR Flip ---
        # Negating the X and Y axes to correct the backwards physical mount
        pts_x = -pts[:, 0]
        pts_y = -pts[:, 1]
        
        # Vectorised rotation + translation using the flipped points
        map_x = self.car_x + pts_x * cos_y - pts_y * sin_y
        map_y = self.car_y + pts_x * sin_y + pts_y * cos_y
        map_z = pts[:, 2]
        
        return np.column_stack([map_x, map_y, map_z])

    # ── Costmap update ────────────────────────────────────────

    def _build_base_grid(self) -> np.ndarray:
        g = np.zeros((GRID_NR, GRID_NC), dtype=np.uint8)
        w = max(1, int(1.0 / GRID_RES_M))
        g[:w, :] = g[-w:, :] = g[:, :w] = g[:, -w:] = 1
        return g

    def _update_grid(self, map_pts: np.ndarray):
        """Refresh dynamic obstacles — keep boundary walls."""
        wall = max(1, int(1.0 / GRID_RES_M))
        self.grid[wall:-wall, wall:-wall] = 0
        for p in map_pts:
            c, r = world_to_cell(p[0], p[1])
            if in_bounds_grid(c, r):
                self.grid[r, c] = 1

    # ── BEV image rendering ───────────────────────────────────

    def _render_bev(self, map_pts: np.ndarray) -> np.ndarray:
        """
        Render a top-down BEV image centred on the vehicle.

        Convention (important for VLM prompt):
          - Vehicle is at image centre, pointing UP (North in image = vehicle forward)
          - Each pixel = BEV_RES_M meters
          - Obstacles: bright white
          - Free space: dark background
          - Vehicle footprint: green rectangle at centre
          - Compass rose + scale bar overlaid
        """
        img = np.zeros((BEV_SIZE_PX, BEV_SIZE_PX, 3), dtype=np.uint8)
        img[:, :] = (20, 25, 40)   # dark blue-grey background

        cx_px = BEV_SIZE_PX // 2
        cy_px = BEV_SIZE_PX // 2

        # Rotate map points into vehicle-centred frame
        # Vehicle frame: forward=up in image, right=right in image
        cos_y = math.cos(-self.car_yaw)   # negative = rotate world to vehicle frame
        sin_y = math.sin(-self.car_yaw)

        # Draw obstacle points
        for p in map_pts:
            # Relative position in ENU
            dx = p[0] - self.car_x
            dy = p[1] - self.car_y
            
            # FIX: Convert to vehicle base_link frame (x=forward, y=left)
            v_forward = dx * cos_y - dy * sin_y
            v_left    = dx * sin_y + dy * cos_y

            # FIX: Map to image pixels (image x=right, image y=down)
            px = int(cx_px - v_left / BEV_RES_M)
            py = int(cy_px - v_forward / BEV_RES_M)

            if 0 <= px < BEV_SIZE_PX and 0 <= py < BEV_SIZE_PX:
                # Colour by height: low=orange, high=white
                z_norm = max(0.0, min(1.0, (p[2] - Z_MIN) / (Z_MAX - Z_MIN)))
                b = int(200 + 55 * z_norm)
                g_c = int(150 + 50 * (1 - z_norm))
                r_c = int(80  + 80 * (1 - z_norm))
                # CHANGE TO RADIUS 1 to make it look less chunky
                cv2.circle(img, (px, py), 1, (r_c, g_c, b), -1)

        # Draw lot boundary as thin rectangle
        # Convert lot corners to vehicle frame
        corners_enu = [
            (X_MIN, Y_MIN), (X_MAX, Y_MIN),
            (X_MAX, Y_MAX), (X_MIN, Y_MAX)]
        corners_px  = []
        for (ex, ey) in corners_enu:
            dx = ex - self.car_x; dy = ey - self.car_y
            # FIX: Apply same exact rotation mapping here
            v_forward = dx * cos_y - dy * sin_y
            v_left    = dx * sin_y + dy * cos_y
            px = int(cx_px - v_left / BEV_RES_M)
            py = int(cy_px - v_forward / BEV_RES_M)
            corners_px.append((px, py))
        cv2.polylines(img, [np.array(corners_px)], True, (80, 120, 160), 1)

        # Draw vehicle footprint at centre (green rectangle)
        car_l_px = int(CAR_LENGTH / BEV_RES_M)
        car_w_px = int(CAR_WIDTH  / BEV_RES_M)
        car_rect = np.array([
            [-car_w_px//2, -car_l_px//2],
            [ car_w_px//2, -car_l_px//2],
            [ car_w_px//2,  car_l_px//2],
            [-car_w_px//2,  car_l_px//2],
        ]) + np.array([cx_px, cy_px])
        cv2.fillPoly(img, [car_rect.astype(np.int32)], (40, 200, 80))
        cv2.polylines(img, [car_rect.astype(np.int32)], True, (255, 255, 255), 1)

        # Forward direction arrow (vehicle points up)
        arrow_len = int(1.5 / BEV_RES_M)
        cv2.arrowedLine(img,
            (cx_px, cy_px),
            (cx_px, cy_px - arrow_len),
            (255, 255, 100), 2, tipLength=0.3)

        # Compass: "FWD" label at top
        cv2.putText(img, "FWD", (cx_px - 15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)

        # Scale bar: 5m
        bar_px  = int(5.0 / BEV_RES_M)
        bar_y   = BEV_SIZE_PX - 15
        bar_x0  = 15
        cv2.line(img, (bar_x0, bar_y), (bar_x0 + bar_px, bar_y),
                 (200, 200, 200), 2)
        cv2.putText(img, "5m", (bar_x0 + bar_px + 5, bar_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Distance rings at 5m and 10m
        for ring_m in [5.0, 10.0]:
            r_px = int(ring_m / BEV_RES_M)
            cv2.circle(img, (cx_px, cy_px), r_px,
                       (50, 60, 80), 1)
            cv2.putText(img, f"{int(ring_m)}m",
                        (cx_px + r_px + 2, cy_px),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 100, 120), 1)

        return img

    # ── Publishers ────────────────────────────────────────────

    def _publish_bev(self, img: np.ndarray):
        try:
            msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            self.bev_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"BEV publish failed: {e}")

    def _publish_costmap(self):
        msg = OccupancyGrid()
        msg.header.stamp             = self.get_clock().now().to_msg()
        msg.header.frame_id          = 'map'
        msg.info.resolution          = float(GRID_RES_M)
        msg.info.width               = GRID_NC
        msg.info.height              = GRID_NR
        msg.info.origin.position.x   = float(X_MIN)
        msg.info.origin.position.y   = float(Y_MIN)
        msg.info.origin.orientation.w = 1.0
        msg.data = (self.grid * 100).astype(np.int8).flatten().tolist()
        self.grid_pub.publish(msg)


# ═══════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = LidarBEVNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()