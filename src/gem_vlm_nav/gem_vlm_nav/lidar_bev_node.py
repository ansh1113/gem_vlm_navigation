#!/usr/bin/env python3
"""
lidar_bev_node.py — LiDAR Bird's-Eye-View Image + Costmap Publisher
=====================================================================
ROS1 (Noetic) simulator port of the original ROS2 lidar_bev_node.

Runs at 5Hz. Produces two outputs every cycle:

1. /lidar_bev_image  (sensor_msgs/Image)
   Top-down BEV image rendered from the Ouster LiDAR pointcloud.
   Vehicle-centred, shows BEV_RANGE_M metres in every direction.

2. /vlm_costmap  (nav_msgs/OccupancyGrid)
   Live binary occupancy grid for A* / obstacle avoidance.
   Covers the full navigatable area at GRID_RES_M resolution.

Topics (in):
  /ouster/points                — Ouster LiDAR pointcloud
  /gps/fix                      — Simulated GPS (sensor_msgs/NavSatFix)
  /septentrio_gnss/insnavgeod   — Simulated INS heading

Topics (out):
  /lidar_bev_image              — BEV image (frame: base_link)
  /vlm_costmap                  — OccupancyGrid (frame: world)

Usage:
  source devel/setup.bash
  python3 lidar_bev_node.py

NOTE: Change X_MIN/X_MAX/Y_MIN/Y_MAX to adjust the navigatable area.
      Everything else derives from those four values automatically.
"""

import math
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import PointCloud2, NavSatFix, Image
from nav_msgs.msg import OccupancyGrid
from septentrio_gnss_driver.msg import INSNavGeod
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import pymap3d as pm

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# ENU origin — matches the Gazebo world's <spherical_coordinates>
# so that GPS ENU coords align with Gazebo world frame coords.
ORIGIN_LAT = 40.0928381
ORIGIN_LON = -88.2356367

# Navigatable area bounds in ENU / Gazebo world frame (metres).
# Change these four values and everything else adapts automatically.
X_MIN = -50
X_MAX =  40
Y_MIN = -12
Y_MAX =  5

# BEV image — vehicle centred, shows BEV_RANGE_M metres around car
BEV_RANGE_M  = 20.0    # metres in each direction from vehicle
BEV_RES_M    = 0.15    # metres per pixel
BEV_SIZE_PX  = int(2 * BEV_RANGE_M / BEV_RES_M)   # square image side (px)

# Costmap grid — full area coverage, coarser resolution for A*
GRID_RES_M   = 0.5
GRID_NC      = int(np.ceil((X_MAX - X_MIN) / GRID_RES_M))   # columns (x)
GRID_NR      = int(np.ceil((Y_MAX - Y_MIN) / GRID_RES_M))   # rows    (y)

# LiDAR height filter (metres above sensor origin)
Z_MIN =  -1.5   # ignore ground returns below this
Z_MAX =  3.00   # ignore ceiling / overhead structure above this

# GEM e4 footprint for BEV overlay
CAR_LENGTH = 2.9   # metres
CAR_WIDTH  = 1.4   # metres

# Publish rate
BEV_HZ = 5.0


# ═══════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════

def ins_heading_to_yaw(h: float) -> float:
    """
    Convert Septentrio INS heading (degrees, 0=North, clockwise)
    to ROS yaw (radians, 0=East, counter-clockwise).
    Matches the original ROS2 conversion exactly.
    """
    return math.radians(90.0 - h) if h < 270.0 else math.radians(450.0 - h)

def world_to_cell(x, y):
    """Convert world-frame (x, y) to costmap (col, row). No bounds check."""
    c = int((x - X_MIN) / GRID_RES_M)
    r = int((y - Y_MIN) / GRID_RES_M)
    return c, r

def in_bounds_grid(c, r):
    return 0 <= r < GRID_NR and 0 <= c < GRID_NC


# ═══════════════════════════════════════════════════════════════
#  NODE
# ═══════════════════════════════════════════════════════════════

class LidarBEVNode:
    def __init__(self):
        rospy.init_node('lidar_bev_node', anonymous=False)

        self.bridge    = CvBridge()
        self.latest_pc = None
        self.car_x     = None
        self.car_y     = None
        self.car_yaw   = 0.0

        # Occupancy grid — pre-fill boundary walls, updated each cycle
        self.grid = self._build_base_grid()

        # ── Subscribers ──────────────────────────────────────
        # Large buff_size for pointcloud — ROS1 drops messages without it
        rospy.Subscriber('/ouster/points', PointCloud2, self.lidar_cb,
                         queue_size=1, buff_size=2**24)

        # GPS → ENU position (same message type as original /navsatfix)
        rospy.Subscriber('/gps/fix', NavSatFix, self.gps_cb,
                         queue_size=1)

        # INS heading (same message type and conversion as original)
        rospy.Subscriber('/septentrio_gnss/insnavgeod', INSNavGeod, self.ins_cb,
                         queue_size=1)

        # ── Publishers ───────────────────────────────────────
        self.bev_pub  = rospy.Publisher('/lidar_bev_image', Image,
                                        queue_size=1)
        self.grid_pub = rospy.Publisher('/vlm_costmap', OccupancyGrid,
                                        queue_size=1, latch=True)

        # ── Timer ────────────────────────────────────────────
        rospy.Timer(rospy.Duration(1.0 / BEV_HZ), self.publish_cycle)

        rospy.loginfo(
            "[lidar_bev] Ready.  BEV: %dpx @ %.2fm/px  Range: +/-%.0fm  "
            "Grid: %dx%d @ %.1fm/cell  Area: x=[%.1f,%.1f] y=[%.1f,%.1f]",
            BEV_SIZE_PX, BEV_RES_M, BEV_RANGE_M,
            GRID_NC, GRID_NR, GRID_RES_M,
            X_MIN, X_MAX, Y_MIN, Y_MAX
        )

    # ── Callbacks ──────────────────────────────────────────────

    def gps_cb(self, msg: NavSatFix):
        """
        Convert GPS lat/lon to ENU using the Gazebo world's spherical
        coordinate origin. This aligns GPS ENU with Gazebo world frame.
        Identical logic to original ROS2 gps_cb.
        """
        try:
            e, n, _ = pm.geodetic2enu(
                msg.latitude, msg.longitude, 0,
                ORIGIN_LAT, ORIGIN_LON, 0)
            self.car_x = float(e)
            self.car_y = float(n)
        except Exception as ex:
            rospy.logwarn_throttle(5.0, "[lidar_bev] GPS conversion failed: %s", str(ex))

    def ins_cb(self, msg: INSNavGeod):
        """
        Extract yaw from INS heading.
        Identical logic to original ROS2 ins_cb.
        """
        if msg.heading is not None and not math.isnan(msg.heading):
            self.car_yaw = ins_heading_to_yaw(msg.heading)

    def lidar_cb(self, msg: PointCloud2):
        """Store latest pointcloud — processed on the publish timer."""
        self.latest_pc = msg

    # ── Main publish cycle ─────────────────────────────────────

    def publish_cycle(self, _event):
        if self.latest_pc is None or self.car_x is None:
            return

        points = self._parse_pointcloud(self.latest_pc)
        if points is None:
            return

        # Height filter
        obs_mask = (points[:, 2] > Z_MIN) & (points[:, 2] < Z_MAX)
        obs_pts  = points[obs_mask]

        # Transform sensor-frame points → ENU world frame
        map_pts = self._sensor_to_map(obs_pts)

        # Update costmap with new obstacles
        self._update_grid(map_pts)

        # Render and publish BEV image
        bev_img = self._render_bev(map_pts)
        self._publish_bev(bev_img)

        # Publish costmap
        self._publish_costmap()

    # ── Pointcloud parsing ─────────────────────────────────────

    def _parse_pointcloud(self, msg: PointCloud2):
        pts_list = list(pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True))
        if not pts_list:
            return None
        pts = np.array([[p[0], p[1], p[2]] for p in pts_list], dtype=np.float32)
        pts = pts[~np.isnan(pts).any(axis=1)]
        return pts if len(pts) > 0 else None

    # ── Coordinate transform ───────────────────────────────────

    def _sensor_to_map(self, pts: np.ndarray) -> np.ndarray:
        """
        Transform Ouster sensor-frame points → ENU map frame.
        Negates X and Y to correct the 180° backwards physical mount.
        Identical logic to original ROS2 _sensor_to_map.
        """
        cos_y = math.cos(self.car_yaw)
        sin_y = math.sin(self.car_yaw)

        # Negate X and Y to correct backwards physical mount
        pts_x = -pts[:, 0]
        pts_y = -pts[:, 1]

        map_x = self.car_x + pts_x * cos_y - pts_y * sin_y
        map_y = self.car_y + pts_x * sin_y + pts_y * cos_y
        map_z = pts[:, 2]

        return np.column_stack([map_x, map_y, map_z])

    # ── Costmap ────────────────────────────────────────────────

    def _build_base_grid(self) -> np.ndarray:
        """
        Build static base grid with 1m boundary walls pre-filled.
        Driven entirely by X_MIN/X_MAX/Y_MIN/Y_MAX and GRID_RES_M.
        """
        g = np.zeros((GRID_NR, GRID_NC), dtype=np.uint8)
        w = max(1, int(1.0 / GRID_RES_M))
        g[:w,  :]  = 1   # south wall
        g[-w:, :]  = 1   # north wall
        g[:,  :w]  = 1   # west  wall
        g[:, -w:]  = 1   # east  wall
        return g

    def _update_grid(self, map_pts: np.ndarray):
        """
        Clear interior dynamic obstacles then re-fill from current scan.
        Boundary walls are never cleared.
        """
        wall = max(1, int(1.0 / GRID_RES_M))
        self.grid[wall:-wall, wall:-wall] = 0
        for p in map_pts:
            c, r = world_to_cell(p[0], p[1])
            if in_bounds_grid(c, r):
                self.grid[r, c] = 1

    # ── BEV image rendering ────────────────────────────────────

    def _render_bev(self, map_pts: np.ndarray) -> np.ndarray:
        """
        Render top-down BEV image centred on the vehicle.

        Convention:
          - Vehicle at image centre, pointing UP (forward = up in image)
          - Each pixel = BEV_RES_M metres
          - Obstacles: coloured by height (orange=low, white/blue=high)
          - Free space: dark blue-grey background
          - Vehicle footprint: green rectangle at centre
          - Navigatable area boundary rectangle
          - Forward arrow, FWD label, scale bar, distance rings

        All pixel math derives from BEV_RANGE_M and BEV_RES_M.
        Boundary rectangle derives from X_MIN/X_MAX/Y_MIN/Y_MAX.
        """
        img = np.zeros((BEV_SIZE_PX, BEV_SIZE_PX, 3), dtype=np.uint8)
        img[:, :] = (20, 25, 40)   # dark blue-grey background

        cx_px = BEV_SIZE_PX // 2
        cy_px = BEV_SIZE_PX // 2

        # Rotation: ENU world frame → vehicle-centred image frame
        cos_y = math.cos(-self.car_yaw)
        sin_y = math.sin(-self.car_yaw)

        # ── Obstacle points ──────────────────────────────────
        for p in map_pts:
            dx = p[0] - self.car_x
            dy = p[1] - self.car_y

            v_forward = dx * cos_y - dy * sin_y
            v_left    = dx * sin_y + dy * cos_y

            px = int(cx_px - v_left    / BEV_RES_M)
            py = int(cy_px - v_forward / BEV_RES_M)

            if 0 <= px < BEV_SIZE_PX and 0 <= py < BEV_SIZE_PX:
                z_norm = max(0.0, min(1.0, (p[2] - Z_MIN) / (Z_MAX - Z_MIN)))
                b   = int(200 + 55 * z_norm)
                g_c = int(150 + 50 * (1 - z_norm))
                r_c = int(80  + 80 * (1 - z_norm))
                cv2.circle(img, (px, py), 1, (r_c, g_c, b), -1)

        # ── Navigatable area boundary rectangle ──────────────
        corners_world = [
            (X_MIN, Y_MIN), (X_MAX, Y_MIN),
            (X_MAX, Y_MAX), (X_MIN, Y_MAX),
        ]
        corners_px = []
        for (wx, wy) in corners_world:
            dx = wx - self.car_x
            dy = wy - self.car_y
            v_forward = dx * cos_y - dy * sin_y
            v_left    = dx * sin_y + dy * cos_y
            px = int(cx_px - v_left    / BEV_RES_M)
            py = int(cy_px - v_forward / BEV_RES_M)
            corners_px.append((px, py))
        cv2.polylines(img, [np.array(corners_px, dtype=np.int32)],
                      True, (80, 120, 160), 1)

        # ── Vehicle footprint ────────────────────────────────
        car_l_px = int(CAR_LENGTH / BEV_RES_M)
        car_w_px = int(CAR_WIDTH  / BEV_RES_M)
        car_rect = np.array([
            [-car_w_px // 2, -car_l_px // 2],
            [ car_w_px // 2, -car_l_px // 2],
            [ car_w_px // 2,  car_l_px // 2],
            [-car_w_px // 2,  car_l_px // 2],
        ]) + np.array([cx_px, cy_px])
        cv2.fillPoly(img,  [car_rect.astype(np.int32)], (40, 200, 80))
        cv2.polylines(img, [car_rect.astype(np.int32)], True, (255, 255, 255), 1)

        # ── Forward direction arrow ──────────────────────────
        arrow_len = int(1.5 / BEV_RES_M)
        cv2.arrowedLine(img,
            (cx_px, cy_px),
            (cx_px, cy_px - arrow_len),
            (255, 255, 100), 2, tipLength=0.3)

        # ── FWD label ────────────────────────────────────────
        cv2.putText(img, "FWD", (cx_px - 15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)

        # ── Scale bar (5m) ───────────────────────────────────
        bar_px = int(5.0 / BEV_RES_M)
        bar_y  = BEV_SIZE_PX - 15
        bar_x0 = 15
        cv2.line(img, (bar_x0, bar_y), (bar_x0 + bar_px, bar_y),
                 (200, 200, 200), 2)
        cv2.putText(img, "5m", (bar_x0 + bar_px + 5, bar_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # ── Distance rings ───────────────────────────────────
        for ring_m in [5.0, 10.0]:
            r_px = int(ring_m / BEV_RES_M)
            cv2.circle(img, (cx_px, cy_px), r_px, (50, 60, 80), 1)
            cv2.putText(img, f"{int(ring_m)}m",
                        (cx_px + r_px + 2, cy_px),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 100, 120), 1)

        return img

    # ── Publishers ─────────────────────────────────────────────

    def _publish_bev(self, img: np.ndarray):
        try:
            msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            msg.header.stamp    = rospy.Time.now()
            msg.header.frame_id = 'base_link'
            self.bev_pub.publish(msg)
        except Exception as e:
            rospy.logwarn("[lidar_bev] BEV publish failed: %s", str(e))

    def _publish_costmap(self):
        msg = OccupancyGrid()
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = 'world'

        msg.info.resolution = float(GRID_RES_M)
        msg.info.width      = GRID_NC
        msg.info.height     = GRID_NR

        # Origin = SW corner of the navigatable area in world frame
        # Adapts automatically when X_MIN/Y_MIN change
        msg.info.origin.position.x    = float(X_MIN)
        msg.info.origin.position.y    = float(Y_MIN)
        msg.info.origin.position.z    = 0.0
        msg.info.origin.orientation.w = 1.0

        msg.data = (self.grid * 100).astype(np.int8).flatten().tolist()
        self.grid_pub.publish(msg)


# ═══════════════════════════════════════════════════════════════

def main():
    node = LidarBEVNode()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
