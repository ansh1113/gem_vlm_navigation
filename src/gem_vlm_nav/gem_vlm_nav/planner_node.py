#!/usr/bin/env python3
"""
planner_node.py — VLM Path Planning + PACMod Control Node for GEM e4
======================================================================
Drop-in replacement for pure_pursuit_ros2.py.

Replaces:  CSV waypoint source + fixed pure pursuit loop
With:      VLM goal → A* planner → smoothed path → pure pursuit → PACMod

The low-level PACMod interface is IDENTICAL to pure_pursuit_ros2.py:
  - Same publishers, same message types, same topic names
  - Same joystick enable/disable logic (LB+RB to enable, LB alone to disable)
  - Same front2steer() steering wheel angle conversion
  - Same PID speed controller with Butterworth speed filter
  - Same INSNavGeod heading conversion formula

Topics (in):
  /navsatfix                  — Septentrio GNSS position
  /insnavgeod                 — Septentrio INS heading (degrees)
  /ouster/points              — Ouster OS1-128 LiDAR pointcloud
  /pacmod/enabled             — PACMod enable status
  /pacmod/vehicle_speed_rpt   — Measured vehicle speed
  /vlm_goal                   — PoseStamped ENU goal from perception_node

Topics (out):  [identical to pure_pursuit_ros2.py]
  /pacmod/global_cmd          — Enable/disable PACMod
  /pacmod/shift_cmd           — Gear (DRIVE=3)
  /pacmod/steering_cmd        — Steering wheel angle + rate
  /pacmod/accel_cmd           — Throttle [0, max_accel]
  /pacmod/brake_cmd           — Brake [0, 1]
  /pacmod/turn_cmd            — Turn signal

Visualization (optional, view in RViz):
  /vlm_path                   — nav_msgs/Path of planned waypoints
  /vlm_costmap                — nav_msgs/OccupancyGrid live obstacle map
"""

import os
import math
import heapq

import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
from scipy.interpolate import splprep, splev
import pymap3d as pm
import pygame

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from std_msgs.msg import Bool
from sensor_msgs.msg import NavSatFix, PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, OccupancyGrid
import sensor_msgs_py.point_cloud2 as pc2

# PACMod messages — same as pure_pursuit_ros2.py
from pacmod2_msgs.msg import (
    PositionWithSpeed,
    VehicleSpeedRpt,
    GlobalCmd,
    SystemCmdFloat,
    SystemCmdInt,
)
from septentrio_gnss_driver.msg import INSNavGeod


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION  (mirrors pure_pursuit_ros2.py parameter defaults)
# ═══════════════════════════════════════════════════════════════

# ENU origin — must match perception_node and pure pursuit controller
ORIGIN_LAT  = 40.0927422
ORIGIN_LON  = -88.2359639

# Highbay operating area (meters ENU)
X_MIN, X_MAX = -25.0, 75.0
Y_MIN, Y_MAX =  -5.0, 20.0

# Occupancy grid
RESOLUTION  = 0.5    # m/cell
INFLATE_R   = 1.5    # obstacle inflation radius (m); 3 cells @ 0.5m

# LiDAR height filter
LIDAR_Z_MIN =  0.1   # m — ignore ground returns
LIDAR_Z_MAX =  2.5   # m — ignore ceiling / overhead structure

# Pure pursuit (mirrors pure_pursuit_ros2.py defaults)
LOOK_AHEAD   = 5.0   # m base lookahead (same as 'look_ahead' parameter)
WHEELBASE    = 2.57  # m
OFFSET       = 1.26  # m — GNSS antenna offset from rear axle (same as original)
DESIRED_SPEED = 2.0  # m/s (capped at 5.0 in original)
MAX_ACCEL    = 0.5   # throttle limit (capped at 2.0 in original)
GOAL_TOL     = 1.5   # m — arrival tolerance

# PID speed controller (same defaults as pure_pursuit_ros2.py)
PID_KP = 0.6
PID_KI = 0.0
PID_KD = 0.1
PID_WG = 10.0

# Speed filter (Butterworth low-pass, same as original)
FILTER_CUTOFF = 1.2
FILTER_FS     = 30
FILTER_ORDER  = 4

# Steering wheel angle conversion constants (front2steer, same as original)
STEER_A = -0.1084   # quadratic coefficient
STEER_B = 21.775    # linear coefficient
STEER_MAX_DEG = 35  # max front wheel angle (degrees)

# PACMod gear commands
GEAR_NEUTRAL = 2
GEAR_DRIVE   = 3


# ═══════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════

def ins_heading_to_yaw(heading_deg: float) -> float:
    """
    Convert Septentrio heading (0=North, CW, degrees) to ENU yaw
    (0=East, CCW, radians). Exact copy of heading_to_yaw() in
    pure_pursuit_ros2.py.
    """
    if heading_deg < 270.0:
        return math.radians(90.0 - heading_deg)
    else:
        return math.radians(450.0 - heading_deg)

def normalize_angle(a: float) -> float:
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a

def world_to_cell(x, y):
    return int((x - X_MIN) / RESOLUTION), int((y - Y_MIN) / RESOLUTION)

def cell_to_world(col, row):
    return (X_MIN + (col + 0.5) * RESOLUTION,
            Y_MIN + (row + 0.5) * RESOLUTION)

def in_bounds(col, row, shape):
    return 0 <= row < shape[0] and 0 <= col < shape[1]


# ═══════════════════════════════════════════════════════════════
#  PID CONTROLLER  (copied verbatim from pure_pursuit_ros2.py)
# ═══════════════════════════════════════════════════════════════

class PID:
    def __init__(self, kp, ki, kd, wg=None):
        self.kp = kp; self.ki = ki; self.kd = kd; self.wg = wg
        self.iterm = 0; self.last_e = 0; self.last_t = None

    def reset(self):
        self.iterm = 0; self.last_e = 0; self.last_t = None

    def get_control(self, t, e):
        if self.last_t is None:
            dt = de = 0.0
        else:
            dt = t - self.last_t
            de = (e - self.last_e) / dt if dt > 0.0 else 0.0
        self.iterm += e * dt
        if self.wg is not None:
            self.iterm = max(min(self.iterm, self.wg), -self.wg)
        self.last_e = e
        self.last_t = t
        return self.kp * e + self.ki * self.iterm + self.kd * de


# ═══════════════════════════════════════════════════════════════
#  BUTTERWORTH SPEED FILTER  (copied verbatim from pure_pursuit_ros2.py)
# ═══════════════════════════════════════════════════════════════

class OnlineFilter:
    def __init__(self, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = signal.butter(
            order, normal_cutoff, btype='low', analog=False)
        self.z = signal.lfilter_zi(self.b, self.a)

    def get_data(self, data):
        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filted[0]


# ═══════════════════════════════════════════════════════════════
#  OCCUPANCY GRID
# ═══════════════════════════════════════════════════════════════

def build_base_grid() -> np.ndarray:
    """Grid pre-filled with 1m boundary walls so A* never plans outside lot."""
    nc   = int(np.ceil((X_MAX - X_MIN) / RESOLUTION))
    nr   = int(np.ceil((Y_MAX - Y_MIN) / RESOLUTION))
    grid = np.zeros((nr, nc), dtype=np.uint8)
    w    = max(1, int(1.0 / RESOLUTION))
    grid[:w, :] = grid[-w:, :] = grid[:, :w] = grid[:, -w:] = 1
    return grid

def build_inflated_grid(grid: np.ndarray, radius_m: float) -> np.ndarray:
    """Circular inflation — ensures paths clear obstacles by at least radius_m."""
    r      = max(1, int(radius_m / RESOLUTION))
    struct = np.zeros((2*r+1, 2*r+1), dtype=bool)
    for i in range(2*r+1):
        for j in range(2*r+1):
            if (i - r)**2 + (j - r)**2 <= r**2:
                struct[i, j] = True
    return ndimage.binary_dilation(
        grid.astype(bool), structure=struct).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
#  A* PATH PLANNER
# ═══════════════════════════════════════════════════════════════

def astar(grid: np.ndarray, start_w: tuple, goal_w: tuple):
    sc = world_to_cell(*start_w)
    gc = world_to_cell(*goal_w)

    def snap_free(col, row):
        if in_bounds(col, row, grid.shape) and grid[row, col] == 0:
            return col, row
        for r in range(1, 15):
            for dc in range(-r, r+1):
                for dr in range(-r, r+1):
                    nc2, nr2 = col+dc, row+dr
                    if in_bounds(nc2, nr2, grid.shape) and grid[nr2, nc2] == 0:
                        return nc2, nr2
        return col, row

    sc = snap_free(*sc)
    gc = snap_free(*gc)
    if not in_bounds(*sc, grid.shape) or not in_bounds(*gc, grid.shape):
        return None

    nbrs = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),
            (1,1,1.414),(1,-1,1.414),(-1,1,1.414),(-1,-1,1.414)]
    heap   = []
    heapq.heappush(heap, (0.0, sc))
    came   = {}
    gscore = {sc: 0.0}

    while heap:
        _, cur = heapq.heappop(heap)
        if cur == gc:
            path = []
            while cur in came:
                path.append(cell_to_world(*cur))
                cur = came[cur]
            path.append(cell_to_world(*sc))
            path.reverse()
            return path
        col, row = cur
        for dc, dr, cost in nbrs:
            nb = (col+dc, row+dr)
            if not in_bounds(nb[0], nb[1], grid.shape): continue
            if grid[nb[1], nb[0]] == 1:               continue
            t = gscore[cur] + cost
            if t < gscore.get(nb, float('inf')):
                came[nb]   = cur
                gscore[nb] = t
                heapq.heappush(heap,
                    (t + math.hypot(nb[0]-gc[0], nb[1]-gc[1]), nb))
    return None

def smooth_path(path, n_points: int = 200):
    if path is None or len(path) < 3: return path
    pts = np.array(path)
    _, idx = np.unique(pts, axis=0, return_index=True)
    pts = pts[np.sort(idx)]
    if len(pts) < 3: return path
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=2.0, k=min(3, len(pts)-1))
        xs, ys = splev(np.linspace(0, 1, n_points), tck)
        return list(zip(xs.tolist(), ys.tolist()))
    except Exception:
        return path


# ═══════════════════════════════════════════════════════════════
#  PLANNER NODE
# ═══════════════════════════════════════════════════════════════

class VLMPlannerNode(Node):
    def __init__(self):
        super().__init__('vlm_planner_node')

        # ── Joystick (same init as pure_pursuit_ros2.py) ───────
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            self.get_logger().warn(
                "No joystick detected — vehicle enable/disable will not work!")
            self._joystick = None
        else:
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            self.get_logger().info("Joystick ready.")

        # ── Speed controller (same as pure_pursuit_ros2.py) ────
        self.pid_speed   = PID(PID_KP, PID_KI, PID_KD, wg=PID_WG)
        self.speed_filter = OnlineFilter(FILTER_CUTOFF, FILTER_FS, FILTER_ORDER)

        # ── Vehicle state ──────────────────────────────────────
        self.lat          = 0.0
        self.lon          = 0.0
        self.heading      = 0.0   # raw degrees from INSNavGeod (same as PP)
        self.speed        = 0.0   # filtered m/s from VehicleSpeedRpt
        self.pacmod_enable = False

        # Car ENU position + yaw (derived from lat/lon/heading)
        self.car_x   = None
        self.car_y   = None
        self.car_yaw = 0.0

        # ── Planning state ─────────────────────────────────────
        self.global_path = None
        self.target_idx  = 0

        # ── Occupancy grid ─────────────────────────────────────
        self.base_grid = build_base_grid()
        self.grid      = self.base_grid.copy()

        # ── PACMod command objects (same as pure_pursuit_ros2.py) ─
        self.global_cmd = GlobalCmd(enable=False, clear_override=True)
        self.gear_cmd   = SystemCmdInt(command=GEAR_NEUTRAL)
        self.brake_cmd  = SystemCmdFloat(command=0.0)
        self.accel_cmd  = SystemCmdFloat(command=0.0)
        self.turn_cmd   = SystemCmdInt(command=1)   # no turn signal
        self.steer_cmd  = PositionWithSpeed(
            angular_position=0.0, angular_velocity_limit=4.0)

        # ── Subscriptions ──────────────────────────────────────
        self.create_subscription(NavSatFix, '/navsatfix',         self.gnss_cb,   10)
        self.create_subscription(INSNavGeod, '/insnavgeod',       self.ins_cb,    10)
        self.create_subscription(Bool, '/pacmod/enabled',         self.enable_cb, 10)
        self.create_subscription(VehicleSpeedRpt,
                                 '/pacmod/vehicle_speed_rpt',     self.speed_cb,  10)
        self.create_subscription(PointCloud2, '/ouster/points',   self.lidar_cb,  10)

        # Goal from perception node — TRANSIENT_LOCAL so we don't miss it
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseStamped, '/vlm_goal', self.goal_cb, qos)

        # ── PACMod publishers (same topics/types as pure_pursuit_ros2.py) ─
        self.global_pub = self.create_publisher(GlobalCmd,         '/pacmod/global_cmd',   10)
        self.gear_pub   = self.create_publisher(SystemCmdInt,      '/pacmod/shift_cmd',    10)
        self.brake_pub  = self.create_publisher(SystemCmdFloat,    '/pacmod/brake_cmd',    10)
        self.accel_pub  = self.create_publisher(SystemCmdFloat,    '/pacmod/accel_cmd',    10)
        self.turn_pub   = self.create_publisher(SystemCmdInt,      '/pacmod/turn_cmd',     10)
        self.steer_pub  = self.create_publisher(PositionWithSpeed, '/pacmod/steering_cmd', 10)

        # ── Visualization publishers (RViz) ────────────────────
        self.path_pub = self.create_publisher(Path,         '/vlm_path',    10)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/vlm_costmap', 10)

        # ── Control timer (20 Hz — same as pure_pursuit_ros2.py) ─
        self.create_timer(1.0 / 20.0, self.control_loop)

        self.get_logger().info(
            "VLM Planner ready. "
            "Waiting for /vlm_goal — use joystick LB+RB to enable vehicle.")

    # ══════════════════════════════════════════════════════════
    #  SENSOR CALLBACKS
    # ══════════════════════════════════════════════════════════

    def gnss_cb(self, msg: NavSatFix):
        """Store raw lat/lon — same as gnss_callback in pure_pursuit_ros2.py."""
        self.lat = msg.latitude
        self.lon = msg.longitude
        # Also update ENU position for planner use
        try:
            e, n, _ = pm.geodetic2enu(
                self.lat, self.lon, 0, ORIGIN_LAT, ORIGIN_LON, 0)
            self.car_x = float(e)
            self.car_y = float(n)
        except Exception:
            pass

    def ins_cb(self, msg: INSNavGeod):
        """
        Store raw heading degrees — same as ins_callback in pure_pursuit_ros2.py.
        Also compute ENU yaw for planner.
        """
        self.heading = msg.heading
        if msg.heading is not None and not math.isnan(msg.heading):
            self.car_yaw = ins_heading_to_yaw(msg.heading)

    def speed_cb(self, msg: VehicleSpeedRpt):
        """Filtered speed — same as speed_callback in pure_pursuit_ros2.py."""
        self.speed = self.speed_filter.get_data(msg.vehicle_speed)

    def enable_cb(self, msg: Bool):
        """PACMod enable status — same as enable_callback in pure_pursuit_ros2.py."""
        self.pacmod_enable = msg.data

    def lidar_cb(self, msg: PointCloud2):
        """
        Build dynamic occupancy grid from LiDAR scan.
        Transform points from Ouster sensor frame to ENU map frame
        using current GNSS position + INS heading.
        """
        if self.car_x is None:
            return

        pts_list = list(pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True))
        if not pts_list:
            return

        points = np.array(
            [[p[0], p[1], p[2]] for p in pts_list], dtype=np.float32)
        points = points[~np.isnan(points).any(axis=1)]
        if len(points) == 0:
            return

        # Height filter — remove ground + ceiling
        obs = points[(points[:, 2] > LIDAR_Z_MIN) & (points[:, 2] < LIDAR_Z_MAX)]

        # Clear inner grid — keep boundary walls
        wall = max(1, int(1.0 / RESOLUTION))
        self.grid[wall:-wall, wall:-wall] = 0

        # Rotate sensor-frame points to ENU map frame
        cos_y = math.cos(self.car_yaw)
        sin_y = math.sin(self.car_yaw)
        for p in obs:
            map_x = self.car_x + p[0] * cos_y - p[1] * sin_y
            map_y = self.car_y + p[0] * sin_y + p[1] * cos_y
            if math.isnan(map_x) or math.isnan(map_y):
                continue
            col, row = world_to_cell(map_x, map_y)
            if in_bounds(col, row, self.grid.shape):
                self.grid[row, col] = 1

        self._publish_grid()

    # ══════════════════════════════════════════════════════════
    #  GOAL CALLBACK — runs A* when perception sends a new goal
    # ══════════════════════════════════════════════════════════

    def goal_cb(self, msg: PoseStamped):
        gx = msg.pose.position.x
        gy = msg.pose.position.y

        if math.isnan(gx) or math.isnan(gy):
            self.get_logger().error("Received NaN goal — ignoring.")
            return
        if self.car_x is None:
            self.get_logger().warn("Goal received but GPS not ready yet.")
            return

        self.get_logger().info(
            f"Goal received: ({gx:.2f}, {gy:.2f}) ENU. Running A*...")

        inflated = build_inflated_grid(self.grid, INFLATE_R)
        raw      = astar(inflated, (self.car_x, self.car_y), (gx, gy))

        if raw is None:
            self.get_logger().error(
                "A* found no path. Check goal is inside lot bounds "
                "and not blocked by obstacles.")
            return

        self.global_path = smooth_path(raw)
        self.target_idx  = 0
        self.get_logger().info(
            f"Path ready: {len(raw)} raw → {len(self.global_path)} smoothed pts.")

        self._publish_path()

    # ══════════════════════════════════════════════════════════
    #  CONTROL LOOP  (20 Hz — same rate as pure_pursuit_ros2.py)
    # ══════════════════════════════════════════════════════════

    def control_loop(self):
        """
        Mirrors the control_loop structure of pure_pursuit_ros2.py exactly:
          - Check joystick enable/disable
          - If enabled and path exists: run pure pursuit + PID speed
          - Publish to same PACMod topics
        """
        joy = self._check_joystick()

        # ── Joystick ENABLE (LB + RB) ─────────────────────────
        if joy == 1 and not self.pacmod_enable:
            self.global_cmd.enable      = True
            self.global_cmd.clear_override = True
            self.global_pub.publish(self.global_cmd)

            self.gear_cmd.command = GEAR_DRIVE
            self.gear_pub.publish(self.gear_cmd)

            self.brake_cmd.command = 0.0
            self.brake_pub.publish(self.brake_cmd)

            self.accel_cmd.command = 0.0
            self.accel_pub.publish(self.accel_cmd)

            self.turn_cmd.command = 3   # no signal
            self.turn_pub.publish(self.turn_cmd)

            self.get_logger().warn(
                'Pacmod Disabled: Vehicle enabled and forward gear engaged')
            return

        # ── Joystick DISABLE (LB only) ────────────────────────
        if joy == 0 and self.pacmod_enable:
            self.global_cmd.enable = False
            self.global_pub.publish(self.global_cmd)

            self.turn_cmd.command = 1
            self.turn_pub.publish(self.turn_cmd)

            self.get_logger().warn('Joystick Disabled: Vehicle disabled')
            return

        # ── Execute controller ────────────────────────────────
        if joy != 0 and self.pacmod_enable:
            # No path yet — hold still
            if self.global_path is None or self.car_x is None:
                return

            # ── Get current vehicle state (same as get_gem_state) ─
            local_x, local_y = self._wps_to_local_xy(self.lon, self.lat)
            curr_yaw = ins_heading_to_yaw(self.heading)
            # Correct for antenna offset (same as pure_pursuit_ros2.py)
            curr_x = local_x - OFFSET * math.cos(curr_yaw)
            curr_y = local_y - OFFSET * math.sin(curr_yaw)

            # ── Goal reached check ─────────────────────────────
            gx, gy = self.global_path[-1]
            if math.hypot(curr_x - gx, curr_y - gy) < GOAL_TOL:
                self.get_logger().info("Goal reached! Stopping.")
                self._stop_vehicle()
                self.global_path = None
                return

            # ── Pure pursuit steering ──────────────────────────
            px = np.array([p[0] for p in self.global_path])
            py = np.array([p[1] for p in self.global_path])
            n  = len(self.global_path)

            # Distance array to all waypoints
            dist_arr = np.hypot(px - curr_x, py - curr_y)

            # Find closest waypoint, then advance by lookahead
            # (mirrors the original goal-finding loop exactly)
            closest = int(np.argmin(dist_arr))
            ld = LOOK_AHEAD + max(0.0, self.speed - 2.5) * 2.0
            goal_idx = closest
            for i in range(closest, n):
                if dist_arr[i] > ld:
                    goal_idx = i
                    break
            goal_idx = min(goal_idx, n - 1)

            tx  = px[goal_idx]
            ty  = py[goal_idx]

            alpha     = math.atan2(ty - curr_y, tx - curr_x) - curr_yaw
            alpha     = normalize_angle(alpha)
            curvature = 0.0 if self.speed < 0.2 else 2.0 * math.sin(alpha) / ld
            front_angle_deg = math.degrees(math.atan(WHEELBASE * curvature))

            # Convert front wheel angle → steering wheel angle
            # (front2steer from pure_pursuit_ros2.py)
            steer_wheel_deg = self._front2steer(front_angle_deg)
            steer_wheel_rad = math.radians(steer_wheel_deg)

            self.steer_cmd.angular_position = steer_wheel_rad
            self.steer_pub.publish(self.steer_cmd)

            # ── PID speed control (same as pure_pursuit_ros2.py) ─
            now         = self.get_clock().now().nanoseconds * 1e-9
            speed_err   = DESIRED_SPEED - self.speed
            if abs(speed_err) < 0.05:
                speed_err = 0.0
            throttle    = self.pid_speed.get_control(now, speed_err)
            throttle    = max(0.0, min(throttle, MAX_ACCEL))

            self.accel_cmd.command = throttle
            self.brake_cmd.command = 0.0
            self.accel_pub.publish(self.accel_cmd)
            self.brake_pub.publish(self.brake_cmd)

            self.global_cmd.enable = True
            self.global_pub.publish(self.global_cmd)

            self.get_logger().info(
                f"Pos: ({curr_x:.2f}, {curr_y:.2f})  "
                f"Target: ({tx:.2f}, {ty:.2f})  "
                f"Speed: {self.speed:.2f}  "
                f"Throttle: {throttle:.2f}  "
                f"Steer: {steer_wheel_deg:.2f}°  "
                f"Dist→Goal: {math.hypot(curr_x-gx, curr_y-gy):.1f}m")

    # ══════════════════════════════════════════════════════════
    #  HELPERS  (copied / adapted from pure_pursuit_ros2.py)
    # ══════════════════════════════════════════════════════════

    def _check_joystick(self) -> int:
        """
        Mirrors check_joystick_enable() from pure_pursuit_ros2.py.
        Returns: 1=enable, 0=disable, 2=no change
        """
        if self._joystick is None:
            return 2
        pygame.event.pump()
        try:
            lb = self._joystick.get_button(6)
            rb = self._joystick.get_button(7)
        except pygame.error:
            self.get_logger().warn("Joystick read failed.")
            return 2
        if lb and rb:
            return 1   # enable
        if lb and not rb:
            return 0   # disable
        return 2       # no change

    def _front2steer(self, f_angle: float) -> float:
        """
        Convert front wheel angle (degrees) to steering wheel angle (degrees).
        Exact copy of front2steer() from pure_pursuit_ros2.py.
        """
        f_angle = max(min(f_angle, STEER_MAX_DEG), -STEER_MAX_DEG)
        angle   = abs(f_angle)
        steer   = STEER_A * angle**2 + STEER_B * angle
        return round(steer if f_angle >= 0 else -steer, 2)

    def _wps_to_local_xy(self, lon: float, lat: float) -> tuple:
        """
        GPS → local ENU.
        Mirrors wps_to_local_xy() from pure_pursuit_ros2.py.
        """
        x, y, _ = pm.geodetic2enu(lat, lon, 0, ORIGIN_LAT, ORIGIN_LON, 0)
        return float(x), float(y)

    def _stop_vehicle(self):
        """Send zero throttle + disable PACMod cleanly."""
        self.accel_cmd.command = 0.0
        self.brake_cmd.command = 0.3   # light brake to hold position
        self.accel_pub.publish(self.accel_cmd)
        self.brake_pub.publish(self.brake_cmd)
        self.global_cmd.enable = False
        self.global_pub.publish(self.global_cmd)
        self.turn_cmd.command = 1
        self.turn_pub.publish(self.turn_cmd)

    # ── Visualization ──────────────────────────────────────────

    def _publish_path(self):
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for wp in self.global_path:
            ps = PoseStamped()
            ps.header             = msg.header
            ps.pose.position.x    = float(wp[0])
            ps.pose.position.y    = float(wp[1])
            ps.pose.position.z    = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _publish_grid(self):
        msg = OccupancyGrid()
        msg.header.stamp          = self.get_clock().now().to_msg()
        msg.header.frame_id       = 'map'
        msg.info.resolution       = float(RESOLUTION)
        msg.info.width            = self.grid.shape[1]
        msg.info.height           = self.grid.shape[0]
        msg.info.origin.position.x = float(X_MIN)
        msg.info.origin.position.y = float(Y_MIN)
        msg.info.origin.orientation.w = 1.0
        msg.data = (self.grid * 100).astype(np.int8).flatten().tolist()
        self.grid_pub.publish(msg)


# ═══════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = VLMPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()