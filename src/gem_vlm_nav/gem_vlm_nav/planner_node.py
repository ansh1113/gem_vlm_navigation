#!/usr/bin/env python3
"""
planner_node.py — VLM Path Planning + PACMod Control Node for GEM e4
======================================================================
ROS 2 node for the **real vehicle**.

Architecture:
  VLM goal (/vlm_goal)  →  A* global plan (gem_planner_core)
                         →  pure-pursuit steering (gem_planner_core)
                         →  PACMod commands

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

import math

import numpy as np
import scipy.signal as signal
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

# Shared planning core
from gem_vlm_nav.gem_planner_core import GemPlannerCore


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# ENU origin — must match perception_node and pure pursuit controller
ORIGIN_LAT  = 40.0927422
ORIGIN_LON  = -88.2359639

# Highbay operating area (metres ENU)
X_MIN, X_MAX = -25.0, 75.0
Y_MIN, Y_MAX =  -5.0, 20.0

# LiDAR height filter
LIDAR_Z_MIN =  0.1   # m — ignore ground returns
LIDAR_Z_MAX =  2.5   # m — ignore ceiling / overhead structure

# Grid resolution
RESOLUTION = 0.5      # m/cell

# Vehicle geometry
WHEELBASE = 2.57      # m
OFFSET    = 1.26      # m — GNSS antenna offset from rear axle

# Speed / control
DESIRED_SPEED = 2.0   # m/s
MAX_ACCEL     = 0.5   # throttle cap
GOAL_TOL      = 1.5   # m — arrival tolerance

# PID speed controller (same as pure_pursuit_ros2.py)
PID_KP = 0.6
PID_KI = 0.0
PID_KD = 0.1
PID_WG = 10.0

# Speed filter (Butterworth low-pass)
FILTER_CUTOFF = 1.2
FILTER_FS     = 30
FILTER_ORDER  = 4

# Steering wheel angle conversion (front2steer)
STEER_A       = -0.1084   # quadratic coefficient
STEER_B       = 21.775    # linear coefficient
STEER_MAX_DEG = 35        # max front wheel angle (degrees)

# PACMod gear commands
GEAR_NEUTRAL = 2
GEAR_DRIVE   = 3


# ═══════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════

def ins_heading_to_yaw(heading_deg: float) -> float:
    """
    Convert Septentrio heading (0=North, CW, degrees) to ENU yaw
    (0=East, CCW, radians).  Exact copy of heading_to_yaw() in
    pure_pursuit_ros2.py.
    """
    if heading_deg < 270.0:
        return math.radians(90.0 - heading_deg)
    else:
        return math.radians(450.0 - heading_deg)


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

        # ── Shared planning core ──────────────────────────────
        self.planner = GemPlannerCore(
            x_min=X_MIN, x_max=X_MAX,
            y_min=Y_MIN, y_max=Y_MAX,
            resolution=RESOLUTION,
            wheelbase=WHEELBASE,
        )

        # ── Speed controller (same as pure_pursuit_ros2.py) ───
        self.pid_speed    = PID(PID_KP, PID_KI, PID_KD, wg=PID_WG)
        self.speed_filter = OnlineFilter(FILTER_CUTOFF, FILTER_FS, FILTER_ORDER)

        # ── Vehicle state ─────────────────────────────────────
        self.lat          = 0.0
        self.lon          = 0.0
        self.heading      = 0.0   # raw degrees from INSNavGeod
        self.speed        = 0.0   # filtered m/s
        self.pacmod_enable = False

        # ENU position + yaw (derived from lat/lon/heading)
        self.car_x   = None
        self.car_y   = None
        self.car_yaw = 0.0

        # ── Planning state ────────────────────────────────────
        self.has_plan = False

        # ── PACMod command objects (same as pure_pursuit_ros2.py)
        self.global_cmd = GlobalCmd(enable=False, clear_override=True)
        self.gear_cmd   = SystemCmdInt(command=GEAR_NEUTRAL)
        self.brake_cmd  = SystemCmdFloat(command=0.0)
        self.accel_cmd  = SystemCmdFloat(command=0.0)
        self.turn_cmd   = SystemCmdInt(command=1)   # no turn signal
        self.steer_cmd  = PositionWithSpeed(
            angular_position=0.0, angular_velocity_limit=4.0)

        # ── Subscriptions ─────────────────────────────────────
        self.create_subscription(NavSatFix, '/navsatfix',         self.gnss_cb,   10)
        self.create_subscription(INSNavGeod, '/insnavgeod',       self.ins_cb,    10)
        self.create_subscription(Bool, '/pacmod/enabled',         self.enable_cb, 10)
        self.create_subscription(VehicleSpeedRpt,
                                 '/pacmod/vehicle_speed_rpt',     self.speed_cb,  10)
        self.create_subscription(PointCloud2, '/ouster/points',   self.lidar_cb,  10)

        # Goal from perception node — TRANSIENT_LOCAL so we don't miss it
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseStamped, '/vlm_goal', self.goal_cb, qos)

        # ── PACMod publishers (same topics/types as pure_pursuit_ros2.py)
        self.global_pub = self.create_publisher(GlobalCmd,         '/pacmod/global_cmd',   10)
        self.gear_pub   = self.create_publisher(SystemCmdInt,      '/pacmod/shift_cmd',    10)
        self.brake_pub  = self.create_publisher(SystemCmdFloat,    '/pacmod/brake_cmd',    10)
        self.accel_pub  = self.create_publisher(SystemCmdFloat,    '/pacmod/accel_cmd',    10)
        self.turn_pub   = self.create_publisher(SystemCmdInt,      '/pacmod/turn_cmd',     10)
        self.steer_pub  = self.create_publisher(PositionWithSpeed, '/pacmod/steering_cmd', 10)

        # ── Visualization publishers (RViz) ───────────────────
        self.path_pub = self.create_publisher(Path,          '/vlm_path',    10)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/vlm_costmap', 10)

        # ── Control timer (20 Hz) ─────────────────────────────
        self.create_timer(1.0 / 20.0, self.control_loop)

        self.get_logger().info(
            "VLM Planner ready (pure pursuit).  "
            "Waiting for /vlm_goal — use joystick LB+RB to enable vehicle.")

    # ══════════════════════════════════════════════════════════
    #  SENSOR CALLBACKS
    # ══════════════════════════════════════════════════════════

    def gnss_cb(self, msg: NavSatFix):
        self.lat = msg.latitude
        self.lon = msg.longitude
        try:
            e, n, _ = pm.geodetic2enu(
                self.lat, self.lon, 0, ORIGIN_LAT, ORIGIN_LON, 0)
            self.car_x = float(e)
            self.car_y = float(n)
        except Exception:
            pass

    def ins_cb(self, msg: INSNavGeod):
        self.heading = msg.heading
        if msg.heading is not None and not math.isnan(msg.heading):
            self.car_yaw = ins_heading_to_yaw(msg.heading)

    def speed_cb(self, msg: VehicleSpeedRpt):
        self.speed = self.speed_filter.get_data(msg.vehicle_speed)

    def enable_cb(self, msg: Bool):
        self.pacmod_enable = msg.data

    def lidar_cb(self, msg: PointCloud2):
        """Build dynamic occupancy grid from Ouster LiDAR pointcloud."""
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

        # Reset dynamic grid, keep boundaries
        self.planner.reset_obstacles()

        # Rotate sensor-frame points to ENU
        cos_y = math.cos(self.car_yaw)
        sin_y = math.sin(self.car_yaw)
        for p in obs:
            map_x = self.car_x + p[0] * cos_y - p[1] * sin_y
            map_y = self.car_y + p[0] * sin_y + p[1] * cos_y
            if math.isnan(map_x) or math.isnan(map_y):
                continue
            col, row = self.planner.world_to_cell(map_x, map_y)
            if self.planner.in_bounds(col, row, self.planner.dyn_grid.shape):
                self.planner.dyn_grid[row, col] = 1

        # Rebuild costmap once after all points are added
        self.planner.costmap = self.planner._build_costmap(self.planner.dyn_grid)

        self._publish_grid()

    # ══════════════════════════════════════════════════════════
    #  GOAL CALLBACK — triggers A* planning
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
            f"Goal received: ({gx:.2f}, {gy:.2f}) ENU.  Running A*...")

        success = self.planner.plan_global_path(
            (self.car_x, self.car_y), (gx, gy))

        if not success:
            self.get_logger().error(
                "A* found no path.  Check goal is inside bounds "
                "and not blocked by obstacles.")
            self.has_plan = False
            return

        self.has_plan = True
        self.pid_speed.reset()
        self.get_logger().info(
            f"Path ready: {len(self.planner.global_path)} smoothed pts.")
        self._publish_path()

    # ══════════════════════════════════════════════════════════
    #  CONTROL LOOP  (20 Hz)
    # ══════════════════════════════════════════════════════════

    def control_loop(self):
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
                'PACMod enabled — forward gear engaged')
            return

        # ── Joystick DISABLE (LB only) ────────────────────────
        if joy == 0 and self.pacmod_enable:
            self.global_cmd.enable = False
            self.global_pub.publish(self.global_cmd)
            self.turn_cmd.command = 1
            self.turn_pub.publish(self.turn_cmd)
            self.get_logger().warn('Joystick Disabled — vehicle disabled')
            return

        # ── Execute controller ────────────────────────────────
        if joy != 0 and self.pacmod_enable:
            if not self.has_plan or self.car_x is None:
                return

            # Current vehicle state with antenna offset correction
            curr_yaw = ins_heading_to_yaw(self.heading)
            local_x, local_y = self._wps_to_local_xy(self.lon, self.lat)
            curr_x = local_x - OFFSET * math.cos(curr_yaw)
            curr_y = local_y - OFFSET * math.sin(curr_yaw)

            car_state = (curr_x, curr_y, curr_yaw)

            # Goal reached check
            gx, gy = self.planner.global_path[-1]
            if math.hypot(curr_x - gx, curr_y - gy) < GOAL_TOL:
                self.get_logger().info("Goal reached!  Stopping.")
                self._stop_vehicle()
                self.has_plan = False
                return

            # Pure pursuit steering
            front_steer, target_speed, emergency = \
                self.planner.get_local_command(car_state, DESIRED_SPEED)

            if emergency:
                self.get_logger().warn(
                    "Obstacle blocking path!  Stopping and replanning...")
                self._stop_vehicle()

                success = self.planner.plan_global_path(
                    (curr_x, curr_y), self.planner.goal)
                if success:
                    self.get_logger().info("Replan successful.")
                    self._publish_path()
                else:
                    self.get_logger().error("Replan failed — path blocked.")
                    self.has_plan = False
                return

            # Convert front wheel angle → steering wheel angle
            steer_wheel_deg = self._front2steer(math.degrees(front_steer))
            steer_wheel_rad = math.radians(steer_wheel_deg)

            self.steer_cmd.angular_position = steer_wheel_rad
            self.steer_pub.publish(self.steer_cmd)

            # PID speed control
            now       = self.get_clock().now().nanoseconds * 1e-9
            speed_err = target_speed - self.speed
            if abs(speed_err) < 0.05:
                speed_err = 0.0
            throttle = self.pid_speed.get_control(now, speed_err)
            throttle = max(0.0, min(throttle, MAX_ACCEL))

            self.accel_cmd.command = throttle
            self.brake_cmd.command = 0.0
            self.accel_pub.publish(self.accel_cmd)
            self.brake_pub.publish(self.brake_cmd)

            self.global_cmd.enable = True
            self.global_pub.publish(self.global_cmd)

            self.get_logger().info(
                f"Pos: ({curr_x:.2f}, {curr_y:.2f})  "
                f"Speed: {self.speed:.2f}  "
                f"Throttle: {throttle:.2f}  "
                f"Steer(front): {math.degrees(front_steer):.1f}°  "
                f"Dist→Goal: {math.hypot(curr_x-gx, curr_y-gy):.1f}m")

    # ══════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════

    def _check_joystick(self) -> int:
        """1=enable, 0=disable, 2=no change."""
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
            return 1
        if lb and not rb:
            return 0
        return 2

    def _front2steer(self, f_angle: float) -> float:
        """
        Convert front wheel angle (degrees) → steering wheel angle (degrees).
        Exact copy of front2steer() from pure_pursuit_ros2.py.
        """
        f_angle = max(min(f_angle, STEER_MAX_DEG), -STEER_MAX_DEG)
        angle   = abs(f_angle)
        steer   = STEER_A * angle**2 + STEER_B * angle
        return round(steer if f_angle >= 0 else -steer, 2)

    def _wps_to_local_xy(self, lon: float, lat: float) -> tuple:
        """GPS → local ENU."""
        x, y, _ = pm.geodetic2enu(lat, lon, 0, ORIGIN_LAT, ORIGIN_LON, 0)
        return float(x), float(y)

    def _stop_vehicle(self):
        """Zero throttle + light brake."""
        self.accel_cmd.command = 0.0
        self.brake_cmd.command = 0.3
        self.accel_pub.publish(self.accel_cmd)
        self.brake_pub.publish(self.brake_cmd)
        self.global_cmd.enable = True
        self.global_pub.publish(self.global_cmd)

    # ── Visualization ─────────────────────────────────────────

    def _publish_path(self):
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for wp in self.planner.global_path:
            ps = PoseStamped()
            ps.header             = msg.header
            ps.pose.position.x    = float(wp[0])
            ps.pose.position.y    = float(wp[1])
            ps.pose.position.z    = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _publish_grid(self):
        grid = self.planner.dyn_grid
        msg  = OccupancyGrid()
        msg.header.stamp          = self.get_clock().now().to_msg()
        msg.header.frame_id       = 'map'
        msg.info.resolution       = float(RESOLUTION)
        msg.info.width            = grid.shape[1]
        msg.info.height           = grid.shape[0]
        msg.info.origin.position.x = float(X_MIN)
        msg.info.origin.position.y = float(Y_MIN)
        msg.info.origin.orientation.w = 1.0
        msg.data = (grid * 100).astype(np.int8).flatten().tolist()
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