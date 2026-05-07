#!/usr/bin/env python3

#================================================================
# File name: planner_node.py
# Description: VLM goal tracker using A* + pure pursuit in ROS2
#              Low-level PACMod interface identical to pure_pursuit.py
# Based on: pure_pursuit.py by Jiaming Zhang, Hang Cui (2025-06-03)
#================================================================

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
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Path, OccupancyGrid
from pacmod2_msgs.msg import (
    PositionWithSpeed, VehicleSpeedRpt,
    GlobalCmd, SystemCmdFloat, SystemCmdInt,
)
from septentrio_gnss_driver.msg import INSNavGeod

# ── Joystick init (same as pure_pursuit.py) ───────────────────
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise RuntimeError("No joystick connected")
joystick = pygame.joystick.Joystick(0)
joystick.init()


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION  (same values as pure_pursuit.py)
# ═══════════════════════════════════════════════════════════════

ORIGIN_LAT = 40.0927422
ORIGIN_LON = -88.2359639

# Grid bounds (metres ENU) — highbay operating area
X_MIN, X_MAX = -25.0, 75.0
Y_MIN, Y_MAX =  -5.0, 20.0

RESOLUTION  = 0.5    # m/cell
INFLATE_R   = 1.5    # m — obstacle inflation radius (car half-width + margin)

LOOK_AHEAD    = 5.0
WHEELBASE     = 2.57
OFFSET        = 1.26
DESIRED_SPEED = 2.0
MAX_ACCEL     = 0.5

GOAL_TOL = 1.5   # m — stop when this close to goal

# PID (same as pure_pursuit.py defaults)
PID_KP, PID_KI, PID_KD, PID_WG = 0.6, 0.0, 0.1, 10.0

# Filter (same as pure_pursuit.py defaults)
FILTER_CUTOFF, FILTER_FS, FILTER_ORDER = 1.2, 30, 4

# front2steer polynomial (same as pure_pursuit.py)
STEER_A       = -0.1084
STEER_B       = 21.775
STEER_MAX_DEG = 35

GEAR_NEUTRAL = 2
GEAR_DRIVE   = 3


# ═══════════════════════════════════════════════════════════════
#  PID — verbatim from pure_pursuit.py
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
#  FILTER — verbatim from pure_pursuit.py
# ═══════════════════════════════════════════════════════════════

class OnlineFilter:
    def __init__(self, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        self.z = signal.lfilter_zi(self.b, self.a)

    def get_data(self, data):
        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filted[0]


# ═══════════════════════════════════════════════════════════════
#  GRID + A*
# ═══════════════════════════════════════════════════════════════

def world_to_cell(x, y):
    return int((x - X_MIN) / RESOLUTION), int((y - Y_MIN) / RESOLUTION)

def cell_to_world(c, r):
    return X_MIN + (c + 0.5) * RESOLUTION, Y_MIN + (r + 0.5) * RESOLUTION

def in_bounds(c, r, shape):
    return 0 <= r < shape[0] and 0 <= c < shape[1]

def build_base_grid():
    nc = int(np.ceil((X_MAX - X_MIN) / RESOLUTION))
    nr = int(np.ceil((Y_MAX - Y_MIN) / RESOLUTION))
    g  = np.zeros((nr, nc), dtype=np.uint8)
    # Wall border (1 m)
    w = max(1, int(1.0 / RESOLUTION))
    g[:w, :] = g[-w:, :] = g[:, :w] = g[:, -w:] = 1
    return g

def inflate_grid(grid, r_m):
    r = max(1, int(r_m / RESOLUTION))
    s = np.zeros((2*r+1, 2*r+1), dtype=bool)
    for i in range(2*r+1):
        for j in range(2*r+1):
            if (i-r)**2 + (j-r)**2 <= r**2:
                s[i, j] = True
    return ndimage.binary_dilation(grid.astype(bool), structure=s).astype(np.uint8)

def astar(grid, start_world, goal_world):
    sc = world_to_cell(*start_world)
    gc = world_to_cell(*goal_world)

    # Snap to nearest free cell if start/goal lands in obstacle
    def snap(c, r):
        if in_bounds(c, r, grid.shape) and grid[r, c] == 0:
            return c, r
        for rd in range(1, 15):
            for dc in range(-rd, rd+1):
                for dr in range(-rd, rd+1):
                    nc2, nr2 = c+dc, r+dr
                    if in_bounds(nc2, nr2, grid.shape) and grid[nr2, nc2] == 0:
                        return nc2, nr2
        return c, r

    sc = snap(*sc)
    gc = snap(*gc)

    if not in_bounds(*sc, grid.shape) or not in_bounds(*gc, grid.shape):
        return None

    nbrs = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),
            (1,1,1.414),(1,-1,1.414),(-1,1,1.414),(-1,-1,1.414)]
    heap = [(0.0, sc)]
    came = {}
    gs   = {sc: 0.0}

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
        c, r = cur
        for dc, dr, cost in nbrs:
            nb = (c+dc, r+dr)
            if not in_bounds(nb[0], nb[1], grid.shape):
                continue
            if grid[nb[1], nb[0]] == 1:
                continue
            t = gs[cur] + cost
            if t < gs.get(nb, 1e9):
                came[nb] = cur
                gs[nb]   = t
                heapq.heappush(heap,
                    (t + math.hypot(nb[0]-gc[0], nb[1]-gc[1]), nb))
    return None

def smooth_path(path, n=200):
    if not path or len(path) < 3:
        return path
    pts = np.array(path)
    _, idx = np.unique(pts, axis=0, return_index=True)
    pts = pts[np.sort(idx)]
    if len(pts) < 3:
        return path
    try:
        tck, _ = splprep([pts[:,0], pts[:,1]], s=2.0, k=min(3, len(pts)-1))
        xs, ys = splev(np.linspace(0, 1, n), tck)
        return list(zip(xs.tolist(), ys.tolist()))
    except Exception:
        return path


# ═══════════════════════════════════════════════════════════════
#  PLANNER NODE
# ═══════════════════════════════════════════════════════════════

class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')

        # ── Parameters — same as pure_pursuit.py ──────────────
        self.declare_parameter('rate_hz',          20)
        self.declare_parameter('look_ahead',        5.0)
        self.declare_parameter('wheelbase',         2.57)
        self.declare_parameter('offset',            1.26)
        self.declare_parameter('origin_lat',        40.0927422)
        self.declare_parameter('origin_lon',       -88.2359639)
        self.declare_parameter('desired_speed',     2.0)
        self.declare_parameter('max_acceleration',  0.5)
        self.declare_parameter('pid/kp',            0.6)
        self.declare_parameter('pid/ki',            0.0)
        self.declare_parameter('pid/kd',            0.1)
        self.declare_parameter('pid/wg',            10)
        self.declare_parameter('filter/cutoff',     1.2)
        self.declare_parameter('filter/fs',         30)
        self.declare_parameter('filter/order',      4)
        self.declare_parameter('vehicle_name',      '')

        vehicle_name = self.get_parameter('vehicle_name').value
        if vehicle_name == '':
            self.get_logger().warn(
                "No vehicle_name parameter — defaulting to e4 parameters.")
        else:
            self.get_logger().info(f"Using vehicle config: {vehicle_name}")

        self.rate_hz       = self.get_parameter('rate_hz').value
        self.look_ahead    = self.get_parameter('look_ahead').value
        self.wheelbase     = self.get_parameter('wheelbase').value
        self.offset        = self.get_parameter('offset').value
        self.olat          = self.get_parameter('origin_lat').value
        self.olon          = self.get_parameter('origin_lon').value
        self.desired_speed = min(5.0, self.get_parameter('desired_speed').value)
        self.max_accel     = min(2.0, self.get_parameter('max_acceleration').value)

        self.pid_speed = PID(
            kp=self.get_parameter('pid/kp').value,
            ki=self.get_parameter('pid/ki').value,
            kd=self.get_parameter('pid/kd').value,
            wg=self.get_parameter('pid/wg').value,
        )
        self.speed_filter = OnlineFilter(
            cutoff=self.get_parameter('filter/cutoff').value,
            fs=self.get_parameter('filter/fs').value,
            order=self.get_parameter('filter/order').value,
        )

        # ── Vehicle state — same as pure_pursuit.py ───────────
        self.lat           = 0.0
        self.lon           = 0.0
        self.heading       = 0.0
        self.speed         = 0.0
        self.pacmod_enable = False
        self.gem_enable    = False

        # ENU position (derived)
        self.car_x = None
        self.car_y = None

        # ── Planning state ────────────────────────────────────
        self.grid     = build_base_grid()   # static obstacle grid
        self.path     = []                  # smoothed A* path [(x,y), ...]
        self.goal_idx = 0                   # current lookahead index
        self.has_plan = False
        self.goal_xy  = None               # (gx, gy) of current goal

        # ── PACMod commands — same as pure_pursuit.py ─────────
        self.global_cmd = GlobalCmd(enable=False, clear_override=True)
        self.gear_cmd   = SystemCmdInt(command=GEAR_NEUTRAL)
        self.brake_cmd  = SystemCmdFloat(command=0.0)
        self.accel_cmd  = SystemCmdFloat(command=0.0)
        self.turn_cmd   = SystemCmdInt(command=1)
        self.steer_cmd  = PositionWithSpeed(
            angular_position=0.0, angular_velocity_limit=4.0)

        # ── Subscriptions — pure_pursuit.py topics + goal ─────
        self.create_subscription(NavSatFix,       '/navsatfix',
                                 self.gnss_callback,   10)
        self.create_subscription(INSNavGeod,      '/insnavgeod',
                                 self.ins_callback,    10)
        self.create_subscription(Bool,            '/pacmod/enabled',
                                 self.enable_callback, 10)
        self.create_subscription(VehicleSpeedRpt, '/pacmod/vehicle_speed_rpt',
                                 self.speed_callback,  10)

        # Goal from VLM/perception — TRANSIENT_LOCAL so we don't miss it
        goal_qos = QoSProfile(depth=10,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseStamped, '/vlm_goal',
                                 self.goal_callback, goal_qos)

        # Also accept a PoseArray of waypoints (for multi-waypoint VLM output)
        self.create_subscription(PoseArray, '/vlm_waypoints',
                                 self.waypoints_callback, goal_qos)

        # Optional: live costmap from external node
        self.create_subscription(OccupancyGrid, '/vlm_costmap',
                                 self.costmap_callback, 10)

        # ── Publishers — identical to pure_pursuit.py ─────────
        self.global_pub = self.create_publisher(GlobalCmd,         '/pacmod/global_cmd',   10)
        self.gear_pub   = self.create_publisher(SystemCmdInt,      '/pacmod/shift_cmd',    10)
        self.brake_pub  = self.create_publisher(SystemCmdFloat,    '/pacmod/brake_cmd',    10)
        self.accel_pub  = self.create_publisher(SystemCmdFloat,    '/pacmod/accel_cmd',    10)
        self.turn_pub   = self.create_publisher(SystemCmdInt,      '/pacmod/turn_cmd',     10)
        self.steer_pub  = self.create_publisher(PositionWithSpeed, '/pacmod/steering_cmd', 10)

        # Visualization
        self.path_pub = self.create_publisher(Path, '/vlm_path', 10)

        # ── Control timer — same rate as pure_pursuit.py ──────
        self.timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)

        self.get_logger().info(
            "Planner node ready. Waiting for /vlm_goal or /vlm_waypoints. "
            "Use joystick LB+RB to enable.")

    # ══════════════════════════════════════════════════════════
    #  SENSOR CALLBACKS — verbatim from pure_pursuit.py
    # ══════════════════════════════════════════════════════════

    def gnss_callback(self, msg):
        self.lat = msg.latitude
        self.lon = msg.longitude
        # Also keep ENU position up to date for planning
        try:
            e, n, _ = pm.geodetic2enu(
                self.lat, self.lon, 0, self.olat, self.olon, 0)
            self.car_x = float(e)
            self.car_y = float(n)
        except Exception:
            pass

    def ins_callback(self, msg):
        self.heading = msg.heading

    def speed_callback(self, msg):
        self.speed = self.speed_filter.get_data(msg.vehicle_speed)

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def costmap_callback(self, msg: OccupancyGrid):
        """Accept a live costmap from an external node (e.g. lidar_bev_node)."""
        nr   = msg.info.height
        nc   = msg.info.width
        data = np.array(msg.data, dtype=np.int8).reshape(nr, nc)
        self.grid = (data > 50).astype(np.uint8)

    # ══════════════════════════════════════════════════════════
    #  GOAL CALLBACKS
    # ══════════════════════════════════════════════════════════

    def goal_callback(self, msg: PoseStamped):
        """Single ENU goal from VLM/perception node."""
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
        self._plan_to(gx, gy)

    def waypoints_callback(self, msg: PoseArray):
        """Multi-waypoint list from VLM node — plan to first waypoint."""
        if not msg.poses:
            return
        if self.car_x is None:
            self.get_logger().warn("Waypoints received but GPS not ready yet.")
            return

        # Plan a single A* path through all waypoints in order
        # by chaining: current → wp0 → wp1 → ... → wpN
        all_wps = [(p.position.x, p.position.y) for p in msg.poses]
        self.get_logger().info(
            f"Waypoints received ({len(all_wps)}): "
            + "  ".join(f"({x:.1f},{y:.1f})" for x, y in all_wps))

        # Plan to first waypoint from current position
        self._plan_to(*all_wps[0])

    def _plan_to(self, gx: float, gy: float):
        """Run A* from current ENU position to (gx, gy) and store path."""
        inf_grid = inflate_grid(self.grid, INFLATE_R)
        raw = astar(inf_grid, (self.car_x, self.car_y), (gx, gy))

        if raw is None:
            self.get_logger().error(
                f"A* found no path to ({gx:.1f}, {gy:.1f}). "
                "Check goal is inside bounds and not blocked.")
            self.has_plan = False
            return

        self.path     = smooth_path(raw)
        self.goal_idx = 0
        self.goal_xy  = (gx, gy)
        self.has_plan = True
        self.pid_speed.reset()

        self.get_logger().info(
            f"Path ready: {len(self.path)} smoothed pts → "
            f"goal ({gx:.1f}, {gy:.1f})")
        self._publish_path()

    # ══════════════════════════════════════════════════════════
    #  HELPERS — verbatim / minimal delta from pure_pursuit.py
    # ══════════════════════════════════════════════════════════

    def heading_to_yaw(self, heading):
        """Exact copy from pure_pursuit.py."""
        return np.radians(90 - heading) if heading < 270 else np.radians(450 - heading)

    def wps_to_local_xy(self, lon, lat):
        """Exact copy from pure_pursuit.py."""
        x, y, _ = pm.geodetic2enu(lat, lon, 0, self.olat, self.olon, 0)
        return x, y

    def get_gem_state(self):
        """Exact copy from pure_pursuit.py."""
        local_x, local_y = self.wps_to_local_xy(self.lon, self.lat)
        yaw = self.heading_to_yaw(self.heading)
        x = local_x - self.offset * math.cos(yaw)
        y = local_y - self.offset * math.sin(yaw)
        return x, y, yaw

    def front2steer(self, f_angle):
        """Exact copy from pure_pursuit.py."""
        f_angle = max(min(f_angle, 35), -35)
        angle = abs(f_angle)
        steer_angle = -0.1084 * angle ** 2 + 21.775 * angle
        return round(steer_angle if f_angle >= 0 else -steer_angle, 2)

    def check_joystick_enable(self):
        """Exact copy from pure_pursuit.py."""
        pygame.event.pump()
        try:
            lb = joystick.get_button(6)
            rb = joystick.get_button(7)
        except pygame.error:
            self.get_logger().warn("Joystick read failed")
            return 2
        if lb and rb:
            return 1
        elif lb and not rb:
            return 0
        return 2

    def _publish_path(self):
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for wp in self.path:
            ps = PoseStamped()
            ps.header             = msg.header
            ps.pose.position.x    = float(wp[0])
            ps.pose.position.y    = float(wp[1])
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    # ══════════════════════════════════════════════════════════
    #  CONTROL LOOP — pure_pursuit.py structure, A* path instead of CSV
    # ══════════════════════════════════════════════════════════

    def control_loop(self):
        joy_enable = self.check_joystick_enable()

        # ── ENABLE (LB + RB) — identical to pure_pursuit.py ──
        if joy_enable == 1 and not self.pacmod_enable:
            self.global_cmd.enable = True
            self.global_cmd.clear_override = True
            self.global_pub.publish(self.global_cmd)

            self.gear_cmd.command = GEAR_DRIVE
            self.gear_pub.publish(self.gear_cmd)

            self.brake_cmd.command = 0.0
            self.brake_pub.publish(self.brake_cmd)

            self.accel_cmd.command = 0.0
            self.accel_pub.publish(self.accel_cmd)

            self.turn_cmd.command = 3
            self.turn_pub.publish(self.turn_cmd)

            self.get_logger().warn(
                'Pacmod Disabled: Vehicle enabled and forward gear engaged')

        # ── DISABLE (LB only) — identical to pure_pursuit.py ─
        elif joy_enable == 0 and self.pacmod_enable:
            self.global_cmd.enable = False
            self.global_pub.publish(self.global_cmd)

            self.turn_cmd.command = 1
            self.turn_pub.publish(self.turn_cmd)

            self.get_logger().warn('Joystick Disabled: Vehicle disabled')

        # ── EXECUTE — pure_pursuit.py structure ───────────────
        elif joy_enable != 0 and self.pacmod_enable:

            curr_x, curr_y, curr_yaw = self.get_gem_state()

            # No plan yet — hold position, keep PACMod alive
            if not self.has_plan:
                self.get_logger().info(
                    "Waiting for /vlm_goal or /vlm_waypoints...",
                    throttle_duration_sec=5.0)
                self.accel_cmd.command = 0.0
                self.brake_cmd.command = 0.0
                self.accel_pub.publish(self.accel_cmd)
                self.brake_pub.publish(self.brake_cmd)
                self.global_cmd.enable = True
                self.global_pub.publish(self.global_cmd)
                self.gear_cmd.command = GEAR_DRIVE
                self.gear_pub.publish(self.gear_cmd)
                return

            path_x = np.array([p[0] for p in self.path])
            path_y = np.array([p[1] for p in self.path])
            n      = len(self.path)

            # Goal reached check
            gx, gy = self.goal_xy
            if math.hypot(curr_x - gx, curr_y - gy) < GOAL_TOL:
                self.get_logger().info("Goal reached! Stopping.")
                self.accel_cmd.command = 0.0
                self.brake_cmd.command = 0.3
                self.accel_pub.publish(self.accel_cmd)
                self.brake_pub.publish(self.brake_cmd)
                self.has_plan = False
                return

            # Pure pursuit — same logic as pure_pursuit.py,
            # operating on A* path instead of CSV waypoints
            dist_arr = np.hypot(path_x - curr_x, path_y - curr_y)
            closest  = int(np.argmin(dist_arr))

            ld = self.look_ahead + max(0.0, self.speed - 2.5) * 2.0

            self.goal_idx = closest
            for i in range(closest, n):
                if dist_arr[i] > ld:
                    self.goal_idx = i
                    break

            target_x = path_x[self.goal_idx]
            target_y = path_y[self.goal_idx]

            alpha     = math.atan2(target_y - curr_y, target_x - curr_x) - curr_yaw
            curvature = 0.0 if self.speed < 0.2 else 2.0 * math.sin(alpha) / ld
            steering_angle = math.atan(self.wheelbase * curvature)
            steering_wheel_angle = self.front2steer(math.degrees(steering_angle))

            self.steer_cmd.angular_position = math.radians(steering_wheel_angle)
            self.steer_pub.publish(self.steer_cmd)

            # Speed control — identical to pure_pursuit.py
            now         = self.get_clock().now().nanoseconds * 1e-9
            speed_error = self.desired_speed - self.speed
            if abs(speed_error) < 0.05:
                speed_error = 0.0
            throttle_cmd = self.pid_speed.get_control(now, speed_error)
            throttle_cmd = max(0.0, min(throttle_cmd, self.max_accel))

            self.accel_cmd.command = throttle_cmd
            self.brake_cmd.command = 0.0
            self.accel_pub.publish(self.accel_cmd)
            self.brake_pub.publish(self.brake_cmd)

            self.global_cmd.enable = True
            self.global_pub.publish(self.global_cmd)

            self.get_logger().info(
                f"Pos: ({curr_x:.2f}, {curr_y:.2f}), "
                f"Target: ({target_x:.2f}, {target_y:.2f}), "
                f"Speed: {self.speed:.2f}, "
                f"Throttle: {throttle_cmd:.2f}, "
                f"Steering: {steering_wheel_angle:.2f}, "
                f"Dist→Goal: {math.hypot(curr_x-gx, curr_y-gy):.1f}m")


# ═══════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
