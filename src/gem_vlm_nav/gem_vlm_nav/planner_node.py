#!/usr/bin/env python3
"""
planner_node.py — Receding-Horizon Path Planner + PACMod Controller
====================================================================
Subscribes to /vlm_waypoints (PoseArray from vlm_node).
On each new waypoint list: IMMEDIATELY preempts current plan,
runs A* from current position to waypoint[0] on live costmap,
and begins tracking. Advances through waypoint list as each
is reached. When list is exhausted and VLM hasn't replied yet:
enters CREEP mode (0.5 m/s, hold heading).

Low-level output is IDENTICAL to pure_pursuit_ros2.py:
  - Same PACMod topics and message types
  - Same joystick enable/disable (LB+RB / LB)
  - Same front2steer() polynomial conversion
  - Same PID speed controller + Butterworth speed filter
  - Same INSNavGeod heading conversion

Topics (in):
  /vlm_waypoints             — PoseArray from vlm_node
  /vlm_costmap               — OccupancyGrid from lidar_bev_node
  /vlm_status                — String status from vlm_node
  /navsatfix                 — Septentrio GNSS
  /insnavgeod                — Septentrio INS heading
  /pacmod/enabled            — PACMod enable status
  /pacmod/vehicle_speed_rpt  — Measured vehicle speed

Topics (out):  [identical to pure_pursuit_ros2.py]
  /pacmod/global_cmd
  /pacmod/shift_cmd
  /pacmod/steering_cmd
  /pacmod/accel_cmd
  /pacmod/brake_cmd
  /pacmod/turn_cmd

Visualization:
  /vlm_current_path          — nav_msgs/Path of current A* path (RViz)
"""

import math
import heapq
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
from scipy.interpolate import splprep, splev
import pymap3d as pm
import pygame
from visualization_msgs.msg import Marker, MarkerArray

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from std_msgs.msg import Bool, String
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from pacmod2_msgs.msg import (
    PositionWithSpeed, VehicleSpeedRpt,
    GlobalCmd, SystemCmdFloat, SystemCmdInt,
)
from septentrio_gnss_driver.msg import INSNavGeod


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

ORIGIN_LAT = 40.0927422
ORIGIN_LON = -88.2359639

X_MIN, X_MAX = -25.0, 75.0
Y_MIN, Y_MAX =  -5.0, 20.0

RESOLUTION  = 0.5    # m/cell — must match lidar_bev_node
INFLATE_R   = 1.5    # m — car half-width + safety margin

# Pure pursuit (matches pure_pursuit_ros2.py defaults)
LOOK_AHEAD    = 5.0
WHEELBASE     = 2.57
OFFSET        = 1.26   # GNSS antenna to rear axle
DESIRED_SPEED = 2.0
MAX_ACCEL     = 0.5
MAX_STEER_RAD = 0.6

# Creep mode
CREEP_SPEED   = 0.5    # m/s when waiting for next VLM plan
CREEP_TIMEOUT = 15.0   # s — full stop if no VLM update for this long

WP_TOL        = 1.8    # m — waypoint arrival tolerance
GOAL_TOL      = 1.5    # m — final goal tolerance

# PID (same as pure_pursuit_ros2.py)
PID_KP, PID_KI, PID_KD, PID_WG = 0.6, 0.0, 0.1, 10.0
FILTER_CUTOFF, FILTER_FS, FILTER_ORDER = 1.2, 30, 4

# front2steer polynomial (same as pure_pursuit_ros2.py)
STEER_A       = -0.1084
STEER_B       = 21.775
STEER_MAX_DEG = 35

GEAR_NEUTRAL  = 2
GEAR_DRIVE    = 3
CONTROL_HZ    = 20


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def ins_heading_to_yaw(h: float) -> float:
    return math.radians(90.0 - h) if h < 270.0 else math.radians(450.0 - h)

def normalize_angle(a: float) -> float:
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def world_to_cell(x, y):
    return int((x - X_MIN) / RESOLUTION), int((y - Y_MIN) / RESOLUTION)

def cell_to_world(c, r):
    return X_MIN + (c + 0.5) * RESOLUTION, Y_MIN + (r + 0.5) * RESOLUTION

def in_bounds(c, r, shape):
    return 0 <= r < shape[0] and 0 <= c < shape[1]


# ═══════════════════════════════════════════════════════════════
#  PID + FILTER  (verbatim from pure_pursuit_ros2.py)
# ═══════════════════════════════════════════════════════════════

class PID:
    def __init__(self, kp, ki, kd, wg=None):
        self.kp=kp; self.ki=ki; self.kd=kd; self.wg=wg
        self.iterm=0; self.last_e=0; self.last_t=None

    def reset(self):
        self.iterm=0; self.last_e=0; self.last_t=None

    def get_control(self, t, e):
        if self.last_t is None:
            dt = de = 0.0
        else:
            dt = t - self.last_t
            de = (e - self.last_e) / dt if dt > 0 else 0.0
        self.iterm += e * dt
        if self.wg:
            self.iterm = max(min(self.iterm, self.wg), -self.wg)
        self.last_e = e; self.last_t = t
        return self.kp * e + self.ki * self.iterm + self.kd * de


class OnlineFilter:
    def __init__(self, cutoff, fs, order):
        nyq = 0.5 * fs
        self.b, self.a = signal.butter(order, cutoff / nyq, btype='low', analog=False)
        self.z = signal.lfilter_zi(self.b, self.a)

    def get_data(self, v):
        out, self.z = signal.lfilter(self.b, self.a, [v], zi=self.z)
        return out[0]


# ═══════════════════════════════════════════════════════════════
#  GRID + A*
# ═══════════════════════════════════════════════════════════════

def build_base_grid():
    nc = int(np.ceil((X_MAX - X_MIN) / RESOLUTION))
    nr = int(np.ceil((Y_MAX - Y_MIN) / RESOLUTION))
    g  = np.zeros((nr, nc), dtype=np.uint8)
    w  = max(1, int(1.0 / RESOLUTION))
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

def astar(grid, sw, gw):
    sc = world_to_cell(*sw)
    gc = world_to_cell(*gw)

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
            if not in_bounds(nb[0], nb[1], grid.shape): continue
            if grid[nb[1], nb[0]] == 1: continue
            t = gs[cur] + cost
            if t < gs.get(nb, 1e9):
                came[nb] = cur
                gs[nb]   = t
                heapq.heappush(heap, (t + math.hypot(nb[0]-gc[0], nb[1]-gc[1]), nb))
    return None

def smooth_path(path, n=200):
    if not path or len(path) < 3: return path
    pts = np.array(path)
    _, idx = np.unique(pts, axis=0, return_index=True)
    pts = pts[np.sort(idx)]
    if len(pts) < 3: return path
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
        super().__init__('vlm_planner_node')

        # ── Joystick ───────────────────────────────────────────
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            self.get_logger().warn("No joystick — vehicle enable won't work!")
            self._joy = None
        else:
            self._joy = pygame.joystick.Joystick(0)
            self._joy.init()

        # ── Speed controller ───────────────────────────────────
        self.pid  = PID(PID_KP, PID_KI, PID_KD, wg=PID_WG)
        self.filt = OnlineFilter(FILTER_CUTOFF, FILTER_FS, FILTER_ORDER)

        # ── Vehicle state ──────────────────────────────────────
        self.lat = self.lon = 0.0
        self.heading      = 0.0
        self.speed        = 0.0
        self.car_x        = None
        self.car_y        = None
        self.car_yaw      = 0.0
        self.pacmod_enable = False
        self.gem_enable    = False   # True after we send PACMod enable sequence

        # ── Mission state ──────────────────────────────────────
        self.wp_queue     = []      # list of (x,y) from VLM
        self.wp_idx       = 0       # which waypoint we're heading to
        self.path         = None    # current A* smoothed path
        self.path_idx     = 0
        self.mode         = 'idle'  # idle | navigating | creep | arrived
        self.vlm_status   = 'waiting'
        self.last_wp_time = 0.0

        # ── Grid ───────────────────────────────────────────────
        self.grid = build_base_grid()
        self.car_marker_pub  = self.create_publisher(Marker, '/vlm_car_marker', 10)
        self.goal_marker_pub = self.create_publisher(MarkerArray, '/vlm_goal_markers', 10)

        # ── PACMod command objects ─────────────────────────────
        self.global_cmd = GlobalCmd(enable=False, clear_override=True)
        self.gear_cmd   = SystemCmdInt(command=GEAR_NEUTRAL)
        self.brake_cmd  = SystemCmdFloat(command=0.0)
        self.accel_cmd  = SystemCmdFloat(command=0.0)
        self.turn_cmd   = SystemCmdInt(command=1)
        self.steer_cmd  = PositionWithSpeed(
            angular_position=0.0, angular_velocity_limit=4.0)

        # ── Subscriptions ──────────────────────────────────────
        self.create_subscription(NavSatFix,       '/navsatfix',               self.gps_cb,    10)
        self.create_subscription(INSNavGeod,      '/insnavgeod',              self.ins_cb,    10)
        self.create_subscription(Bool,            '/pacmod/enabled',          self.enable_cb, 10)
        self.create_subscription(VehicleSpeedRpt, '/pacmod/vehicle_speed_rpt',self.speed_cb,  10)
        self.create_subscription(OccupancyGrid,   '/vlm_costmap',             self.costmap_cb,10)
        self.create_subscription(String,          '/vlm_status',              self.status_cb, 10)

        wp_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseArray, '/vlm_waypoints', self.waypoints_cb, wp_qos)

        # ── PACMod publishers ──────────────────────────────────
        self.global_pub = self.create_publisher(GlobalCmd,         '/pacmod/global_cmd',    10)
        self.gear_pub   = self.create_publisher(SystemCmdInt,      '/pacmod/shift_cmd',     10)
        self.brake_pub  = self.create_publisher(SystemCmdFloat,    '/pacmod/brake_cmd',     10)
        self.accel_pub  = self.create_publisher(SystemCmdFloat,    '/pacmod/accel_cmd',     10)
        self.turn_pub   = self.create_publisher(SystemCmdInt,      '/pacmod/turn_cmd',      10)
        self.steer_pub  = self.create_publisher(PositionWithSpeed, '/pacmod/steering_cmd',  10)
        self.path_pub   = self.create_publisher(Path,              '/vlm_current_path',     10)

        self.create_timer(1.0 / CONTROL_HZ, self.control_loop)
        self.get_logger().info("VLM Planner ready. Joystick LB+RB to enable.")

    # ══════════════════════════════════════════════════════════
    #  SENSOR CALLBACKS
    # ══════════════════════════════════════════════════════════

    def gps_cb(self, msg):
        self.lat = msg.latitude
        self.lon = msg.longitude
        try:
            e, n, _ = pm.geodetic2enu(
                self.lat, self.lon, 0, ORIGIN_LAT, ORIGIN_LON, 0)
            self.car_x, self.car_y = float(e), float(n)
        except Exception:
            pass

    def ins_cb(self, msg):
        self.heading = msg.heading
        if msg.heading is not None and not math.isnan(msg.heading):
            self.car_yaw = ins_heading_to_yaw(msg.heading)

    def speed_cb(self, msg):
        self.speed = self.filt.get_data(msg.vehicle_speed)

    def enable_cb(self, msg):
        self.pacmod_enable = msg.data

    def status_cb(self, msg):
        self.vlm_status = msg.data
        if msg.data == 'arrived':
            self.mode = 'arrived'

    def costmap_cb(self, msg: OccupancyGrid):
        """Receive live costmap from lidar_bev_node and rebuild grid."""
        nr   = msg.info.height
        nc   = msg.info.width
        data = np.array(msg.data, dtype=np.int8).reshape(nr, nc)
        self.grid = (data > 50).astype(np.uint8)

    # ══════════════════════════════════════════════════════════
    #  WAYPOINT CALLBACK — preempt and replan
    # ══════════════════════════════════════════════════════════

    def waypoints_cb(self, msg: PoseArray):
        if not msg.poses:
            return
        if self.car_x is None:
            self.get_logger().warn("Waypoints received but GPS not ready.")
            return

        new_wps = [(p.position.x, p.position.y) for p in msg.poses]
        self.get_logger().info(
            f"VLM waypoints ({len(new_wps)}): "
            + "  ".join(f"({x:.1f},{y:.1f})" for x, y in new_wps))

        # Preempt current plan immediately
        self.wp_queue     = new_wps
        self.wp_idx       = 0
        self.last_wp_time = self.get_clock().now().nanoseconds * 1e-9
        self.mode         = 'navigating'

        self._plan_to_current_wp()

    def _plan_to_current_wp(self):
        """Run A* to wp_queue[wp_idx]."""
        if self.wp_idx >= len(self.wp_queue) or self.car_x is None:
            return

        target   = self.wp_queue[self.wp_idx]
        inf_grid = inflate_grid(self.grid, INFLATE_R)
        raw      = astar(inf_grid, (self.car_x, self.car_y), target)

        if raw is None:
            self.get_logger().error(
                f"A* failed to WP{self.wp_idx} ({target[0]:.1f},{target[1]:.1f}). "
                "Entering creep mode.")
            self.mode = 'creep'
            return

        self.path     = smooth_path(raw)
        self.path_idx = 0
        self._publish_path_viz()
        self.get_logger().info(
            f"A* to WP{self.wp_idx} ({target[0]:.1f},{target[1]:.1f}): "
            f"{len(self.path)} smoothed pts.")

    # ══════════════════════════════════════════════════════════
    #  CONTROL LOOP  (20 Hz)
    # ══════════════════════════════════════════════════════════

    def _enable_pacmod(self):
        """Send the PACMod enable + DRIVE gear sequence."""
        self.global_cmd.enable         = True
        self.global_cmd.clear_override = True
        self.global_pub.publish(self.global_cmd)
        self.gear_cmd.command = GEAR_DRIVE
        self.gear_pub.publish(self.gear_cmd)
        self.brake_cmd.command = 0.0; self.brake_pub.publish(self.brake_cmd)
        self.accel_cmd.command = 0.0; self.accel_pub.publish(self.accel_cmd)
        self.turn_cmd.command  = 3;   self.turn_pub.publish(self.turn_cmd)
        self.gem_enable = True
        self.get_logger().warn('PACMod enabled, DRIVE gear engaged')

    def _keep_pacmod_alive(self):
        """Re-assert enable + DRIVE every cycle so PACMod stays engaged."""
        self.global_cmd.enable = True
        self.global_pub.publish(self.global_cmd)
        self.gear_cmd.command = GEAR_DRIVE
        self.gear_pub.publish(self.gear_cmd)

    def control_loop(self):
        self._publish_dashboard_markers()
        joy = self._check_joystick()

        # ── Joystick ENABLE (LB + RB) ─────────────────────────
        if joy == 1 and not self.pacmod_enable:
            self._enable_pacmod()

        # ── Joystick DISABLE (LB only) ────────────────────────
        elif joy == 0 and self.pacmod_enable:
            self.global_cmd.enable = False
            self.global_pub.publish(self.global_cmd)
            self.turn_cmd.command = 1; self.turn_pub.publish(self.turn_cmd)
            self.gem_enable = False
            self.get_logger().warn('PACMod disabled by joystick')

        # ── No joystick: auto-enable once GPS is ready ────────
        elif self._joy is None and not self.gem_enable and self.car_x is not None:
            self._enable_pacmod()

        # ── Execute controller ────────────────────────────────
        #    Runs when joystick isn't disabling AND PACMod is
        #    enabled (either by callback or by our own enable)
        if joy != 0 and (self.pacmod_enable or self.gem_enable):
            if self.car_x is None:
                self._keep_pacmod_alive()
                return

            if self.mode == 'arrived':
                self.get_logger().info("Mission complete.", once=True)
                self._stop()
                return

            if self.mode == 'idle':
                # Keep PACMod alive in idle so it doesn't disengage
                self._keep_pacmod_alive()
                return

            # ── Check waypoint arrival ─────────────────────────
            if self.wp_queue and self.wp_idx < len(self.wp_queue):
                twx, twy = self.wp_queue[self.wp_idx]
                if math.hypot(self.car_x - twx, self.car_y - twy) < WP_TOL:
                    self.wp_idx += 1
                    if self.wp_idx < len(self.wp_queue):
                        self.get_logger().info(
                            f"WP{self.wp_idx-1} reached → replanning to WP{self.wp_idx}")
                        self._plan_to_current_wp()
                    else:
                        self.get_logger().info(
                            "All waypoints reached → CREEP mode (waiting for VLM)")
                        self.mode = 'creep'

            # ── Creep + timeout ────────────────────────────────
            if self.mode == 'creep':
                now = self.get_clock().now().nanoseconds * 1e-9
                if now - self.last_wp_time > CREEP_TIMEOUT:
                    self.get_logger().warn(
                        f"No VLM update in {CREEP_TIMEOUT}s → full stop")
                    self._stop()
                    return
                self._execute_creep()
                return

            # ── Normal navigation ──────────────────────────────
            if self.path is None:
                self._keep_pacmod_alive()
                return

            # Vehicle state with antenna offset (same as pure_pursuit_ros2.py)
            lx, ly   = self._gps_to_local(self.lon, self.lat)
            curr_yaw = ins_heading_to_yaw(self.heading)
            curr_x   = lx - OFFSET * math.cos(curr_yaw)
            curr_y   = ly - OFFSET * math.sin(curr_yaw)

            # Steering
            steer_deg = self._pure_pursuit(curr_x, curr_y, curr_yaw)
            self.steer_cmd.angular_position = math.radians(steer_deg)
            self.steer_pub.publish(self.steer_cmd)

            # Speed PID
            now       = self.get_clock().now().nanoseconds * 1e-9
            speed_err = DESIRED_SPEED - self.speed
            if abs(speed_err) < 0.05: speed_err = 0.0
            throttle  = max(0.0, min(self.pid.get_control(now, speed_err), MAX_ACCEL))
            self.accel_cmd.command = throttle
            self.brake_cmd.command = 0.0
            self.accel_pub.publish(self.accel_cmd)
            self.brake_pub.publish(self.brake_cmd)

            # Re-assert enable + gear every cycle
            self._keep_pacmod_alive()

            # Log
            if self.path_idx % 20 == 0 and self.wp_queue:
                twx, twy = self.wp_queue[min(self.wp_idx, len(self.wp_queue)-1)]
                self.get_logger().info(
                    f"({curr_x:.1f},{curr_y:.1f}) → WP{self.wp_idx}"
                    f"({twx:.1f},{twy:.1f}) "
                    f"dist:{math.hypot(curr_x-twx,curr_y-twy):.1f}m "
                    f"spd:{self.speed:.2f} thr:{throttle:.2f} "
                    f"steer:{steer_deg:.1f}° [{self.mode}/{self.vlm_status}]")

    # ══════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════

    def _pure_pursuit(self, cx, cy, cyaw) -> float:
        """
        Pure pursuit. Returns steering WHEEL angle in degrees.
        Mirrors control logic of pure_pursuit_ros2.py exactly.
        """
        px = np.array([p[0] for p in self.path])
        py = np.array([p[1] for p in self.path])
        n  = len(self.path)

        dist_arr = np.hypot(px - cx, py - cy)
        closest  = int(np.argmin(dist_arr))
        ld       = LOOK_AHEAD + max(0.0, self.speed - 2.5) * 2.0

        goal_idx = closest
        for i in range(closest, n):
            if dist_arr[i] > ld:
                goal_idx = i
                break
        self.path_idx = min(goal_idx, n - 1)

        tx = px[self.path_idx]
        ty = py[self.path_idx]

        alpha     = math.atan2(ty - cy, tx - cx) - cyaw
        alpha     = normalize_angle(alpha)
        curvature = 0.0 if self.speed < 0.2 else 2.0 * math.sin(alpha) / ld
        front_deg = math.degrees(math.atan(WHEELBASE * curvature))

        return self._front2steer(front_deg)

    def _execute_creep(self):
        """Hold heading, drive at CREEP_SPEED while waiting for VLM."""
        self.steer_cmd.angular_position = 0.0
        self.steer_pub.publish(self.steer_cmd)
        now      = self.get_clock().now().nanoseconds * 1e-9
        throttle = max(0.0, min(
            self.pid.get_control(now, CREEP_SPEED - self.speed), MAX_ACCEL))
        self.accel_cmd.command = throttle
        self.brake_cmd.command = 0.0
        self.accel_pub.publish(self.accel_cmd)
        self.brake_pub.publish(self.brake_cmd)
        self.global_cmd.enable = True
        self.global_pub.publish(self.global_cmd)

    def _stop(self):
        self.accel_cmd.command = 0.0; self.accel_pub.publish(self.accel_cmd)
        self.brake_cmd.command = 0.3; self.brake_pub.publish(self.brake_cmd)
        # Keep PACMod enabled so the vehicle can resume when new goals arrive
        self.global_cmd.enable = True; self.global_pub.publish(self.global_cmd)
        self.gear_cmd.command  = GEAR_DRIVE; self.gear_pub.publish(self.gear_cmd)

    def _check_joystick(self) -> int:
        if self._joy is None: return 2
        pygame.event.pump()
        try:
            lb = self._joy.get_button(6)
            rb = self._joy.get_button(7)
        except pygame.error:
            return 2
        if lb and rb:      return 1
        if lb and not rb:  return 0
        return 2

    def _front2steer(self, f_deg: float) -> float:
        """Exact copy of front2steer() from pure_pursuit_ros2.py."""
        f_deg = max(min(f_deg, STEER_MAX_DEG), -STEER_MAX_DEG)
        a     = abs(f_deg)
        s     = STEER_A * a**2 + STEER_B * a
        return round(s if f_deg >= 0 else -s, 2)

    def _gps_to_local(self, lon, lat):
        x, y, _ = pm.geodetic2enu(lat, lon, 0, ORIGIN_LAT, ORIGIN_LON, 0)
        return float(x), float(y)

    def _publish_path_viz(self):
        if not self.path: return
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

    def _publish_dashboard_markers(self):
        # 1. Red Marker for the Car
        if self.car_x is not None:
            car_marker = Marker()
            car_marker.header.stamp = self.get_clock().now().to_msg()
            car_marker.header.frame_id = 'map'
            car_marker.ns = 'car'
            car_marker.id = 0
            car_marker.type = Marker.CUBE
            car_marker.action = Marker.ADD
            car_marker.pose.position.x = self.car_x
            car_marker.pose.position.y = self.car_y
            car_marker.pose.position.z = 0.5
            car_marker.pose.orientation.z = math.sin(self.car_yaw / 2.0)
            car_marker.pose.orientation.w = math.cos(self.car_yaw / 2.0)
            car_marker.scale.x = 2.9  # Physical length of GEM e4
            car_marker.scale.y = 1.4  # Physical width
            car_marker.scale.z = 1.0
            car_marker.color.r = 1.0  # RED
            car_marker.color.g = 0.0
            car_marker.color.b = 0.0
            car_marker.color.a = 0.8
            self.car_marker_pub.publish(car_marker)

        # 2. Yellow Markers for the Goals
        if self.wp_queue:
            goal_array = MarkerArray()
            for i, (wx, wy) in enumerate(self.wp_queue):
                m = Marker()
                m.header.stamp = self.get_clock().now().to_msg()
                m.header.frame_id = 'map'
                m.ns = 'goals'
                m.id = i
                m.type = Marker.CYLINDER
                m.action = Marker.ADD
                m.pose.position.x = float(wx)
                m.pose.position.y = float(wy)
                m.pose.position.z = 0.1
                m.scale.x = 1.5
                m.scale.y = 1.5
                m.scale.z = 0.2
                m.color.r = 1.0  # YELLOW
                m.color.g = 1.0
                m.color.b = 0.0
                m.color.a = 0.9
                goal_array.markers.append(m)
            self.goal_marker_pub.publish(goal_array)


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