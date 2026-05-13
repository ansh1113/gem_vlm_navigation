#!/usr/bin/env python3
"""
planner_node.py — Segment-Replanning A* Planner + Stanley Controller
=====================================================================

Architecture
------------
                        ┌─────────────────────────────────────┐
  /vlm_waypoints  ──────►                                     │
  /vlm_costmap    ──────►   PlannerNode                       │
  /gazebo/...     ──────►                                     │
                        │  1. A* from current_pos through     │
                        │     all VLM waypoints (chained)     │
                        │  2. Spline-smooth the raw path      │
                        │  3. Stanley tracks the smooth path  │
                        │  4. On arrival at each intermediate │
                        │     VLM waypoint → replan with      │
                        │     fresh costmap                   │
                        │  5. On arrival at final waypoint    │
                        │     → stop, publish "arrived"       │
                        └─────────────────────────────────────┘
                                        │
                                  /ackermann_cmd
                                  /planner_path
                                  /planner_status

Topics (in):
    /vlm_waypoints          — PoseArray, ENU world frame
    /vlm_costmap            — OccupancyGrid, ENU world frame
    /gazebo/get_model_state — ground-truth pose + twist

Topics (out):
    /ackermann_cmd          — AckermannDrive
    /planner_path           — nav_msgs/Path  (RViz visualisation)
    /planner_status         — String  ("planning"|"tracking"|"arrived"|"idle")

Usage:
    source devel/setup.bash
    python3 planner_node.py
"""

import csv
import math
import heapq
import os
import tempfile
import threading
import time
from collections import defaultdict

import numpy as np
import rospy
from scipy.interpolate import splprep, splev

from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String
from gazebo_msgs.srv import GetModelState
from tf.transformations import euler_from_quaternion


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Gazebo model name — verify with: rostopic echo /gazebo/model_states | grep name
MODEL_NAME = 'gem_e4'

# Navigatable zone — MUST match lidar_bev_node.py and vlm_node.py exactly
X_MIN = -50
X_MAX =  40
Y_MIN = -12
Y_MAX =  5

# Vehicle
WHEELBASE = 1.75        # metres — matches the working stanley_sim.py

# Stanley
STANLEY_K         = 0.45   # cross-track gain
STANLEY_HZ        = 20.0   # control loop rate (Hz)
SPEED_BASE        = 2.8    # normal cruise speed (m/s)
SPEED_SLOW        = 1.5    # near-obstacle speed
SPEED_TURN        = 1.8    # sharp-turn speed
SPEED_APPROACH    = 1.2    # final-approach ramp top
SPEED_MIN         = 0.3    # absolute floor while moving
MAX_STEER_RAD     = 0.6    # steering clamp

# Speed trigger thresholds
SLOW_OBSTACLE_M   = 3.0    # metres — slow when nearest obstacle < this
SLOW_STEER_RAD    = 0.35   # radians — slow when |steer| > this
SLOW_FINAL_M      = 5.0    # metres — begin approach ramp inside this

# Arrival radii
WP_ARRIVE_M       = 2.5    # metres — advance to next VLM waypoint
FINAL_ARRIVE_M    = 1.5    # metres — declare "arrived" at final goal

# A* / costmap
INFLATE_CELLS     = 1      # obstacle inflation in grid cells (1 cell = 0.5 m)
YAW_PENALTY       = 2.0    # A* turn-cost weight (from reference planner)
SPLINE_POINTS     = 250    # output points from splprep smoother
REPLAN_INTERVAL_S = 3.0    # replan mid-segment every N seconds while tracking


# ═══════════════════════════════════════════════════════════════
#  ZONE CLAMPING
# ═══════════════════════════════════════════════════════════════

def _clamp_to_zone(x, y, margin=0.0):
    return (
        max(X_MIN + margin, min(X_MAX - margin, float(x))),
        max(Y_MIN + margin, min(Y_MAX - margin, float(y))),
    )


# ═══════════════════════════════════════════════════════════════
#  COSTMAP WRAPPER
# ═══════════════════════════════════════════════════════════════

class CostmapInfo:
    """Wraps a nav_msgs/OccupancyGrid for coordinate conversion and lookup."""

    def __init__(self, msg):
        self.res  = msg.info.resolution
        self.nc   = msg.info.width
        self.nr   = msg.info.height
        self.ox   = msg.info.origin.position.x
        self.oy   = msg.info.origin.position.y
        raw       = np.array(msg.data, dtype=np.int8).reshape(self.nr, self.nc)
        self.grid = (raw > 0).astype(np.uint8)   # 1=occupied 0=free

    def world_to_cell(self, wx, wy):
        c = int((wx - self.ox) / self.res)
        r = int((wy - self.oy) / self.res)
        return r, c

    def cell_to_world(self, r, c):
        return (self.ox + (c + 0.5) * self.res,
                self.oy + (r + 0.5) * self.res)

    def in_bounds(self, r, c):
        return 0 <= r < self.nr and 0 <= c < self.nc

    def nearest_obstacle_dist(self, wx, wy):
        r0, c0 = self.world_to_cell(wx, wy)
        search  = int(SLOW_OBSTACLE_M / self.res) + 2
        best    = SLOW_OBSTACLE_M + 1.0
        for dr in range(-search, search + 1):
            for dc in range(-search, search + 1):
                rr, cc = r0 + dr, c0 + dc
                if self.in_bounds(rr, cc) and self.grid[rr, cc]:
                    d = math.hypot(dr, dc) * self.res
                    if d < best:
                        best = d
        return best


# ═══════════════════════════════════════════════════════════════
#  A*
# ═══════════════════════════════════════════════════════════════

def _inflate(grid, radius):
    if radius == 0:
        return grid.copy()
    out = grid.copy()
    nr, nc = grid.shape
    for r, c in zip(*np.where(grid > 0)):
        out[max(0, r-radius):min(nr, r+radius+1),
            max(0, c-radius):min(nc, c+radius+1)] = 1
    return out


def _snap_to_free(grid, r, c):
    """Spiral outward from (r,c) to find nearest free cell."""
    nr, nc = grid.shape
    if 0 <= r < nr and 0 <= c < nc and grid[r, c] == 0:
        return r, c
    for radius in range(1, 20):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < nr and 0 <= cc < nc and grid[rr, cc] == 0:
                    return rr, cc
    return r, c


def _angle_diff(a, b):
    d = a - b
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


def astar(grid, start, goal, start_yaw=0.0, yaw_penalty=YAW_PENALTY):
    """
    Yaw-penalised 8-connected A* on a binary grid.
    Incorporates turn cost so the raw path already prefers gentle curves.

    Parameters
    ----------
    grid        : (NR, NC) uint8 — 1=obstacle, 0=free
    start/goal  : (row, col)
    start_yaw   : vehicle heading at start (radians, ENU)
    yaw_penalty : cost weight per radian of heading change

    Returns list of (row, col), or [] on failure.
    """
    nr, nc = grid.shape

    MOVES = [
        ( 0,  1, 1.0,          0.0),
        ( 0, -1, 1.0,          math.pi),
        ( 1,  0, 1.0,          math.pi / 2),
        (-1,  0, 1.0,         -math.pi / 2),
        ( 1,  1, math.sqrt(2), math.pi / 4),
        ( 1, -1, math.sqrt(2),-math.pi / 4),
        (-1,  1, math.sqrt(2), 3 * math.pi / 4),
        (-1, -1, math.sqrt(2),-3 * math.pi / 4),
    ]

    def valid(r, c):
        return 0 <= r < nr and 0 <= c < nc and grid[r, c] == 0

    start = _snap_to_free(grid, *start)
    goal  = _snap_to_free(grid, *goal)

    if not valid(*start) or not valid(*goal):
        return []
    if start == goal:
        return [start]

    def heur(r, c):
        dr = abs(r - goal[0]); dc = abs(c - goal[1])
        return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)

    g_cost  = defaultdict(lambda: float('inf'))
    g_cost[start] = 0.0
    parent  = {}
    heading = {start: start_yaw}
    heap    = [(heur(*start), start[0], start[1])]

    while heap:
        f, r, c = heapq.heappop(heap)
        cur = (r, c)

        if cur == goal:
            path = []
            while cur in parent:
                path.append(cur)
                cur = parent[cur]
            path.append(start)
            path.reverse()
            return path

        if f > g_cost[cur] + heur(*cur) + 1e-6:
            continue

        cur_h = heading[cur]
        for dr, dc, dist, move_h in MOVES:
            nb = (r + dr, c + dc)
            if not valid(*nb):
                continue
            ng = g_cost[cur] + dist + yaw_penalty * abs(_angle_diff(move_h, cur_h))
            if ng < g_cost[nb]:
                g_cost[nb]  = ng
                parent[nb]  = cur
                heading[nb] = move_h
                heapq.heappush(heap, (ng + heur(*nb), nb[0], nb[1]))

    return []


# ═══════════════════════════════════════════════════════════════
#  PATH SMOOTHING + HELPERS
# ═══════════════════════════════════════════════════════════════

def _smooth(world_pts, n_points=SPLINE_POINTS):
    """
    splprep/splev B-spline smoothing (same approach as reference Planner._smooth).
    Deduplicates, fits spline with s=2.0, evaluates at n_points.
    Clamps output to navigatable zone. Falls back to raw on failure.
    """
    if not world_pts or len(world_pts) < 3:
        return [_clamp_to_zone(x, y) for x, y in world_pts]

    pts = np.array(world_pts, dtype=np.float64)
    _, idx = np.unique(pts, axis=0, return_index=True)
    pts = pts[np.sort(idx)]

    if len(pts) < 3:
        return [_clamp_to_zone(*p) for p in pts]

    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]],
                         s=2.0, k=min(3, len(pts) - 1))
        xs, ys = splev(np.linspace(0, 1, n_points), tck)
        return [_clamp_to_zone(float(x), float(y)) for x, y in zip(xs, ys)]
    except Exception as e:
        rospy.logwarn("[planner] Spline failed (%s) — using raw path.", str(e))
        return [_clamp_to_zone(x, y) for x, y in world_pts]


def _compute_yaws(path):
    if len(path) < 2:
        return [0.0] * len(path)
    yaws = []
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        yaws.append(math.atan2(dy, dx))
    yaws.append(yaws[-1])
    return yaws


def _pi_2_pi(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


# ═══════════════════════════════════════════════════════════════
#  STANLEY CONTROLLER
# ═══════════════════════════════════════════════════════════════

class Stanley:
    """
    Stanley controller that tracks an in-memory list of (x, y) waypoints.

    Logic follows stanley_sim.py exactly:
      - Front-axle reference point
      - WHEELBASE = 1.75 m
      - model_name = 'gem_e4'
      - target_index clamped to len-2
      - Braking loop on halt

    Speed scheduling (obstacle, sharp turn, final approach) layered on top.
    """

    def __init__(self, get_state_fn, ackermann_pub):
        self._get_state = get_state_fn
        self._pub       = ackermann_pub
        self.path_x     = []
        self.path_y     = []

    def load_path(self, path):
        self.path_x = [p[0] for p in path]
        self.path_y = [p[1] for p in path]

    def get_vehicle_state(self):
        try:
            resp = self._get_state(model_name=MODEL_NAME,
                                   relative_entity_name='world')
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(5.0, "[stanley] get_model_state: %s", str(e))
            return None, None, None, None, None
        p = resp.pose.position
        o = resp.pose.orientation
        v = resp.twist.linear
        _, _, yaw = euler_from_quaternion([o.x, o.y, o.z, o.w])
        return (round(float(p.x), 4), round(float(p.y), 4),
                round(float(v.x), 4), round(float(v.y), 4),
                round(float(yaw),  4))

    def _send(self, speed, steer):
        msg = AckermannDrive()
        msg.speed          = float(speed)
        msg.steering_angle = float(steer)
        self._pub.publish(msg)

    def stop(self):
        self._send(0.0, 0.0)

    def brake_to_halt(self, rate):
        """Negative-speed braking loop from stanley_sim.py."""
        rospy.loginfo("[stanley] Braking to halt.")
        while not rospy.is_shutdown():
            _, _, xd, yd, _ = self.get_vehicle_state()
            if xd is None:
                break
            if math.hypot(xd, yd) < 0.05:
                self._send(0.0, 0.0)
                break
            self._send(-2.8, 0.0)
            rate.sleep()

    def step(self, costmap=None):
        """
        One Stanley control tick.

        Returns 'tracking', 'arrived', or None (state unavailable).
        """
        if not self.path_x:
            return 'arrived'

        curr_x, curr_y, xd, yd, curr_yaw = self.get_vehicle_state()
        if curr_x is None:
            return None

        # Front axle position
        front_x = curr_x + WHEELBASE * math.cos(curr_yaw)
        front_y = curr_y + WHEELBASE * math.sin(curr_yaw)

        # Closest path point (by front-axle distance) — clamped to len-2
        dx = [front_x - px for px in self.path_x]
        dy = [front_y - py for py in self.path_y]
        target_index = int(np.argmin(np.hypot(dx, dy)))
        target_index = min(target_index, len(self.path_x) - 2)

        # Cross-track error (signed, via rotated front-axle vector)
        front_axle_vec = np.array([math.cos(curr_yaw - math.pi / 2.0),
                                   math.sin(curr_yaw - math.pi / 2.0)])
        vec_t2f = np.array([dx[target_index], dy[target_index]])
        ef = float(np.dot(vec_t2f, front_axle_vec))

        # Heading error from path tangent
        theta_p = math.atan2(
            self.path_y[target_index + 1] - self.path_y[target_index],
            self.path_x[target_index + 1] - self.path_x[target_index])
        theta_e = _pi_2_pi(theta_p - curr_yaw)

        f_vel = max(math.hypot(xd, yd), 0.1)

        # Stanley law
        delta = theta_e + math.atan2(STANLEY_K * ef, f_vel)
        delta = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, delta))

        # ── Speed schedule ────────────────────────────────────
        speed = SPEED_BASE

        if costmap is not None:
            obs_d = costmap.nearest_obstacle_dist(curr_x, curr_y)
            if obs_d < SLOW_OBSTACLE_M:
                t = obs_d / SLOW_OBSTACLE_M
                speed = min(speed, SPEED_SLOW + t * (SPEED_BASE - SPEED_SLOW))

        if abs(delta) > SLOW_STEER_RAD:
            speed = min(speed, SPEED_TURN)

        dist_end = math.hypot(curr_x - self.path_x[-1],
                              curr_y - self.path_y[-1])
        if dist_end < SLOW_FINAL_M:
            t = dist_end / SLOW_FINAL_M
            speed = min(speed, SPEED_MIN + t * (SPEED_APPROACH - SPEED_MIN))

        speed = max(speed, SPEED_MIN)

        self._send(speed, delta)

        rospy.logdebug(
            "[stanley] ef=%.3f  θe=%.1f°  δ=%.3f  spd=%.2f",
            ef, math.degrees(theta_e), delta, speed)

        return 'tracking'


# ═══════════════════════════════════════════════════════════════
#  PLANNER NODE
# ═══════════════════════════════════════════════════════════════

class PlannerNode:

    def __init__(self):
        rospy.init_node('planner_node', anonymous=False)

        # ── State ─────────────────────────────────────────────
        self.vlm_waypoints  = []     # list of (x,y) clamped to zone
        self.latest_costmap = None   # CostmapInfo, updated by subscriber
        self._lock          = threading.Lock()

        # ── Gazebo service ────────────────────────────────────
        rospy.loginfo("[planner] Waiting for /gazebo/get_model_state …")
        rospy.wait_for_service('/gazebo/get_model_state')
        self._get_state = rospy.ServiceProxy('/gazebo/get_model_state',
                                             GetModelState)
        rospy.loginfo("[planner] Gazebo service ready.")

        # ── Publishers ────────────────────────────────────────
        self._ack_pub    = rospy.Publisher('/ackermann_cmd',  AckermannDrive,
                                           queue_size=1)
        self._path_pub   = rospy.Publisher('/planner_path',   Path,
                                           queue_size=1, latch=True)
        self._status_pub = rospy.Publisher('/planner_status', String,
                                           queue_size=1, latch=True)

        # ── Stanley ───────────────────────────────────────────
        self._stanley      = Stanley(self._get_state, self._ack_pub)
        self._stanley_rate = rospy.Rate(STANLEY_HZ)

        # ── Subscribers (start after everything else is ready) ─
        rospy.Subscriber('/vlm_waypoints', PoseArray,     self._wp_cb,      queue_size=1)
        rospy.Subscriber('/vlm_costmap',   OccupancyGrid, self._costmap_cb, queue_size=1)

        self._publish_status("idle")
        rospy.loginfo("[planner] Ready.  Waiting for /vlm_waypoints …")

        # Main loop runs in this thread
        self._run()

    # ── ROS callbacks ─────────────────────────────────────────

    def _wp_cb(self, msg):
        wps = [_clamp_to_zone(p.position.x, p.position.y) for p in msg.poses]
        with self._lock:
            self.vlm_waypoints = wps
        if wps:
            rospy.loginfo("[planner] New VLM waypoints (%d). Will replan.", len(wps))
            for i, (wx, wy) in enumerate(wps):
                rospy.loginfo("[planner]   wp[%d] = (%.3f, %.3f)", i, wx, wy)

    def _costmap_cb(self, msg):
        with self._lock:
            self.latest_costmap = CostmapInfo(msg)

    # ── Main loop ─────────────────────────────────────────────

    def _run(self):
        """
        Outer loop:
          1. Wait for waypoints + costmap.
          2. A* from current pos through ALL remaining VLM waypoints (chained).
          3. Smooth → load into Stanley.
          4. Stanley tracks until it arrives at waypoints[0].
          5. Pop waypoints[0]; if more remain → replan with fresh costmap.
          6. If none remain → brake + "arrived".
        """
        while not rospy.is_shutdown():

            with self._lock:
                wps = list(self.vlm_waypoints)
                cm  = self.latest_costmap

            if not wps or cm is None:
                self._stanley_rate.sleep()
                continue

            # Get current pose
            car_x, car_y, _, _, car_yaw = self._stanley.get_vehicle_state()
            if car_x is None:
                self._stanley_rate.sleep()
                continue

            # Already at final goal?
            fx, fy = wps[-1]
            dist_to_final = math.hypot(car_x - fx, car_y - fy)
            rospy.loginfo(
                "[planner] car=(%.3f, %.3f)  final_wp=(%.3f, %.3f)  dist=%.3f  threshold=%.3f",
                car_x, car_y, fx, fy, dist_to_final, FINAL_ARRIVE_M)
            if dist_to_final < FINAL_ARRIVE_M:
                rospy.loginfo("[planner] Already at final goal — done.")
                self._stanley.brake_to_halt(self._stanley_rate)
                self._publish_status("arrived")
                with self._lock:
                    self.vlm_waypoints = []
                continue

            # ── Plan ──────────────────────────────────────────
            self._publish_status("planning")
            path = self._plan_path(car_x, car_y, car_yaw, wps, cm)

            if path is None or len(path) < 2:
                rospy.logwarn_throttle(2.0,
                    "[planner] A* failed — retrying next cycle.")
                self._stanley_rate.sleep()
                continue

            # ── Load + track ───────────────────────────────────
            self._stanley.load_path(path)
            self._publish_path(path)
            self._publish_status("tracking")
            rospy.loginfo("[planner] Tracking %d-pt path → first wp (%.1f, %.1f).",
                          len(path), wps[0][0], wps[0][1])

            reached = self._track_to_waypoint(wps[0])

            # ── Advance ────────────────────────────────────────
            if reached:
                with self._lock:
                    if self.vlm_waypoints:
                        self.vlm_waypoints.pop(0)
                    remaining = list(self.vlm_waypoints)

                if not remaining:
                    self._stanley.brake_to_halt(self._stanley_rate)
                    self._publish_status("arrived")
                    rospy.loginfo("[planner] Final VLM waypoint reached.")
                else:
                    rospy.loginfo(
                        "[planner] Intermediate wp reached. "
                        "%d remaining. Replanning …", len(remaining))
                    # Loop continues → fresh costmap + replan

    # ── Planning ──────────────────────────────────────────────

    def _plan_path(self, car_x, car_y, car_yaw, wps, cm):
        """
        A* from current_pos → wps[0] → … → wps[-1], all chained.
        Returns smoothed list of (x, y) world points, or None on failure.
        """
        inflated = _inflate(cm.grid, INFLATE_CELLS)
        chain    = [_clamp_to_zone(car_x, car_y)] + list(wps)
        full     = []
        t0       = time.time()

        for i in range(len(chain) - 1):
            sx, sy = chain[i]
            gx, gy = chain[i + 1]

            sr, sc = cm.world_to_cell(sx, sy)
            gr, gc = cm.world_to_cell(gx, gy)

            sr = max(0, min(cm.nr - 1, sr)); sc = max(0, min(cm.nc - 1, sc))
            gr = max(0, min(cm.nr - 1, gr)); gc = max(0, min(cm.nc - 1, gc))

            seg_yaw = car_yaw if i == 0 else 0.0
            cells   = astar(inflated, (sr, sc), (gr, gc),
                            start_yaw=seg_yaw, yaw_penalty=YAW_PENALTY)

            if not cells:
                rospy.logwarn("[planner] A* failed segment %d→%d "
                              "(%.1f,%.1f)→(%.1f,%.1f).",
                              i, i+1, sx, sy, gx, gy)
                return None

            seg = [_clamp_to_zone(*cm.cell_to_world(r, c)) for r, c in cells]
            if full:
                seg = seg[1:]    # drop duplicate junction point
            full.extend(seg)

        smooth = _smooth(full, SPLINE_POINTS)
        rospy.loginfo("[planner] A*+smooth %.3f s → %d pts",
                      time.time() - t0, len(smooth))
        return smooth

    # ── Tracking ──────────────────────────────────────────────

    def _track_to_waypoint(self, target_wp):
        """
        Run Stanley at STANLEY_HZ until the vehicle reaches target_wp
        within WP_ARRIVE_M, or until waypoints are replaced by a new
        VLM command, or rospy shuts down.

        Every REPLAN_INTERVAL_S seconds, replans from the current pose
        to the remaining VLM waypoints using the latest costmap, so the
        path stays clear of obstacles that appear mid-segment.

        Returns True = reached, False = interrupted.
        """
        tx, ty = target_wp
        last_replan = time.time()

        while not rospy.is_shutdown():

            # Detect new VLM command (waypoints replaced)
            with self._lock:
                current_wps = list(self.vlm_waypoints)
            if not current_wps or current_wps[0] != target_wp:
                rospy.loginfo("[planner] Waypoints changed — interrupting track.")
                return False

            # Arrival check
            car_x, car_y, _, _, car_yaw = self._stanley.get_vehicle_state()
            if car_x is not None:
                if math.hypot(car_x - tx, car_y - ty) < WP_ARRIVE_M:
                    return True

            # Periodic mid-segment replan
            now = time.time()
            if car_x is not None and (now - last_replan) >= REPLAN_INTERVAL_S:
                with self._lock:
                    cm  = self.latest_costmap
                    wps = list(self.vlm_waypoints)
                if cm is not None and wps:
                    rospy.loginfo("[planner] Mid-segment replan …")
                    new_path = self._plan_path(car_x, car_y, car_yaw, wps, cm)
                    if new_path and len(new_path) >= 2:
                        self._stanley.load_path(new_path)
                        self._publish_path(new_path)
                        rospy.loginfo("[planner] Mid-segment replan → %d pts.",
                                      len(new_path))
                    else:
                        rospy.logwarn("[planner] Mid-segment replan failed — "
                                      "keeping current path.")
                last_replan = now

            # Stanley step with latest costmap
            with self._lock:
                cm = self.latest_costmap

            result = self._stanley.step(costmap=cm)
            if result == 'arrived':   # Stanley exhausted its path
                return True

            self._stanley_rate.sleep()

        return False

    # ── Utilities ─────────────────────────────────────────────

    def _publish_status(self, s):
        msg = String()
        msg.data = s
        self._status_pub.publish(msg)
        rospy.loginfo("[planner] Status → %s", s)

    def _publish_path(self, path):
        msg = Path()
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = 'world'
        for x, y in path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x    = x
            ps.pose.position.y    = y
            ps.pose.position.z    = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._path_pub.publish(msg)


# ═══════════════════════════════════════════════════════════════

def main():
    PlannerNode()   # _run() blocks; rospy.spin() not needed


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
