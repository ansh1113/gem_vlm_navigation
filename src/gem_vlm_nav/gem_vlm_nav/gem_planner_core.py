#!/usr/bin/env python3
"""
gem_planner_core.py — Shared Planning Core for GEM e4 VLM Navigation
======================================================================
ROS-agnostic.  Used by both:
  • planner_node.py      (ROS 2 / real vehicle / PACMod)
  • planner_sim_node.py  (ROS 1 / Gazebo simulator)

Pipeline
--------
1.  Occupancy grid  →  inflated binary grid  →  continuous EDT costmap
2.  A* on the inflated grid  →  raw waypoints  →  cubic-spline smoothing
3.  Pure-pursuit controller  →  (front_wheel_steer, target_speed, emergency)

The pure-pursuit controller replaces the previous DWA-lite trajectory
rollout.  It is geometrically simpler, cheaper, and well-suited to the
low-speed structured environments the GEM e4 operates in.
"""

import math
import heapq
from collections import defaultdict

import numpy as np
import scipy.ndimage as ndimage
from scipy.interpolate import splprep, splev


# ═══════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════

def normalize_angle(a):
    """Wrap angle to [-π, π]."""
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


# ═══════════════════════════════════════════════════════════════
#  PLANNER CORE
# ═══════════════════════════════════════════════════════════════

class GemPlannerCore:
    """
    Occupancy-grid A* planner + pure-pursuit local controller.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float
        World-frame bounding box of the operating area (metres).
    resolution : float
        Grid cell size (m/cell).  Default 0.5 m.
    wheelbase : float
        GEM e4 wheelbase (m).  Default 2.57 m.
    """

    # ── construction ───────────────────────────────────────────
    def __init__(self, *,
                 x_min=-50.0, x_max=150.0,
                 y_min=-50.0, y_max=150.0,
                 resolution=0.5,
                 wheelbase=2.57):

        # Grid geometry
        self.X_MIN = x_min
        self.X_MAX = x_max
        self.Y_MIN = y_min
        self.Y_MAX = y_max
        self.RESOLUTION = resolution

        # Vehicle geometry
        self.WHEELBASE  = wheelbase
        self.CAR_WIDTH  = 1.4       # m — full body width
        self.MAX_STEER  = 0.61      # rad ≈ 35° max front-wheel angle

        # A* inflation
        self.INFLATE_R = 1.5        # m — obstacle inflation for global plan

        # Pure-pursuit tuning
        self.BASE_LOOKAHEAD = 4.0   # m — minimum lookahead at zero speed
        self.K_LOOKAHEAD    = 0.5   # m per (m/s) speed gain
        self.MIN_LOOKAHEAD  = 2.5   # m — hard floor

        # Obstacle-clearance check
        self.OBSTACLE_CHECK_PTS = 20  # number of waypoints ahead to check
        self.CLEARANCE_MARGIN   = 0.4 # m — added to half car width

        # Internal state
        self.base_grid  = self._build_empty_grid()
        self.dyn_grid   = self.base_grid.copy()
        self.costmap    = self._build_costmap(self.dyn_grid)
        self.global_path = None
        self.target_idx  = 0
        self.goal        = None

    # ── grid helpers ───────────────────────────────────────────

    def _build_empty_grid(self):
        nc = int(math.ceil((self.X_MAX - self.X_MIN) / self.RESOLUTION))
        nr = int(math.ceil((self.Y_MAX - self.Y_MIN) / self.RESOLUTION))
        g  = np.zeros((nr, nc), dtype=np.uint8)
        w  = max(1, int(1.0 / self.RESOLUTION))
        g[:w, :] = g[-w:, :] = g[:, :w] = g[:, -w:] = 1   # boundary walls
        return g

    def _build_costmap(self, grid):
        """EDT — each cell = distance in metres to the nearest obstacle."""
        return ndimage.distance_transform_edt(grid == 0) * self.RESOLUTION

    def world_to_cell(self, x, y):
        return (int((x - self.X_MIN) / self.RESOLUTION),
                int((y - self.Y_MIN) / self.RESOLUTION))

    def cell_to_world(self, col, row):
        return (self.X_MIN + (col + 0.5) * self.RESOLUTION,
                self.Y_MIN + (row + 0.5) * self.RESOLUTION)

    def in_bounds(self, col, row, shape):
        return 0 <= row < shape[0] and 0 <= col < shape[1]

    def get_clearance(self, x, y):
        c, r = self.world_to_cell(x, y)
        if self.in_bounds(c, r, self.costmap.shape):
            return float(self.costmap[r, c])
        return 0.0

    # ── obstacle management ────────────────────────────────────

    def add_obstacle(self, cx, cy, radius=1.2):
        """Mark circular obstacle on the dynamic grid; rebuild costmap."""
        c0, r0 = self.world_to_cell(cx - radius, cy - radius)
        c1, r1 = self.world_to_cell(cx + radius, cy + radius)
        added = False
        for row in range(max(0, r0), min(self.dyn_grid.shape[0], r1 + 1)):
            for col in range(max(0, c0), min(self.dyn_grid.shape[1], c1 + 1)):
                wx, wy = self.cell_to_world(col, row)
                if (wx - cx)**2 + (wy - cy)**2 <= radius**2:
                    if self.dyn_grid[row, col] == 0:
                        self.dyn_grid[row, col] = 1
                        added = True
        if added:
            self.costmap = self._build_costmap(self.dyn_grid)
        return added

    def reset_obstacles(self):
        """Clear all dynamic obstacles, keep boundary walls."""
        self.dyn_grid = self.base_grid.copy()
        self.costmap  = self._build_costmap(self.dyn_grid)

    # ── A* inflation ──────────────────────────────────────────

    def _inflate_grid(self, grid, radius_m):
        r = max(1, int(radius_m / self.RESOLUTION))
        struct = np.zeros((2*r+1, 2*r+1), dtype=bool)
        for i in range(2*r+1):
            for j in range(2*r+1):
                if (i - r)**2 + (j - r)**2 <= r**2:
                    struct[i, j] = True
        return ndimage.binary_dilation(
            grid.astype(bool), structure=struct).astype(np.uint8)

    # ── A* global planner ──────────────────────────────────────

    def plan_global_path(self, start_xy, goal_xy):
        """
        Run A* on the inflated grid from *start_xy* to *goal_xy*.

        Returns True on success (path stored in self.global_path).
        """
        inf_grid = self._inflate_grid(self.dyn_grid, self.INFLATE_R)

        sc = self.world_to_cell(*start_xy)
        gc = self.world_to_cell(*goal_xy)

        # Snap occupied cells to nearest free cell
        def snap_free(col, row):
            if self.in_bounds(col, row, inf_grid.shape) and inf_grid[row, col] == 0:
                return col, row
            for r in range(1, 15):
                for dc in range(-r, r + 1):
                    for dr in range(-r, r + 1):
                        nc, nr = col + dc, row + dr
                        if (self.in_bounds(nc, nr, inf_grid.shape)
                                and inf_grid[nr, nc] == 0):
                            return nc, nr
            return col, row

        sc = snap_free(*sc)
        gc = snap_free(*gc)
        if (not self.in_bounds(*sc, inf_grid.shape)
                or not self.in_bounds(*gc, inf_grid.shape)):
            return False

        # 8-connected A*
        nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        heap = []
        heapq.heappush(heap, (0.0, sc))
        came   = {}
        gscore = defaultdict(lambda: float('inf'))
        gscore[sc] = 0.0

        raw_path = None
        while heap:
            _, cur = heapq.heappop(heap)
            if cur == gc:
                path = []
                while cur in came:
                    path.append(self.cell_to_world(*cur))
                    cur = came[cur]
                path.append(self.cell_to_world(*sc))
                path.reverse()
                raw_path = path
                break
            col, row = cur
            for dc, dr, cost in nbrs:
                nb = (col + dc, row + dr)
                if (not self.in_bounds(nb[0], nb[1], inf_grid.shape)
                        or inf_grid[nb[1], nb[0]] == 1):
                    continue
                t = gscore[cur] + cost
                if t < gscore[nb]:
                    came[nb]   = cur
                    gscore[nb] = t
                    h = math.hypot(nb[0] - gc[0], nb[1] - gc[1])
                    heapq.heappush(heap, (t + h, nb))

        if raw_path is None:
            return False

        # Smooth with cubic spline
        self.global_path = self._smooth_path(raw_path)
        self.target_idx  = 0
        self.goal        = goal_xy
        return True

    def _smooth_path(self, path, n_points=250):
        if path is None or len(path) < 3:
            return path
        pts = np.array(path)
        _, idx = np.unique(pts, axis=0, return_index=True)
        pts = pts[np.sort(idx)]
        if len(pts) < 3:
            return path
        try:
            tck, _ = splprep([pts[:, 0], pts[:, 1]],
                             s=2.0, k=min(3, len(pts) - 1))
            xs, ys = splev(np.linspace(0, 1, n_points), tck)
            return list(zip(xs.tolist(), ys.tolist()))
        except Exception:
            return path

    # ── pure-pursuit local controller ──────────────────────────

    def get_local_command(self, car_state, desired_speed):
        """
        Pure-pursuit controller.

        Parameters
        ----------
        car_state : (x, y, yaw)
            Current vehicle pose in the world frame.
        desired_speed : float
            Cruise speed (m/s).

        Returns
        -------
        front_steer_rad : float
            Front wheel steering angle (radians, + left).
        target_speed : float
            Commanded speed (m/s).  Zero during emergency.
        emergency : bool
            True → path is blocked; caller should stop and replan.
        """
        if not self.global_path:
            return 0.0, 0.0, False

        x, y, yaw = car_state
        n  = len(self.global_path)
        px = np.array([p[0] for p in self.global_path])
        py = np.array([p[1] for p in self.global_path])

        # ── 1. advance the closest-point index (forward only) ──
        search_end = min(n, self.target_idx + 80)
        dists = np.hypot(px[self.target_idx:search_end] - x,
                         py[self.target_idx:search_end] - y)
        if len(dists) == 0:
            return 0.0, 0.0, False
        self.target_idx += int(np.argmin(dists))

        # Skip waypoints that are already behind the car
        while (self.target_idx < n - 1
               and np.hypot(x - px[self.target_idx],
                            y - py[self.target_idx]) < 1.0):
            self.target_idx += 1

        # ── 2. compute speed-adaptive lookahead distance ───────
        ld = max(self.MIN_LOOKAHEAD,
                 self.BASE_LOOKAHEAD + self.K_LOOKAHEAD * desired_speed)

        # ── 3. find the lookahead point on the path ────────────
        goal_idx = self.target_idx
        for i in range(self.target_idx, n):
            if np.hypot(px[i] - x, py[i] - y) >= ld:
                goal_idx = i
                break
        else:
            goal_idx = n - 1   # path end

        tx, ty = px[goal_idx], py[goal_idx]

        # ── 4. pure-pursuit steering geometry ──────────────────
        alpha = normalize_angle(math.atan2(ty - y, tx - x) - yaw)

        # Actual ld used for the formula is distance to the target point
        ld_actual = max(0.1, math.hypot(tx - x, ty - y))
        steer = math.atan2(2.0 * self.WHEELBASE * math.sin(alpha),
                           ld_actual)
        steer = max(-self.MAX_STEER, min(self.MAX_STEER, steer))

        # ── 5. obstacle clearance check along path ahead ───────
        half_width = self.CAR_WIDTH / 2.0 + self.CLEARANCE_MARGIN
        emergency  = False

        check_end = min(n, self.target_idx + self.OBSTACLE_CHECK_PTS)
        for i in range(self.target_idx, check_end):
            clr = self.get_clearance(px[i], py[i])
            if clr < half_width:
                emergency = True
                break

        if emergency:
            return steer, 0.0, True

        # ── 6. slow down near the goal ─────────────────────────
        if self.goal is not None:
            dist_to_goal = math.hypot(x - self.goal[0], y - self.goal[1])
            if dist_to_goal < 5.0:
                # Linear ramp-down in the last 5 m
                desired_speed = max(0.5, desired_speed * (dist_to_goal / 5.0))

        return steer, desired_speed, False
