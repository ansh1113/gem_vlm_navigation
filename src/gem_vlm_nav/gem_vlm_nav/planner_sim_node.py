#!/usr/bin/env python3
"""
planner_sim_node.py — VLM Path Planning for GEM e4 (POLARIS_GEM_Simulator)
===========================================================================
ROS 1 (Noetic) node for the Gazebo simulator.
Run inside the simulator Docker container:
    python3 planner_sim_node.py
  or
    rosrun gem_vlm_nav planner_sim_node.py

Architecture:
  VLM goal (/vlm_goal)  →  A* global plan (gem_planner_core)
                         →  pure-pursuit steering (gem_planner_core)
                         →  AckermannDrive command

Simulator interface (POLARIS_GEM_Simulator):
  /ackermann_cmd              — AckermannDrive (steering + speed)
  /gazebo/get_model_state     — vehicle pose (x, y, yaw)
  /front_laser/scan           — 2-D LaserScan for obstacle detection
  /vlm_goal                   — PoseStamped goal from VLM perception

This node is the ROS-1 counterpart of planner_node.py (ROS 2 / real vehicle).
Both share the same gem_planner_core for planning + pure pursuit.
"""

import os
import sys
import math

import numpy as np

# ── Make gem_planner_core importable regardless of install method ──
# When running standalone (python3 planner_sim_node.py), the core
# module lives in the same directory.
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import rospy
from ackermann_msgs.msg import AckermannDrive
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.srv import GetModelState
from tf.transformations import euler_from_quaternion

from gem_planner_core import GemPlannerCore


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION  (simulator-specific defaults)
# ═══════════════════════════════════════════════════════════════

# Grid bounds — larger than real-vehicle highbay
X_MIN, X_MAX = -100.0, 100.0
Y_MIN, Y_MAX = -100.0, 100.0

RESOLUTION    = 0.5        # m/cell
WHEELBASE     = 2.57       # m (GEM e4)
DESIRED_SPEED = 2.0        # m/s cruise
GOAL_TOL      = 1.5        # m — arrival tolerance
RATE_HZ       = 20         # control loop frequency

# LaserScan
SCAN_RANGE_MAX = 10.0      # m — ignore obstacles beyond this
OBS_RADIUS     = 1.0       # m — obstacle inflation for each scan hit


# ═══════════════════════════════════════════════════════════════
#  PLANNER SIM NODE
# ═══════════════════════════════════════════════════════════════

class PlannerSimNode:
    def __init__(self):
        rospy.init_node('gem_planner_sim_node', anonymous=True)
        self.rate = rospy.Rate(RATE_HZ)

        # Shared planning core
        self.planner = GemPlannerCore(
            x_min=X_MIN, x_max=X_MAX,
            y_min=Y_MIN, y_max=Y_MAX,
            resolution=RESOLUTION,
            wheelbase=WHEELBASE,
        )

        # State
        self.car_state = None      # (x, y, yaw)
        self.goal_xy   = None      # (gx, gy) from VLM
        self.has_plan  = False

        # AckermannDrive publisher
        self.ackermann_msg = AckermannDrive()
        self.ackermann_pub = rospy.Publisher(
            '/ackermann_cmd', AckermannDrive, queue_size=1)

        # Subscriptions
        self.scan_sub = rospy.Subscriber(
            '/front_laser/scan', LaserScan, self.scan_callback)
        self.goal_sub = rospy.Subscriber(
            '/vlm_goal', PoseStamped, self.goal_callback)

    # ── goal ──────────────────────────────────────────────────

    def goal_callback(self, msg):
        new_goal = (msg.pose.position.x, msg.pose.position.y)
        rospy.loginfo(f"Received VLM goal: {new_goal}")
        self.goal_xy  = new_goal
        self.has_plan = False       # trigger replan

    # ── pose from Gazebo ──────────────────────────────────────

    def get_gem_pose(self):
        """Query Gazebo for model state → (x, y, yaw)."""
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=2.0)
            proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            state = proxy(model_name='gem')

            x = state.pose.position.x
            y = state.pose.position.y
            q = state.pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return x, y, yaw
        except rospy.ServiceException as exc:
            rospy.logwarn(f"Gazebo service error: {exc}")
            return None

    # ── LaserScan → obstacles ─────────────────────────────────

    def scan_callback(self, msg):
        if self.car_state is None:
            return

        x, y, yaw = self.car_state

        # Clear previous dynamic obstacles (avoid ghost obstacles)
        self.planner.reset_obstacles()

        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max and r < SCAN_RANGE_MAX:
                # Obstacle position in body frame
                obs_x_local = r * math.cos(angle)
                obs_y_local = r * math.sin(angle)

                # Transform to world frame
                obs_x = x + obs_x_local * math.cos(yaw) - obs_y_local * math.sin(yaw)
                obs_y = y + obs_x_local * math.sin(yaw) + obs_y_local * math.cos(yaw)

                self.planner.add_obstacle(obs_x, obs_y, radius=OBS_RADIUS)
            angle += msg.angle_increment

    # ── main loop ─────────────────────────────────────────────

    def run(self):
        rospy.loginfo("GEM Sim Planner (pure pursuit) — waiting for /vlm_goal")

        while not rospy.is_shutdown():
            # 1. Get current pose from Gazebo
            pose = self.get_gem_pose()
            if pose is None:
                self.rate.sleep()
                continue
            self.car_state = pose

            # 2. Standby until VLM sends a goal
            if self.goal_xy is None:
                self.ackermann_msg.speed = 0.0
                self.ackermann_msg.steering_angle = 0.0
                self.ackermann_pub.publish(self.ackermann_msg)
                self.rate.sleep()
                continue

            # 3. Plan if needed
            if not self.has_plan:
                rospy.loginfo("Planning path to goal...")
                success = self.planner.plan_global_path(
                    (self.car_state[0], self.car_state[1]), self.goal_xy)
                if success:
                    rospy.loginfo(
                        f"Path found — {len(self.planner.global_path)} pts")
                    self.has_plan = True
                else:
                    rospy.logwarn("A* failed — retrying next cycle")
                    self.ackermann_msg.speed = 0.0
                    self.ackermann_pub.publish(self.ackermann_msg)
                    self.rate.sleep()
                    continue

            # 4. Goal reached?
            dist = math.hypot(
                self.car_state[0] - self.goal_xy[0],
                self.car_state[1] - self.goal_xy[1])
            if dist < GOAL_TOL:
                rospy.loginfo("Goal reached!")
                self.ackermann_msg.speed = 0.0
                self.ackermann_msg.steering_angle = 0.0
                self.ackermann_pub.publish(self.ackermann_msg)
                self.goal_xy  = None
                self.has_plan = False
                self.rate.sleep()
                continue

            # 5. Pure-pursuit controller
            front_steer, target_speed, emergency = \
                self.planner.get_local_command(self.car_state, DESIRED_SPEED)

            if emergency:
                rospy.logwarn("Obstacle blocking path — replanning...")
                self.ackermann_msg.speed = 0.0
                self.ackermann_msg.steering_angle = 0.0
                self.ackermann_pub.publish(self.ackermann_msg)

                success = self.planner.plan_global_path(
                    (self.car_state[0], self.car_state[1]), self.goal_xy)
                if success:
                    rospy.loginfo("Replan successful.")
                else:
                    rospy.logerr("Replan failed — path completely blocked.")
                    self.has_plan = False
            else:
                self.ackermann_msg.speed          = target_speed
                self.ackermann_msg.steering_angle = front_steer
                self.ackermann_pub.publish(self.ackermann_msg)

            self.rate.sleep()


# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    try:
        PlannerSimNode().run()
    except rospy.ROSInterruptException:
        pass
