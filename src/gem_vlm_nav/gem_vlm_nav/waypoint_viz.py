#!/usr/bin/env python3
"""
waypoint_visualizer.py — VLM Waypoint Marker Publisher
=======================================================
Subscribes to /vlm_waypoints (geometry_msgs/PoseArray) and publishes
a visualization_msgs/MarkerArray on /vlm_waypoint_markers.

Displays:
  - Yellow sphere at each waypoint
  - Numbered text label above each sphere
  - Yellow line strip connecting all waypoints in order

Usage:
    source devel/setup.bash
    python3 waypoint_visualizer.py

RViz:
    Add -> By topic -> /vlm_waypoint_markers -> MarkerArray
"""

import rospy
from geometry_msgs.msg import PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray


SPHERE_RADIUS  = 0.8    # metres
SPHERE_HEIGHT  = 0.5    # z height of sphere centre
LABEL_HEIGHT   = 1.2    # z height of text label
LABEL_SIZE     = 0.5    # text scale
LINE_WIDTH     = 0.1    # line strip width


class WaypointVisualizer:
    def __init__(self):
        rospy.init_node('waypoint_visualizer', anonymous=False)

        self.marker_pub = rospy.Publisher(
            '/vlm_waypoint_markers', MarkerArray,
            queue_size=1, latch=True
        )

        rospy.Subscriber('/vlm_waypoints', PoseArray, self.waypoints_cb,
                         queue_size=1)

        rospy.loginfo("[waypoint_viz] Ready. Listening on /vlm_waypoints.")

    def waypoints_cb(self, msg: PoseArray):
        ma = MarkerArray()

        # Clear all previous markers first
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        delete_all.header.frame_id = 'world'
        delete_all.header.stamp = rospy.Time.now()
        ma.markers.append(delete_all)
        self.marker_pub.publish(ma)
        ma = MarkerArray()

        now = msg.header.stamp

        for i, pose in enumerate(msg.poses):
            wx = pose.position.x
            wy = pose.position.y

            # ── Yellow sphere ─────────────────────────────
            sphere = Marker()
            sphere.header.stamp    = now
            sphere.header.frame_id = 'world'
            sphere.ns              = 'vlm_spheres'
            sphere.id              = i
            sphere.type            = Marker.SPHERE
            sphere.action          = Marker.ADD
            sphere.pose.position.x = wx
            sphere.pose.position.y = wy
            sphere.pose.position.z = SPHERE_HEIGHT
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = SPHERE_RADIUS
            sphere.color.r = 1.0
            sphere.color.g = 1.0
            sphere.color.b = 0.0
            sphere.color.a = 1.0
            sphere.lifetime = rospy.Duration(0)   # 0 = never expire
            ma.markers.append(sphere)

            # ── Numbered label ────────────────────────────
            label = Marker()
            label.header.stamp    = now
            label.header.frame_id = 'world'
            label.ns              = 'vlm_labels'
            label.id              = i
            label.type            = Marker.TEXT_VIEW_FACING
            label.action          = Marker.ADD
            label.pose.position.x = wx
            label.pose.position.y = wy
            label.pose.position.z = LABEL_HEIGHT
            label.pose.orientation.w = 1.0
            label.scale.z         = LABEL_SIZE
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            label.text            = str(i + 1)
            label.lifetime        = rospy.Duration(0)
            ma.markers.append(label)

        # ── Line strip connecting all waypoints ───────────
        if len(msg.poses) > 1:
            line = Marker()
            line.header.stamp    = now
            line.header.frame_id = 'world'
            line.ns              = 'vlm_line'
            line.id              = 0
            line.type            = Marker.LINE_STRIP
            line.action          = Marker.ADD
            line.scale.x         = LINE_WIDTH
            line.color.r         = 1.0
            line.color.g         = 1.0
            line.color.b         = 0.0
            line.color.a         = 1.0
            line.lifetime        = rospy.Duration(0)
            for pose in msg.poses:
                p = Point()
                p.x = pose.position.x
                p.y = pose.position.y
                p.z = SPHERE_HEIGHT
                line.points.append(p)
            ma.markers.append(line)

        self.marker_pub.publish(ma)
        rospy.loginfo("[waypoint_viz] Published %d waypoint markers.",
                      len(msg.poses))


def main():
    node = WaypointVisualizer()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
