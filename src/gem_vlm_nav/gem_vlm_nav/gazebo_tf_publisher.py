#!/usr/bin/env python3
"""
gazebo_tf_publisher.py
======================
Polls /gazebo/get_model_state at 50Hz and broadcasts the ground-truth
transform:  world --> base_footprint

This fills the missing link in the TF tree. Without it, RViz has no idea
where the car is in the world frame, so the costmap and car appear unrelated.

The full chain after this node runs:
    world --> base_footprint --> base_link --> [all sensors/wheels]

Usage:
    source devel/setup.bash
    python3 gazebo_tf_publisher.py

    Optionally change MODEL_NAME below if your model is not named 'gem_e4'.
    Check with: rostopic echo /gazebo/model_states | grep "name" -A1
"""

import rospy
import tf
from gazebo_msgs.srv import GetModelState, GetModelStateRequest

MODEL_NAME      = 'gem_e4'      # must match the model name in Gazebo
REFERENCE_FRAME = 'world'       # Gazebo's global frame
CHILD_FRAME     = 'base_footprint'
PUBLISH_HZ      = 50.0


def main():
    rospy.init_node('gazebo_tf_publisher', anonymous=False)

    rospy.loginfo("[gazebo_tf] Waiting for /gazebo/get_model_state service...")
    rospy.wait_for_service('/gazebo/get_model_state')
    get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    rospy.loginfo("[gazebo_tf] Service ready. Publishing world -> base_footprint at %dHz", int(PUBLISH_HZ))

    broadcaster = tf.TransformBroadcaster()
    req = GetModelStateRequest()
    req.model_name      = MODEL_NAME
    req.relative_entity_name = REFERENCE_FRAME

    rate = rospy.Rate(PUBLISH_HZ)
    while not rospy.is_shutdown():
        try:
            resp = get_state(req)
            if not resp.success:
                rospy.logwarn_throttle(5.0, "[gazebo_tf] get_model_state failed for '%s'", MODEL_NAME)
                rate.sleep()
                continue

            pos = resp.pose.position
            ori = resp.pose.orientation

            broadcaster.sendTransform(
                (pos.x, pos.y, pos.z),
                (ori.x, ori.y, ori.z, ori.w),
                rospy.Time.now(),
                CHILD_FRAME,    # child
                REFERENCE_FRAME # parent
            )

        except rospy.ServiceException as e:
            rospy.logwarn_throttle(5.0, "[gazebo_tf] Service call failed: %s", str(e))

        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
