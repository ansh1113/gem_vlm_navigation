from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Start the Perception Node
        Node(
            package='gem_vlm_nav',
            executable='perception_node',
            name='vlm_perception',
            output='screen'
        ),
        
        # 2. Start the Planner + PACMod Controller Node
        Node(
            package='gem_vlm_nav',
            executable='planner_node',
            name='vlm_planner',
            output='screen'
        )
    ])