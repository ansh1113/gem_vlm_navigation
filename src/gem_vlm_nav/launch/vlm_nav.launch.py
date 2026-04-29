from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Start the LiDAR BEV Map Generator
        Node(
            package='gem_vlm_nav',
            executable='lidar_bev_node',
            name='lidar_bev_node',
            output='screen'
        ),

        # 2. Start the VLM Brain (formerly perception_node)
        Node(
            package='gem_vlm_nav',
            executable='vlm_node',
            name='vlm_node',
            output='screen'
        ),
        
        # 3. Start the Planner + PACMod Controller Node
        Node(
            package='gem_vlm_nav',
            executable='planner_node',
            name='vlm_planner',
            output='screen'
        )
    ])