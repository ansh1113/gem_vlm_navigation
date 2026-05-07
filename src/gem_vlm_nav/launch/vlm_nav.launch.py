from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Load the same e4 vehicle config that pure_pursuit uses
    vehicle_env = os.environ.get('VEHICLE_NAME', 'e4')
    config_file = vehicle_env + '_pp.yaml'
    config_path = os.path.join(
        get_package_share_directory('gem_gnss_control'),
        'config',
        config_file
    )

    return LaunchDescription([
        # 1. Start the Perception Node
        Node(
            package='gem_vlm_nav',
            executable='perception_node',
            name='vlm_perception',
            output='screen'
        ),

        # 2. Start the Planner + PACMod Controller Node
        #    Load vehicle config so PID/filter/offset params are available
        Node(
            package='gem_vlm_nav',
            executable='planner_node',
            name='vlm_planner',
            output='screen',
            parameters=[config_path]
        )
    ])