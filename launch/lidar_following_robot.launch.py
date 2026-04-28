from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([
        Node(
            package='tb3_nav2_slam',
            executable='lidar_following_robot',
            name='lidar_following_robot',
            output='screen',
            parameters=[
                {
                    'scan_topic': '/scan',
                    'cmd_vel_topic': '/cmd_vel',

                    'front_angle_deg':45.0,
                    'target_distance': 1.20,
                    'distance_tolerance': 0.15,

                    'min_detect_distance': 0.25,
                    'max_detect_distance': 4.00,

                    'linear_kp': 0.30,
                    'angular_kp': 1.60,

                    'max_linear_speed': 0.18,
                    'max_angular_speed': 0.80,

                    'allow_reverse': False,
                    'control_rate_hz': 10.0,
                }
            ],
        )
    ])