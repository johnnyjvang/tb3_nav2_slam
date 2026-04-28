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
                    'scan_topic': '/tb3_1/scan',
                    'cmd_vel_topic': '/tb3_1/cmd_vel',

                    'front_angle_deg': 20.0,
                    'target_distance': 0.90,
                    'distance_tolerance': 0.10,

                    'min_detect_distance': 0.25,
                    'max_detect_distance': 2.50,

                    'linear_kp': 0.35,
                    'angular_kp': 1.20,

                    'max_linear_speed': 0.10,
                    'max_angular_speed': 0.45,

                    'allow_reverse': False,
                    'control_rate_hz': 10.0,
                }
            ],
        ),
    ])