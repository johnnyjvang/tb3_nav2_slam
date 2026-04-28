from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='tb3_1_scan_bridge',
            output='screen',
            arguments=[
                '/tb3_1/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            ],
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='tb3_1_cmd_vel_bridge',
            output='screen',
            arguments=[
                '/tb3_1/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            ],
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='tb3_1_odom_bridge',
            output='screen',
            arguments=[
                '/tb3_1/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            ],
        ),
    ])