import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    tb3_1_model = os.path.expanduser(
        '~/turtlebot3_ws/src/tb3_nav2_slam/models/tb3_1_burger/model.sdf'
    )

    return LaunchDescription([

        Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_tb3_1',
            output='screen',
            arguments=[
                '-name', 'tb3_1',
                '-file', tb3_1_model,
                '-x', '-2.25',
                '-y', '-0.5',
                '-z', '0.01',
            ],
        ),
    ])