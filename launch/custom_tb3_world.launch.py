import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')

    world_file = os.path.join(
        turtlebot3_gazebo_dir,
        'worlds',
        'turtlebot3_world.world'
    )

    custom_models_dir = os.path.expanduser(
        '~/turtlebot3_ws/src/tb3_nav2_slam/models'
    )

    tb3_1_model = os.path.join(
        custom_models_dir,
        'tb3_1_burger',
        'model.sdf'
    )

    tb3_2_model = os.path.join(
        custom_models_dir,
        'tb3_2_burger',
        'model.sdf'
    )

    return LaunchDescription([

        # Allow Gazebo to find both model folders
        SetEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=':'.join([
                os.path.join(turtlebot3_gazebo_dir, 'models'),
                custom_models_dir,
            ])
        ),

        # Launch Gazebo
        ExecuteProcess(
            cmd=[
                'gz',
                'sim',
                '-r',
                world_file,
            ],
            output='screen',
        ),

        # -------------------------
        # Spawn tb3_1 (camera robot)
        # -------------------------
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
                '-z', '0.05',
                '-Y', '0.0',
            ],
        ),

        # -------------------------
        # Spawn tb3_2 (target robot)
        # -------------------------
        Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_tb3_2',
            output='screen',
            arguments=[
                '-name', 'tb3_2',
                '-file', tb3_2_model,
                '-x', '-1.80',   # ~0.45–0.5 m in front
                '-y', '-0.5',
                '-z', '0.05',
                '-Y', '0.0',
            ],
        ),

        # -------------------------
        # Global bridge (simulation time)
        # -------------------------
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='clock_bridge',
            output='screen',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
            ],
        ),

        # =========================
        # tb3_1 bridges
        # =========================

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

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='tb3_1_imu_bridge',
            output='screen',
            arguments=[
                '/tb3_1/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
            ],
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='tb3_1_camera_bridge',
            output='screen',
            arguments=[
                '/tb3_1/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
            ],
        ),

        # =========================
        # tb3_2 bridges
        # =========================

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='tb3_2_scan_bridge',
            output='screen',
            arguments=[
                '/tb3_2/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            ],
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='tb3_2_cmd_vel_bridge',
            output='screen',
            arguments=[
                '/tb3_2/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            ],
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='tb3_2_odom_bridge',
            output='screen',
            arguments=[
                '/tb3_2/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            ],
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='tb3_2_imu_bridge',
            output='screen',
            arguments=[
                '/tb3_2/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
            ],
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='tb3_2_camera_bridge',
            output='screen',
            arguments=[
                '/tb3_2/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
            ],
        ),
    ])