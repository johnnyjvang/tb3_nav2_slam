#!/usr/bin/env python3

# ============================================================
# goal_from_list.py Summary
# ------------------------------------------------------------
# This node sends 1 to 3 waypoint goals to Nav2 using
# goThroughPoses(). The user may provide custom CLI goals or
# let the script use built-in defaults.
#
# Supported input formats:
# - 0 args -> default goals
# - 3 args -> 1 goal
# - 6 args -> 2 goals
# - 9 args -> 3 goals
#
# Each goal is entered as:
# x y yaw_deg
#
# Input validation ensures values are numeric and yaw stays
# within a reasonable range.
# ============================================================

import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult


# ============================================================
# Yaw → Quaternion Conversion
# ------------------------------------------------------------
# ROS does NOT represent orientation using a single yaw angle.
# Instead, it uses quaternions (x, y, z, w) for 3D rotations.
#
# Since TurtleBot3 is a 2D robot (rotates only around Z axis),
# we can simplify the quaternion:
# - x = 0
# - y = 0
# - z = sin(yaw / 2)
# - w = cos(yaw / 2)
#
# This function converts a yaw angle (in radians) into the
# quaternion components (z, w) required for PoseStamped.
# ============================================================
def yaw_to_quaternion_z_w(yaw_rad: float):
    qz = math.sin(yaw_rad / 2.0)
    qw = math.cos(yaw_rad / 2.0)
    return qz, qw


# ============================================================
# build_pose Function Explanation
# ------------------------------------------------------------
# This function creates a PoseStamped message used as a navigation
# goal for Nav2. It converts human-readable inputs (x, y, yaw in
# degrees) into a properly formatted ROS pose.
#
# Key steps:
# - Sets frame_id = 'map' → defines global coordinate system
# - Uses navigator clock → ensures correct ROS/sim time sync
# - Assigns position (x, y, z) → where the robot should go
# - Converts yaw (degrees) → quaternion (z, w) for orientation
#
# PoseStamped is required because Nav2 needs both spatial context
# (frame) and time (timestamp) to correctly interpret the goal.
# ============================================================
def build_pose(navigator: BasicNavigator, x: float, y: float, yaw_deg: float) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()

    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = 0.0

    yaw_rad = math.radians(yaw_deg)
    qz, qw = yaw_to_quaternion_z_w(yaw_rad)
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw

    return pose


# ============================================================
# Float Parsing Helper
# ------------------------------------------------------------
# Converts string input to float and raises a clear error
# message if the value is not numeric.
# ============================================================
def parse_float(value: str, name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f'Invalid {name}: "{value}" is not a number.') from exc


# ============================================================
# Yaw Validation Helper
# ------------------------------------------------------------
# Ensures yaw stays within a reasonable range for user input.
# ============================================================
def validate_yaw_deg(yaw_deg: float, name: str) -> None:
    if yaw_deg < -360.0 or yaw_deg > 360.0:
        raise ValueError(
            f'Invalid {name}: {yaw_deg} is outside expected range [-360, 360].'
        )


# ============================================================
# Optional CLI Argument Handling
# ------------------------------------------------------------
# This parser supports:
# - 0 arguments  -> use default waypoint goals
# - 3 arguments  -> 1 pose  (x1 y1 yaw1)
# - 6 arguments  -> 2 poses (x1 y1 yaw1 x2 y2 yaw2)
# - 9 arguments  -> 3 poses (x1 y1 yaw1 x2 y2 yaw2 x3 y3 yaw3)
#
# Any other number of arguments is rejected.
# ============================================================
def parse_goals_from_cli(args: list[str]) -> list[tuple[float, float, float]]:
    if len(args) == 0:
        return []

    if len(args) not in (3, 6, 9):
        raise ValueError(
            'Invalid number of arguments. Expected 0, 3, 6, or 9 values:\n'
            'x1 y1 yaw1 [x2 y2 yaw2] [x3 y3 yaw3]'
        )

    parsed: list[tuple[float, float, float]] = []
    num_goals = len(args) // 3

    for i in range(num_goals):
        base = i * 3
        x = parse_float(args[base], f'x{i + 1}')
        y = parse_float(args[base + 1], f'y{i + 1}')
        yaw_deg = parse_float(args[base + 2], f'yaw{i + 1}')
        validate_yaw_deg(yaw_deg, f'yaw{i + 1}')
        parsed.append((x, y, yaw_deg))

    return parsed


# ============================================================
# main() Function Summary
# ------------------------------------------------------------
# This function initializes ROS 2, sets the robot’s starting
# position (initial pose), and sends 1 to 3 waypoint goals using
# Nav2. It waits until the navigation stack is ready, allows
# localization time to settle, then commands the robot to move
# through the requested goals.
#
# While the robot is moving, it periodically checks for feedback
# and waits until the task is complete. Once finished, it reports
# whether the navigation succeeded, failed, or was canceled,
# then shuts down the system cleanly.
# ============================================================
def main():
    # Initialize ROS 2 system
    rclpy.init()

    # Create high-level Nav2 helper
    navigator = BasicNavigator()

    # Default goals if no CLI input is provided
    default_goal_values = [
        (0.0, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (1.0, -0.5, 0.0),
    ]

    try:
        # Parse optional CLI goals
        cli_goal_values = parse_goals_from_cli(sys.argv[1:])

        if len(cli_goal_values) == 0:
            goal_values = default_goal_values
            navigator.info('No CLI goals provided. Using default waypoint goals.')
        else:
            goal_values = cli_goal_values
            navigator.info(f'Using {len(goal_values)} CLI-provided goal(s).')

        # Set initial pose to the known TurtleBot3 starting pose
        # in the default Gazebo world/map
        init_pose = build_pose(navigator, 0.0, 0.0, 0.0)
        navigator.setInitialPose(init_pose)

        # Wait until Nav2 lifecycle nodes are active
        navigator.info('Waiting for Nav2 to become active...')
        navigator.waitUntilNav2Active()

        # Give AMCL/localization a short time to settle before sending goals
        navigator.info('Waiting for localization to settle...')
        time.sleep(2.0)

        # Build PoseStamped goals
        goals = [
            build_pose(navigator, x, y, yaw_deg)
            for x, y, yaw_deg in goal_values
        ]

        # Log the goals being sent
        navigator.info(
            'Sending goals: '
            + ', '.join(
                f'({x:.2f}, {y:.2f}, {yaw_deg:.1f} deg)'
                for x, y, yaw_deg in goal_values
            )
        )

        # Start waypoint navigation
        navigator.goThroughPoses(goals)

        # Track time for periodic feedback printing
        last_feedback_time = time.time()

        # Monitor navigation progress until complete
        while not navigator.isTaskComplete():
            feedback = navigator.getFeedback()
            now = time.time()

            if feedback is not None and (now - last_feedback_time) > 1.0:
                navigator.info('Waypoint navigation still running...')
                last_feedback_time = now

        # Retrieve final mission result
        result = navigator.getResult()

        if result == TaskResult.SUCCEEDED:
            navigator.info('Waypoint mission succeeded!')
        elif result == TaskResult.CANCELED:
            navigator.warn('Waypoint mission was canceled.')
        elif result == TaskResult.FAILED:
            navigator.error('Waypoint mission failed.')
        else:
            navigator.error('Waypoint mission returned an unknown result.')

    except ValueError as exc:
        navigator.error(str(exc))
        navigator.info(
            'Usage:\n'
            'ros2 run tb3_nav2_slam goal_from_list x1 y1 yaw1 [x2 y2 yaw2] [x3 y3 yaw3]'
        )

    finally:
        # Clean up Nav2 helper node
        navigator.destroyNode()

        # Shut down ROS 2
        rclpy.shutdown()


# ============================================================
# Python Entry Point Check
# ------------------------------------------------------------
# This ensures main() only runs when this file is executed
# directly, not when imported into another script.
# ============================================================
if __name__ == '__main__':
    main()