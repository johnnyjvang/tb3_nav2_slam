#!/usr/bin/env python3

# ============================================================
# single_goal_return.py Summary
# ------------------------------------------------------------
# This node sends the TurtleBot3 to a single goal pose using
# Nav2, then sends it back to a predefined home pose.
#
# This is useful for repeatable testing because each run ends
# with the robot returning near its original starting location.
#
# Optional CLI input:
# ros2 run tb3_nav2_slam single_goal_return goal_x goal_y goal_yaw_deg
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
# ROS uses quaternions for orientation, not a single yaw value.
# For a 2D robot, only z and w are needed.
# ============================================================
def yaw_to_quaternion_z_w(yaw_rad: float):
    qz = math.sin(yaw_rad / 2.0)
    qw = math.cos(yaw_rad / 2.0)
    return qz, qw


# ============================================================
# build_pose Function Explanation
# ------------------------------------------------------------
# Creates a PoseStamped message from x, y, and yaw_deg.
# This is the format Nav2 requires for navigation goals.
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
# Goal Runner Helper
# ------------------------------------------------------------
# Sends a single goal to Nav2 and waits for the task to finish.
# Returns the TaskResult so the caller can decide what to do next.
# ============================================================
def run_goal(navigator: BasicNavigator, goal_pose: PoseStamped, label: str) -> TaskResult:
    navigator.info(f'Sending {label} goal...')
    navigator.goToPose(goal_pose)

    last_feedback_time = time.time()

    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        now = time.time()

        if feedback is not None and (now - last_feedback_time) > 1.0:
            navigator.info(f'{label} navigation still running...')
            last_feedback_time = now

    result = navigator.getResult()

    if result == TaskResult.SUCCEEDED:
        navigator.info(f'{label} goal succeeded!')
    elif result == TaskResult.CANCELED:
        navigator.warn(f'{label} goal was canceled.')
    elif result == TaskResult.FAILED:
        navigator.error(f'{label} goal failed.')
    else:
        navigator.error(f'{label} goal returned an unknown result.')

    return result


# ============================================================
# main() Function Summary
# ------------------------------------------------------------
# Initializes ROS 2 and Nav2, sends the robot to a requested
# goal, then sends it back to the home pose.
# ============================================================
def main():
    rclpy.init()
    navigator = BasicNavigator()

    # --------------------------------------------------------
    # Home pose: choose a known, repeatable starting location
    # inside the map. Adjust this if your actual robot start
    # in Gazebo/map is different.
    # --------------------------------------------------------
    home_x = 0.0
    home_y = 0.0
    home_yaw_deg = 0.0

    # --------------------------------------------------------
    # Default outbound goal if no CLI values are provided
    # --------------------------------------------------------
    goal_x = 0.5
    goal_y = -0.5
    goal_yaw_deg = 0.0

    # Optional CLI:
    # ros2 run tb3_nav2_slam single_goal_return 1.0 0.0 90
    if len(sys.argv) == 4:
        try:
            goal_x = float(sys.argv[1])
            goal_y = float(sys.argv[2])
            goal_yaw_deg = float(sys.argv[3])
        except ValueError:
            navigator.error('Invalid CLI input. Expected: goal_x goal_y goal_yaw_deg')
            navigator.destroyNode()
            rclpy.shutdown()
            return
    elif len(sys.argv) not in (1, 4):
        navigator.error('Usage: ros2 run tb3_nav2_slam single_goal_return [goal_x goal_y goal_yaw_deg]')
        navigator.destroyNode()
        rclpy.shutdown()
        return

    try:
        # ----------------------------------------------------
        # Set initial pose only if you know the robot is
        # actually starting at the home pose.
        # ----------------------------------------------------
        init_pose = build_pose(navigator, home_x, home_y, home_yaw_deg)
        navigator.setInitialPose(init_pose)

        navigator.info('Waiting for Nav2 to become active...')
        navigator.waitUntilNav2Active()

        navigator.info('Waiting for localization to settle...')
        time.sleep(2.0)

        outbound_goal = build_pose(navigator, goal_x, goal_y, goal_yaw_deg)
        home_goal = build_pose(navigator, home_x, home_y, home_yaw_deg)

        navigator.info(
            f'Outbound goal: x={goal_x:.2f}, y={goal_y:.2f}, yaw_deg={goal_yaw_deg:.1f}'
        )
        outbound_result = run_goal(navigator, outbound_goal, 'Outbound')

        if outbound_result != TaskResult.SUCCEEDED:
            navigator.error('Skipping return-home because outbound goal did not succeed.')
            return

        navigator.info('Outbound goal complete. Returning to home pose...')
        run_goal(navigator, home_goal, 'Return-home')

    finally:
        navigator.destroyNode()
        rclpy.shutdown()


# ============================================================
# Python Entry Point Check
# ------------------------------------------------------------
# Runs main() only when this script is executed directly.
# ============================================================
if __name__ == '__main__':
    main()