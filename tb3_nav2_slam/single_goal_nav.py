#!/usr/bin/env python3

import math
import sys
import time
# ============================================================
# rclpy (ROS 2 Python Client Library) Explanation
# ------------------------------------------------------------
# rclpy is the core Python library used to interact with ROS 2.
# It provides the foundation for creating nodes, handling
# communication (topics, services, actions), and managing the
# lifecycle of a ROS-based program.
#
# In this script, rclpy is used to:
# - initialize the ROS 2 system → rclpy.init()
# - allow the node to communicate with the ROS graph
# - cleanly shut everything down → rclpy.shutdown()
#
# Typical functions used (especially in this setup):
# - rclpy.init() → starts ROS 2 communication (must be called first)
# - rclpy.shutdown() → safely shuts down ROS 2 when done
#
# In more advanced nodes (not directly shown here), rclpy also supports:
# - rclpy.spin(node) → keeps a node alive and processes callbacks
# - rclpy.spin_once(node) → processes a single callback iteration
# - rclpy.ok() → checks if ROS is still running
#
# In this Nav2 setup, BasicNavigator internally manages the node,
# so we do NOT manually create a Node or call rclpy.spin(). Instead,
# rclpy is mainly used to initialize and shut down the system.
# ============================================================
import rclpy
# ============================================================
# PoseStamped Explanation
# ------------------------------------------------------------
# PoseStamped (geometry_msgs.msg) represents a robot pose with
# additional context required for navigation:
# - header.frame_id → defines the coordinate frame (e.g., 'map')
# - header.stamp → timestamp used for TF/time synchronization
# - pose.position → x, y, z location
# - pose.orientation → quaternion rotation (z, w used for 2D yaw)
#
# PoseStamped is REQUIRED for Nav2 because the system must know:
# where the goal is (frame) and when it was generated (time).
#
# A regular Pose only includes position and orientation, but has
# NO frame or timestamp, so Nav2 cannot correctly interpret it.
# ============================================================
from geometry_msgs.msg import PoseStamped
# ============================================================
# BasicNavigator Explanation
# ------------------------------------------------------------
# BasicNavigator is a high-level Nav2 helper that wraps action
# clients, lifecycle nodes, and communication into a simple API.
# It allows sending navigation goals without manually managing
# Nav2 internals.
#
# TaskResult is used to interpret the result of a navigation task
# (SUCCEEDED, FAILED, CANCELED).
#
# This class supports multiple behaviors such as:
# - goToPose() for single goal navigation
# - goThroughPoses() for waypoint navigation
# - getPath() for planning only
# ============================================================
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
#
# A regular yaw value cannot be used directly because ROS
# messages do not have a yaw field — only quaternion format.
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
#
# This function acts as a helper to standardize goal creation
# before sending it to navigator.goToPose().
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
# main() Function Summary
# ------------------------------------------------------------
# This function initializes ROS 2, sets the robot’s starting
# position (initial pose), and sends a navigation goal using
# Nav2. It waits until the navigation stack is ready, then
# commands the robot to move to a specified (x, y, yaw) goal.
#
# While the robot is moving, it periodically checks for feedback
# and waits until the task is complete. Once finished, it reports
# whether the navigation succeeded, failed, or was canceled,
# then shuts down the system cleanly.
# ============================================================
def main():
    # Initialize ROS 2 system (must be called before using ROS)
    rclpy.init()

    # Create a high-level Nav2 controller object
    navigator = BasicNavigator()

    # ------------------------------------------------------------
    # Default goal (used if no CLI arguments are provided)
    # ------------------------------------------------------------
    goal_x = 1.5
    goal_y = 0.5
    goal_yaw_deg = 90.0

    # Allow user to override goal via command line
    # Example:
    # ros2 run tb3_nav2_slam single_goal_nav 1.0 0.0 90
    if len(sys.argv) == 4:
        goal_x = float(sys.argv[1])
        goal_y = float(sys.argv[2])
        goal_yaw_deg = float(sys.argv[3])

    # ------------------------------------------------------------
    # Set initial pose (VERY IMPORTANT for localization)
    # ------------------------------------------------------------
    # This should match the robot's actual position in the map
    init_pose = build_pose(navigator, -2.0, -0.5, 0.0)

    # Send initial pose to AMCL
    navigator.setInitialPose(init_pose)

    # Wait until Nav2 stack (planner, controller, AMCL) is ready
    navigator.info('Waiting for Nav2 to become active...')
    navigator.waitUntilNav2Active()

    # ------------------------------------------------------------
    # Build goal pose
    # ------------------------------------------------------------
    goal_pose = build_pose(navigator, goal_x, goal_y, goal_yaw_deg)

    # Log goal information
    navigator.info(
        f'Sending goal: x={goal_x:.2f}, y={goal_y:.2f}, yaw_deg={goal_yaw_deg:.1f}'
    )

    # Send goal to Nav2
    navigator.goToPose(goal_pose)

    # Track time for periodic feedback logging
    last_feedback_time = time.time()

    # ------------------------------------------------------------
    # Main loop: wait for navigation to complete
    # ------------------------------------------------------------
    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        now = time.time()

        # Print status every 1 second
        if feedback is not None and (now - last_feedback_time) > 1.0:
            navigator.info('Navigation task still running...')
            last_feedback_time = now

    # ------------------------------------------------------------
    # Get final result of navigation
    # ------------------------------------------------------------
    result = navigator.getResult()

    # Check result status
    if result == TaskResult.SUCCEEDED:
        navigator.info('Goal succeeded!')
    elif result == TaskResult.CANCELED:
        navigator.warn('Goal was canceled.')
    elif result == TaskResult.FAILED:
        navigator.error('Goal failed.')
    else:
        navigator.error('Goal returned an unknown result.')

    # Clean up Nav2 node
    navigator.destroyNode()

    # Shut down ROS 2
    rclpy.shutdown()

# ============================================================
# Python Entry Point Check
# ------------------------------------------------------------
# This condition ensures that main() only runs when this script
# is executed directly (e.g., via "ros2 run").
#
# __name__ is a built-in Python variable:
# - '__main__' → when the script is run directly
# - module name → when the script is imported
#
# This prevents main() from running unintentionally when the
# file is imported into another Python script.
# ============================================================
if __name__ == '__main__':
    main()