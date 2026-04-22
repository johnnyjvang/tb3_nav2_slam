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
# - process AMCL subscription callbacks → rclpy.spin_once()
# - cleanly shut everything down → rclpy.shutdown()
#
# Typical functions used in this setup:
# - rclpy.init() → starts ROS 2 communication
# - rclpy.spin_once(node, timeout_sec=...) → processes one callback cycle
# - rclpy.shutdown() → safely shuts down ROS 2
#
# In this Nav2 setup, BasicNavigator internally manages the node,
# so we do NOT manually create a separate Node. Instead, we use
# the navigator itself to subscribe to /amcl_pose and to interact
# with Nav2.
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
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

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
# - setInitialPose() for localization initialization
# - goToPose() for single goal navigation
# - goThroughPoses() for waypoint navigation
# - getPath() for planning only
# ============================================================
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy


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
# Quaternion → Yaw Conversion
# ------------------------------------------------------------
# This converts a planar quaternion back into yaw in degrees.
# It is used when AMCL already has a valid pose and we want to
# reuse that pose instead of forcing a hardcoded start pose.
# ============================================================
def quaternion_z_w_to_yaw_deg(qz: float, qw: float) -> float:
    yaw_rad = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
    return math.degrees(yaw_rad)


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
# AMCL Pose Reader
# ------------------------------------------------------------
# This helper listens to /amcl_pose and stores the most recent
# localized pose in the map frame. If AMCL is already localized,
# we can reuse that pose instead of always forcing a hardcoded
# initial pose like (0, 0, 0).
# ============================================================
class AmclPoseReader:
    def __init__(self, navigator: BasicNavigator):
        self.latest_amcl_pose: PoseWithCovarianceStamped | None = None

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub = navigator.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.callback,
            qos,
        )

    def callback(self, msg: PoseWithCovarianceStamped) -> None:
        self.latest_amcl_pose = msg

    def wait_for_pose(self, navigator: BasicNavigator, timeout_sec: float = 2.0) -> bool:
        start = time.time()

        while time.time() - start < timeout_sec:
            rclpy.spin_once(navigator, timeout_sec=0.1)
            if self.latest_amcl_pose is not None:
                return True

        return False


# ============================================================
# main() Function Summary
# ------------------------------------------------------------
# This function initializes ROS 2, checks whether AMCL already
# has a valid current pose, and then sends a navigation goal
# using Nav2.
#
# Behavior:
# - If /amcl_pose is already available, reuse the current
#   localized pose instead of forcing a static initial pose.
# - If /amcl_pose is not available, fall back to the known
#   starting pose at (0.0, 0.0, 0.0).
#
# This makes repeated runs safer because the script does not
# blindly reset localization back to the original start pose
# after the robot has already moved.
# ============================================================
def main():
    # Initialize ROS 2 system
    rclpy.init()

    # Create high-level Nav2 controller object
    navigator = BasicNavigator()

    # Create AMCL reader so we can check if localization already exists
    amcl_reader = AmclPoseReader(navigator)

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
    # Initial pose handling
    # ------------------------------------------------------------
    # If AMCL already has a valid pose, reuse that localized pose.
    # If not, fall back to the known TurtleBot3 start pose.
    if amcl_reader.wait_for_pose(navigator, timeout_sec=2.0):
        amcl_msg = amcl_reader.latest_amcl_pose

        if amcl_msg is not None:
            x = amcl_msg.pose.pose.position.x
            y = amcl_msg.pose.pose.position.y
            qz = amcl_msg.pose.pose.orientation.z
            qw = amcl_msg.pose.pose.orientation.w
            yaw_deg = quaternion_z_w_to_yaw_deg(qz, qw)

            navigator.info(
                f'AMCL pose detected. Reusing current pose: '
                f'x={x:.2f}, y={y:.2f}, yaw_deg={yaw_deg:.1f}'
            )

            init_pose = build_pose(navigator, x, y, yaw_deg)
        else:
            navigator.warn('AMCL wait returned True but pose is None (unexpected). Using default.')
            init_pose = build_pose(navigator, 0.0, 0.0, 0.0)
    else:
        navigator.info(
            'No AMCL pose available yet. Falling back to default initial pose.'
        )
        init_pose = build_pose(navigator, 0.0, 0.0, 0.0)

    # Send initial pose to AMCL
    navigator.setInitialPose(init_pose)

    # Wait until Nav2 stack (planner, controller, AMCL) is ready
    navigator.info('Waiting for Nav2 to become active...')
    navigator.waitUntilNav2Active()

    # Give localization a short moment to settle
    navigator.info('Waiting for localization to settle...')
    time.sleep(2.0)

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