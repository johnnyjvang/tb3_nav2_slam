#!/usr/bin/env python3

# ============================================================
# goal_from_list.py Summary
# ------------------------------------------------------------
# This node sends 1 to 3 goals to Nav2 sequentially using
# goToPose(). The user may provide custom CLI goals or let the
# script use built-in defaults.
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
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
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
# ============================================================
def yaw_to_quaternion_z_w(yaw_rad: float):
    qz = math.sin(yaw_rad / 2.0)
    qw = math.cos(yaw_rad / 2.0)
    return qz, qw


# ============================================================
# Quaternion → Yaw Conversion
# ------------------------------------------------------------
# Converts a planar quaternion back into yaw in degrees.
# This is used when AMCL already has a valid pose and we want
# to reuse that localized pose instead of forcing (0,0,0).
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
# - 0 arguments  -> use default goals
# - 3 arguments  -> 1 pose
# - 6 arguments  -> 2 poses
# - 9 arguments  -> 3 poses
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
# run_goal Helper
# ------------------------------------------------------------
# Sends one goal using goToPose() and waits for completion.
# This matches the single-goal behavior that already works in
# RViz and in single_goal_nav.
# ============================================================
def run_goal(
    navigator: BasicNavigator,
    goal_pose: PoseStamped,
    label: str,
) -> TaskResult:
    navigator.info(f'Sending {label}...')
    navigator.goToPose(goal_pose)

    last_feedback_time = time.time()

    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        now = time.time()

        if feedback is not None and (now - last_feedback_time) > 1.0:
            navigator.info(f'{label} still running...')
            last_feedback_time = now

    result = navigator.getResult()

    if result == TaskResult.SUCCEEDED:
        navigator.info(f'{label} succeeded!')
    elif result == TaskResult.CANCELED:
        navigator.warn(f'{label} was canceled.')
    elif result == TaskResult.FAILED:
        navigator.error(f'{label} failed.')
    else:
        navigator.error(f'{label} returned an unknown result.')

    return result


# ============================================================
# main() Function Summary
# ------------------------------------------------------------
# This function initializes ROS 2, checks whether AMCL already
# has a valid current pose, and then sends 1 to 3 goals
# sequentially using goToPose().
#
# Behavior:
# - If /amcl_pose is already available, reuse the current
#   localized pose instead of forcing a static initial pose.
# - If /amcl_pose is not available, fall back to the known
#   TurtleBot3 starting pose at (0.0, 0.0, 0.0).
#
# This makes reruns safer because the script does not always
# reset localization back to the original start pose after the
# robot has already moved.
# ============================================================
def main():
    rclpy.init()
    navigator = BasicNavigator()
    amcl_reader = AmclPoseReader(navigator)

    default_goal_values = [
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    ]

    try:
        cli_goal_values = parse_goals_from_cli(sys.argv[1:])

        if len(cli_goal_values) == 0:
            goal_values = default_goal_values
            navigator.info('No CLI goals provided. Using default goals.')
        else:
            goal_values = cli_goal_values
            navigator.info(f'Using {len(goal_values)} CLI-provided goal(s).')

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
                navigator.warn(
                    'AMCL wait returned True but pose is None (unexpected). Using default.'
                )
                init_pose = build_pose(navigator, 0.0, 0.0, 0.0)
        else:
            navigator.info(
                'No AMCL pose available yet. Falling back to default initial pose.'
            )
            init_pose = build_pose(navigator, 0.0, 0.0, 0.0)

        navigator.setInitialPose(init_pose)

        navigator.info('Waiting for Nav2 to become active...')
        navigator.waitUntilNav2Active()

        navigator.info('Waiting for localization to settle...')
        time.sleep(2.0)

        goals = [
            build_pose(navigator, x, y, yaw_deg)
            for x, y, yaw_deg in goal_values
        ]

        navigator.info(
            'Prepared goals: '
            + ', '.join(
                f'({x:.2f}, {y:.2f}, {yaw_deg:.1f} deg)'
                for x, y, yaw_deg in goal_values
            )
        )

        for idx, goal_pose in enumerate(goals, start=1):
            result = run_goal(navigator, goal_pose, f'Goal {idx}')

            if result != TaskResult.SUCCEEDED:
                navigator.error(
                    f'Stopping mission because Goal {idx} did not succeed.'
                )
                break
        else:
            navigator.info('All goals completed successfully!')

    except ValueError as exc:
        navigator.error(str(exc))
        navigator.info(
            'Usage:\n'
            'ros2 run tb3_nav2_slam goal_from_list x1 y1 yaw1 [x2 y2 yaw2] [x3 y3 yaw3]'
        )

    finally:
        navigator.destroyNode()
        rclpy.shutdown()


# ============================================================
# Python Entry Point Check
# ------------------------------------------------------------
# This ensures main() only runs when this file is executed
# directly, not when imported into another script.
# ============================================================
if __name__ == '__main__':
    main()