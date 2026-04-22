#!/usr/bin/env python3

import math
import sys
import time

from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy


def yaw_to_quaternion_z_w(yaw: float):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return qz, qw


def build_goal(x: float, y: float, yaw_deg: float) -> PoseStamped:
    goal = PoseStamped()
    goal.header.frame_id = 'map'
    goal.header.stamp = rclpy.clock.Clock().now().to_msg()

    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = 0.0

    yaw_rad = math.radians(yaw_deg)
    qz, qw = yaw_to_quaternion_z_w(yaw_rad)
    goal.pose.orientation.z = qz
    goal.pose.orientation.w = qw

    return goal


def main():
    rclpy.init()

    navigator = BasicNavigator()

    # Wait until Nav2 is active before sending a goal
    navigator.waitUntilNav2Active()

    # Default goal if no CLI args are provided
    x = 1.0
    y = 0.0
    yaw_deg = 0.0

    # Optional CLI usage:
    # ros2 run tb3_nav2_slam single_goal_nav 1.0 0.5 90
    if len(sys.argv) == 4:
        x = float(sys.argv[1])
        y = float(sys.argv[2])
        yaw_deg = float(sys.argv[3])

    goal = build_goal(x, y, yaw_deg)

    navigator.info(f'Sending goal: x={x:.2f}, y={y:.2f}, yaw_deg={yaw_deg:.1f}')
    navigator.goToPose(goal)

    last_feedback_time = time.time()

    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        now = time.time()

        if feedback is not None and (now - last_feedback_time) > 1.0:
            navigator.info('Navigation task still running...')
            last_feedback_time = now

    result = navigator.getResult()

    if result == TaskResult.SUCCEEDED:
        navigator.info('Goal succeeded!')
    elif result == TaskResult.CANCELED:
        navigator.warn('Goal was canceled.')
    elif result == TaskResult.FAILED:
        navigator.error('Goal failed.')
    else:
        navigator.error('Goal returned an unknown result.')

    navigator.lifecycleShutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()