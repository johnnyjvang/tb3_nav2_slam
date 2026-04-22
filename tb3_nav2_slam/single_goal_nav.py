#!/usr/bin/env python3

import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult


def yaw_to_quaternion_z_w(yaw_rad: float):
    qz = math.sin(yaw_rad / 2.0)
    qw = math.cos(yaw_rad / 2.0)
    return qz, qw


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


def main():
    rclpy.init()
    navigator = BasicNavigator()

    # Optional CLI:
    # ros2 run tb3_nav2_slam single_goal_nav goal_x goal_y goal_yaw_deg
    goal_x = 1.5
    goal_y = 0.5
    goal_yaw_deg = 90.0

    if len(sys.argv) == 4:
        goal_x = float(sys.argv[1])
        goal_y = float(sys.argv[2])
        goal_yaw_deg = float(sys.argv[3])

    # Set a known initial pose near the TB3 world start
    init_pose = build_pose(navigator, -2.0, -0.5, 0.0)
    navigator.setInitialPose(init_pose)

    navigator.info('Waiting for Nav2 to become active...')
    navigator.waitUntilNav2Active()

    goal_pose = build_pose(navigator, goal_x, goal_y, goal_yaw_deg)

    navigator.info(
        f'Sending goal: x={goal_x:.2f}, y={goal_y:.2f}, yaw_deg={goal_yaw_deg:.1f}'
    )
    navigator.goToPose(goal_pose)

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

    navigator.destroyNode()
    rclpy.shutdown()


if __name__ == '__main__':
    main()