#!/usr/bin/env python3

# ============================================================
# random_safe_goal_explorer.py Summary
# ------------------------------------------------------------
# This node samples valid navigation goals from the map and
# sends them sequentially to Nav2 using goToPose().
#
# Behavior:
# - Waits for /map and Nav2 to be ready
# - Finds free-space points from OccupancyGrid
# - Filters out points too close to walls
# - Randomly picks goals from valid points
# - Sends goals one at a time
# - If a goal fails, pick another one
# - Saves a PNG map with goal locations
# - Saves a CSV summary with goal position and time
# ============================================================

import csv
import math
import os
import random
import time
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)


class RandomSafeExplorer(Node):

    def __init__(self):
        super().__init__('random_safe_goal_explorer')

        # ------------------------------------------------------------
        # PARAMETERS TO TUNE
        # ------------------------------------------------------------

        self.num_goals = 3
        self.safe_radius = 0.40
        self.override_resolution: Optional[float] = None

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = self.resolve_output_directory()

        self.get_logger().info(f'Saving results to: {self.output_dir}')

        self.map_data: Optional[OccupancyGrid] = None
        self.map_received = False

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos,
        )

    # ------------------------------------------------------------
    # RESOLVE OUTPUT DIRECTORY
    # ------------------------------------------------------------
    def resolve_output_directory(self):

        preferred_dir = os.path.expanduser(
            '~/tb3_nav2_slam/tb3_nav2_slam/results'
        )

        fallback_dirs = [
            preferred_dir,
            os.path.expanduser('~/tb3_nav2_slam/tb3_nav2_slam'),
            os.path.expanduser('~/tb3_nav2_slam'),
            os.getcwd(),
        ]

        for directory in fallback_dirs:
            if os.path.isdir(directory):
                return directory

        return os.getcwd()

    # ------------------------------------------------------------
    # MAP CALLBACK
    # ------------------------------------------------------------
    def map_callback(self, msg: OccupancyGrid):
        self.map_data = msg
        self.map_received = True
        self.get_logger().info('Map received.')

    # ------------------------------------------------------------
    # CHECK IF CELL IS SAFE
    # ------------------------------------------------------------
    def is_safe_cell(self, x_idx, y_idx, width, height, data, radius_cells):

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):

                nx = x_idx + dx
                ny = y_idx + dy

                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    return False

                index = ny * width + nx

                if data[index] != 0:
                    return False

        return True

    # ------------------------------------------------------------
    # EXTRACT SAFE POINTS
    # ------------------------------------------------------------
    def get_safe_points(self):

        if self.map_data is None:
            self.get_logger().error('Map data is None.')
            return []

        msg = self.map_data

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = msg.info.origin
        data = msg.data

        if self.override_resolution is not None:
            resolution = self.override_resolution
            self.get_logger().info(f'Using overridden resolution: {resolution}')
        else:
            self.get_logger().info(f'Using map resolution from /map: {resolution}')

        radius_cells = int(self.safe_radius / resolution)

        self.get_logger().info(
            f'Safe radius set to {self.safe_radius} m (~{radius_cells} cells)'
        )

        safe_points = []

        for y in range(height):
            for x in range(width):

                index = y * width + x

                if data[index] != 0:
                    continue

                if not self.is_safe_cell(x, y, width, height, data, radius_cells):
                    continue

                wx = origin.position.x + x * resolution
                wy = origin.position.y + y * resolution

                safe_points.append((wx, wy))

        self.get_logger().info(f'Total safe points found: {len(safe_points)}')

        return safe_points

    # ------------------------------------------------------------
    # CREATE GOAL POSE
    # ------------------------------------------------------------
    def create_pose(self, x, y, theta=0.0):

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y

        pose.pose.orientation.z = math.sin(theta / 2.0)
        pose.pose.orientation.w = math.cos(theta / 2.0)

        return pose

    # ------------------------------------------------------------
    # PRINT SUMMARY
    # ------------------------------------------------------------
    def print_goal_summary(self, selected_goals):

        self.get_logger().info('------------------------------')
        self.get_logger().info('Random safe goal summary')
        self.get_logger().info('------------------------------')

        if len(selected_goals) == 0:
            self.get_logger().warn('No goals completed.')
            return

        total_time = 0.0

        for goal in selected_goals:
            total_time += goal['elapsed_time']

            self.get_logger().info(
                f"Goal {goal['goal_num']}: "
                f"x={goal['x']:.2f}, "
                f"y={goal['y']:.2f}, "
                f"theta={goal['theta']:.2f}, "
                f"time={goal['elapsed_time']:.2f} sec"
            )

        self.get_logger().info(f'Total navigation time: {total_time:.2f} sec')
        self.get_logger().info(
            f'Average time per goal: {total_time / len(selected_goals):.2f} sec'
        )

    # ------------------------------------------------------------
    # SAVE CSV
    # ------------------------------------------------------------
    def save_goal_summary_csv(self, selected_goals):

        csv_name = f'random_safe_goal_summary_{self.timestamp}.csv'
        csv_path = os.path.join(self.output_dir, csv_name)

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([
                'goal_num',
                'x',
                'y',
                'theta',
                'elapsed_time_sec',
                'result',
            ])

            for goal in selected_goals:
                writer.writerow([
                    goal['goal_num'],
                    f"{goal['x']:.3f}",
                    f"{goal['y']:.3f}",
                    f"{goal['theta']:.3f}",
                    f"{goal['elapsed_time']:.3f}",
                    goal['result'],
                ])

        self.get_logger().info(f'Saved CSV: {csv_path}')

    # ------------------------------------------------------------
    # SAVE PNG MAP
    # ------------------------------------------------------------
    def save_goal_map_png(self, selected_goals):

        if self.map_data is None:
            self.get_logger().warn('No map available to save.')
            return

        png_name = f'random_safe_goal_map_{self.timestamp}.png'
        png_path = os.path.join(self.output_dir, png_name)

        msg = self.map_data

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = msg.info.origin

        data = np.array(msg.data).reshape((height, width))

        image = np.zeros((height, width))
        image[data == 0] = 1.0
        image[data == -1] = 0.5
        image[data > 0] = 0.0

        x_min = origin.position.x
        x_max = origin.position.x + width * resolution
        y_min = origin.position.y
        y_max = origin.position.y + height * resolution

        plt.figure(figsize=(8, 8))
        plt.imshow(
            image,
            cmap='gray',
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
        )

        for goal in selected_goals:
            x = goal['x']
            y = goal['y']
            goal_num = goal['goal_num']

            plt.scatter(x, y, s=120, marker=MarkerStyle('o'))
            plt.text(
                x,
                y,
                str(goal_num),
                fontsize=12,
                ha='center',
                va='center',
            )

        plt.title('Random Safe Goals on Map')
        plt.xlabel('x position (m)')
        plt.ylabel('y position (m)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

        self.get_logger().info(f'Saved PNG: {png_path}')


def main(args=None):

    rclpy.init(args=args)

    node = RandomSafeExplorer()
    navigator = BasicNavigator()

    node.get_logger().info('Waiting for map...')
    while rclpy.ok() and not node.map_received:
        rclpy.spin_once(node, timeout_sec=0.1)

    if not rclpy.ok():
        return

    node.get_logger().info('Waiting for Nav2 to become active...')
    navigator.waitUntilNav2Active()

    safe_points = node.get_safe_points()

    if len(safe_points) == 0:
        node.get_logger().error('No safe points found. Exiting.')
        rclpy.shutdown()
        return

    goals_sent = 0
    selected_goals = []

    while rclpy.ok() and goals_sent < node.num_goals:

        x, y = random.choice(safe_points)
        theta = 0.0

        node.get_logger().info(
            f'Navigating to goal {goals_sent + 1}: '
            f'x="{x:.2f}", y="{y:.2f}", theta="{theta:.2f}"'
        )

        goal_pose = node.create_pose(x, y, theta)

        start_time = time.time()
        navigator.goToPose(goal_pose)

        while rclpy.ok() and not navigator.isTaskComplete():
            feedback = navigator.getFeedback()

            if feedback is not None:
                node.get_logger().info(
                    f'Distance remaining: {feedback.distance_remaining:.2f} m'
                )

            rclpy.spin_once(node, timeout_sec=0.1)

        elapsed_time = time.time() - start_time
        result = navigator.getResult()

        if result == TaskResult.SUCCEEDED:
            node.get_logger().info(
                f'Goal {goals_sent + 1} succeeded in {elapsed_time:.2f} sec.'
            )

            selected_goals.append({
                'goal_num': goals_sent + 1,
                'x': x,
                'y': y,
                'theta': theta,
                'elapsed_time': elapsed_time,
                'result': 'SUCCEEDED',
            })

            goals_sent += 1

        elif result == TaskResult.FAILED:
            node.get_logger().warn(
                f'Goal {goals_sent + 1} failed after '
                f'{elapsed_time:.2f} sec. Picking a new point...'
            )

        elif result == TaskResult.CANCELED:
            node.get_logger().warn(
                f'Goal {goals_sent + 1} canceled after {elapsed_time:.2f} sec.'
            )
            break

        else:
            node.get_logger().warn(
                f'Goal {goals_sent + 1} ended with unknown result '
                f'after {elapsed_time:.2f} sec.'
            )

    node.print_goal_summary(selected_goals)
    node.save_goal_summary_csv(selected_goals)
    node.save_goal_map_png(selected_goals)

    node.get_logger().info('Completed random safe goal run. Shutting down.')

    rclpy.shutdown()


if __name__ == '__main__':
    main()