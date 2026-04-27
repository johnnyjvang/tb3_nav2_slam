#!/usr/bin/env python3

# ============================================================
# timer_based_patrol_explorer.py Summary
# ------------------------------------------------------------
# This node samples valid patrol goals from the map and sends
# them sequentially to Nav2 using goToPose().
#
# Behavior:
# - Waits for /map and Nav2 to be ready
# - Finds free-space points from OccupancyGrid
# - Filters out points too close to walls
# - Dynamically scales patrol spacing based on map size
# - Sends a new patrol goal on a timed interval
# - Avoids picking goals near recently visited goals
# - Avoids repeating the same general area
# - If a goal fails, pick another one
# - Saves a PNG map with patrol goal locations
# - Saves a CSV summary with goal position and time
# - Optionally rotates the saved PNG map for display alignment
# - Marks the world origin at x=0, y=0 on the saved PNG
# - Color-codes start, patrol, final, path, and origin markers
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


class TimerBasedPatrolExplorer(Node):

    def __init__(self):
        super().__init__('timer_based_patrol_explorer')

        # ------------------------------------------------------------
        # PARAMETERS TO TUNE
        # ------------------------------------------------------------

        self.num_goals = 5

        # Minimum wall clearance for a sampled patrol point.
        self.safe_radius = 0.40

        # If True, distance rules are automatically calculated from map size.
        self.use_dynamic_spacing = True

        # Dynamic spacing ratios based on the smaller side of the map.
        self.last_goal_spacing_ratio = 0.25
        self.recent_goal_spacing_ratio = 0.20

        # Safety limits for dynamic spacing.
        self.min_dynamic_spacing = 1.50
        self.max_dynamic_spacing = 5.00

        # These are overwritten if dynamic spacing is enabled.
        self.min_distance_from_last_goal = 2.50
        self.min_distance_from_recent_goals = 2.00

        # How many previous successful goals to avoid.
        self.recent_goal_memory = 3

        # Delay before selecting the next patrol goal.
        self.patrol_interval_sec = 5.0

        # Maximum number of random attempts before relaxing location rules.
        self.max_goal_attempts = 300

        # Optional override if the map resolution is incorrect.
        self.override_resolution: Optional[float] = None

        # ------------------------------------------------------------
        # MAP PLOT SETTINGS
        # ------------------------------------------------------------
        # This only changes the saved PNG visualization.
        # It does NOT affect Nav2, goals, TF, localization, or /map.
        #
        # Options:
        #   0    = no rotation
        #   90   = rotate image counterclockwise 90 degrees
        #   -90  = rotate image clockwise 90 degrees
        #   180  = rotate image 180 degrees
        #
        # If your saved PNG map looks 90 degrees clockwise compared
        # to the goal points, try 90 first.
        self.map_plot_rotation_deg = 90

        # Show world origin on saved PNG.
        self.show_origin_marker = True

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
    # LIMIT VALUE BETWEEN MIN AND MAX
    # ------------------------------------------------------------
    def clamp(self, value, min_value, max_value):

        return max(min_value, min(value, max_value))

    # ------------------------------------------------------------
    # CONFIGURE DYNAMIC SPACING
    # ------------------------------------------------------------
    def configure_dynamic_spacing(self):

        if self.map_data is None:
            self.get_logger().warn(
                'Map data is None. Dynamic spacing cannot be configured.'
            )
            return

        msg = self.map_data

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        if self.override_resolution is not None:
            resolution = self.override_resolution

        map_width_m = width * resolution
        map_height_m = height * resolution

        map_scale = min(map_width_m, map_height_m)

        if self.use_dynamic_spacing:
            last_goal_spacing = map_scale * self.last_goal_spacing_ratio
            recent_goal_spacing = map_scale * self.recent_goal_spacing_ratio

            self.min_distance_from_last_goal = self.clamp(
                last_goal_spacing,
                self.min_dynamic_spacing,
                self.max_dynamic_spacing,
            )

            self.min_distance_from_recent_goals = self.clamp(
                recent_goal_spacing,
                self.min_dynamic_spacing,
                self.max_dynamic_spacing,
            )

        self.get_logger().info('------------------------------')
        self.get_logger().info('Dynamic patrol spacing settings')
        self.get_logger().info('------------------------------')
        self.get_logger().info(f'Map width: {map_width_m:.2f} m')
        self.get_logger().info(f'Map height: {map_height_m:.2f} m')
        self.get_logger().info(f'Map scale used: {map_scale:.2f} m')
        self.get_logger().info(
            f'Min distance from last goal: {self.min_distance_from_last_goal:.2f} m'
        )
        self.get_logger().info(
            f'Min distance from recent goals: {self.min_distance_from_recent_goals:.2f} m'
        )

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
    # DISTANCE BETWEEN TWO POINTS
    # ------------------------------------------------------------
    def distance_between_points(self, point_a, point_b):

        ax, ay = point_a
        bx, by = point_b

        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    # ------------------------------------------------------------
    # CHECK IF GOAL IS FAR FROM RECENT GOALS
    # ------------------------------------------------------------
    def is_far_from_recent_goals(self, candidate, selected_goals):

        if len(selected_goals) == 0:
            return True

        last_goal = selected_goals[-1]
        last_point = (last_goal['x'], last_goal['y'])

        distance_from_last = self.distance_between_points(
            candidate,
            last_point,
        )

        if distance_from_last < self.min_distance_from_last_goal:
            return False

        recent_goals = selected_goals[-self.recent_goal_memory:]

        for goal in recent_goals:
            recent_point = (goal['x'], goal['y'])

            distance_from_recent = self.distance_between_points(
                candidate,
                recent_point,
            )

            if distance_from_recent < self.min_distance_from_recent_goals:
                return False

        return True

    # ------------------------------------------------------------
    # SELECT PATROL GOAL
    # ------------------------------------------------------------
    def select_patrol_goal(self, safe_points, selected_goals):

        for attempt in range(self.max_goal_attempts):

            candidate = random.choice(safe_points)

            if self.is_far_from_recent_goals(candidate, selected_goals):
                self.get_logger().info(
                    f'Selected patrol point after {attempt + 1} attempt(s).'
                )
                return candidate

        self.get_logger().warn(
            'Could not find a patrol point far from recent goals. '
            'Relaxing repeat-area rule and choosing a random safe point.'
        )

        return random.choice(safe_points)

    # ------------------------------------------------------------
    # ROTATE MAP IMAGE FOR PNG DISPLAY
    # ------------------------------------------------------------
    def rotate_map_image_for_display(self, image):

        if self.map_plot_rotation_deg == 90:
            return np.rot90(image, k=1)

        if self.map_plot_rotation_deg == -90:
            return np.rot90(image, k=-1)

        if self.map_plot_rotation_deg == 180:
            return np.rot90(image, k=2)

        return image

    # ------------------------------------------------------------
    # PRINT SUMMARY
    # ------------------------------------------------------------
    def print_goal_summary(self, selected_goals):

        self.get_logger().info('------------------------------')
        self.get_logger().info('Timer-based patrol summary')
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
                f"time={goal['elapsed_time']:.2f} sec, "
                f"wait={goal['wait_before_goal_sec']:.2f} sec"
            )

        self.get_logger().info(f'Total navigation time: {total_time:.2f} sec')
        self.get_logger().info(
            f'Average time per goal: {total_time / len(selected_goals):.2f} sec'
        )

    # ------------------------------------------------------------
    # SAVE CSV
    # ------------------------------------------------------------
    def save_goal_summary_csv(self, selected_goals):

        csv_name = f'timer_based_patrol_summary_{self.timestamp}.csv'
        csv_path = os.path.join(self.output_dir, csv_name)

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([
                'goal_num',
                'x',
                'y',
                'theta',
                'wait_before_goal_sec',
                'elapsed_time_sec',
                'result',
                'min_distance_from_last_goal',
                'min_distance_from_recent_goals',
                'map_plot_rotation_deg',
            ])

            for goal in selected_goals:
                writer.writerow([
                    goal['goal_num'],
                    f"{goal['x']:.3f}",
                    f"{goal['y']:.3f}",
                    f"{goal['theta']:.3f}",
                    f"{goal['wait_before_goal_sec']:.3f}",
                    f"{goal['elapsed_time']:.3f}",
                    goal['result'],
                    f"{self.min_distance_from_last_goal:.3f}",
                    f"{self.min_distance_from_recent_goals:.3f}",
                    self.map_plot_rotation_deg,
                ])

        self.get_logger().info(f'Saved CSV: {csv_path}')

    # ------------------------------------------------------------
    # SAVE PNG MAP
    # ------------------------------------------------------------
    def save_goal_map_png(self, selected_goals):

        if self.map_data is None:
            self.get_logger().warn('No map available to save.')
            return

        png_name = f'timer_based_patrol_map_{self.timestamp}.png'
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

        # ------------------------------------------------------------
        # OPTIONAL MAP IMAGE ROTATION FOR PNG DISPLAY ONLY
        # ------------------------------------------------------------
        image = self.rotate_map_image_for_display(image)

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

        # ------------------------------------------------------------
        # PLOT PATROL GOALS
        # ------------------------------------------------------------
        for goal in selected_goals:
            x = goal['x']
            y = goal['y']
            goal_num = goal['goal_num']

            if goal_num == 1:
                marker_color = 'green'
                marker_label = 'Start goal'
            elif goal_num == len(selected_goals):
                marker_color = 'red'
                marker_label = 'Final goal'
            else:
                marker_color = 'blue'
                marker_label = 'Patrol goal'

            plt.scatter(
                x,
                y,
                s=120,
                marker=MarkerStyle('o'),
                color=marker_color,
                label=marker_label if goal_num in [1, len(selected_goals)] else None,
            )

            plt.text(
                x,
                y,
                str(goal_num),
                fontsize=12,
                ha='center',
                va='center',
                color='white',
            )

        # ------------------------------------------------------------
        # PLOT PATROL PATH
        # ------------------------------------------------------------
        if len(selected_goals) > 1:
            path_x = [goal['x'] for goal in selected_goals]
            path_y = [goal['y'] for goal in selected_goals]

            plt.plot(
                path_x,
                path_y,
                linewidth=2,
                color='blue',
                alpha=0.7,
                label='Patrol path',
            )

        # ------------------------------------------------------------
        # SHOW MAP ORIGIN
        # ------------------------------------------------------------
        if self.show_origin_marker:
            plt.scatter(
                0.0,
                0.0,
                s=180,
                marker=MarkerStyle('x'),
                color='purple',
                label='Map origin',
            )

            plt.text(
                0.0,
                0.0,
                ' origin',
                fontsize=10,
                ha='left',
                va='bottom',
                color='purple',
            )

        plt.title('Timer-Based Patrol Goals on Map')
        plt.xlabel('x position (m)')
        plt.ylabel('y position (m)')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

        self.get_logger().info(f'Saved PNG: {png_path}')


def main(args=None):

    rclpy.init(args=args)

    node = TimerBasedPatrolExplorer()
    navigator = BasicNavigator()

    node.get_logger().info('Waiting for map...')
    while rclpy.ok() and not node.map_received:
        rclpy.spin_once(node, timeout_sec=0.1)

    if not rclpy.ok():
        return

    node.configure_dynamic_spacing()

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

        if goals_sent > 0:
            node.get_logger().info(
                f'Waiting {node.patrol_interval_sec:.1f} sec before next patrol goal...'
            )

            wait_start = time.time()

            while rclpy.ok() and time.time() - wait_start < node.patrol_interval_sec:
                rclpy.spin_once(node, timeout_sec=0.1)

            wait_before_goal = time.time() - wait_start
        else:
            wait_before_goal = 0.0

        x, y = node.select_patrol_goal(safe_points, selected_goals)

        theta = random.uniform(-math.pi, math.pi)

        node.get_logger().info(
            f'Navigating to patrol goal {goals_sent + 1}: '
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
                f'Patrol goal {goals_sent + 1} succeeded in {elapsed_time:.2f} sec.'
            )

            selected_goals.append({
                'goal_num': goals_sent + 1,
                'x': x,
                'y': y,
                'theta': theta,
                'wait_before_goal_sec': wait_before_goal,
                'elapsed_time': elapsed_time,
                'result': 'SUCCEEDED',
            })

            goals_sent += 1

        elif result == TaskResult.FAILED:
            node.get_logger().warn(
                f'Patrol goal {goals_sent + 1} failed after '
                f'{elapsed_time:.2f} sec. Picking a new point...'
            )

        elif result == TaskResult.CANCELED:
            node.get_logger().warn(
                f'Patrol goal {goals_sent + 1} canceled after {elapsed_time:.2f} sec.'
            )
            break

        else:
            node.get_logger().warn(
                f'Patrol goal {goals_sent + 1} ended with unknown result '
                f'after {elapsed_time:.2f} sec.'
            )

    node.print_goal_summary(selected_goals)
    node.save_goal_summary_csv(selected_goals)
    node.save_goal_map_png(selected_goals)

    node.get_logger().info('Completed timer-based patrol run. Shutting down.')

    rclpy.shutdown()


if __name__ == '__main__':
    main()