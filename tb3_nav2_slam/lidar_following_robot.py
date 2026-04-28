#!/usr/bin/env python3

import math
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


Point = Tuple[float, float, float, float]
# Point = x, y, distance, angle


class LidarFollowingRobot(Node):

    def __init__(self):
        super().__init__('lidar_following_robot')

        # ------------------------------------------------------------
        # Topic parameters
        # ------------------------------------------------------------
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        # ------------------------------------------------------------
        # Detection parameters
        # ------------------------------------------------------------
        self.declare_parameter('front_angle_deg', 12.0)
        self.declare_parameter('min_detect_distance', 0.25)
        self.declare_parameter('max_detect_distance', 2.50)

        # TurtleBot-sized target cluster width.
        self.declare_parameter('target_min_width', 0.10)
        self.declare_parameter('target_max_width', 0.45)

        # Gap between adjacent scan points before starting new cluster.
        self.declare_parameter('cluster_gap_threshold', 0.12)
        self.declare_parameter('min_points_per_cluster', 4)

        # ------------------------------------------------------------
        # Following control parameters
        # ------------------------------------------------------------
        self.declare_parameter('target_distance', 0.90)
        self.declare_parameter('distance_tolerance', 0.10)

        self.declare_parameter('linear_kp', 0.35)
        self.declare_parameter('angular_kp', 1.20)

        self.declare_parameter('max_linear_speed', 0.10)
        self.declare_parameter('max_angular_speed', 0.45)

        self.declare_parameter('allow_reverse', False)
        self.declare_parameter('control_rate_hz', 10.0)

        # Stop if target has not been seen recently.
        self.declare_parameter('target_timeout_sec', 0.50)

        # ------------------------------------------------------------
        # Read parameters with Pylance-friendly types
        # ------------------------------------------------------------
        self.scan_topic: str = (
            self.get_parameter('scan_topic').get_parameter_value().string_value
        )

        self.cmd_vel_topic: str = (
            self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        )

        self.front_angle_deg: float = (
            self.get_parameter('front_angle_deg').get_parameter_value().double_value
        )

        self.min_detect_distance: float = (
            self.get_parameter('min_detect_distance').get_parameter_value().double_value
        )

        self.max_detect_distance: float = (
            self.get_parameter('max_detect_distance').get_parameter_value().double_value
        )

        self.target_min_width: float = (
            self.get_parameter('target_min_width').get_parameter_value().double_value
        )

        self.target_max_width: float = (
            self.get_parameter('target_max_width').get_parameter_value().double_value
        )

        self.cluster_gap_threshold: float = (
            self.get_parameter('cluster_gap_threshold').get_parameter_value().double_value
        )

        self.min_points_per_cluster: int = (
            self.get_parameter('min_points_per_cluster').get_parameter_value().integer_value
        )

        self.target_distance: float = (
            self.get_parameter('target_distance').get_parameter_value().double_value
        )

        self.distance_tolerance: float = (
            self.get_parameter('distance_tolerance').get_parameter_value().double_value
        )

        self.linear_kp: float = (
            self.get_parameter('linear_kp').get_parameter_value().double_value
        )

        self.angular_kp: float = (
            self.get_parameter('angular_kp').get_parameter_value().double_value
        )

        self.max_linear_speed: float = (
            self.get_parameter('max_linear_speed').get_parameter_value().double_value
        )

        self.max_angular_speed: float = (
            self.get_parameter('max_angular_speed').get_parameter_value().double_value
        )

        self.allow_reverse: bool = (
            self.get_parameter('allow_reverse').get_parameter_value().bool_value
        )

        self.control_rate_hz: float = (
            self.get_parameter('control_rate_hz').get_parameter_value().double_value
        )

        self.target_timeout_sec: float = (
            self.get_parameter('target_timeout_sec').get_parameter_value().double_value
        )

        # ------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------
        self.latest_scan: Optional[LaserScan] = None
        self.last_target_time_sec: Optional[float] = None

        self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10,
        )

        self.timer = self.create_timer(
            1.0 / self.control_rate_hz,
            self.control_loop,
        )

        self.get_logger().info('LiDAR cluster-following robot started.')
        self.get_logger().info(f'Scan topic: {self.scan_topic}')
        self.get_logger().info(f'Cmd topic: {self.cmd_vel_topic}')
        self.get_logger().info(f'Front cone: +/- {self.front_angle_deg:.1f} deg')
        self.get_logger().info(
            f'Target cluster width: '
            f'{self.target_min_width:.2f} m to {self.target_max_width:.2f} m'
        )
        self.get_logger().info(f'Target follow distance: {self.target_distance:.2f} m')

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------
    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def now_sec(self) -> float:
        now = self.get_clock().now()
        return float(now.nanoseconds) / 1e9

    def clamp(self, value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(value, max_value))

    # ------------------------------------------------------------
    # Convert front LiDAR scan into XY points
    # ------------------------------------------------------------
    def get_front_points(self, scan: LaserScan) -> List[Point]:
        front_angle_rad = math.radians(self.front_angle_deg)
        points: List[Point] = []

        for i, distance in enumerate(scan.ranges):
            if math.isnan(distance) or math.isinf(distance):
                continue

            if distance < self.min_detect_distance:
                continue

            if distance > self.max_detect_distance:
                continue

            angle = scan.angle_min + float(i) * scan.angle_increment

            if abs(angle) > front_angle_rad:
                continue

            x = distance * math.cos(angle)
            y = distance * math.sin(angle)

            points.append((x, y, distance, angle))

        return points

    # ------------------------------------------------------------
    # Group adjacent points into clusters
    # ------------------------------------------------------------
    def cluster_points(self, points: List[Point]) -> List[List[Point]]:
        if len(points) == 0:
            return []

        clusters: List[List[Point]] = []
        current_cluster: List[Point] = [points[0]]

        for point in points[1:]:
            prev_x, prev_y, _, _ = current_cluster[-1]
            x, y, _, _ = point

            gap = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)

            if gap <= self.cluster_gap_threshold:
                current_cluster.append(point)
            else:
                clusters.append(current_cluster)
                current_cluster = [point]

        clusters.append(current_cluster)
        return clusters

    # ------------------------------------------------------------
    # Estimate target from compact cluster
    # ------------------------------------------------------------
    def find_front_target(self, scan: LaserScan) -> Optional[Tuple[float, float, float, int]]:
        """
        Returns:
            distance, angle, width, point_count

        The target is selected from compact clusters in front of the robot.
        This helps reject walls/pillars that are too wide or not TurtleBot-sized.
        """
        points = self.get_front_points(scan)

        if len(points) == 0:
            return None

        clusters = self.cluster_points(points)

        best_target: Optional[Tuple[float, float, float, int]] = None
        best_score: Optional[float] = None

        for cluster in clusters:
            point_count = len(cluster)

            if point_count < self.min_points_per_cluster:
                continue

            xs = [p[0] for p in cluster]
            ys = [p[1] for p in cluster]
            distances = [p[2] for p in cluster]
            angles = [p[3] for p in cluster]

            width = math.sqrt(
                (max(xs) - min(xs)) ** 2 +
                (max(ys) - min(ys)) ** 2
            )

            if width < self.target_min_width:
                continue

            if width > self.target_max_width:
                continue

            avg_distance = sum(distances) / float(point_count)
            avg_angle = sum(angles) / float(point_count)

            # Prefer targets near the center and reasonably close.
            center_penalty = abs(avg_angle) * 0.50
            distance_score = avg_distance
            score = distance_score + center_penalty

            if best_score is None or score < best_score:
                best_score = score
                best_target = (
                    avg_distance,
                    avg_angle,
                    width,
                    point_count,
                )

        return best_target

    # ------------------------------------------------------------
    # Command publishing
    # ------------------------------------------------------------
    def publish_cmd(self, linear_x: float, angular_z: float) -> None:
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        self.cmd_pub.publish(cmd)

    def stop_robot(self) -> None:
        self.publish_cmd(0.0, 0.0)

    # ------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------
    def control_loop(self) -> None:
        if self.latest_scan is None:
            self.get_logger().warn('Waiting for scan...')
            self.stop_robot()
            return

        target = self.find_front_target(self.latest_scan)

        if target is None:
            self.stop_robot()
            self.get_logger().info('No TurtleBot-sized target detected. Stopping.')
            return

        distance, angle, width, point_count = target
        self.last_target_time_sec = self.now_sec()

        distance_error = distance - self.target_distance

        if abs(distance_error) <= self.distance_tolerance:
            linear_x = 0.0
        else:
            linear_x = self.linear_kp * distance_error

        if not self.allow_reverse:
            linear_x = max(0.0, linear_x)

        linear_x = self.clamp(
            linear_x,
            -self.max_linear_speed,
            self.max_linear_speed,
        )

        angular_z = self.angular_kp * angle

        angular_z = self.clamp(
            angular_z,
            -self.max_angular_speed,
            self.max_angular_speed,
        )

        self.get_logger().info(
            f'target distance={distance:.2f} m, '
            f'angle={math.degrees(angle):.1f} deg, '
            f'width={width:.2f} m, '
            f'points={point_count}, '
            f'linear={linear_x:.2f}, '
            f'angular={angular_z:.2f}'
        )

        self.publish_cmd(linear_x, angular_z)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = LidarFollowingRobot()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt. Stopping follower.')
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()