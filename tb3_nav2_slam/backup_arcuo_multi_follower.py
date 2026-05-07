#!/usr/bin/env python3

# ============================================================
# Code is Purely back version of aruco_multi_follower.py
# Does not contain left/right avoidance
# ============================================================

import json
import math
from typing import Any

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class ArucoMultiFollower(Node):
    def __init__(self) -> None:
        super().__init__("aruco_multi_follower")

        self.declare_parameter("detection_topic", "/aruco_multi/detections")
        self.declare_parameter("cmd_vel_topic", "/tb3_1/cmd_vel")

        self.declare_parameter("follow_distance", 0.35)
        self.declare_parameter("linear_gain", 0.35)
        self.declare_parameter("angular_gain", 1.50)
        self.declare_parameter("max_linear_speed", 0.08)
        self.declare_parameter("max_angular_speed", 0.50)
        self.declare_parameter("lost_timeout", 0.75)

        # P-control uses the current tracking error to create velocity commands.
        # linear_gain controls forward speed response to distance error.
        # angular_gain controls turning response to left/right camera error.
        self.declare_parameter("use_p_control", True)

        # EMA smoothing factor for noisy ArUco readings.
        # Lower = smoother/slower, higher = faster/more reactive.
        self.declare_parameter("ema_alpha", 0.30)

        # LiDAR emergency stop parameters.
        # If an obstacle is closer than this in front, block forward motion.
        self.declare_parameter("scan_topic", "/tb3_1/scan")
        self.declare_parameter("emergency_stop_distance", 0.35)
        self.declare_parameter("front_angle_deg", 30.0)

        self.detection_topic = self.get_string_parameter(
            "detection_topic",
            "/aruco_multi/detections",
        )
        self.cmd_vel_topic = self.get_string_parameter(
            "cmd_vel_topic",
            "/tb3_1/cmd_vel",
        )

        self.follow_distance = self.get_float_parameter("follow_distance", 0.35)
        self.linear_gain = self.get_float_parameter("linear_gain", 0.35)
        self.angular_gain = self.get_float_parameter("angular_gain", 1.50)
        self.max_linear_speed = self.get_float_parameter("max_linear_speed", 0.08)
        self.max_angular_speed = self.get_float_parameter("max_angular_speed", 0.50)
        self.lost_timeout = self.get_float_parameter("lost_timeout", 0.75)
        self.use_p_control = self.get_bool_parameter("use_p_control", True)

        self.ema_alpha = self.get_float_parameter("ema_alpha", 0.30)
        self.ema_alpha = self.clamp(self.ema_alpha, 0.01, 1.0)

        self.scan_topic = self.get_string_parameter("scan_topic", "/tb3_1/scan")
        self.emergency_stop_distance = self.get_float_parameter(
            "emergency_stop_distance",
            0.35,
        )
        self.front_angle_deg = self.get_float_parameter("front_angle_deg", 30.0)

        self.last_detection_time = None
        self.latest_detection = None

        # Filtered target values used for smoother control.
        self.filtered_distance = None
        self.filtered_error_x = None
        self.filtered_tag_id = None

        # Latest closest obstacle distance in the front LiDAR sector.
        self.front_obstacle_distance = None

        self.det_sub = self.create_subscription(
            String,
            self.detection_topic,
            self.detection_callback,
            10,
        )

        # Subscribe to LiDAR scan data.
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            self.cmd_vel_topic,
            10,
        )

        self.timer = self.create_timer(0.1, self.watchdog_callback)

        self.get_logger().info("ArUco multi-follower started with EMA smoothing.")
        self.get_logger().info(f"Subscribing to: {self.detection_topic}")
        self.get_logger().info(f"Subscribing to LiDAR: {self.scan_topic}")
        self.get_logger().info(f"Publishing TwistStamped to: {self.cmd_vel_topic}")
        self.get_logger().info(f"Follow distance: {self.follow_distance:.2f} m")
        self.get_logger().info(f"EMA alpha: {self.ema_alpha:.2f}")
        self.get_logger().info(f"P-control enabled: {self.use_p_control}")
        self.get_logger().info(
            f"Emergency stop distance: {self.emergency_stop_distance:.2f} m"
        )
        self.get_logger().info(f"Front LiDAR angle: +/- {self.front_angle_deg:.1f} deg")
        self.get_logger().info("Selection priority: back > left/right > front")

    def get_string_parameter(self, name: str, default_value: str) -> str:
        value: Any = self.get_parameter(name).value
        if isinstance(value, str):
            return value
        return default_value

    def get_float_parameter(self, name: str, default_value: float) -> float:
        value: Any = self.get_parameter(name).value
        if isinstance(value, (float, int)):
            return float(value)
        return default_value

    def get_bool_parameter(self, name: str, default_value: bool) -> bool:
        value: Any = self.get_parameter(name).value
        if isinstance(value, bool):
            return value
        return default_value

    def clamp(self, value: float, min_value: float, max_value: float) -> float:
        return max(min(value, max_value), min_value)

    def reset_filter(self) -> None:
        # Clear stale filter values when target tracking is lost.
        self.filtered_distance = None
        self.filtered_error_x = None
        self.filtered_tag_id = None

    def apply_ema_filter(
        self,
        raw_distance: float,
        raw_error_x: float,
        tag_id: int,
    ) -> tuple[float, float]:
        # Smooth distance and horizontal error before control.

        # Reset filter if the selected tag changes.
        if self.filtered_tag_id != tag_id:
            self.filtered_distance = raw_distance
            self.filtered_error_x = raw_error_x
            self.filtered_tag_id = tag_id
            return raw_distance, raw_error_x

        # Initialize filter on first valid reading.
        if self.filtered_distance is None or self.filtered_error_x is None:
            self.filtered_distance = raw_distance
            self.filtered_error_x = raw_error_x
        else:
            # EMA: alpha * new reading + (1 - alpha) * previous filtered value.
            self.filtered_distance = (
                self.ema_alpha * raw_distance
                + (1.0 - self.ema_alpha) * self.filtered_distance
            )

            self.filtered_error_x = (
                self.ema_alpha * raw_error_x
                + (1.0 - self.ema_alpha) * self.filtered_error_x
            )

        return self.filtered_distance, self.filtered_error_x

    def compute_p_control(
        self,
        distance: float,
        error_x: float,
    ) -> tuple[float, float, float, float]:
        # P-control update:
        # Convert current distance/angle error directly into speed commands.
        # Bigger error = stronger correction.

        distance_error = distance - self.follow_distance

        # Positive distance error means the robot is too far from the tag.
        linear_x = self.linear_gain * distance_error

        # Deadband prevents tiny distance noise from making the robot creep.
        if distance_error <= 0.02:
            linear_x = 0.0

        linear_x = self.clamp(
            linear_x,
            0.0,
            self.max_linear_speed,
        )

        # error_x is the tag offset from camera center.
        # Negative sign turns the robot back toward the tag.
        angle_error = -error_x

        angular_z = self.angular_gain * angle_error

        angular_z = self.clamp(
            angular_z,
            -self.max_angular_speed,
            self.max_angular_speed,
        )

        return linear_x, angular_z, distance_error, angle_error

    def apply_lidar_emergency_stop(self, linear_x: float) -> tuple[float, bool]:
        # Emergency stop layer:
        # ArUco/P-control can request forward motion, but LiDAR can block it.
        # This prevents the robot from driving into a close front obstacle.

        if self.front_obstacle_distance is None:
            return linear_x, False

        if self.front_obstacle_distance < self.emergency_stop_distance:
            return 0.0, True

        return linear_x, False

    def scan_callback(self, msg: LaserScan) -> None:
        # Read only the front LiDAR sector and save the closest valid distance.

        front_angle_rad = math.radians(self.front_angle_deg)
        valid_front_ranges = []

        for index, distance in enumerate(msg.ranges):
            if not math.isfinite(distance):
                continue

            if distance < msg.range_min or distance > msg.range_max:
                continue

            angle = msg.angle_min + (index * msg.angle_increment)

            if abs(angle) <= front_angle_rad:
                valid_front_ranges.append(float(distance))

        if valid_front_ranges:
            self.front_obstacle_distance = min(valid_front_ranges)
        else:
            self.front_obstacle_distance = None

    def choose_best_detection(self, detections: list[dict]) -> dict | None:
        if not detections:
            return None

        back_tags = [d for d in detections if d.get("side") == "back"]
        side_tags = [
            d for d in detections
            if d.get("side") in ("left", "right")
        ]
        front_tags = [d for d in detections if d.get("side") == "front"]

        if back_tags:
            return min(back_tags, key=lambda d: abs(float(d["normalized_error_x"])))

        if side_tags:
            return min(side_tags, key=lambda d: abs(float(d["normalized_error_x"])))

        if front_tags:
            return min(front_tags, key=lambda d: abs(float(d["normalized_error_x"])))

        return min(detections, key=lambda d: float(d["distance"]))

    def detection_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            detections = data.get("detections", [])

            selected_detection = self.choose_best_detection(detections)

            if selected_detection is None:
                return

            self.latest_detection = selected_detection
            self.last_detection_time = self.get_clock().now()

            raw_distance = float(self.latest_detection["distance"])
            raw_error_x = float(self.latest_detection["normalized_error_x"])
            tag_id = int(self.latest_detection["id"])
            side = str(self.latest_detection["side"])

            # Use filtered values for smoother velocity commands.
            distance, error_x = self.apply_ema_filter(
                raw_distance,
                raw_error_x,
                tag_id,
            )

            # P-control update:
            # Use the current tracking error to calculate forward and turn speed.
            linear_x, angular_z, distance_error, angle_error = self.compute_p_control(
                distance,
                error_x,
            )

            # LiDAR emergency stop update:
            # Keep turning allowed, but block forward motion if front is too close.
            linear_x, emergency_stop_active = self.apply_lidar_emergency_stop(linear_x)

            if emergency_stop_active:
                angular_z = 0.0

            self.publish_cmd(linear_x, angular_z)

            visible_summary = [
                f"{int(d['id'])}:{d.get('side', 'unknown')}"
                for d in detections
            ]

            front_distance_text = (
                f"{self.front_obstacle_distance:.3f}"
                if self.front_obstacle_distance is not None
                else "None"
            )

            self.get_logger().info(
                f"Visible={visible_summary} | "
                f"Selected ID {tag_id} ({side}) | "
                f"raw_distance={raw_distance:.3f} m | "
                f"filtered_distance={distance:.3f} m | "
                f"raw_error_x={raw_error_x:.3f} | "
                f"filtered_error_x={error_x:.3f} | "
                f"distance_error={distance_error:.3f} | "
                f"angle_error={angle_error:.3f} | "
                f"front_obstacle={front_distance_text} m | "
                f"estop={emergency_stop_active} | "
                f"cmd_linear={linear_x:.3f} | "
                f"cmd_angular={angular_z:.3f}",
                throttle_duration_sec=0.5,
            )

        except Exception as exc:
            self.get_logger().error(f"Detection processing error: {exc}")
            self.stop_robot()

    def watchdog_callback(self) -> None:
        if self.last_detection_time is None:
            return

        now = self.get_clock().now()
        elapsed = (now - self.last_detection_time).nanoseconds / 1e9

        if elapsed > self.lost_timeout:
            self.stop_robot()
            self.latest_detection = None

            # Reset filter after target loss.
            self.reset_filter()

            self.get_logger().warn(
                "Tag detection timeout. Stopping robot and resetting EMA filter.",
                throttle_duration_sec=1.0,
            )

    def publish_cmd(self, linear_x: float, angular_z: float) -> None:
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"

        cmd.twist.linear.x = float(linear_x)
        cmd.twist.angular.z = float(angular_z)

        self.cmd_pub.publish(cmd)

    def stop_robot(self) -> None:
        self.publish_cmd(0.0, 0.0)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = ArucoMultiFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()