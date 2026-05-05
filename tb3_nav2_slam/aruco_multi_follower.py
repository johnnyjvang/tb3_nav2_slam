#!/usr/bin/env python3

import json
from typing import Any

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
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

        self.last_detection_time = None
        self.latest_detection = None

        self.det_sub = self.create_subscription(
            String,
            self.detection_topic,
            self.detection_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            self.cmd_vel_topic,
            10,
        )

        self.timer = self.create_timer(0.1, self.watchdog_callback)

        self.get_logger().info("ArUco multi-follower started.")
        self.get_logger().info(f"Subscribing to: {self.detection_topic}")
        self.get_logger().info(f"Publishing TwistStamped to: {self.cmd_vel_topic}")
        self.get_logger().info(f"Follow distance: {self.follow_distance:.2f} m")
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

    def clamp(self, value: float, min_value: float, max_value: float) -> float:
        return max(min(value, max_value), min_value)

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

            distance = float(self.latest_detection["distance"])
            error_x = float(self.latest_detection["normalized_error_x"])
            tag_id = int(self.latest_detection["id"])
            side = str(self.latest_detection["side"])

            distance_error = distance - self.follow_distance

            linear_x = self.linear_gain * distance_error

            if distance_error <= 0.02:
                linear_x = 0.0

            linear_x = self.clamp(
                linear_x,
                0.0,
                self.max_linear_speed,
            )

            angular_z = -self.angular_gain * error_x

            angular_z = self.clamp(
                angular_z,
                -self.max_angular_speed,
                self.max_angular_speed,
            )

            self.publish_cmd(linear_x, angular_z)

            visible_summary = [
                f"{int(d['id'])}:{d.get('side', 'unknown')}"
                for d in detections
            ]

            self.get_logger().info(
                f"Visible={visible_summary} | "
                f"Selected ID {tag_id} ({side}) | "
                f"distance={distance:.3f} m | "
                f"error_x={error_x:.3f} | "
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
            self.get_logger().warn(
                "Tag detection timeout. Stopping robot.",
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