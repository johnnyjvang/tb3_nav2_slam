#!/usr/bin/env python3

import json
import math
from typing import Any

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class ArucoVfhFollower(Node):
    def __init__(self) -> None:
        super().__init__("aruco_vfh_follower")

        self.declare_parameter("detection_topic", "/aruco_multi/detections")
        self.declare_parameter("cmd_vel_topic", "/tb3_1/cmd_vel")

        self.declare_parameter("follow_distance", 0.35)
        self.declare_parameter("linear_gain", 0.35)
        self.declare_parameter("angular_gain", 1.50)
        self.declare_parameter("max_linear_speed", 0.08)
        self.declare_parameter("max_angular_speed", 0.50)
        self.declare_parameter("lost_timeout", 0.75)

        # P-control uses the current tracking error to create velocity commands
        # linear_gain controls forward speed response to distance error
        # angular_gain controls turning response to left/right camera error
        self.declare_parameter("use_p_control", True)

        # EMA smoothing factor for noisy ArUco readings
        # Lower = smoother/slower
        # Higher = faster/more reactive
        self.declare_parameter("ema_alpha", 0.30)

        # LiDAR emergency stop parameters
        # If an obstacle is closer than this in front, block forward motion
        self.declare_parameter("scan_topic", "/tb3_1/scan")
        self.declare_parameter("emergency_stop_distance", 0.35)
        self.declare_parameter("front_angle_deg", 30.0)

        # VFH-lite avoidance parameters
        # VFH checks many LiDAR angle sectors instead of only left vs right
        self.declare_parameter("avoidance_distance", 0.55)
        self.declare_parameter("avoidance_linear_speed", 0.02)
        self.declare_parameter("vfh_sector_deg", 10.0)
        self.declare_parameter("vfh_angle_limit_deg", 90.0)
        self.declare_parameter("vfh_clearance_distance", 0.40)
        self.declare_parameter("vfh_turn_gain", 1.25)
        self.declare_parameter("camera_horizontal_fov_deg", 60.0)

        # Side zones are still kept for recovery backup decisions
        self.declare_parameter("side_angle_min_deg", 30.0)
        self.declare_parameter("side_angle_max_deg", 90.0)

        # Recovery behavior parameters
        # If the robot gets too close, slowly back up and turn toward clearer space
        self.declare_parameter("enable_recovery_backup", True)
        self.declare_parameter("recovery_backup_speed", -0.03)
        self.declare_parameter("recovery_turn_speed", 0.20)

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

        # Load VFH-lite settings from ROS parameters
        self.avoidance_distance = self.get_float_parameter(
            "avoidance_distance",
            0.55,
        )
        self.avoidance_linear_speed = self.get_float_parameter(
            "avoidance_linear_speed",
            0.02,
        )
        self.vfh_sector_deg = self.get_float_parameter("vfh_sector_deg", 10.0)
        self.vfh_angle_limit_deg = self.get_float_parameter(
            "vfh_angle_limit_deg",
            90.0,
        )
        self.vfh_clearance_distance = self.get_float_parameter(
            "vfh_clearance_distance",
            0.40,
        )
        self.vfh_turn_gain = self.get_float_parameter("vfh_turn_gain", 1.25)
        self.camera_horizontal_fov_deg = self.get_float_parameter(
            "camera_horizontal_fov_deg",
            60.0,
        )

        self.side_angle_min_deg = self.get_float_parameter(
            "side_angle_min_deg",
            30.0,
        )
        self.side_angle_max_deg = self.get_float_parameter(
            "side_angle_max_deg",
            90.0,
        )

        # Load recovery backup settings from ROS parameters
        self.enable_recovery_backup = self.get_bool_parameter(
            "enable_recovery_backup",
            True,
        )
        self.recovery_backup_speed = self.get_float_parameter(
            "recovery_backup_speed",
            -0.03,
        )
        self.recovery_turn_speed = self.get_float_parameter(
            "recovery_turn_speed",
            0.20,
        )

        self.last_detection_time = None
        self.latest_detection = None

        # Filtered target values used for smoother control
        self.filtered_distance = None
        self.filtered_error_x = None
        self.filtered_tag_id = None

        # Latest closest obstacle distance in the front LiDAR sector
        self.front_obstacle_distance = None

        # Track closest obstacle distances on the left and right
        # These are used by recovery backup when the robot gets too close
        self.left_obstacle_distance = None
        self.right_obstacle_distance = None

        # VFH sector list stores angle and clearance values from LiDAR
        # Each sector represents a possible steering direction
        self.vfh_sectors = []

        self.det_sub = self.create_subscription(
            String,
            self.detection_topic,
            self.detection_callback,
            10,
        )

        # Subscribe to LiDAR scan data
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

        self.get_logger().info("ArUco VFH follower started")
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

        # Log VFH parameters so tuning is visible at runtime
        self.get_logger().info(f"Avoidance distance: {self.avoidance_distance:.2f} m")
        self.get_logger().info(
            f"Avoidance linear speed: {self.avoidance_linear_speed:.2f} m/s"
        )
        self.get_logger().info(f"VFH sector size: {self.vfh_sector_deg:.1f} deg")
        self.get_logger().info(
            f"VFH angle limit: +/- {self.vfh_angle_limit_deg:.1f} deg"
        )
        self.get_logger().info(
            f"VFH clearance distance: {self.vfh_clearance_distance:.2f} m"
        )
        self.get_logger().info(f"VFH turn gain: {self.vfh_turn_gain:.2f}")

        # Log recovery parameters so tuning is visible at runtime
        self.get_logger().info(
            f"Recovery backup enabled: {self.enable_recovery_backup}"
        )
        self.get_logger().info(
            f"Recovery backup speed: {self.recovery_backup_speed:.2f} m/s"
        )
        self.get_logger().info(
            f"Recovery turn speed: {self.recovery_turn_speed:.2f} rad/s"
        )

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
        # Clear stale filter values when target tracking is lost
        self.filtered_distance = None
        self.filtered_error_x = None
        self.filtered_tag_id = None

    def apply_ema_filter(
        self,
        raw_distance: float,
        raw_error_x: float,
        tag_id: int,
    ) -> tuple[float, float]:
        # Smooth distance and horizontal error before control

        # Reset filter if the selected tag changes
        if self.filtered_tag_id != tag_id:
            self.filtered_distance = raw_distance
            self.filtered_error_x = raw_error_x
            self.filtered_tag_id = tag_id
            return raw_distance, raw_error_x

        # Initialize filter on first valid reading
        if self.filtered_distance is None or self.filtered_error_x is None:
            self.filtered_distance = raw_distance
            self.filtered_error_x = raw_error_x
        else:
            # EMA = alpha * new reading + (1 - alpha) * previous filtered value
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
        # Convert current distance/angle error directly into speed commands
        # Bigger error = stronger correction

        distance_error = distance - self.follow_distance

        # Positive distance error means the robot is too far from the tag
        linear_x = self.linear_gain * distance_error

        # Deadband prevents tiny distance noise from making the robot creep
        if distance_error <= 0.02:
            linear_x = 0.0

        linear_x = self.clamp(
            linear_x,
            0.0,
            self.max_linear_speed,
        )

        # error_x is the tag offset from camera center
        # Negative sign turns the robot back toward the tag
        angle_error = -error_x

        angular_z = self.angular_gain * angle_error

        angular_z = self.clamp(
            angular_z,
            -self.max_angular_speed,
            self.max_angular_speed,
        )

        return linear_x, angular_z, distance_error, angle_error

    def apply_lidar_emergency_stop(self, linear_x: float) -> tuple[float, bool]:
        # LiDAR can override forward motion when an obstacle is too close

        if self.front_obstacle_distance is None:
            return linear_x, False

        if self.front_obstacle_distance < self.emergency_stop_distance:
            return 0.0, True

        return linear_x, False

    def get_clearance_value(self, distance: float | None) -> float:
        # Convert missing LiDAR data into zero clearance
        # This prevents unknown space from being treated as safely open
        if distance is None:
            return 0.0

        return distance

    def apply_recovery_backup(self) -> tuple[float, float, bool, str]:
        # Recovery backup behavior
        # If the robot is too close to an obstacle, reverse slowly instead of freezing
        # The robot also turns toward the side with more open space

        if not self.enable_recovery_backup:
            return 0.0, 0.0, True, "EMERGENCY_STOP"

        left_clearance = self.get_clearance_value(self.left_obstacle_distance)
        right_clearance = self.get_clearance_value(self.right_obstacle_distance)

        linear_x = self.recovery_backup_speed

        if left_clearance > right_clearance:
            # Left side is more open so back up while turning left
            angular_z = abs(self.recovery_turn_speed)
            return linear_x, angular_z, True, "RECOVERY_BACKUP_LEFT"

        # Right side is more open or tied so back up while turning right
        angular_z = -abs(self.recovery_turn_speed)
        return linear_x, angular_z, True, "RECOVERY_BACKUP_RIGHT"

    def get_target_heading_deg(self, error_x: float) -> float:
        # Convert ArUco camera error into an approximate desired heading
        # error_x is normalized from left to right in the image
        # Negative error_x means the tag is left of center
        half_fov = self.camera_horizontal_fov_deg / 2.0

        target_heading_deg = -error_x * half_fov

        return self.clamp(
            target_heading_deg,
            -self.vfh_angle_limit_deg,
            self.vfh_angle_limit_deg,
        )

    def build_vfh_sectors(self, msg: LaserScan) -> list[dict]:
        # Build VFH sectors from the LiDAR scan
        # Each sector stores the closest obstacle in that angle window

        sector_count = int((2.0 * self.vfh_angle_limit_deg) / self.vfh_sector_deg) + 1
        sector_data = []

        start_angle_deg = -self.vfh_angle_limit_deg

        for sector_index in range(sector_count):
            center_deg = start_angle_deg + (sector_index * self.vfh_sector_deg)
            sector_data.append(
                {
                    "center_deg": center_deg,
                    "min_distance": None,
                }
            )

        for index, distance in enumerate(msg.ranges):
            if not math.isfinite(distance):
                continue

            if distance < msg.range_min or distance > msg.range_max:
                continue

            angle_rad = msg.angle_min + (index * msg.angle_increment)
            angle_deg = math.degrees(angle_rad)

            # Normalize angle into -180 to 180 degrees
            while angle_deg > 180.0:
                angle_deg -= 360.0

            while angle_deg < -180.0:
                angle_deg += 360.0

            if angle_deg < -self.vfh_angle_limit_deg:
                continue

            if angle_deg > self.vfh_angle_limit_deg:
                continue

            sector_index = int(
                round((angle_deg + self.vfh_angle_limit_deg) / self.vfh_sector_deg)
            )
            sector_index = max(0, min(sector_index, sector_count - 1))

            current_min = sector_data[sector_index]["min_distance"]

            if current_min is None or distance < current_min:
                sector_data[sector_index]["min_distance"] = float(distance)

        return sector_data

    def choose_vfh_sector(self, target_heading_deg: float) -> dict | None:
        # Choose the open sector closest to the ArUco target direction
        # This replaces hard left/right avoidance with many possible headings

        if not self.vfh_sectors:
            return None

        open_sectors = []

        for sector in self.vfh_sectors:
            min_distance = sector["min_distance"]

            if min_distance is None:
                continue

            if min_distance >= self.vfh_clearance_distance:
                open_sectors.append(sector)

        if not open_sectors:
            return None

        return min(
            open_sectors,
            key=lambda sector: abs(sector["center_deg"] - target_heading_deg),
        )

    def apply_vfh_avoidance(
        self,
        linear_x: float,
        angular_z: float,
        error_x: float,
    ) -> tuple[float, float, bool, str, float | None]:
        # VFH-lite avoidance layer
        #
        # Priority
        # 1 Recovery backup if obstacle is too close
        # 2 Normal follow if front is clear
        # 3 Choose safest LiDAR sector closest to ArUco target direction

        linear_x, emergency_stop_active = self.apply_lidar_emergency_stop(linear_x)

        if emergency_stop_active:
            return (*self.apply_recovery_backup(), None)

        if self.front_obstacle_distance is None:
            return linear_x, angular_z, False, "FOLLOW", None

        if self.front_obstacle_distance >= self.avoidance_distance:
            return linear_x, angular_z, False, "FOLLOW", None

        target_heading_deg = self.get_target_heading_deg(error_x)
        best_sector = self.choose_vfh_sector(target_heading_deg)

        if best_sector is None:
            # If no safe sector exists, use recovery backup as fallback
            return (*self.apply_recovery_backup(), None)

        selected_heading_deg = float(best_sector["center_deg"])
        selected_heading_rad = math.radians(selected_heading_deg)

        # Limit forward speed during VFH steering
        linear_x = min(linear_x, self.avoidance_linear_speed)

        # Turn toward the selected safe sector
        angular_z = self.vfh_turn_gain * selected_heading_rad

        angular_z = self.clamp(
            angular_z,
            -self.max_angular_speed,
            self.max_angular_speed,
        )

        if selected_heading_deg >= 0.0:
            mode = "VFH_LEFT"
        else:
            mode = "VFH_RIGHT"

        return linear_x, angular_z, False, mode, selected_heading_deg

    def scan_callback(self, msg: LaserScan) -> None:
        # Read LiDAR sectors and save closest valid distances

        front_angle_rad = math.radians(self.front_angle_deg)
        side_angle_min_rad = math.radians(self.side_angle_min_deg)
        side_angle_max_rad = math.radians(self.side_angle_max_deg)

        valid_front_ranges = []
        valid_left_ranges = []
        valid_right_ranges = []

        # Build VFH sectors from the full front LiDAR region
        self.vfh_sectors = self.build_vfh_sectors(msg)

        for index, distance in enumerate(msg.ranges):
            if not math.isfinite(distance):
                continue

            if distance < msg.range_min or distance > msg.range_max:
                continue

            angle = msg.angle_min + (index * msg.angle_increment)

            # Normalize angle into -pi to pi radians
            while angle > math.pi:
                angle -= 2.0 * math.pi

            while angle < -math.pi:
                angle += 2.0 * math.pi

            if abs(angle) <= front_angle_rad:
                valid_front_ranges.append(float(distance))

            # Positive angles are treated as the left side
            if side_angle_min_rad <= angle <= side_angle_max_rad:
                valid_left_ranges.append(float(distance))

            # Negative angles are treated as the right side
            if -side_angle_max_rad <= angle <= -side_angle_min_rad:
                valid_right_ranges.append(float(distance))

        if valid_front_ranges:
            self.front_obstacle_distance = min(valid_front_ranges)
        else:
            self.front_obstacle_distance = None

        # Save closest obstacle distance on each side
        if valid_left_ranges:
            self.left_obstacle_distance = min(valid_left_ranges)
        else:
            self.left_obstacle_distance = None

        if valid_right_ranges:
            self.right_obstacle_distance = min(valid_right_ranges)
        else:
            self.right_obstacle_distance = None

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

            # Use filtered values for smoother velocity commands
            distance, error_x = self.apply_ema_filter(
                raw_distance,
                raw_error_x,
                tag_id,
            )

            # Use the current tracking error to calculate forward and turn speed
            linear_x, angular_z, distance_error, angle_error = self.compute_p_control(
                distance,
                error_x,
            )

            # Apply emergency stop first then VFH steering if front is blocked
            linear_x, angular_z, emergency_stop_active, avoid_mode, vfh_heading = (
                self.apply_vfh_avoidance(linear_x, angular_z, error_x)
            )

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

            left_distance_text = (
                f"{self.left_obstacle_distance:.3f}"
                if self.left_obstacle_distance is not None
                else "None"
            )
            right_distance_text = (
                f"{self.right_obstacle_distance:.3f}"
                if self.right_obstacle_distance is not None
                else "None"
            )
            vfh_heading_text = (
                f"{vfh_heading:.1f}"
                if vfh_heading is not None
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
                f"left_obstacle={left_distance_text} m | "
                f"right_obstacle={right_distance_text} m | "
                f"vfh_heading={vfh_heading_text} deg | "
                f"mode={avoid_mode} | "
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

            # Reset filter after target loss
            self.reset_filter()

            self.get_logger().warn(
                "Tag detection timeout resetting EMA filter",
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

    node = ArucoVfhFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()