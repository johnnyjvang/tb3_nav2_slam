#!/usr/bin/env python3

import rclpy

from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node


class ArucoFollower(Node):
    def __init__(self):
        super().__init__('aruco_follower')

        self.pose_topic = '/tb3_1/aruco/target_pose'
        self.cmd_vel_topic = '/tb3_1/cmd_vel'

        # Stop this far away from the tag.
        self.target_distance_m = 0.35

        # Safe speed limits.
        self.max_linear_speed = 0.08
        self.max_angular_speed = 0.50

        # Controller gains.
        self.linear_gain = 0.35
        self.angular_gain = 1.50

        # If the tag disappears, stop after this timeout.
        self.tag_timeout_sec = 0.75
        self.last_pose_time = None

        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            self.cmd_vel_topic,
            10,
        )

        self.timer = self.create_timer(0.1, self.watchdog_callback)

        self.get_logger().info('ArUco follower started.')
        self.get_logger().info(f'Subscribing to {self.pose_topic}')
        self.get_logger().info(f'Publishing TwistStamped to {self.cmd_vel_topic}')
        self.get_logger().info(f'Target stopping distance: {self.target_distance_m:.2f} m')

    def clamp(self, value, min_value, max_value):
        return max(min(value, max_value), min_value)

    def publish_cmd(self, linear_x, angular_z):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        cmd.twist.linear.x = float(linear_x)
        cmd.twist.angular.z = float(angular_z)

        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)

    def pose_callback(self, msg: PoseStamped):
        self.last_pose_time = self.get_clock().now()

        forward_distance = msg.pose.position.x
        lateral_offset = msg.pose.position.y

        distance_error = forward_distance - self.target_distance_m

        linear_x = self.linear_gain * distance_error

        if distance_error <= 0.02:
            linear_x = 0.0

        linear_x = self.clamp(
            linear_x,
            0.0,
            self.max_linear_speed,
        )

        # If the marker is off-center, rotate toward it.
        # If the turn direction is backwards in Gazebo, change this to + instead of -.
        angular_z = -self.angular_gain * lateral_offset

        angular_z = self.clamp(
            angular_z,
            -self.max_angular_speed,
            self.max_angular_speed,
        )

        self.publish_cmd(linear_x, angular_z)

        self.get_logger().info(
            f'forward={forward_distance:.3f} m, '
            f'lateral={lateral_offset:.3f} m, '
            f'cmd_linear={linear_x:.3f}, '
            f'cmd_angular={angular_z:.3f}',
            throttle_duration_sec=0.5,
        )

    def watchdog_callback(self):
        if self.last_pose_time is None:
            return

        now = self.get_clock().now()
        elapsed = (now - self.last_pose_time).nanoseconds / 1e9

        if elapsed > self.tag_timeout_sec:
            self.stop_robot()
            self.get_logger().warn(
                'Tag pose timeout. Stopping robot.',
                throttle_duration_sec=1.0,
            )


def main(args=None):
    rclpy.init(args=args)

    node = ArucoFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()