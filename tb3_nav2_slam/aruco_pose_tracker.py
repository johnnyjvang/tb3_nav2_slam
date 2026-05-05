#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image


def make_detector_parameters():
    if hasattr(cv2.aruco, 'DetectorParameters_create'):
        return cv2.aruco.DetectorParameters_create()

    return cv2.aruco.DetectorParameters()


class ArucoPoseTracker(Node):
    def __init__(self):
        super().__init__('aruco_pose_tracker')

        self.bridge = CvBridge()

        self.image_topic = '/tb3_1/camera/image'
        self.pose_topic = '/tb3_1/aruco/target_pose'
        self.target_marker_id = 0

        # Marker size in meters.
        # 0.0762 m = 3 inches.
        self.marker_size_m = 0.0762

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.pose_topic,
            10,
        )

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        self.aruco_params = make_detector_parameters()

        self.get_logger().info('ArUco pose tracker started.')
        self.get_logger().info(f'Listening on {self.image_topic}')
        self.get_logger().info(f'Publishing pose on {self.pose_topic}')
        self.get_logger().info(f'Target marker ID: {self.target_marker_id}')
        self.get_logger().info(f'Marker size: {self.marker_size_m:.4f} m')

    def get_camera_calibration(self, frame):
        height, width = frame.shape[:2]

        # Simple approximate camera intrinsics for Gazebo.
        #
        # This is NOT a real calibrated camera model.
        # It assumes:
        # - focal length is approximately equal to image width in pixels
        # - optical center is at the center of the image
        # - no lens distortion
        #
        # This is good enough for early Gazebo testing, but later a real
        # camera_info topic or calibration file should be used.
        focal_length_px = width

        camera_matrix = np.array(
            [
                [focal_length_px, 0.0, width / 2.0],
                [0.0, focal_length_px, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        return camera_matrix, dist_coeffs

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8',
            )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params,
            )

            if ids is None:
                self.get_logger().info(
                    'No ArUco marker detected.',
                    throttle_duration_sec=2.0,
                )
                return

            ids = ids.flatten()

            camera_matrix, dist_coeffs = self.get_camera_calibration(frame)

            # Estimate marker pose relative to the CAMERA frame.
            #
            # OpenCV camera frame:
            #   x = left/right in the image
            #   y = up/down in the image
            #   z = forward distance away from the camera
            #
            # ROS base_link frame usually means:
            #   x = forward
            #   y = left/right
            #   z = up
            #
            # For easier robot control, this node publishes a PoseStamped
            # using a robot-friendly layout:
            #   pose.position.x = OpenCV z  -> forward distance
            #   pose.position.y = OpenCV x  -> left/right offset
            #   pose.position.z = OpenCV y  -> up/down offset
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.marker_size_m,
                camera_matrix,
                dist_coeffs,
            )

            for i, marker_id in enumerate(ids):
                if marker_id != self.target_marker_id:
                    continue

                tvec = tvecs[i][0]

                camera_x = float(tvec[0])
                camera_y = float(tvec[1])
                camera_z = float(tvec[2])

                forward_distance = camera_z
                lateral_offset = camera_x
                vertical_offset = camera_y

                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = 'tb3_1_camera_frame'

                pose_msg.pose.position.x = forward_distance
                pose_msg.pose.position.y = lateral_offset
                pose_msg.pose.position.z = vertical_offset
                pose_msg.pose.orientation.w = 1.0

                self.pose_pub.publish(pose_msg)

                self.get_logger().info(
                    f'Target ID {marker_id}: '
                    f'forward={forward_distance:.3f} m, '
                    f'lateral={lateral_offset:.3f} m, '
                    f'vertical={vertical_offset:.3f} m',
                    throttle_duration_sec=1.0,
                )
                return

        except Exception as exc:
            self.get_logger().error(f'Image processing error: {exc}')


def main(args=None):
    rclpy.init(args=args)

    node = ArucoPoseTracker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()