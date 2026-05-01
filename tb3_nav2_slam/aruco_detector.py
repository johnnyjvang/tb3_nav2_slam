#!/usr/bin/env python3

# ============================================================
# aruco_detector.py Summary
# ------------------------------------------------------------
# This node subscribes to the TurtleBot3 camera feed and detects
# ArUco markers using OpenCV.
#
# Behavior:
# - Listens to /tb3_1/camera/image
# - Converts ROS Image → OpenCV format
# - Detects ArUco markers
# - Draws bounding boxes around detected markers
# - Prints detected marker IDs to the terminal
#
# This version uses the older OpenCV ArUco API:
# cv2.aruco.detectMarkers()
# ============================================================

import cv2
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


# ============================================================
# make_detector_parameters Helper
# ------------------------------------------------------------
# OpenCV versions differ:
# - Some use cv2.aruco.DetectorParameters_create()
# - Some use cv2.aruco.DetectorParameters()
#
# This helper tries both so the script works across versions.
# ============================================================
def make_detector_parameters():
    if hasattr(cv2.aruco, 'DetectorParameters_create'):
        return cv2.aruco.DetectorParameters_create()

    return cv2.aruco.DetectorParameters()


# ============================================================
# ArucoDetector Node
# ------------------------------------------------------------
# This ROS2 node:
# - Subscribes to the camera topic
# - Processes incoming images
# - Runs ArUco detection
# ============================================================
class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        # ------------------------------------------------------------
        # CvBridge Setup
        # ------------------------------------------------------------
        # Converts ROS Image messages into OpenCV images.
        # ------------------------------------------------------------
        self.bridge = CvBridge()

        # ------------------------------------------------------------
        # Image Subscriber
        # ------------------------------------------------------------
        # Subscribes to the camera feed from tb3_1.
        # ------------------------------------------------------------
        self.image_sub = self.create_subscription(
            Image,
            '/tb3_1/camera/image',
            self.image_callback,
            10,
        )

        # ------------------------------------------------------------
        # ArUco Dictionary Setup
        # ------------------------------------------------------------
        # DICT_4X4_50 means:
        # - 4x4 marker pattern
        # - 50 possible marker IDs
        #
        # This must match the dictionary used to generate the tag.
        # ------------------------------------------------------------
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        # ------------------------------------------------------------
        # ArUco Detection Parameters
        # ------------------------------------------------------------
        # Uses a helper because OpenCV versions expose this differently.
        # ------------------------------------------------------------
        self.aruco_params = make_detector_parameters()

        self.get_logger().info('ArUco detector started.')
        self.get_logger().info('Listening on /tb3_1/camera/image')
        self.get_logger().info('Using OpenCV cv2.aruco.detectMarkers() API.')

    # ============================================================
    # Image Callback
    # ------------------------------------------------------------
    # Runs every time a new camera frame is received.
    #
    # Steps:
    # 1. Convert ROS image → OpenCV image
    # 2. Convert frame to grayscale
    # 3. Detect ArUco markers
    # 4. Draw bounding boxes around detected markers
    # 5. Print detected marker IDs
    # 6. Display the annotated image
    # ============================================================
    def image_callback(self, msg: Image) -> None:
        try:
            # ------------------------------------------------------------
            # Convert ROS Image → OpenCV Image
            # ------------------------------------------------------------
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8',
            )

            # ------------------------------------------------------------
            # Convert to Grayscale
            # ------------------------------------------------------------
            # ArUco detection generally works on grayscale images.
            # ------------------------------------------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ------------------------------------------------------------
            # Detect ArUco Markers
            # ------------------------------------------------------------
            # Older OpenCV API returns:
            # - corners: list of corner points
            # - ids: marker IDs
            # - rejected: rejected candidate marker regions
            # ------------------------------------------------------------
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params,
            )

            # ------------------------------------------------------------
            # If Markers Are Detected
            # ------------------------------------------------------------
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                detected_ids = ids.flatten().tolist()

                self.get_logger().info(
                    f'Detected ArUco IDs: {detected_ids}'
                )

                for marker_id in detected_ids:
                    if marker_id == 0:
                        self.get_logger().info(
                            'Target tag ID 0 is visible.'
                        )

            # ------------------------------------------------------------
            # If No Markers Are Detected
            # ------------------------------------------------------------
            else:
                self.get_logger().info('No ArUco marker detected.')

            # ------------------------------------------------------------
            # Display Image
            # ------------------------------------------------------------
            # Shows the live camera feed with marker boxes drawn.
            # ------------------------------------------------------------
            cv2.imshow('TB3_1 ArUco Detection', frame)
            cv2.waitKey(1)

        except Exception as exc:
            self.get_logger().error(f'Image processing error: {exc}')


# ============================================================
# main() Function
# ------------------------------------------------------------
# Initializes ROS2, creates the detector node, and keeps it alive.
# ============================================================
def main(args=None):
    rclpy.init(args=args)

    node = ArucoDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


# ============================================================
# Python Entry Point
# ------------------------------------------------------------
# Ensures main() only runs when this file is executed directly.
# ============================================================
if __name__ == '__main__':
    main()