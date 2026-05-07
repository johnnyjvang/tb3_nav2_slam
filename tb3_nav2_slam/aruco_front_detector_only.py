#!/usr/bin/env python3

# ============================================================
# aruco_front_detector_only.py Summary
# ------------------------------------------------------------
# This node subscribes to the TurtleBot3 camera feed and detects
# ArUco markers using OpenCV
#
# Behavior:
# - Listens to /tb3_1/camera/image
# - Converts ROS Image → OpenCV format
# - Detects ArUco markers
# - Estimates marker pose using rvec and tvec
# - Filters out markers that appear to be viewed from the back side
# - Draws bounding boxes around front-facing detected markers
# - Prints detected marker IDs to the terminal
#
# This version uses the older OpenCV ArUco API:
# cv2.aruco.detectMarkers()
# ============================================================

import cv2
import numpy as np
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
# This helper tries both so the script works across versions
# ============================================================
def make_detector_parameters():
    if hasattr(cv2.aruco, 'DetectorParameters_create'):
        return cv2.aruco.DetectorParameters_create()

    return cv2.aruco.DetectorParameters()


# ============================================================
# ArucoFrontDetectorOnly Node
# ------------------------------------------------------------
# This ROS2 node:
# - Subscribes to the camera topic
# - Processes incoming images
# - Runs ArUco detection
# - Rejects detections that appear to come from the marker back side
# ============================================================
class ArucoFrontDetectorOnly(Node):
    def __init__(self):
        super().__init__('aruco_front_detector_only')

        # ------------------------------------------------------------
        # CvBridge Setup
        # ------------------------------------------------------------
        # Converts ROS Image messages into OpenCV images
        # ------------------------------------------------------------
        self.bridge = CvBridge()

        # ------------------------------------------------------------
        # Camera and Marker Settings
        # ------------------------------------------------------------
        # marker_length should match the physical/simulated tag size in meters
        # facing_threshold controls how strict the front-face filter is
        # ------------------------------------------------------------
        self.marker_length = 0.10
        self.facing_threshold = -0.3

        # ------------------------------------------------------------
        # Image Subscriber
        # ------------------------------------------------------------
        # Subscribes to the camera feed from tb3_1
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
        # This must match the dictionary used to generate the tag
        # ------------------------------------------------------------
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        # ------------------------------------------------------------
        # ArUco Detection Parameters
        # ------------------------------------------------------------
        # Uses a helper because OpenCV versions expose this differently
        # ------------------------------------------------------------
        self.aruco_params = make_detector_parameters()

        self.get_logger().info('ArUco front-only detector started')
        self.get_logger().info('Listening on /tb3_1/camera/image')
        self.get_logger().info('Using OpenCV cv2.aruco.detectMarkers() API')
        self.get_logger().info(f'Marker length: {self.marker_length:.3f} m')
        self.get_logger().info(f'Facing threshold: {self.facing_threshold:.3f}')

    # ============================================================
    # get_camera_calibration Helper
    # ------------------------------------------------------------
    # Creates a simple approximate camera matrix from the image size
    #
    # This works for simulation testing, but real hardware should use
    # actual camera calibration values
    # ============================================================
    def get_camera_calibration(self, frame):
        height, width = frame.shape[:2]

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

    # ============================================================
    # get_marker_facing_score Helper
    # ------------------------------------------------------------
    # Uses the marker rotation vector to estimate which way the
    # marker face is pointing
    #
    # rvec tells marker orientation
    # tvec tells marker position
    #
    # The facing score is used to reject markers detected from the
    # wrong side
    # ============================================================
    def get_marker_facing_score(self, rvec) -> float:
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        marker_normal = rotation_matrix[:, 2]

        facing_score = float(marker_normal[2])

        return facing_score

    # ============================================================
    # is_marker_facing_camera Helper
    # ------------------------------------------------------------
    # Returns True only when the marker appears to face the camera
    #
    # If valid tags are rejected, tune facing_threshold or flip the
    # comparison sign below
    # ============================================================
    def is_marker_facing_camera(self, rvec) -> tuple[bool, float]:
        facing_score = self.get_marker_facing_score(rvec)

        is_facing = facing_score < self.facing_threshold

        return is_facing, facing_score

    # ============================================================
    # Image Callback
    # ------------------------------------------------------------
    # Runs every time a new camera frame is received
    #
    # Steps:
    # 1. Convert ROS image → OpenCV image
    # 2. Convert frame to grayscale
    # 3. Detect ArUco markers
    # 4. Estimate pose using rvec and tvec
    # 5. Reject markers viewed from the back side
    # 6. Draw bounding boxes around front-facing detected markers
    # 7. Print detected marker IDs
    # 8. Display the annotated image
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
            # ArUco detection generally works on grayscale images
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
                camera_matrix, dist_coeffs = self.get_camera_calibration(frame)

                # ------------------------------------------------------------
                # Estimate Marker Pose
                # ------------------------------------------------------------
                # rvecs tell marker rotation/orientation
                # tvecs tell marker translation/position
                #
                # The rotation vector is used to filter backside detections
                # ------------------------------------------------------------
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners,
                    self.marker_length,
                    camera_matrix,
                    dist_coeffs,
                )

                detected_ids = ids.flatten().tolist()
                front_facing_corners = []
                front_facing_ids = []
                rejected_ids = []

                for i, marker_id in enumerate(detected_ids):
                    # ------------------------------------------------------------
                    # Check Marker Facing Direction
                    # ------------------------------------------------------------
                    # Keeps detections only if the marker appears to face camera
                    # ------------------------------------------------------------
                    rvec = rvecs[i][0]
                    is_facing, facing_score = self.is_marker_facing_camera(rvec)

                    if not is_facing:
                        rejected_ids.append(marker_id)

                        self.get_logger().info(
                            f'Rejected ArUco ID {marker_id} from backside | '
                            f'facing_score={facing_score:.3f}',
                            throttle_duration_sec=1.0,
                        )

                        continue

                    front_facing_corners.append(corners[i])
                    front_facing_ids.append([marker_id])

                    self.get_logger().info(
                        f'Detected front-facing ArUco ID {marker_id} | '
                        f'facing_score={facing_score:.3f}'
                    )

                    if marker_id == 0:
                        self.get_logger().info(
                            'Target tag ID 0 is visible from the front side'
                        )

                # ------------------------------------------------------------
                # Draw Only Front-Facing Markers
                # ------------------------------------------------------------
                # This makes the window show only accepted detections
                # ------------------------------------------------------------
                if front_facing_ids:
                    cv2.aruco.drawDetectedMarkers(
                        frame,
                        front_facing_corners,
                        np.array(front_facing_ids, dtype=np.int32),
                    )
                else:
                    self.get_logger().info(
                        f'No front-facing markers accepted | rejected={rejected_ids}',
                        throttle_duration_sec=1.0,
                    )

            # ------------------------------------------------------------
            # If No Markers Are Detected
            # ------------------------------------------------------------
            else:
                self.get_logger().info('No ArUco marker detected')

            # ------------------------------------------------------------
            # Display Image
            # ------------------------------------------------------------
            # Shows the live camera feed with accepted marker boxes drawn
            # ------------------------------------------------------------
            cv2.imshow('TB3_1 ArUco Front-Only Detection', frame)
            cv2.waitKey(1)

        except Exception as exc:
            self.get_logger().error(f'Image processing error: {exc}')


# ============================================================
# main() Function
# ------------------------------------------------------------
# Initializes ROS2, creates the detector node, and keeps it alive
# ============================================================
def main(args=None):
    rclpy.init(args=args)

    node = ArucoFrontDetectorOnly()

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
# Ensures main() only runs when this file is executed directly
# ============================================================
if __name__ == '__main__':
    main()