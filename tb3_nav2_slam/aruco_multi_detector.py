#!/usr/bin/env python3

import json
import math
from typing import Any

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


def make_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()

    return cv2.aruco.DetectorParameters()


class ArucoMultiDetector(Node):
    def __init__(self) -> None:
        super().__init__("aruco_multi_detector")

        self.bridge = CvBridge()

        self.declare_parameter("image_topic", "/tb3_1/camera/image")
        self.declare_parameter("detection_topic", "/aruco_multi/detections")
        self.declare_parameter("marker_length", 0.10)
        self.declare_parameter("valid_ids", [0, 1, 2, 3])

        self.image_topic = self.get_string_parameter(
            "image_topic",
            "/tb3_1/camera/image",
        )
        self.detection_topic = self.get_string_parameter(
            "detection_topic",
            "/aruco_multi/detections",
        )
        self.marker_length = self.get_float_parameter("marker_length", 0.10)
        self.valid_ids = self.get_int_list_parameter("valid_ids", [0, 1, 2, 3])

        self.tag_sides = {
            0: "back",
            1: "front",
            2: "left",
            3: "right",
        }

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.detection_pub = self.create_publisher(
            String,
            self.detection_topic,
            10,
        )

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        self.aruco_params = make_detector_parameters()

        self.get_logger().info("ArUco multi-detector started.")
        self.get_logger().info(f"Listening on: {self.image_topic}")
        self.get_logger().info(f"Publishing detections on: {self.detection_topic}")
        self.get_logger().info(f"Valid IDs: {self.valid_ids}")
        self.get_logger().info(f"Marker length: {self.marker_length:.3f} m")

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

    def get_int_list_parameter(
        self,
        name: str,
        default_value: list[int],
    ) -> list[int]:
        value: Any = self.get_parameter(name).value

        if isinstance(value, (list, tuple)):
            return [int(item) for item in value]

        return default_value

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

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params,
            )

            detections = []

            if ids is None:
                self.publish_detections(detections)
                self.get_logger().info(
                    "No ArUco marker detected.",
                    throttle_duration_sec=2.0,
                )
                return

            ids = ids.flatten()

            camera_matrix, dist_coeffs = self.get_camera_calibration(frame)

            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.marker_length,
                camera_matrix,
                dist_coeffs,
            )

            image_width = frame.shape[1]
            image_center_x = image_width / 2.0

            for i, marker_id_raw in enumerate(ids):
                marker_id = int(marker_id_raw)

                if marker_id not in self.valid_ids:
                    continue

                corner = corners[i][0]
                center_x = float(np.mean(corner[:, 0]))
                center_y = float(np.mean(corner[:, 1]))

                error_x = center_x - image_center_x
                normalized_error_x = error_x / image_center_x

                tvec = tvecs[i][0]
                camera_x = float(tvec[0])
                camera_y = float(tvec[1])
                camera_z = float(tvec[2])

                distance = math.sqrt(
                    (camera_x * camera_x)
                    + (camera_y * camera_y)
                    + (camera_z * camera_z)
                )

                detections.append(
                    {
                        "id": marker_id,
                        "side": self.tag_sides.get(marker_id, "unknown"),
                        "center_x": center_x,
                        "center_y": center_y,
                        "error_x": float(error_x),
                        "normalized_error_x": float(normalized_error_x),
                        "distance": float(distance),
                        "x": camera_x,
                        "y": camera_y,
                        "z": camera_z,
                    }
                )

            self.publish_detections(detections)

            if detections:
                closest = min(detections, key=lambda d: d["distance"])
                self.get_logger().info(
                    f"Detected ID {closest['id']} | "
                    f"side={closest['side']} | "
                    f"distance={closest['distance']:.2f} m | "
                    f"error_x={closest['normalized_error_x']:.2f}",
                    throttle_duration_sec=0.5,
                )

        except Exception as exc:
            self.get_logger().error(f"Image processing error: {exc}")

    def publish_detections(self, detections: list[dict]) -> None:
        output = {
            "stamp": self.get_clock().now().nanoseconds,
            "detections": detections,
        }

        msg_out = String()
        msg_out.data = json.dumps(output)
        self.detection_pub.publish(msg_out)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = ArucoMultiDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()