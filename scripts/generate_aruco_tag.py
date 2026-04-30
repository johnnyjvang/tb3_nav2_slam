#!/usr/bin/env python3

import argparse
import os

import cv2
import numpy as np


def generate_aruco_marker(
    marker_id: int,
    marker_size_px: int,
    border_px: int,
    output_path: str,
) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    marker = cv2.aruco.generateImageMarker(
        dictionary,
        marker_id,
        marker_size_px,
    )

    marker_with_border = cv2.copyMakeBorder(
        marker,
        border_px,
        border_px,
        border_px,
        border_px,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, marker_with_border)

    print(f"Saved ArUco marker ID {marker_id} to:")
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--border", type=int, default=80)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.expanduser(
            "~/turtlebot3_ws/src/tb3_nav2_slam/models/tb3_1_burger/materials/textures/aruco_0.png"
        ),
    )

    args = parser.parse_args()

    generate_aruco_marker(
        marker_id=args.id,
        marker_size_px=args.size,
        border_px=args.border,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()