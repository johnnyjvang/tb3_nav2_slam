#!/usr/bin/env python3

import argparse
import os

import cv2


def generate_aruco_marker(
    marker_id: int,
    marker_size_px: int,
    border_px: int,
    output_dirs: list,
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

    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"aruco_{marker_id}.png")
        cv2.imwrite(output_path, marker_with_border)

        print(f"Saved ArUco marker ID {marker_id} to:")
        print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()

    # Single ID (optional)
    parser.add_argument("--id", type=int, default=None)

    # Multi-ID
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)

    # Marker properties
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--border", type=int, default=80)

    # Output directories (multiple!)
    parser.add_argument(
        "--output_dirs",
        nargs="+",
        default=[
            os.path.expanduser(
                "~/turtlebot3_ws/src/tb3_nav2_slam/models/tb3_1_burger/materials/textures/"
            ),
            os.path.expanduser(
                "~/turtlebot3_ws/src/tb3_nav2_slam/models/tb3_2_burger/materials/textures/"
            ),
        ],
    )

    args = parser.parse_args()

    # Single ID mode
    if args.id is not None:
        generate_aruco_marker(
            marker_id=args.id,
            marker_size_px=args.size,
            border_px=args.border,
            output_dirs=args.output_dirs,
        )
        return

    # Batch mode
    for i in range(args.start_id, args.start_id + args.count):
        generate_aruco_marker(
            marker_id=i,
            marker_size_px=args.size,
            border_px=args.border,
            output_dirs=args.output_dirs,
        )


if __name__ == "__main__":
    main()