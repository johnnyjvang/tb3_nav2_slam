# tb3_nav2_slam

ROS 2 package for TurtleBot3 navigation, SLAM, goal exploration, robot following, and ArUco-based robot detection experiments in Gazebo and on TurtleBot3 hardware.

This project is currently used as a hands-on mobile robotics testbed for learning and validating Nav2, SLAM, multi-robot simulation, robot following behavior, and perception-based tracking.

## Current Status

The package currently includes working scripts and launch files for:

- TurtleBot3 Nav2 goal navigation
- Single-goal navigation and return behavior
- Navigation through a list of goals
- Random safe goal exploration
- Timer-based patrol exploration
- LiDAR-based robot following
- Two TurtleBot3 simulation setup in Gazebo
- Custom TurtleBot3 world launch support
- ArUco marker detection using an RGB camera
- Gazebo simulation support for robot-to-robot detection experiments

## What Has Been Done

### Navigation and SLAM

- Set up TurtleBot3 navigation using ROS 2 and Nav2.
- Successfully tested SLAM with TurtleBot3.
- Added scripts for sending single goals, returning from a goal, and navigating through predefined goals.
- Added random safe goal exploration for testing autonomous movement inside a mapped environment.
- Added timer-based patrol behavior for repeated exploration-style movement.

### Multi-Robot Simulation

- Added launch support for running two TurtleBot3 robots in Gazebo.
- Added custom TurtleBot3 world launch files.
- Added bridge/spawn launch files for simulation setup.
- Used simulation to test ideas before running on physical robots.

### Robot Following

- Added a LiDAR-based following node.
- Tested basic following behavior where one robot attempts to track another robot or object using scan data.
- Identified a practical limitation: TurtleBot3 robots are similar in height, making LiDAR-only tracking difficult in some cases.

### ArUco + RGB Camera Detection

- Added ArUco marker support for robot detection.
- Added Gazebo model updates for using an ArUco tag on a TurtleBot3.
- Added an RGB-camera-based detection node for identifying the marker in simulation.
- This supports future robot-to-robot tracking where one robot can visually identify another robot.

### Results and Logging

- Added result-saving behavior for random safe goal exploration.
- Updated output paths so generated results are saved into the project directory structure instead of random execution locations.
- Added timestamped summary files so multiple test runs can be saved without overwriting prior results.

## Repository Structure

```text
tb3_nav2_slam/
├── launch/
│   ├── custom_tb3_world.launch.py
│   ├── lidar_following_robot.launch.py
│   ├── lidar_following_robot_sim.launch.py
│   ├── spawn_tb3_1_with_bridge.launch.py
│   ├── tb3_1_bridge.launch.py
│   └── two_tb3_sim.launch.py
├── models/
├── scripts/
├── tb3_nav2_slam/
│   ├── aruco_detector.py
│   ├── goal_from_list.py
│   ├── lidar_following_robot.py
│   ├── random_safe_goal_explorer.py
│   ├── single_goal_nav.py
│   ├── single_goal_return.py
│   └── timer_based_patrol_explorer.py
├── package.xml
├── setup.py
└── LICENSE
```

## Main ROS 2 Nodes

| Node | Purpose |
|---|---|
| `single_goal_nav` | Sends one Nav2 goal to the robot. |
| `single_goal_return` | Sends a goal and returns from that goal. |
| `goal_from_list` | Runs navigation goals from a predefined list. |
| `random_safe_goal_explorer` | Selects safe random goals for exploration testing. |
| `timer_based_patrol_explorer` | Runs patrol/exploration behavior for a timed session. |
| `lidar_following_robot` | Uses LiDAR scan data for basic following behavior. |
| `aruco_detector` | Uses camera input and OpenCV ArUco detection for robot/tag tracking. |

## Build Instructions

From the ROS 2 workspace:

```bash
cd ~/turtlebot3_ws
colcon build --packages-select tb3_nav2_slam --symlink-install
source install/setup.bash
```

If using TurtleBot3 Burger:

```bash
export TURTLEBOT3_MODEL=burger
```

## Example Commands

### Launch TurtleBot3 Gazebo World

```bash
ros2 launch tb3_nav2_slam custom_tb3_world.launch.py
```

### Launch Two TurtleBot3 Robots in Simulation

```bash
ros2 launch tb3_nav2_slam two_tb3_sim.launch.py
```

### Run Single Goal Navigation

```bash
ros2 run tb3_nav2_slam single_goal_nav
```

### Run Goal List Navigation

```bash
ros2 run tb3_nav2_slam goal_from_list
```

### Run Random Safe Goal Explorer

```bash
ros2 run tb3_nav2_slam random_safe_goal_explorer
```

### Run Timer-Based Patrol Explorer

```bash
ros2 run tb3_nav2_slam timer_based_patrol_explorer
```

### Run LiDAR Following Robot Node

```bash
ros2 run tb3_nav2_slam lidar_following_robot
```

### Run ArUco Detector

```bash
ros2 run tb3_nav2_slam aruco_detector
```

## Current Working Features

- ROS 2 Python package builds successfully with `ament_python`.
- Nav2 goal scripts are installed as ROS 2 console commands.
- Gazebo simulation launch files are included.
- Two-robot simulation setup is included.
- LiDAR following and ArUco detection nodes are included.
- MIT license is included.

## Work in Progress

- Improving robot-to-robot tracking reliability.
- Testing ArUco detection with RGB camera placement in Gazebo.
- Refining multi-robot following behavior.
- Expanding physical TurtleBot3 tests beyond simulation.
- Improving README images, diagrams, and example output screenshots.

## Notes

This project is experimental and actively being developed. It is intended for learning, testing, and documenting applied mobile robotics workflows using TurtleBot3, ROS 2, Nav2, SLAM, Gazebo, LiDAR, and camera-based perception.

## Author

Johnny J. Vang  
Syracuse, NY  
GitHub: [johnnyjvang](https://github.com/johnnyjvang)

## License

MIT License
