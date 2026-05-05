# tb3_nav2_slam

ROS 2 package for TurtleBot3 navigation, SLAM, goal exploration, robot following, and ArUco-based robot detection experiments in Gazebo and on TurtleBot3 hardware.

This project is currently used as a hands-on mobile robotics testbed for learning and validating Nav2, SLAM, multi-robot simulation, robot following behavior, and perception-based tracking.

---

## Current Status

The package currently includes working scripts and launch files for:

- TurtleBot3 Nav2 goal navigation
- Single-goal navigation and return behavior
- Navigation through a list of goals
- Random safe goal exploration
- Timer-based patrol exploration
- LiDAR-based robot following
- ArUco-based robot following using RGB camera and pose estimation
- Two TurtleBot3 simulation setup in Gazebo
- Custom TurtleBot3 world launch support
- Gazebo simulation support for robot-to-robot detection experiments

---

## Featured Work

### Timer-Based Patrol Explorer

The `timer_based_patrol_explorer` node performs autonomous exploration by sending navigation goals at timed intervals.

<p align="center">
  <img src="img/timer_based_patrol_explorer.gif" width="700">
</p>

<p align="center">
  <img src="img/timer_based_patrol_map_20260504_212637.png" width="700">
</p>

This node:
- Dispatches goals at fixed time intervals instead of purely random sampling
- Uses Nav2 for path planning and execution
- Generates a map showing visited goals and path coverage
- Saves results with timestamps for repeatable testing

Behavior characteristics:
- Goals are selected periodically to ensure continuous movement
- Navigation success/failure is monitored
- Exploration coverage can be visually validated through saved output maps

Safety considerations:
- Relies on Nav2 costmaps to avoid obstacles
- Avoids sending invalid or unreachable goals
- Stops or retries when navigation fails
- Maintains stable motion through Nav2 constraints

#### How to Run

Terminal 1:
```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

Terminal 2:
```bash
ros2 launch turtlebot3_navigation2 navigation2.launch.py map:=$HOME/map_turtlebot3_world.yaml use_sim_time:=true
```

Terminal 3:
```bash
ros2 run tb3_nav2_slam timer_based_patrol_explorer.py
```

---

### ArUco-Based Robot Following

This project demonstrates vision-based robot-to-robot tracking using ArUco markers and an RGB camera.

<p align="center">
  <img src="img/aruco_follower.gif" width="700">
</p>

System pipeline:

```text
Camera -> ArUco Detection -> Pose Estimation -> Pose Topic -> Follower Controller -> cmd_vel
```

System behavior:
- Two TurtleBot3 robots are spawned in Gazebo
- The target robot (tb3_2) is teleoperated
- The follower robot (tb3_1) uses its camera to detect ArUco markers
- The pose of the detected marker is estimated using OpenCV
- Pose is converted into a robot-friendly coordinate system
- A follower node generates velocity commands based on:
  - Forward distance to the target
  - Lateral offset from the center of the image

Control characteristics:
- Moves forward when the target is far
- Rotates toward the target when off-center
- Stops at a defined distance threshold
- Stops automatically if the marker is lost

#### How to Run

Terminal 1:
```bash
ros2 launch tb3_nav2_slam custom_tb3_world.launch.py
```

Terminal 2:
```bash
ros2 run turtlebot3_teleop teleop_keyboard --ros-args -r cmd_vel:=/tb3_2/cmd_vel
```

Terminal 3:
```bash
ros2 run tb3_nav2_slam aruco_detector
```

Terminal 4:
```bash
ros2 run tb3_nav2_slam aruco_pose_tracker
```

Terminal 5:
```bash
ros2 run tb3_nav2_slam aruco_follower
```

---

## Multi-ArUco Robot Following

<p align="center">
  <img src="img/aruco_multi_follower.gif" width="700">
</p>

This feature extends the original ArUco pipeline to support **multiple markers and 360° tracking**.

### System Pipeline

Camera -> Multi Detector -> Detection Topic -> Multi Follower -> cmd_vel

### Marker Layout

0 = back  
1 = front  
2 = left  
3 = right  

### Multi-Detector Logic

- Subscribes to: `/tb3_1/camera/image`
- Detects all visible markers
- Estimates pose using OpenCV
- Publishes structured detection data

### Multi-Follower Logic

Instead of selecting the closest tag, the system uses priority:

1. back tag  
2. left/right tags  
3. front tag  

This allows:
- stable tracking when multiple tags are visible
- smooth transitions during robot rotation
- realistic behavior for real-world deployment

### Handling Multiple Visible Tags

When multiple tags are detected:
- system selects best candidate based on alignment and priority
- avoids rapid switching between tags
- maintains smooth motion control

---

## How to Run (Multi-ArUco)

Terminal 1:
```bash
ros2 launch tb3_nav2_slam custom_tb3_world.launch.py
```

Terminal 2:
```bash
ros2 run turtlebot3_teleop teleop_keyboard --ros-args -r cmd_vel:=/tb3_2/cmd_vel
```

Terminal 3:
```bash
ros2 run tb3_nav2_slam aruco_multi_detector
```

Terminal 4:
```bash
ros2 run tb3_nav2_slam aruco_multi_follower
```

Terminal 5 (only used to see camera view):
```bash
ros2 run tb3_nav2_slam aruco_detector 
```

---


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
│   ├── aruco_pose_tracker.py
│   ├── aruco_follower.py
│   ├── goal_from_list.py
│   ├── lidar_following_robot.py
│   ├── random_safe_goal_explorer.py
│   ├── single_goal_nav.py
│   ├── single_goal_return.py
│   └── timer_based_patrol_explorer.py
├── img/
├── package.xml
├── setup.py
└── LICENSE
```

---

## Main ROS 2 Nodes

| Node | Purpose |
|---|---|
| `single_goal_nav` | Sends one Nav2 goal to the robot. |
| `single_goal_return` | Sends a goal and returns from that goal. |
| `goal_from_list` | Runs navigation goals from a predefined list. |
| `random_safe_goal_explorer` | Selects safe random goals for exploration testing. |
| `timer_based_patrol_explorer` | Runs patrol/exploration behavior for a timed session. |
| `lidar_following_robot` | Uses LiDAR scan data for basic following behavior. |
| `aruco_detector` | Detects ArUco markers from camera input. |
| `aruco_pose_tracker` | Estimates marker pose and publishes target position. |
| `aruco_follower` | Follows target robot using pose-based control. |

---

## Build Instructions

```bash
cd ~/turtlebot3_ws
colcon build --packages-select tb3_nav2_slam --symlink-install
source install/setup.bash
```

---

## Example Commands

### Launch TurtleBot3 Gazebo World

```bash
ros2 launch tb3_nav2_slam custom_tb3_world.launch.py
```

### Launch Two TurtleBot3 Robots in Simulation

```bash
ros2 launch tb3_nav2_slam two_tb3_sim.launch.py
```

### Run Timer-Based Patrol Explorer

```bash
ros2 run tb3_nav2_slam timer_based_patrol_explorer
```

### Run ArUco Detection and Following

```bash
ros2 run tb3_nav2_slam aruco_detector
ros2 run tb3_nav2_slam aruco_pose_tracker
ros2 run tb3_nav2_slam aruco_follower
```

---

## Current Working Features

- ROS 2 Python package builds successfully with `ament_python`
- Nav2 goal scripts are installed as ROS 2 console commands
- Gazebo simulation launch files are included
- Two-robot simulation setup is included
- Timer-based patrol exploration with result visualization
- ArUco-based detection, pose estimation, and robot following
- MIT license is included

---

## Work in Progress

- Improving robustness of ArUco tracking (multi-marker support)
- Improving robot-to-robot tracking reliability
- Refining control behavior for smoother following
- Expanding testing on physical TurtleBot3 hardware
- Improving documentation with additional visuals and diagrams

---

## Notes

This project is experimental and actively being developed. It is intended for learning, testing, and documenting applied mobile robotics workflows using TurtleBot3, ROS 2, Nav2, SLAM, Gazebo, LiDAR, and camera-based perception.

---

## License

MIT License