# GEM e4 Vision-Language Model (VLM) Navigation

This repository replaces standard static waypoint tracking with a dynamic, language-conditioned autonomous navigation stack for the Polaris GEM e4 platform.

It utilizes a Vision-Language Model (GPT-4o) to identify physical targets from natural language commands, generates a dynamic BEV costmap from an Ouster LiDAR, calculates collision-free paths using kinematic A*, and executes the path directly via the vehicle's PACMod system.

# Demo
https://drive.google.com/file/d/1hyys7mZEofxt-i_umuBVRV_u26QL9Jn8/view?usp=sharing
https://drive.google.com/file/d/1htV_sITShUnw7c4KM_CDd5hqNqXCuZVi/view?usp=sharing

## System Architecture
* **lidar_bev_node:** Generates a top-down 2D occupancy grid and a visual rendering of the environment for the VLM.
* **vlm_node:** Queries OpenAI with front RGB and top-down LiDAR images at 0.2Hz to parse natural language commands into ENU target coordinates.
* **planner_node:** Receding-horizon A* planner running at 20Hz. Generates smooth trajectories and publishes steering/throttle commands directly to PACMod.

---

## 1. Installation & Setup

Clone the repository and compile the workspace. Note that you also need the `pacmod2_msgs` definitions.

```bash
mkdir -p ~/CS588/vlm_group
cd ~/CS588/vlm_group

# Clone the unified VLM repository
git clone https://github.com/ansh1113/gem_vlm_navigation.git
cd gem_ws/src

# Clone PACMod2 messages (required for the planner node)
git clone https://github.com/astuff/pacmod2_msgs.git
cd ..

# Compile the workspace
colcon build --symlink-install
```

## 2. Vehicle Hardware Bring-up
Before launching the autonomy stack, initialize the physical sensors and drive-by-wire hardware. Open a new terminal for each of these commands.

Terminal 1: Launch Sensors

```bash
source install/setup.bash
ros2 launch basic_launch sensor_init.launch.py
```
IMPORTANT: Because of the e4 incident, the corner cameras are not compatible with this backup PC, so corner cameras will not work. Do not run the corner camera launch file.

Terminal 2: Launch GNSS

```bash
source install/setup.bash
ros2 launch basic_launch visualization.launch.py
```
IMPORTANT: Make sure that the heading in GNSS control is correct. Relaunch GNSS or restart the machine if you have to.

Terminal 3: Launch PACMod (Drive-by-Wire)
Ensure the physical PACMod switch on the dashboard is flipped to ENABLED, and make sure the manual DBW joystick control node is NOT running.

```bash
source install/setup.bash
ros2 launch pacmod2 pacmod2.launch.xml
```

## 3. Launching the VLM Brain
Once the hardware is publishing data, launch the navigation stack. You must provide an OpenAI API key for the vision model to process the scene.

Terminal 4: VLM Core

```bash
source install/setup.bash
export OPENAI_API_KEY='your-api-key-here'

# This launches the BEV generator, the VLM node, and the A* Planner simultaneously
ros2 launch gem_vlm_nav vlm_nav.launch.py
```
(Optional) Open RViz2 on a separate monitor to view the VLM Command Center dashboard.

Red Box: Live vehicle footprint and heading.

Yellow Cylinders: VLM-generated target waypoints.

Green Line: Active A* trajectory avoiding LiDAR obstacles.

## 4. Execution & Safety
The vehicle will not move until safety conditions are met and a language command is received.

Engage the Controller: The safety operator must hold LB + RB on the Logitech joystick to pass control to the autonomous system. Releasing RB will instantly disable PACMod and stop the vehicle.

Send a Command: Open a final terminal to issue your navigation prompt:

Terminal 5: Trigger Command

```bash
source install/setup.bash
ros2 topic pub --once /vlm_command std_msgs/String "{data: 'navigate to the open space on the right'}"
```
