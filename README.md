# GEM e4 VLM Navigation — Simulator Stack

Natural language navigation for the Polaris GEM e4 in Gazebo.
Issue a plain-English command; the vehicle plans and drives there autonomously.

```
User command → VLM (GPT-4o) → ENU waypoints → A* on LiDAR costmap → Stanley controller → /ackermann_cmd
```

---

## System Architecture

| Node | File | Role |
|---|---|---|
| Gazebo world | `high_bay_3d.world` | Simulated indoor environment |
| `gazebo_tf_publisher` | `gazebo_tf_publisher.py` | Broadcasts `world → base_footprint` TF at 50 Hz |
| `lidar_bev_node` | `lidar_bev_node.py` | Builds LiDAR BEV image + `/vlm_costmap` OccupancyGrid at 5 Hz |
| `vlm_node` | `vlm_node.py` | Queries GPT-4o with RGB + BEV images; outputs locked ENU waypoints |
| Waypoint visualiser | `waypoint_viz.py` | Displays VLM waypoints in RViz |
| `planner_node` | `planner_node.py` | A* on costmap → spline smooth → Stanley controller → `/ackermann_cmd` |

---

## Prerequisites

```bash
# Python dependencies
pip install pymap3d opencv-python scipy openai
```

Set your OpenAI API key once per session:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

---

## Launch Order

Each command goes in its own terminal. Source your workspace in every terminal before running.

```bash
source devel/setup.bash
```

---

### Terminal 1 — Gazebo World

```bash
roslaunch gem_launch gem_init.launch world_name:="high_bay_3d.world" x:=[START_X] y:=[START Y] yaw:=[START YAW]
```
> Replace `START_X, START_Y,` and `START_YAW` with the desired coordinates you want your vehicle initialized to
> `START_X` should be in [-40, 50], and `START_Y` should be in [-5, 12] to ensure the vehicle spawns in the Highbay Lot
> Wait until Gazebo is fully loaded and the model appears before continuing.

---

### Terminal 2 — TF Publisher

Consistently publishes a TF from global to local frame (`world` frame to `base_footprint`)
```bash
python3 gazebo_tf_publisher.py
```
---

### Terminal 3 — LiDAR BEV Node

Subscribes to the Ouster pointcloud, GPS, and INS heading.
Publishes `/lidar_bev_image` and `/vlm_costmap`.

```bash
python3 lidar_bev_node.py
```
> The costmap must be publishing before the planner starts.
> Verify with: `rostopic hz /vlm_costmap`
> You can visualize the costmap in your RViz window as well

---

### Terminal 4 — VLM Node

Waits for a `/vlm_command` string, then queries GPT-4o once and locks
a set of ENU waypoints on `/vlm_waypoints`.

```bash
export OPENAI_API_KEY='your-api-key-here'
python3 vlm_node.py
```

---

### Terminal 5 — Waypoint Visualiser

```bash
python3 YOUR_VIZ_NODE.py
```

---

### Terminal 6 — Planner Node

Runs A* on the live costmap, smooths the path with a B-spline, and
drives the vehicle via Stanley control. Replans every 3 seconds
mid-segment to react to new obstacles.

```bash
python3 planner_node.py
```
---

## Sending a Command

Once all six nodes are running, issue a navigation command from any terminal:

```bash
rostopic pub --once /vlm_command std_msgs/String "data: 'drive forward 10 meters'"
```

If waypoints are being visualized in RViz, a yellow sphere should appear along a goal trajectory, and the vehicle should begin moving towards these waypoints.

---

## RViz Setup

Add these displays to see what the stack is doing:

| Display | Topic | Notes |
|---|---|---|
| Path | `/planner_path` | Green line — active A* trajectory |
| PoseArray | `/vlm_waypoints` | Yellow arrows — VLM target waypoints |
| Image | `/lidar_bev_image` | Top-down LiDAR view |
| Map | `/vlm_costmap` | Occupancy grid with obstacle inflation |

Optional: Set **Fixed Frame** to `world` if global visualization is desired.

---

## Adding Obstacles to the World

Add static obstacles directly in `high_bay_3d.world` inside the `<world>` tag.
Gazebo world-frame coordinates align exactly with ENU (`+X = East, +Y = North`).

```xml
<!-- Box obstacle — z = half the height so it sits on the floor -->
<model name='obstacle_1'>
  <static>1</static>
  <link name='link'>
    <collision name='collision'>
      <geometry><box><size>1.0 1.0 1.5</size></box></geometry>
    </collision>
    <visual name='visual'>
      <geometry><box><size>1.0 1.0 1.5</size></box></geometry>
      <material><ambient>0.8 0.2 0.2 1</ambient></material>
    </visual>
  </link>
  <pose>15 5 0.75 0 0 0</pose>
</model>
```

The planner will route around any obstacles that it may hit (i.e. are at the relevant z-coordinates)
