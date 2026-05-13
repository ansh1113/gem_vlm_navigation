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
| Waypoint visualiser | *(your viz node)* | Displays VLM waypoints in RViz |
| `planner_node` | `planner_node.py` | A* on costmap → spline smooth → Stanley controller → `/ackermann_cmd` |

---

## Prerequisites

```bash
# Python dependencies
pip install pymap3d opencv-python scipy openai

# ROS dependencies (should already be in your workspace)
# - septentrio_gnss_driver
# - ackermann_msgs
# - cv_bridge
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
roslaunch gazebo_ros empty_world.launch \
    world_name:=$(rospack find YOUR_PACKAGE)/worlds/high_bay_3d.world \
    paused:=false \
    use_sim_time:=true
```

> Replace `YOUR_PACKAGE` with the ROS package that contains your world file.
> Wait until Gazebo is fully loaded and the model appears before continuing.

---

### Terminal 2 — TF Publisher

Bridges the Gazebo ground-truth pose into the ROS TF tree
(`world → base_footprint`). Everything downstream depends on this.

```bash
python3 gazebo_tf_publisher.py
```

Expected output:
```
[gazebo_tf] Service ready. Publishing world -> base_footprint at 50Hz
```

---

### Terminal 3 — LiDAR BEV Node

Subscribes to the Ouster pointcloud, GPS, and INS heading.
Publishes `/lidar_bev_image` and `/vlm_costmap`.

```bash
python3 lidar_bev_node.py
```

Expected output:
```
[lidar_bev] Ready.  BEV: 267px @ 0.15m/px  Range: +/-20m  Grid: 275x157 @ 0.5m/cell
```

> The costmap must be publishing before the planner starts.
> Verify with: `rostopic hz /vlm_costmap`

---

### Terminal 4 — VLM Node

Waits for a `/vlm_command` string, then queries GPT-4o once and locks
a set of ENU waypoints on `/vlm_waypoints`.

```bash
export OPENAI_API_KEY='your-api-key-here'
python3 vlm_node.py
```

Expected output:
```
[vlm] Ready. One-shot VLM mode. New command = new fixed waypoint set.
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

Expected output:
```
[planner] Gazebo service ready.
[planner] Ready.  Waiting for /vlm_waypoints …
[planner] Status → idle
```

---

## Sending a Command

Once all six nodes are running, issue a navigation command from any terminal:

```bash
rostopic pub --once /vlm_command std_msgs/String "data: 'drive to the open space on the left'"
```

The stack will respond in sequence:

```
[vlm]     New command received: 'drive to the open space on the left'
[vlm]     LOCKED fixed waypoints in 2.3s. status=navigating plan_type=local_goal
[planner] New VLM waypoints (1). Will replan.
[planner] Status → planning
[planner] A*+smooth 0.021s → 250 pts
[planner] Status → tracking
[stanley] ef=0.012  θe=1.2°  δ=0.031  spd=2.80
```

---

## RViz Setup

Add these displays to see what the stack is doing:

| Display | Topic | Notes |
|---|---|---|
| Path | `/planner_path` | Green line — active A* trajectory |
| PoseArray | `/vlm_waypoints` | Yellow arrows — VLM target waypoints |
| Image | `/lidar_bev_image` | Top-down LiDAR view |
| Map | `/vlm_costmap` | Occupancy grid with obstacle inflation |

Set **Fixed Frame** to `world`.

---

## Key Parameters

All tunable constants live at the top of each file.

**`lidar_bev_node.py`**

| Parameter | Default | Description |
|---|---|---|
| `ORIGIN_LAT / LON` | `40.0928381 / -88.2356367` | Must match `<spherical_coordinates>` in world file |
| `X_MIN/X_MAX/Y_MIN/Y_MAX` | see file | Navigatable zone bounds |
| `Z_MIN / Z_MAX` | `0.0 / 3.0` | LiDAR height filter |
| `BEV_RANGE_M` | `20.0` | Metres shown around vehicle in BEV image |

**`vlm_node.py`**

| Parameter | Default | Description |
|---|---|---|
| `GPT_MODEL` | `gpt-4o` | Vision model used for planning |
| `MAX_WAYPOINTS` | `5` | Maximum waypoints per VLM call |
| `ARRIVAL_RADIUS_M` | `2.0` | Radius to declare a waypoint reached |

**`planner_node.py`**

| Parameter | Default | Description |
|---|---|---|
| `WHEELBASE` | `1.75` m | GEM e4 wheelbase |
| `SPEED_BASE` | `2.8` m/s | Cruise speed |
| `INFLATE_CELLS` | `1` | Obstacle inflation (1 cell = 0.5 m) |
| `REPLAN_INTERVAL_S` | `3.0` s | Mid-segment replan frequency |
| `WP_ARRIVE_M` | `2.5` m | Intermediate waypoint arrival radius |
| `FINAL_ARRIVE_M` | `1.5` m | Final goal arrival radius |

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

The LiDAR will detect any object with height between `Z_MIN=0.0` and `Z_MAX=3.0` m
and the planner will route around it automatically.

---

## Troubleshooting

**Planner says "Already at final goal" immediately**
The VLM waypoint landed within `FINAL_ARRIVE_M` of the spawn position.
Check that `ORIGIN_LAT/LON` in all three Python files matches `<spherical_coordinates>`
in the world file exactly. Run:
```bash
rostopic echo /vlm_waypoints
rostopic echo /gazebo/model_states
```
and confirm the coordinates are in the same frame.

**A* fails every cycle**
The goal cell is inside an inflated obstacle. Reduce `INFLATE_CELLS` from 1 to 0,
or check that the VLM waypoint is inside the `X_MIN/X_MAX/Y_MIN/Y_MAX` zone.

**No `/vlm_costmap` messages**
Confirm the LiDAR is publishing: `rostopic hz /ouster/points`.
The BEV node needs at least one pointcloud before it publishes the costmap.

**VLM node hangs**
Check your `OPENAI_API_KEY` is exported and valid.
Set `DRY_RUN = True` in `vlm_node.py` to test the full stack without an API call.
