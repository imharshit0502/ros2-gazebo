# my_robot_navigation — mapping, localization and autonomous shelf patrol

ROS 2 Jazzy + Gazebo Harmonic + Nav2. Three stages, each with one launch file:

| Stage | Launch file | What happens |
|---|---|---|
| 1. Build the map (done once) | `slam_mapping.launch.xml` | SLAM Toolbox maps the empty store while `mapping_tour.py` drives every aisle |
| 2. Drive by keyboard | `localization.launch.xml` | AMCL finds the robot on the saved map; you drive with teleop |
| 3. Autonomous patrol | `patrol.launch.xml` | Nav2 drives HOME → every aisle (up + down) → HOME, avoiding shoppers |

The saved map is already in `maps/retail_store.{pgm,yaml}`. You only need stage 1 again if
the store layout changes.

---

## 0. One-time setup (every new terminal)

```bash
cd ~/ros2_rws
colcon build                       # only after changing files
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

> **Why the export?** Your shell profile sets `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, but that
> library is not installed, so every `ros2` command would fail. Alternative permanent fix:
> `sudo apt install ros-jazzy-rmw-cyclonedds-cpp`.

> **Use a normal Ubuntu terminal (Ctrl+Alt+T), not the VS Code terminal.** VS Code is a snap
> package and leaks snap library paths into its terminal; the Gazebo window then crashes with
> `symbol lookup error ... /snap/core20/...libpthread.so.0`, and Gazebo's launcher kills the
> simulation with it.

---

## 3. Autonomous shelf patrol (the main demo)

**Terminal 1**
```bash
ros2 launch my_robot_navigation patrol.launch.xml
```

What starts: Gazebo (store with 3 walking shoppers) → robot → map server + AMCL → Nav2 →
RViz → `shelf_patrol.py`. After ~20 s the patrol prints:

```
========== SHELF PATROL - round 1/1 : 16 legs ===========
[leg 1/16] transit           -> driving to Aisle 1 - south end
   ... 7.88 m to go
   reached Aisle 1 - south end
[leg 2/16] INSPECTING WALL_W       -> driving to Aisle 1 - north end
...
========== PATROL ROUND 1 COMPLETE: 16/16 legs OK, 9.3 min (sim time) - robot is back HOME ==========
```

**Terminal 2 (optional) — what the CV node will receive**
```bash
ros2 topic echo /patrol/status
# data: state=INSPECTING leg=5/16 target=Aisle_2_-_north_end shelves=S1-E,N1-E
```

### What to point at in RViz

| Display | What it shows | Why it matters |
|---|---|---|
| **Saved Map** (grey/black) | the map from stage 1 | |
| **Patrol Route** (orange line, numbered balls) | blue = HOME, green = end of an inspection leg, yellow = transit, **red = current goal** | the mission plan |
| **Global Plan** (green line) | path from the planner (NavFn), re-planned every second | bends around shoppers |
| **Global / Local Costmap** (colours) | how "dangerous" each cell is; shoppers appear as moving blobs in the local costmap | how the robot "sees" obstacles |
| **Robot Footprint** (yellow box) | the 0.6 × 0.5 m rectangle Nav2 uses for collision checks | |
| **Slow-down Zone** (orange box ahead of robot) | if anything is inside, speed drops to 40 % | safety around people |
| **LiDAR Scan** (red dots) | live 270° laser | should sit exactly on the map walls → localization is good |
| **AMCL Particles** (blue arrows) | localization guesses, tight cluster = confident | |
| **Shelf Camera** (image panel) | the mast camera looking at the shelf on the robot's left | input for void detection |
| **MPPI Trajectories** (off by default, tick it) | the hundreds of futures MPPI tries each cycle | nice slide for "how the controller thinks" |

### How it avoids shelves and shoppers (talking points)
1. **Shelves / walls / checkouts** are in the saved map → the global planner never routes through
   them, and the wide inflation keeps paths in the middle of each aisle.
2. **Shoppers** are not on the map. The LiDAR sees them live → they appear in the local and
   global costmaps → the planner re-plans around them every second and MPPI steers around them.
3. **Collision monitor** (last safety layer, independent of the planner): anything inside the
   orange slow-down box ahead → 40 % speed; predicted contact within 2 s → stop.
4. Measured over a full headless round (16/16 legs, 9.5 min): whenever a shopper was closer than
   0.8 m, the robot moved toward them at most 0.15 m/s — it stops, yields or steers away.
   Gazebo shoppers are scripted and blind (they never step aside), so they can still walk up to
   a robot that has stopped; real people would not.

### Useful variations
```bash
ros2 launch my_robot_navigation patrol.launch.xml loops:=0             # patrol forever
ros2 launch my_robot_navigation patrol.launch.xml headless:=true       # no Gazebo window (much less CPU)
ros2 launch my_robot_navigation patrol.launch.xml auto_patrol:=false   # Nav2 only: click "2D Goal Pose" in RViz
```

### Changing the route
Edit `config/patrol_route.yaml` (comments explain every field), then `colcon build`.
The shelf camera looks **left**, so driving north shows the west-side shelf and driving
south shows the east-side shelf — that is why every aisle is driven up *and* down.

---

## 2. Drive by keyboard on the saved map

```bash
# Terminal 1
ros2 launch my_robot_navigation localization.launch.xml
# Terminal 2 (keep focused while driving)
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
Keys: `i` forward, `,` back, `j`/`l` turn, `k` stop, `z` slower (press ~4× first — 0.5 m/s is fast
for a 2.68 m aisle). If the red scan dots drift off the map walls, click **2D Pose Estimate** in
RViz and drag from the robot's true position in its facing direction.

---

## 1. Re-build the map (only if the store changes)

```bash
ros2 launch my_robot_navigation slam_mapping.launch.xml      # automatic tour, ~10 min
# when the terminal prints MAPPING TOUR COMPLETE:
ros2 run nav2_map_server map_saver_cli -f ~/ros2_rws/src/my_robot_navigation/maps/retail_store
colcon build --packages-select my_robot_navigation
```
Mapping uses `retail_store_mapping.world` (same store, no shoppers) so people are not drawn
into the map.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `failed to load shared library 'librmw_cyclonedds_cpp.so'` | run the `export RMW_IMPLEMENTATION=...` line above |
| Gazebo window crashes with `/snap/core20/...` | launched from VS Code terminal — use a normal terminal |
| Robot does not move, patrol says "Waiting for Nav2 + AMCL" | give it ~30 s; Nav2 starts 11 servers. Check terminal 1 for red errors |
| Everything is slow / jerky | Gazebo renders 3 sensors + 2400 product visuals. Use `headless:=true` and watch RViz only |
| A leg prints `could NOT reach ... skipping` | something blocked the goal for too long; the patrol retries once, then continues |

## Files

```
config/slam_toolbox_mapping.yaml   SLAM settings (strict loop closure - aisles all look alike)
config/amcl_localization.yaml      map server + AMCL
config/nav2_params.yaml            planner, MPPI controller, costmaps, collision monitor ...
config/patrol_route.yaml           the patrol waypoints  <- edit this to change the route
launch/slam_mapping.launch.xml     stage 1
launch/localization.launch.xml     stage 2
launch/patrol.launch.xml           stage 3
scripts/mapping_tour.py            simple go-to-point driver used while mapping
scripts/shelf_patrol.py            patrol mission on top of Nav2 (nav2_simple_commander)
maps/retail_store.*                the saved map (+ SLAM pose graph)
rviz/*.rviz                        one RViz layout per stage
```
