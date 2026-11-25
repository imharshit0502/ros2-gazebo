# my_robot_perception — shelf-void detection

This package finds **empty shelf spaces (voids)** in the robot's mast camera, draws
**bounding boxes** around them, places each void on the map, and **alerts the supervisor**.

It has two interchangeable detector "backends". Both return the same thing: a list of boxes.

| backend | what it is | needs | use it for |
|---|---|---|---|
| `classical` *(default)* | OpenCV: shelf-rail bands + "grey from board to board" + width test | nothing extra | the live Gazebo demo. Tuned on simulator frames |
| `shelfvoid` | your EE-511 notebook model (`shelfvoid-1.ipynb`): frozen **DINOv2 ViT-S/14** + product/void prototype head → void mask → boxes | PyTorch + `shelfvoid_head.pt` | showing the learned model; real-photo images |

---

## 1. Run it

```bash
cd ~/ros2_rws && colcon build && source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Full demo: the patrol now starts the void detector too (void_detection:=true by default)
ros2 launch my_robot_navigation patrol.launch.xml
```

In RViz:
* **Void Detection (camera + boxes)**: the mast camera with red boxes.
  `checking 2/4` = candidate (seen twice so far). `VOID #3 A3 S2-E L1` = confirmed void.
* **Detected Voids**: red cubes on the map where voids were confirmed (orange = still checking).

In the terminal, each void is reported **once**:
```
🚨 SHELF VOID #3 DETECTED  aisle 3 | shelf S2-E | level 1 | ~1.05 m wide | at (-1.34, -1.21, 0.69) [map]
   ✔ matches ground truth V02 (breakfast)
   -> please restock
```
At the end of the patrol it prints a **summary with precision / recall** against
`my_robot_bringup/config/shelf_voids.yaml`, the list of voids that were deliberately placed in the world.

Other ways to run it:
```bash
# detector alone, next to any running simulation (teleop, SLAM, ...)
ros2 launch my_robot_perception void_detection.launch.xml viewer:=true

# look at the topics
ros2 topic echo /void_detection/alerts          # one JSON message per confirmed void
ros2 run rqt_image_view rqt_image_view /void_detection/image_annotated
ros2 run rqt_image_view rqt_image_view /void_detection/debug_image     # what the detector "sees"

# no ROS at all: run on saved pictures (makes slide figures)
ros2 run my_robot_perception detect_voids_offline \
     ~/ros2_rws/src/my_robot_perception/test_images -o ~/void_results --debug
```

## 2. Topics

| Topic | Type | Content |
|---|---|---|
| `/shelf_camera/image_raw` (in) | sensor_msgs/Image | mast camera, 1024×768 @ 10 Hz |
| `/shelf_camera/camera_info` (in) | sensor_msgs/CameraInfo | focal length, used to turn pixels into rays |
| `/scan` (in) | sensor_msgs/LaserScan | distance to the shelf front |
| `/patrol/status` (in) | std_msgs/String | which shelf faces the patrol is inspecting → shelf name in alerts |
| `/void_detection/image_annotated` | sensor_msgs/Image | camera + red boxes + status banner |
| `/void_detection/debug_image` | sensor_msgs/Image | classical: grey = "empty" pixels, yellow = rails. shelfvoid: P(void) heat-map |
| `/void_detection/detections` | vision_msgs/Detection2DArray | per-frame boxes (pixels + score), standard message for other nodes |
| `/void_detection/alerts` | std_msgs/String (JSON) | one message per **confirmed** void: aisle, shelf, level, position, width, GT match |
| `/void_detection/markers` | visualization_msgs/MarkerArray | cubes + labels in RViz |

## 3. How it works

```
camera frame ─► detector backend ─► boxes (pixels)
                                        │  TF map←camera + LiDAR range → ray ∩ shelf plane
                                        ▼
                               void position (x, y, z) in the map
                                        │  same place seen again? → merge ("track")
                                        ▼
                    seen ≥ confirm_hits (4) times → ALERT once, red cube in RViz
```

**Why confirm over several frames?** The robot drives past each void, so the void is in view for
several seconds. A single bad frame (a shopper walking past, a steep view at an aisle end) does
not pile up 4 sightings at the same map position, so it never becomes an alert. It also gives the
supervisor **one** alert per void instead of 10 per second.

**Why TF + LiDAR?** The camera only tells the *direction* of the void. The LiDAR beam that points
the same way as the box measures how far away the shelf front is. Direction + distance = 3-D
position, which tells us the aisle and level (from the height) and lets us check against the ground truth.
The LiDAR also acts as a **"is there a shelf here?" check**: if the beam finds nothing at aisle distance,
the camera is looking past a shelf end at a far-away aisle. Boxes seen there are ignored.

**Safety rules that came out of testing** (each one fixed a real false or duplicate alert):
| Rule | Fixes |
|---|---|
| A box is localised only if the LiDAR sees a shelf at aisle distance in its direction | grey far-away shelves seen through the cross aisle |
| A shelf level must be ≥ 10 % of the image height | thin fake "levels" where a far shelf's rails mix with ours |
| A void must be ≥ 0.35 m wide in the world (`min_void_width_m`) | normal gaps between products |
| Sightings are matched *across the camera ray*, not in plain x/y/z | the same void alerted twice because its distance estimate was off |
| A box cut by the image border can confirm a void but never starts one | half-visible voids with a shifted centre |
| If only 3 rails are visible, a 4th level is added below the lowest one | the bottom price strip is often hidden by z-fighting (it sits exactly on the deck face) |

### The classical detector, step by step (`detectors/shelf_rail_detector.py`)
1. **Rails**: every shelf level has a yellow price strip. Keep yellow pixels that form long, thin,
   horizontal lines, then find the image rows full of them.
2. **Bands**: the space between two rails is one shelf level (the same "shelf-row band" idea as
   the DBSCAN step in notebook cell 5).
3. **Empty pixels**: grey with almost no colour (HSV saturation < 20) and not dark. That is the
   back panel or the shelf board. Products, even the black ones, keep a slight colour tint.
4. **Empty columns**: ≥ 85 % of the band is empty, i.e. grey *from board to board*. A short product
   with headroom above it does not count.
5. **Width test**: a run of empty columns wider than **1.0 × band height** (~40 cm) is a void. Normal
   gaps between products in this store are < 30 cm, and the voids are > 55 cm.
6. Only columns with a rail underneath count, so the floor and the next aisle are ignored.
7. Levels thinner than 10 % of the image height are skipped: they belong to a shelf far away.

All thresholds are ratios of the band height, so they keep working if the robot is a bit closer to
or farther from the shelf.

### Measured results (full headless patrol, 16 legs, 9.5 min sim time, 3 walking shoppers)
| Run | Found | False alarms | Duplicate alerts | Notes |
|---|---|---|---|---|
| 1 (first version) | 11 / 12 | 0 | 4 | missed V01: bottom rail hidden |
| 2 (+ bottom band, ray matching) | 12 / 12 | 9 | 0 | false alarms looking past shelf ends |
| **3 (final)** | **12 / 12** | **0** | **0** | precision 1.00, recall 1.00 |

## 4. Using the ShelfVoid (DINOv2) notebook model

`detectors/shelfvoid_dinov2.py` is a line-by-line port of notebook cells 3, 4, 7 and 9 (same layer
names, so the saved head loads directly). The only new step is **mask → bounding boxes**
(`mask_to_boxes`, one box per connected blob of void pixels).

1. Get the weights: on Kaggle, cell 14 saves `/kaggle/working/shelfvoid_results/shelfvoid_head.pt`.
   Copy it to `my_robot_perception/models/`.
2. Install PyTorch (not on this machine yet; pip itself is missing):
   ```bash
   sudo apt install python3-pip
   pip install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
   The first run needs internet: `torch.hub` downloads DINOv2 (~90 MB) into `~/.cache/torch/hub`.
3. Run:
   ```bash
   ros2 launch my_robot_navigation patrol.launch.xml void_backend:=shelfvoid \
        void_weights:=$HOME/ros2_rws/src/my_robot_perception/models/shelfvoid_head.pt
   ```
   On CPU one frame takes ~1 s, so set `process_rate_hz: 1.0` in `config/void_detector.yaml`.
   If PyTorch or the weights are missing, the node logs an error and **falls back to `classical`**,
   so the demo never breaks.

**Expect a domain gap.** The head was trained on real SKU-110K photos (mean IoU 0.11 on Kaggle).
Gazebo products are flat-coloured boxes, which look nothing like real packaging. To make the
learned model work well in simulation, re-run the notebook's own self-supervised recipe on simulator
frames: product boxes come for free from the world generator, and synthetic voids from
`cv2.inpaint` exactly as in cell 6. That makes a good "future work" slide, and a fair comparison
slide (learned vs. classical) for the presentation.

## 5. Files
```
my_robot_perception/void_detector_node.py      ROS node: detect → localise → confirm → alert
my_robot_perception/detect_voids_offline.py    same detectors on image files (no ROS)
my_robot_perception/detectors/base.py          VoidBox, mask_to_boxes, draw_boxes
my_robot_perception/detectors/shelf_rail_detector.py   classical backend
my_robot_perception/detectors/shelfvoid_dinov2.py      notebook (DINOv2 ShelfVoid) backend
config/void_detector.yaml                      all parameters, commented
launch/void_detection.launch.xml               detector (+ optional rqt_image_view)
models/                                        put shelfvoid_head.pt here
test_images/                                   real frames from the Gazebo mast camera
```
