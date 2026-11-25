#!/usr/bin/env python3
"""
void_detector_node - finds EMPTY shelf spaces in the mast camera and alerts the supervisor.

PIPELINE (one camera frame)
  /shelf_camera/image_raw ──► detector backend ──► 2-D boxes (pixels)
                                (classical | shelfvoid)        │
                                                               ▼
     TF (map ← camera) + /scan (distance to the shelf) ──► 3-D position of each box
                                                               │
                                                               ▼
     "tracks" in the map: the same void seen in many frames (the robot drives past it)
     is merged into ONE track. When a track was seen `confirm_hits` times ──► ALERT
                                                               │
     ┌─────────────────────────────┬───────────────────────────┼──────────────────────────┐
     ▼                             ▼                           ▼                          ▼
 /void_detection/image_annotated  /void_detection/detections  /void_detection/alerts  /void_detection/markers
 camera image + red boxes          vision_msgs Detection2DArray  std_msgs/String (JSON)   red cubes in RViz map

WHY CONFIRM OVER SEVERAL FRAMES?
  A single frame can be wrong (motion blur, a shopper walking past, a shelf seen
  at a steep angle at the aisle end). Requiring several sightings at the same map
  position removes most one-off false alarms, and one alert per void (not 10 per
  second) is what a store supervisor actually wants.

WHY TF + LIDAR?
  The camera only gives a direction (a ray). The LiDAR beam pointing the same way
  as the camera tells how far away the shelf front is. Ray ∩ shelf plane = the
  void's (x, y, z) in the map, which is used to (a) merge sightings, (b) tell the
  supervisor WHERE (aisle / shelf / level), (c) compare with the ground truth file.
"""

import json
import math
import os
import time

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from my_robot_perception.detectors import draw_boxes, make_detector

RED, GREEN, YELLOW, CYAN, BOLD, RESET = "\033[91m", "\033[92m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"


# ───────────────────────────── small geometry helpers ─────────────────────────────
def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tf_to_rt(tf):
    t = tf.transform.translation
    return quat_to_rot(tf.transform.rotation), np.array([t.x, t.y, t.z])


class Track:
    """One void in the world, built from many sightings."""

    def __init__(self, tid, pos, size, now):
        self.id = tid
        self.pos = np.array(pos, float)
        self.size = np.array(size, float)       # (width, height) in metres
        self.hits = 1
        self.first_seen = now
        self.last_seen = now
        self.confirmed = False
        self.label = {}
        self.cam = self.pos.copy()      # camera position of the latest sighting (gives the aisle)

    def update(self, pos, size, now):
        a = 1.0 / (self.hits + 1)                # running average
        self.pos = (1 - a) * self.pos + a * np.asarray(pos)
        self.size = (1 - a) * self.size + a * np.asarray(size)
        self.hits += 1
        self.last_seen = now


class VoidDetectorNode(Node):
    def __init__(self):
        super().__init__("void_detector")
        p = self.declare_parameter
        # --- what to run
        p("backend", "classical")                      # classical | shelfvoid
        p("weights_path", "")                          # shelfvoid only: .../shelfvoid_head.pt
        p("shelfvoid_threshold", 0.5)
        p("dino_hub_repo", "facebookresearch/dinov2")  # or a local checkout path + dino_hub_source=local
        p("dino_hub_source", "github")
        p("classical_min_width_ratio", 1.0)
        # --- topics / frames
        p("image_topic", "/shelf_camera/image_raw")
        p("camera_info_topic", "/shelf_camera/camera_info")
        p("scan_topic", "/scan")
        p("patrol_status_topic", "/patrol/status")
        p("world_frame", "map")                        # falls back to 'odom' if no map
        p("process_rate_hz", 4.0)                      # frames analysed per second
        # --- confirmation / localisation
        p("confirm_hits", 4)                           # sightings before an ALERT
        p("assoc_radius_m", 0.6)                       # sightings closer than this = same void
        p("candidate_timeout_s", 6.0)
        p("min_void_width_m", 0.35)                    # narrower = gap between products, not a void                  # forget unconfirmed tracks after this
        p("default_shelf_distance_m", 1.34)            # used when no LiDAR reading
        p("shelf_levels_m", [0.12, 0.52, 0.92, 1.32])  # top of each shelf board (level 0..3)
        p("aisle_x_m", [-7.16, -3.58, 0.0, 3.58, 7.16])
        p("row_split_y_m", 1.8)                        # y < split -> south row (S*), else north (N*)
        # --- evaluation (optional)
        p("ground_truth_file", "")                     # my_robot_bringup/config/shelf_voids.yaml
        p("gt_match_radius_m", 0.8)
        p("publish_debug_image", True)

        g = lambda n: self.get_parameter(n).value   # noqa: E731
        backend = g("backend")
        try:
            if backend == "shelfvoid":
                self.detector = make_detector(
                    "shelfvoid", weights_path=os.path.expanduser(g("weights_path")),
                    threshold=g("shelfvoid_threshold"),
                    hub_repo=g("dino_hub_repo"), hub_source=g("dino_hub_source"))
            else:
                self.detector = make_detector("classical", min_width_ratio=g("classical_min_width_ratio"))
        except Exception as e:  # missing torch / weights -> keep the demo running
            self.get_logger().error(f"Could not start backend '{backend}': {e}")
            self.get_logger().warn("Falling back to backend 'classical'.")
            self.detector = make_detector("classical")
        self.get_logger().info(f"Void detector backend: {BOLD}{self.detector.name}{RESET}")

        self.world_frame = g("world_frame")
        self.confirm_hits = int(g("confirm_hits"))
        self.assoc_r = float(g("assoc_radius_m"))
        self.cand_timeout = float(g("candidate_timeout_s"))
        self.min_width = float(g("min_void_width_m"))
        self.default_dist = float(g("default_shelf_distance_m"))
        self.levels = list(g("shelf_levels_m"))
        self.aisle_x = list(g("aisle_x_m"))
        self.row_split = float(g("row_split_y_m"))
        self.debug_on = bool(g("publish_debug_image"))

        self.gt = []
        gt_file = os.path.expanduser(g("ground_truth_file"))
        if gt_file and os.path.isfile(gt_file):
            with open(gt_file) as fh:
                self.gt = yaml.safe_load(fh).get("voids", [])
            self.get_logger().info(f"Loaded {len(self.gt)} ground-truth voids from {gt_file} (for scoring)")
        self.gt_r = float(g("gt_match_radius_m"))

        self.bridge = CvBridge()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.K = None
        self.scan = None
        self.patrol_shelves = []
        self.patrol_state = ""
        self.latest_img = None
        self.tracks = []
        self.next_id = 1
        self.frames = 0
        self.warned_tf = False
        self.summary_printed = False
        self.t_start = time.monotonic()

        self.create_subscription(Image, g("image_topic"), self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, g("camera_info_topic"), self.on_info, qos_profile_sensor_data)
        self.create_subscription(LaserScan, g("scan_topic"), self.on_scan, qos_profile_sensor_data)
        self.create_subscription(String, g("patrol_status_topic"), self.on_patrol, 10)

        self.pub_img = self.create_publisher(Image, "/void_detection/image_annotated", 2)
        self.pub_dbg = self.create_publisher(Image, "/void_detection/debug_image", 2)
        self.pub_det = self.create_publisher(Detection2DArray, "/void_detection/detections", 10)
        self.pub_alert = self.create_publisher(String, "/void_detection/alerts", 10)
        self.pub_mark = self.create_publisher(MarkerArray, "/void_detection/markers", 10)

        # Process the NEWEST frame at a fixed rate. If the detector is slow (DINOv2 on CPU),
        # old frames are simply skipped instead of piling up in a queue.
        self.create_timer(1.0 / float(g("process_rate_hz")), self.process)
        self.get_logger().info(f"Listening on {g('image_topic')} ... (alerts on /void_detection/alerts)")

    # ───────────────────────────── subscriptions ─────────────────────────────
    def on_image(self, msg):
        self.latest_img = msg

    def on_info(self, msg):
        self.K = np.array(msg.k, float).reshape(3, 3)

    def on_scan(self, msg):
        self.scan = msg

    def on_patrol(self, msg):
        # "state=INSPECTING leg=5/16 target=... shelves=S1-E,N1-E"
        fields = dict(kv.split("=", 1) for kv in msg.data.split() if "=" in kv)
        self.patrol_state = fields.get("state", "")
        sh = fields.get("shelves", "none")
        self.patrol_shelves = [] if sh in ("none", "") else [s.strip() for s in sh.split(",") if s.strip()]
        if self.patrol_state == "DONE" and not self.summary_printed:
            self.print_summary()

    # ───────────────────────────── localisation ─────────────────────────────
    def lookup(self, target, source, stamp):
        for t in (stamp, Time()):                # exact time first, else latest available
            try:
                return self.tf_buffer.lookup_transform(target, source, t, timeout=Duration(seconds=0.05))
            except Exception:
                continue
        return None

    def shelf_distance(self, o_wc, fwd_w, look_w, stamp):
        """How far the shelf is, in the direction one box is seen (look_w, horizontal).

        Returns the distance of the shelf front from the camera measured along the
        camera's viewing direction fwd_w, or None if the LiDAR sees NO shelf there,
        i.e. the camera is looking past a shelf end at something far away (another
        aisle). Boxes seen there are not voids on the shelf we are inspecting.
        """
        if self.scan is None:
            return self.default_dist
        tf_wl = self.lookup(self.world_frame, self.scan.header.frame_id, stamp)
        if tf_wl is None:
            return self.default_dist
        R_wl, o_wl = tf_to_rt(tf_wl)
        d_l = R_wl.T @ look_w                                      # direction in LiDAR frame
        ang = math.atan2(d_l[1], d_l[0])
        s = self.scan
        i0 = int(round((ang - s.angle_min) / s.angle_increment))
        if not 0 <= i0 < len(s.ranges):
            return self.default_dist                               # outside the LiDAR's view
        win = max(1, int(math.radians(1.5) / s.angle_increment))
        rs = [r for r in s.ranges[max(0, i0 - win):i0 + win + 1] if s.range_min < r < s.range_max]
        if not rs:
            return None                                            # nothing within LiDAR range
        r = float(np.median(rs))
        hit_w = o_wl + R_wl @ np.array([r * math.cos(ang), r * math.sin(ang), 0.0])
        d = float(np.dot(hit_w[:2] - o_wc[:2], fwd_w[:2]))
        if d > self.default_dist + 0.6:
            return None                                            # looking past the shelf end
        # Much closer than a shelf = a shopper or the robot's own mast: keep the nominal distance.
        return d if d > self.default_dist - 0.45 else self.default_dist

    def localise(self, boxes, header):
        """Pixel boxes -> list of (centre_xyz, (width_m, height_m), unit_ray, camera_xyz) in world_frame, or None."""
        if self.K is None:
            return None
        stamp = Time.from_msg(header.stamp)
        tf_wc = self.lookup(self.world_frame, header.frame_id, stamp)
        # No map yet? AMCL needs a few seconds after start-up, so wait for it. Only if there
        # is still no map after 30 s (e.g. teleop without localization) fall back to odom.
        if tf_wc is None and self.world_frame != "odom" and time.monotonic() - self.t_start > 30.0:
            tf_wc = self.lookup("odom", header.frame_id, stamp)
            if tf_wc is not None:
                self.world_frame = "odom"
                self.get_logger().warn("No 'map' frame (no localization running) -> using 'odom'.")
        if tf_wc is None:
            if not self.warned_tf:
                self.get_logger().warn(f"No TF {self.world_frame} <- {header.frame_id} yet; "
                                       "boxes are drawn but not localised / alerted.")
                self.warned_tf = True
            return None
        R, o = tf_to_rt(tf_wc)
        fwd = R @ np.array([0.0, 0.0, 1.0])            # optical z axis = viewing direction
        fwd[2] = 0.0
        fwd /= np.linalg.norm(fwd)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        def pix_to_world(u, v, dist):
            ray = R @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            denom = float(np.dot(ray, fwd))
            if denom <= 1e-6:
                return None
            return o + ray * (dist / denom)            # intersect with the shelf-front plane

        out = []
        for b in boxes:
            look = R @ np.array([(b.cx - cx) / fx, 0.0, 1.0])      # horizontal direction of the box
            look[2] = 0.0
            look /= np.linalg.norm(look)
            dist = self.shelf_distance(o, fwd, look, stamp)
            if dist is None:
                out.append(None)
                continue
            c = pix_to_world(b.cx, b.cy, dist)
            p1, p2 = pix_to_world(b.x1, b.y1, dist), pix_to_world(b.x2, b.y2, dist)
            if c is None or p1 is None or p2 is None:
                out.append(None)
                continue
            width = float(np.linalg.norm((p2 - p1)[:2]))
            height = abs(float(p1[2] - p2[2]))
            # Depth errors move low voids slightly below z=0, so allow a margin, but drop
            # estimates that are clearly impossible (far below the floor / above the shelf).
            if not (-0.25 < c[2] < self.levels[-1] + 0.6):
                out.append(None)
                continue
            ray = c - o
            out.append((c, (width, height), ray / np.linalg.norm(ray), o))
        return out

    # ───────────────────────────── labelling ─────────────────────────────
    def describe(self, pos, cam):
        x, y, z = pos
        # the aisle is where the ROBOT stands (the void's own x is on the shelf between two aisles)
        aisle = 1 + int(np.argmin([abs(cam[0] - a) for a in self.aisle_x]))
        level = 0
        for i, top in enumerate(self.levels):
            if z >= top - 0.05:
                level = i
        shelf = "unknown"
        if self.patrol_shelves:
            row = "S" if y < self.row_split else "N"
            match = [s for s in self.patrol_shelves if s.startswith(row)]
            shelf = match[0] if match else self.patrol_shelves[0]
        return {"aisle": aisle, "level": level, "shelf": shelf}

    def match_gt(self, pos, level):
        best, best_d = None, 1e9
        for v in self.gt:
            cx = v["centre"][0]                      # x of the shelf front
            # anywhere inside the void's y-span counts as distance 0 along the shelf
            ylo, yhi = v["y_range"]
            along = max(0.0, ylo - pos[1], pos[1] - yhi)
            d = math.hypot(pos[0] - cx, along)
            if v["level"] == level and d < self.gt_r and d < best_d:
                best, best_d = v, d
        return best

    # ───────────────────────────── main loop ─────────────────────────────
    def process(self):
        msg, self.latest_img = self.latest_img, None
        if msg is None:
            return
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        boxes = self.detector.detect(bgr)
        self.frames += 1
        now = self.get_clock().now().nanoseconds * 1e-9

        locs = self.localise(boxes, msg.header) if boxes else []
        tags = [""] * len(boxes)
        if locs:
            for i, (b, loc) in enumerate(zip(boxes, locs)):
                if loc is None:
                    continue
                # A box cut by the left/right image border shows only PART of the void,
                # so its centre is shifted. Use it to confirm an existing track, but never to
                # move a track or start a new one (that is what made duplicate alerts).
                partial = b.x1 <= 3 or b.x2 >= msg.width - 4
                if not partial and loc[1][0] < self.min_width:
                    tags[i] = "too narrow"
                    continue
                tr = self.associate(loc[0], loc[1], loc[2], now, partial)
                if tr is not None:
                    tr.cam = loc[3]
                if tr is None:
                    tags[i] = "entering view"
                elif tr.confirmed:
                    tags[i] = f"VOID #{tr.id}  A{tr.label['aisle']} {tr.label['shelf']} L{tr.label['level']}"
                else:
                    tags[i] = f"checking {tr.hits}/{self.confirm_hits}"
        self.tracks = [t for t in self.tracks if t.confirmed or now - t.last_seen < self.cand_timeout]

        n_conf = sum(t.confirmed for t in self.tracks)
        banner = (f"[{self.detector.name}] voids in view: {len(boxes)}   "
                  f"confirmed so far: {n_conf}   {self.patrol_state}")
        out = draw_boxes(bgr, boxes, tags, banner)
        img_msg = self.bridge.cv2_to_imgmsg(out, encoding="bgr8")
        img_msg.header = msg.header
        self.pub_img.publish(img_msg)
        if self.debug_on:
            dbg = self.detector.debug_image()
            if dbg is not None:
                dmsg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
                dmsg.header = msg.header
                self.pub_dbg.publish(dmsg)
        self.publish_detections(boxes, msg.header)
        self.publish_markers()

    def associate(self, pos, size, ray, now, partial=False):
        """Find the track this sighting belongs to (or start a new one).

        The distance to the shelf is the least certain part of a position. A wrong
        distance slides the point ALONG the camera ray (changing both depth and height).
        So we compare the track with this sighting's ray: the part of the difference
        along the ray (depth) may be large, the part across the ray must be small.
        """
        best, best_d = None, 1e9
        for t in self.tracks:
            delta = t.pos - pos
            depth = float(np.dot(delta, ray))
            perp = delta - depth * ray
            sideways = math.hypot(perp[0], perp[1])
            if abs(depth) < 1.0 and sideways < self.assoc_r and abs(perp[2]) < 0.2 and sideways < best_d:
                best, best_d = t, sideways
        if best is None:
            if partial:
                return None
            best = Track(self.next_id, pos, size, now)
            self.next_id += 1
            self.tracks.append(best)
        elif partial:
            best.hits += 1
            best.last_seen = now
        else:
            best.update(pos, size, now)
        if not best.confirmed and best.hits >= self.confirm_hits:
            self.confirm(best)
        return best

    def confirm(self, t):
        t.confirmed = True
        t.label = self.describe(t.pos, t.cam)
        gt = self.match_gt(t.pos, t.label["level"]) if self.gt else None
        t.label["gt"] = gt["id"] if gt else None
        if gt and t.label["shelf"] == "unknown":
            t.label["shelf"] = f"{gt['shelf']}-{gt['face']}"
        alert = {
            "id": t.id,
            "aisle": t.label["aisle"], "shelf": t.label["shelf"], "level": t.label["level"],
            "position_m": [round(float(v), 2) for v in t.pos],
            "frame": self.world_frame,
            "width_m": round(float(t.size[0]), 2),
            "sightings": t.hits,
            "backend": self.detector.name,
            "ground_truth": t.label["gt"],
        }
        self.pub_alert.publish(String(data=json.dumps(alert)))
        x, y, z = t.pos
        gt_txt = ""
        if self.gt:
            gt_txt = (f"{GREEN}   ✔ matches ground truth {gt['id']} ({gt['category']}){RESET}" if gt
                      else f"{YELLOW}   ✘ no ground-truth void here (false alarm?){RESET}")
        self.get_logger().info(
            f"\n{RED}{BOLD}🚨 SHELF VOID #{t.id} DETECTED{RESET}{RED}  "
            f"aisle {t.label['aisle']} | shelf {t.label['shelf']} | level {t.label['level']} | "
            f"~{t.size[0]:.2f} m wide | at ({x:.2f}, {y:.2f}, {z:.2f}) [{self.world_frame}]{RESET}"
            f"{gt_txt}\n   -> please restock")

    # ───────────────────────────── outputs ─────────────────────────────
    def publish_detections(self, boxes, header):
        arr = Detection2DArray()
        arr.header = header
        for b in boxes:
            d = Detection2D()
            d.header = header
            d.bbox.center.position.x = float(b.cx)
            d.bbox.center.position.y = float(b.cy)
            d.bbox.size_x = float(b.w)
            d.bbox.size_y = float(b.h)
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id = "void"
            h.hypothesis.score = float(b.score)
            d.results.append(h)
            arr.detections.append(d)
        self.pub_det.publish(arr)

    def publish_markers(self):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        stamp = self.get_clock().now().to_msg()
        for t in self.tracks:
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = stamp
            m.ns = "voids"
            m.id = t.id
            m.type = Marker.CUBE
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in t.pos)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = max(0.15, float(t.size[0]))
            m.scale.z = max(0.1, float(t.size[1]))
            if t.confirmed:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 0.75
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.6, 0.0, 0.35
            arr.markers.append(m)
            if t.confirmed:
                txt = Marker()
                txt.header = m.header
                txt.ns = "void_labels"
                txt.id = t.id
                txt.type = Marker.TEXT_VIEW_FACING
                txt.pose.position.x, txt.pose.position.y = m.pose.position.x, m.pose.position.y
                txt.pose.position.z = 2.1
                txt.pose.orientation.w = 1.0
                txt.scale.z = 0.3
                txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
                txt.text = f"VOID #{t.id} {t.label['shelf']} L{t.label['level']}"
                arr.markers.append(txt)
        self.pub_mark.publish(arr)

    def print_summary(self):
        self.summary_printed = True
        conf = [t for t in self.tracks if t.confirmed]
        lines = [f"\n{CYAN}{BOLD}========== VOID DETECTION SUMMARY ({self.detector.name}, "
                 f"{self.frames} frames analysed) =========={RESET}"]
        for t in conf:
            lines.append(f"  #{t.id:<3} aisle {t.label['aisle']}  shelf {t.label['shelf']:<8} level {t.label['level']}"
                         f"  ({t.pos[0]:.2f}, {t.pos[1]:.2f})  GT={t.label.get('gt')}")
        if self.gt:
            found = {t.label.get("gt") for t in conf if t.label.get("gt")}
            fa = sum(1 for t in conf if not t.label.get("gt"))
            missed = [v["id"] for v in self.gt if v["id"] not in found]
            prec = len(found) / max(1, len(found) + fa)
            rec = len(found) / max(1, len(self.gt))
            lines.append(f"  found {len(found)}/{len(self.gt)} ground-truth voids | false alarms {fa} | "
                         f"precision {prec:.2f} recall {rec:.2f}")
            if missed:
                lines.append(f"  missed: {', '.join(missed)}")
        else:
            lines.append(f"  {len(conf)} voids reported")
        self.get_logger().info("\n".join(lines))


def main():
    rclpy.init()
    node = VoidDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass                                   # Ctrl+C / launch shutdown: exit quietly
    finally:
        if not node.summary_printed:
            node.print_summary()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
