#!/usr/bin/env python3
"""
shelf_patrol.py - autonomous shelf-inspection patrol using Nav2.

WHAT IT DOES
  1. Waits until Nav2 (and AMCL localization) are fully active.
  2. Raises the camera mast to full height.
  3. Sends the robot to every waypoint in config/patrol_route.yaml, one after another,
     then back HOME. Nav2 plans each path and avoids shelves AND shoppers on the way.
  4. While driving a leg that passes shelves, publishes which shelf faces the camera
     is looking at on /patrol/status. The void-detection (CV) node can subscribe to
     that and tag each detected void with its shelf / aisle.
  5. Shows the route in RViz (/patrol/route_markers) and prints a summary at the end.

WHY goToPose ONE WAYPOINT AT A TIME (instead of Nav2's followWaypoints)?
  It lets this script print a clear message for every leg, publish what is being
  inspected, and retry a waypoint if a shopper blocked the way - all useful for a demo.

/patrol/status format (std_msgs/String), easy to parse with str.split():
  "state=INSPECTING leg=5/16 target=Aisle_2_-_north_end shelves=S1-E,N1-E"
  "state=TRANSIT    leg=4/16 target=... shelves=none"
  "state=DONE" / "state=FAILED ..."
"""

import math
import os
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float64, String
from visualization_msgs.msg import Marker, MarkerArray

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def make_pose(nav, x, y, yaw_deg):
    p = PoseStamped()
    p.header.frame_id = "map"
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x = float(x)
    p.pose.position.y = float(y)
    yaw = math.radians(yaw_deg)
    p.pose.orientation.z = math.sin(yaw / 2.0)
    p.pose.orientation.w = math.cos(yaw / 2.0)
    return p


class PatrolIO(Node):
    """Small helper node: status topic, mast command and RViz markers."""

    def __init__(self):
        super().__init__("shelf_patrol")
        default_route = os.path.join(get_package_share_directory("my_robot_navigation"),
                                     "config", "patrol_route.yaml")
        self.declare_parameter("route_file", default_route)
        self.declare_parameter("loops", 1)              # 0 = patrol forever
        self.declare_parameter("mast_height", 0.40)     # metres, 0.40 = fully extended
        self.declare_parameter("retries", 1)            # extra attempts per waypoint

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, "/patrol/status", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/patrol/route_markers", latched)
        self.mast_pub = self.create_publisher(Float64, "/mast/height_cmd", 10)

    def status(self, text):
        self.status_pub.publish(String(data=text))

    def publish_route(self, home, wps, current=-1):
        arr = MarkerArray()
        pts = [home] + wps + [home]
        line = Marker()
        line.header.frame_id = "map"
        line.ns, line.id, line.type = "route", 0, Marker.LINE_STRIP
        line.scale.x = 0.05
        line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 0.6, 0.0, 0.8
        line.pose.orientation.w = 1.0
        for w in pts:
            line.points.append(Point(x=float(w["x"]), y=float(w["y"]), z=0.05))
        arr.markers.append(line)
        for i, w in enumerate([home] + wps):
            m = Marker()
            m.header.frame_id = "map"
            m.ns, m.id, m.type = "wp", i, Marker.SPHERE
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(w["x"]), float(w["y"]), 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.35 if i == 0 else 0.25
            if i == 0:
                m.color.r, m.color.g, m.color.b = 0.1, 0.4, 1.0        # home = blue
            elif i == current:
                m.color.r, m.color.g, m.color.b = 1.0, 0.1, 0.1        # current goal = red
            elif w.get("inspect"):
                m.color.r, m.color.g, m.color.b = 0.1, 0.9, 0.1        # ends an inspection leg = green
            else:
                m.color.r, m.color.g, m.color.b = 1.0, 0.8, 0.0        # transit = yellow
            m.color.a = 1.0
            arr.markers.append(m)
            t = Marker()
            t.header.frame_id = "map"
            t.ns, t.id, t.type = "label", i, Marker.TEXT_VIEW_FACING
            t.pose.position.x, t.pose.position.y, t.pose.position.z = float(w["x"]) + 0.3, float(w["y"]), 0.6
            t.pose.orientation.w = 1.0
            t.scale.z = 0.35
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = "HOME" if i == 0 else str(i)
            arr.markers.append(t)
        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    io = PatrolIO()
    nav = BasicNavigator()

    route_file = io.get_parameter("route_file").value
    loops = int(io.get_parameter("loops").value)
    retries = int(io.get_parameter("retries").value)
    with open(route_file) as fh:
        route = yaml.safe_load(fh)
    home = route["home"]
    legs = route["waypoints"] + [dict(home, inspect="")]   # always finish at HOME
    n = len(legs)

    io.get_logger().info(f"Loaded {len(route['waypoints'])} waypoints from {route_file}")
    io.publish_route(home, route["waypoints"])
    io.get_logger().info("Waiting for Nav2 + AMCL to become active ...")
    nav.waitUntilNav2Active(localizer="amcl")

    io.get_logger().info(f"Raising camera mast to {io.get_parameter('mast_height').value:.2f} m")
    for _ in range(5):
        io.mast_pub.publish(Float64(data=float(io.get_parameter("mast_height").value)))
        time.sleep(0.1)

    loop = 0
    while rclpy.ok() and (loops == 0 or loop < loops):
        loop += 1
        print(f"\n{BOLD}{CYAN}========== SHELF PATROL - round {loop}"
              f"{'' if loops == 0 else '/' + str(loops)} : {n} legs ==========={RESET}", flush=True)
        t_start = nav.get_clock().now()
        ok, failed = 0, []

        for i, wp in enumerate(legs):
            leg = f"{i + 1}/{n}"
            shelves = wp.get("inspect", "")
            state = "INSPECTING" if shelves else "TRANSIT"
            status = (f"state={state} leg={leg} target={wp['name'].replace(' ', '_')} "
                      f"shelves={shelves.replace(' ', '') or 'none'}")
            io.publish_route(home, route["waypoints"], current=(0 if i == n - 1 else i + 1))
            if shelves:
                print(f"{GREEN}[leg {leg}] INSPECTING {shelves:<12} -> driving to {wp['name']}{RESET}",
                      flush=True)
            else:
                print(f"{YELLOW}[leg {leg}] transit           -> driving to {wp['name']}{RESET}", flush=True)

            result = None
            for attempt in range(retries + 1):
                if attempt > 0:
                    print(f"{YELLOW}   retry {attempt}: clearing costmaps and trying again{RESET}", flush=True)
                    nav.clearAllCostmaps()
                nav.goToPose(make_pose(nav, wp["x"], wp["y"], wp["yaw_deg"]))
                last_print = 0.0
                while not nav.isTaskComplete():
                    rclpy.spin_once(io, timeout_sec=0.0)
                    io.status(status)
                    fb = nav.getFeedback()
                    now = time.time()
                    if fb and now - last_print > 5.0:
                        last_print = now
                        rec = f", recoveries {fb.number_of_recoveries}" if fb.number_of_recoveries else ""
                        print(f"   ... {fb.distance_remaining:5.2f} m to go{rec}", flush=True)
                    time.sleep(0.2)
                result = nav.getResult()
                if result == TaskResult.SUCCEEDED:
                    break

            if result == TaskResult.SUCCEEDED:
                ok += 1
                print(f"{GREEN}   reached {wp['name']}{RESET}", flush=True)
            else:
                failed.append(f"{leg} {wp['name']}")
                print(f"{RED}   could NOT reach {wp['name']} ({result}) - skipping to next waypoint{RESET}",
                      flush=True)

        mins = (nav.get_clock().now() - t_start).nanoseconds / 1e9 / 60.0
        colour = GREEN if not failed else YELLOW
        print(f"\n{BOLD}{colour}========== PATROL ROUND {loop} COMPLETE: {ok}/{n} legs OK, "
              f"{mins:.1f} min (sim time) - robot is back HOME =========={RESET}", flush=True)
        for f in failed:
            print(f"{RED}   failed leg {f}{RESET}", flush=True)
        io.status("state=DONE" if not failed else f"state=DONE_WITH_FAILURES failed={len(failed)}")

    io.publish_route(home, route["waypoints"])
    nav.destroy_node()
    io.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
