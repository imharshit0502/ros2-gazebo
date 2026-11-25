#!/usr/bin/env python3
"""
mapping_tour.py - drive the robot through every aisle so SLAM can map the store.

WHY THIS NODE?
  SLAM only maps what the LiDAR has seen, so someone has to drive the robot around
  during mapping. You can do it by hand with teleop_twist_keyboard. This node does
  the same thing automatically along a fixed route, which gives the same complete
  map every time and is easy to show in a presentation.

HOW IT WORKS (a very simple "go-to-point" controller, NOT Nav2)
  1. Ask TF where the robot is in the `map` frame (pose published by SLAM Toolbox).
  2. Point at the next waypoint: turn on the spot if the heading error is large,
     otherwise drive forward while steering proportionally.
  3. When within `goal_tolerance` of the waypoint, move on to the next one.
  4. Safety: if the LiDAR sees something closer than `stop_distance` straight
     ahead, stop moving forward (turning is still allowed).

  The robot spawns at the world origin facing +x (east), and SLAM starts the map
  at the spawn pose, so map coordinates == Gazebo world coordinates. That is why
  the waypoints below can be read straight off the store layout drawing in
  my_robot_bringup/scripts/generate_retail_store_world.py.

  Slow speeds are deliberate. Scan matching works best when consecutive scans
  overlap a lot, and slow motion also makes the live map growth easy to follow.
"""

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

# (x, y) in metres, map frame == Gazebo world frame. Aisle centres x = -7.16, -3.58, 0, 3.58, 7.16
WAYPOINTS = [
    (0.0, 7.8),      # up aisle 3 to the back corridor
    (-3.58, 7.8),    # west along the back corridor
    (-3.58, -3.8),   # down aisle 2 (full length)
    (-7.16, -3.8),   # west along the front promenade
    (-7.16, 7.8),    # up aisle 1 (beside the west wall shelf)
    (3.58, 7.8),     # long run east along the back corridor (revisits known area -> loop closure)
    (3.58, -3.8),    # down aisle 4
    (7.16, -3.8),    # east along the front promenade
    (7.16, 7.8),     # up aisle 5 (beside the east wall shelf)
    (7.16, -3.8),    # back down aisle 5
    (5.45, -3.8),    # line up with the gap between checkout 2's stool and checkout 3
    (5.45, -7.5),    # through the checkout lane
    (-1.5, -7.5),    # west along the storefront (sees checkouts, exit and entrance gates)
    (-1.5, -3.8),    # back up to the promenade
    (0.0, -3.8),
    (0.0, 0.0),      # home = spawn point (closes the final loop)
]


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class MappingTour(Node):
    def __init__(self):
        super().__init__("mapping_tour")
        self.declare_parameter("linear_speed", 0.35)      # m/s
        self.declare_parameter("angular_speed", 0.6)      # rad/s
        self.declare_parameter("goal_tolerance", 0.25)    # m
        self.declare_parameter("turn_in_place_angle", 0.35)  # rad (~20 deg)
        self.declare_parameter("stop_distance", 0.30)     # m, measured from the LiDAR
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")

        self.v_max = self.get_parameter("linear_speed").value
        self.w_max = self.get_parameter("angular_speed").value
        self.tol = self.get_parameter("goal_tolerance").value
        self.turn_angle = self.get_parameter("turn_in_place_angle").value
        self.stop_dist = self.get_parameter("stop_distance").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)

        self.front_clear = True
        self.idx = 0
        self.done = False
        self.waiting_logged = False
        self.blocked_logged = False
        self.timer = self.create_timer(0.1, self.step)  # 10 Hz control loop
        self.get_logger().info(f"Mapping tour: {len(WAYPOINTS)} waypoints. Waiting for the "
                               f"'{self.map_frame}' frame from SLAM Toolbox ...")

    def on_scan(self, msg: LaserScan):
        # Look at the +/- 25 deg slice straight ahead of the robot
        half = math.radians(25.0)
        closest = float("inf")
        for i, r in enumerate(msg.ranges):
            a = msg.angle_min + i * msg.angle_increment
            if -half <= a <= half and msg.range_min < r < closest:
                closest = r
        self.front_clear = closest > self.stop_dist

    def robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time(),
                                                timeout=Duration(seconds=0.0))
        except TransformException:
            return None
        tr = t.transform
        return tr.translation.x, tr.translation.y, yaw_from_quaternion(tr.rotation)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def step(self):
        if self.done:
            return
        pose = self.robot_pose()
        if pose is None:
            if not self.waiting_logged:
                self.get_logger().info("... no map -> base_footprint TF yet (is SLAM running?)")
                self.waiting_logged = True
            return
        if self.waiting_logged:
            self.get_logger().info("SLAM is up, starting the tour.")
            self.waiting_logged = False

        x, y, yaw = pose
        gx, gy = WAYPOINTS[self.idx]
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)

        if dist < self.tol:
            self.get_logger().info(f"Reached waypoint {self.idx + 1}/{len(WAYPOINTS)}  ({gx:.2f}, {gy:.2f})")
            self.idx += 1
            if self.idx >= len(WAYPOINTS):
                self.stop()
                self.done = True
                self.get_logger().info(
                    "MAPPING TOUR COMPLETE. Save the map now (in another terminal):\n"
                    "  ros2 run nav2_map_server map_saver_cli -f "
                    "~/ros2_rws/src/my_robot_navigation/maps/retail_store")
            return

        err = wrap_angle(math.atan2(dy, dx) - yaw)
        cmd = Twist()
        cmd.angular.z = max(-self.w_max, min(self.w_max, 1.5 * err))
        if abs(err) < self.turn_angle:
            # slow down near the waypoint and while still correcting heading
            cmd.linear.x = min(self.v_max, 0.8 * dist) * (1.0 - abs(err) / self.turn_angle * 0.5)
            if not self.front_clear:
                cmd.linear.x = 0.0
                if not self.blocked_logged:
                    self.get_logger().warn("Obstacle straight ahead - pausing forward motion")
                    self.blocked_logged = True
            else:
                self.blocked_logged = False
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = MappingTour()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
