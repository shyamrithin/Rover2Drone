#!/usr/bin/env python3
# =============================================================================
# File:        src/rover2drone_coordination/rover2drone_coordination/relative_state.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Updated:     2026-09-25  rover pose from /rover/ground_truth (world frame)
#              by default. The old /rover/odometry path added the spawn
#              position but never rotated by the spawn yaw, so the rover
#              drifted away from the latched drone as soon as it drove
#              (slant range grew ~0.9 m per m driven with the -56 deg spawn).
# Depends:     rclpy, nav_msgs, geometry_msgs, std_msgs, px4_msgs
# =============================================================================
"""
relative_state.py
=================

Publishes the relative geometry between the ground rover and the aerial
vehicle, so each has a view of where the other is.

This is the minimal coordination link for the Rover2Drone marsupial
inspection stack. It fuses two position sources that do not share a
convention and reconciles them:

  Rover    nav_msgs/Odometry on /rover/odometry, published by Gazebo's
           DiffDrive plugin in ENU (x East, y North, z Up).

  Drone    px4_msgs/VehicleLocalPosition on /fmu/out/vehicle_local_position,
           published by PX4's EKF2 in NED (x North, y East, z Down) and
           relative to wherever the vehicle booted, NOT the Gazebo origin.

The NED-to-ENU conversion is (x, y, z)_ENU = (y, x, -z)_NED. The drone's
local origin offset is supplied as a parameter because PX4 zeroes its
estimator at boot position; for the demo the drone starts on the rover
deck, so the default offset matches that spawn pose.

Published topics
  /coordination/relative_position   geometry_msgs/PointStamped
      Drone position expressed in the rover's frame (ENU metres).
  /coordination/slant_range         std_msgs/Float32
      Straight-line distance between the two vehicles, metres.
  /coordination/ground_range        std_msgs/Float32
      Horizontal distance only, metres. This is the quantity a tether
      length or an RF link budget actually constrains.
  /coordination/bearing_deg         std_msgs/Float32
      Bearing from rover to drone, degrees, 0 = rover's forward axis,
      positive counter-clockwise.
  /coordination/link_ok             std_msgs/Bool
      False when slant range exceeds max_range_m. A placeholder for the
      RF path-loss model that replaces it later.

Parameters
  drone_origin_x, drone_origin_y, drone_origin_z   PX4 local origin in
      the Gazebo world frame (ENU metres). Defaults match the demo spawn.
  rover_topic      nav_msgs/Odometry source for the rover (default
      /rover/ground_truth, already in the world frame).
  rover_frame      'world' (default): rover_topic is in world ENU.
                   'odom': rover_topic is DiffDrive odometry, which starts
      at the spawn pose with the spawn heading as its x axis; it is rotated
      by rover_origin_yaw and offset by rover_origin_x/y/z.
  max_range_m      Range beyond which link_ok goes false.
  publish_rate_hz  Output rate.

Usage
  ros2 run rover2drone_coordination relative_state --ros-args -p use_sim_time:=true
"""

import math

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import Bool, Float32

from px4_msgs.msg import VehicleLocalPosition


def quat_to_yaw(x, y, z, w):
    """Yaw angle in radians from a quaternion, ignoring roll and pitch."""
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


class RelativeState(Node):
    def __init__(self):
        super().__init__('relative_state')

        self.declare_parameter('drone_origin_x', 8.05)
        self.declare_parameter('drone_origin_y', 0.0)
        self.declare_parameter('drone_origin_z', 0.55)
        # Gazebo's DiffDrive plugin publishes odometry in the rover's own
        # odom frame, which starts at zero wherever the model spawned. The
        # spawn pose must therefore be added back to recover world ENU.
        self.declare_parameter('rover_origin_x', 8.0)
        self.declare_parameter('rover_origin_y', 0.0)
        self.declare_parameter('rover_origin_z', 0.22)
        self.declare_parameter('rover_origin_yaw', 0.0)
        self.declare_parameter('rover_topic', '/rover/ground_truth')
        self.declare_parameter('rover_frame', 'world')
        self.declare_parameter('max_range_m', 60.0)
        # PX4 v1.17 versions some uORB topics on the wire with a _v1 suffix
        # (vehicle_local_position, vehicle_status, home_position), while
        # others (vehicle_attitude, vehicle_odometry) are unversioned.
        # Parameterised so a firmware bump does not require a code change.
        self.declare_parameter('drone_pos_topic',
                               '/fmu/out/vehicle_local_position_v1')
        self.declare_parameter('publish_rate_hz', 10.0)

        self.origin = (
            self.get_parameter('drone_origin_x').value,
            self.get_parameter('drone_origin_y').value,
            self.get_parameter('drone_origin_z').value,
        )
        self.rover_origin = (
            self.get_parameter('rover_origin_x').value,
            self.get_parameter('rover_origin_y').value,
            self.get_parameter('rover_origin_z').value,
        )
        self.max_range = self.get_parameter('max_range_m').value
        rate = self.get_parameter('publish_rate_hz').value

        # PX4 publishes over uXRCE-DDS with best-effort, volatile QoS and a
        # depth of 1. A default reliable subscriber will never match it and
        # will sit silent with no error, which is a common first bug here.
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self.rover_odom = None
        self.drone_pos = None

        self.rover_topic = self.get_parameter('rover_topic').value
        self.rover_frame = self.get_parameter('rover_frame').value
        self.rover_yaw0 = self.get_parameter('rover_origin_yaw').value
        self.create_subscription(
            Odometry, self.rover_topic, self._on_rover, 10)
        self.create_subscription(
            VehicleLocalPosition,
            self.get_parameter('drone_pos_topic').value,
            self._on_drone, px4_qos)

        self.pub_rel = self.create_publisher(
            PointStamped, '/coordination/relative_position', 10)
        self.pub_slant = self.create_publisher(
            Float32, '/coordination/slant_range', 10)
        self.pub_ground = self.create_publisher(
            Float32, '/coordination/ground_range', 10)
        self.pub_bearing = self.create_publisher(
            Float32, '/coordination/bearing_deg', 10)
        self.pub_link = self.create_publisher(
            Bool, '/coordination/link_ok', 10)

        self.create_timer(1.0 / rate, self._tick)
        self._warned = False

        self.get_logger().info(
            f'relative_state up. Rover from {self.rover_topic} '
            f'({self.rover_frame} frame). Drone local origin assumed at '
            f'{self.origin} ENU, max range {self.max_range} m.')

    def _on_rover(self, msg):
        self.rover_odom = msg

    def _on_drone(self, msg):
        self.drone_pos = msg

    def _tick(self):
        if self.rover_odom is None or self.drone_pos is None:
            if not self._warned:
                missing = []
                if self.rover_odom is None:
                    missing.append(self.rover_topic)
                if self.drone_pos is None:
                    missing.append(
                        self.get_parameter('drone_pos_topic').value)
                self.get_logger().warn(f'Waiting for: {", ".join(missing)}')
                self._warned = True
            return

        d = self.drone_pos
        if not (d.xy_valid and d.z_valid):
            return

        # PX4 NED -> world ENU, then offset by the PX4 local origin.
        drone_e = self.origin[0] + d.y
        drone_n = self.origin[1] + d.x
        drone_u = self.origin[2] - d.z

        r = self.rover_odom.pose.pose
        rover_yaw = quat_to_yaw(r.orientation.x, r.orientation.y,
                                r.orientation.z, r.orientation.w)
        if self.rover_frame == 'world':
            rover_e, rover_n, rover_u = r.position.x, r.position.y, r.position.z
        else:
            # DiffDrive odom: x axis = spawn heading, origin = spawn pose.
            c0, s0 = math.cos(self.rover_yaw0), math.sin(self.rover_yaw0)
            rover_e = self.rover_origin[0] + c0 * r.position.x - s0 * r.position.y
            rover_n = self.rover_origin[1] + s0 * r.position.x + c0 * r.position.y
            rover_u = self.rover_origin[2] + r.position.z
            rover_yaw += self.rover_yaw0

        # Vector from rover to drone, in the world frame.
        de = drone_e - rover_e
        dn = drone_n - rover_n
        du = drone_u - rover_u

        # Rotate into the rover's body frame so "ahead" means ahead.
        c, s = math.cos(-rover_yaw), math.sin(-rover_yaw)
        bx = de * c - dn * s
        by = de * s + dn * c

        ground = math.hypot(de, dn)
        slant = math.sqrt(ground * ground + du * du)
        bearing = math.degrees(math.atan2(by, bx))

        stamp = self.get_clock().now().to_msg()

        p = PointStamped()
        p.header.stamp = stamp
        p.header.frame_id = 'rover/base_link'
        p.point.x, p.point.y, p.point.z = bx, by, du
        self.pub_rel.publish(p)

        self.pub_slant.publish(Float32(data=float(slant)))
        self.pub_ground.publish(Float32(data=float(ground)))
        self.pub_bearing.publish(Float32(data=float(bearing)))
        self.pub_link.publish(Bool(data=bool(slant <= self.max_range)))


def main(args=None):
    rclpy.init(args=args)
    node = RelativeState()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # The SIGINT handler may already have shut the context down.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()