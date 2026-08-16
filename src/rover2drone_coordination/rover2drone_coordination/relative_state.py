#!/usr/bin/env python3
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
        self.declare_parameter('max_range_m', 60.0)
        self.declare_parameter('publish_rate_hz', 10.0)

        self.origin = (
            self.get_parameter('drone_origin_x').value,
            self.get_parameter('drone_origin_y').value,
            self.get_parameter('drone_origin_z').value,
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

        self.create_subscription(
            Odometry, '/rover/odometry', self._on_rover, 10)
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
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
            f'relative_state up. Drone local origin assumed at '
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
                    missing.append('/rover/odometry')
                if self.drone_pos is None:
                    missing.append('/fmu/out/vehicle_local_position')
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
        rover_e, rover_n, rover_u = r.position.x, r.position.y, r.position.z
        rover_yaw = quat_to_yaw(r.orientation.x, r.orientation.y,
                                r.orientation.z, r.orientation.w)

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
        rclpy.shutdown()


if __name__ == '__main__':
    main()