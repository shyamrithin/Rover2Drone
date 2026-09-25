#!/usr/bin/env python3
# =============================================================================
# File:        src/rover2drone_nav/rover2drone_nav/rover_ekf.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-25
# Updated:     2026-09-25  reports re-acquisitions and odometry scale
# Depends:     rclpy, nav_msgs, sensor_msgs, std_msgs; ekf.py, geo.py
# =============================================================================
"""
rover_ekf
=========

Runs the planar EKF (ekf.py) on the rover's real sensor streams and
publishes the pose estimate the route follower drives on.

Subscribes  /rover/odometry     nav_msgs/Odometry  wheel speed (twist.linear.x)
            /rover/imu          sensor_msgs/Imu    gyro z (drives the predict step)
            /rover/gnss/fix     sensor_msgs/NavSatFix  (sensor_sim)
            /rover/compass      sensor_msgs/Imu    yaw + variance (sensor_sim)
Publishes   /rover/ekf/odom     nav_msgs/Odometry, frame world, child
                                rover/base_link_ekf, with x/y/yaw covariance
            /rover/ekf/status   std_msgs/String JSON: counts of accepted and
                                gated GNSS / compass updates, sigma_xy

The prediction runs on every IMU message (100 Hz in sim time), using the
latest wheel speed. The output z is the GNSS altitude, low-pass filtered
(the planar filter does not estimate height; only x, y, yaw are used).
"""

import json
import math
import os

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import String

from rover2drone_nav.ekf import EkfParams, PlanarEkf
from rover2drone_nav.geo import LocalFrame, read_origin
from rover2drone_nav.route_io import DEFAULT_REPO


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp_s(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


class RoverEkf(Node):
    def __init__(self):
        super().__init__("rover_ekf")
        self.declare_parameter("repo", DEFAULT_REPO)
        self.declare_parameter("publish_every", 2)   # IMU messages per output
        defaults = EkfParams()
        for f in EkfParams.__dataclass_fields__:
            self.declare_parameter(f, float(getattr(defaults, f)))
        params = EkfParams(**{f: float(self.get_parameter(f).value)
                              for f in EkfParams.__dataclass_fields__})
        repo = self.get_parameter("repo").value
        self.frame = LocalFrame(*read_origin(os.path.join(repo, "config", "turbines.yaml")))
        self.ekf = PlanarEkf(params)
        self.v = 0.0
        self.last_imu_t = None
        self.z = None
        self.n_imu = 0
        self.every = int(self.get_parameter("publish_every").value)

        self.pub = self.create_publisher(Odometry, "/rover/ekf/odom", 20)
        self.status_pub = self.create_publisher(String, "/rover/ekf/status", 10)
        self.create_subscription(Odometry, "/rover/odometry", self.on_odom, 20)
        self.create_subscription(Imu, "/rover/imu", self.on_imu, 50)
        self.create_subscription(NavSatFix, "/rover/gnss/fix", self.on_fix, 10)
        self.create_subscription(Imu, "/rover/compass", self.on_compass, 10)
        self.create_timer(2.0, self.report)
        self.get_logger().info("rover_ekf up: waiting for first GNSS fix + compass")

    def on_odom(self, msg):
        self.v = msg.twist.twist.linear.x

    def on_imu(self, msg):
        t = stamp_s(msg.header)
        if self.last_imu_t is not None:
            dt = t - self.last_imu_t
            if 0.0 < dt < 0.5:
                self.ekf.predict(self.v, msg.angular_velocity.z, dt)
        self.last_imu_t = t
        self.n_imu += 1
        if self.ekf.ready and self.n_imu % self.every == 0:
            self.publish(msg.header.stamp)

    def on_fix(self, msg):
        e, n, u = self.frame.to_enu(msg.latitude, msg.longitude, msg.altitude)
        sig = math.sqrt(max(msg.position_covariance[0], 1e-6))
        was = self.ekf.ready
        self.ekf.update_gnss(e, n, sig)
        self.z = u if self.z is None else 0.9 * self.z + 0.1 * u
        if self.ekf.ready and not was:
            self.get_logger().info(f"EKF initialised at ({e:.1f}, {n:.1f})")

    def on_compass(self, msg):
        was = self.ekf.ready
        self.ekf.update_compass(yaw_from_quat(msg.orientation),
                                math.sqrt(max(msg.orientation_covariance[8], 1e-8)))
        if self.ekf.ready and not was:
            self.get_logger().info("EKF initialised")

    def publish(self, stamp):
        x, y, yaw = self.ekf.pose()
        m = Odometry()
        m.header.stamp = stamp
        m.header.frame_id = "world"
        m.child_frame_id = "rover/base_link_ekf"
        m.pose.pose.position.x, m.pose.pose.position.y = x, y
        m.pose.pose.position.z = self.z or 0.0
        m.pose.pose.orientation.z = math.sin(yaw / 2.0)
        m.pose.pose.orientation.w = math.cos(yaw / 2.0)
        P = self.ekf.P
        cov = [0.0] * 36
        cov[0], cov[1], cov[5] = P[0, 0], P[0, 1], P[0, 2]
        cov[6], cov[7], cov[11] = P[1, 0], P[1, 1], P[1, 2]
        cov[30], cov[31], cov[35] = P[2, 0], P[2, 1], P[2, 2]
        cov[14] = cov[21] = cov[28] = 1e6
        m.pose.covariance = cov
        m.twist.twist.linear.x = self.v
        self.pub.publish(m)

    def report(self):
        e = self.ekf
        if not e.ready:
            return
        sx, sy = e.sigma_xy()
        self.status_pub.publish(String(data=json.dumps({
            "gnss": e.n_gnss, "gnss_gated": e.n_gnss_rej, "compass": e.n_compass,
            "compass_gated": e.n_compass_rej, "reacquired": e.n_reacquire,
            "odom_scale": round(float(e.s[4]), 4), "sigma_x": round(sx, 3),
            "sigma_y": round(sy, 3), "gyro_bias": round(float(e.s[3]), 5)})))


def main():
    rclpy.init()
    node = RoverEkf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
