#!/usr/bin/env python3
# =============================================================================
# File:        src/rover2drone_nav/rover2drone_nav/sensor_sim.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-25
# Depends:     rclpy, nav_msgs, sensor_msgs; sensor_models.py, geo.py,
#              route_io.py
# =============================================================================
"""
sensor_sim
==========

Produces realistic GNSS fixes and compass headings from the rover's
ground-truth pose, using the error models in sensor_models.py (bias that
wanders, white noise, outages, multipath and compass distortion near the
steel turbine towers). Gazebo's NavSat only adds white noise, which makes
localisation look far easier than it is.

Subscribes  /rover/ground_truth      nav_msgs/Odometry (world ENU)
Publishes   /rover/gnss/fix          sensor_msgs/NavSatFix  (no message during
                                     an outage; covariance = the receiver's
                                     claimed accuracy)
            /rover/compass           sensor_msgs/Imu, orientation only (yaw),
                                     REP-145: angular velocity / acceleration
                                     covariance[0] = -1 (not provided)
            /rover/gnss/true_error   std_msgs/Float64, true horizontal error
                                     of each fix (for evaluation only)

Parameters: profile (rtk | m8n | degraded), seed, repo, plus any GnssParams
field (e.g. bias_sigma) or compass_<CompassParams field> to override.
"""

import math
import os

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from std_msgs.msg import Float64

from rover2drone_nav.geo import LocalFrame, read_origin
from rover2drone_nav.route_io import DEFAULT_REPO, load_turbines
from rover2drone_nav.sensor_models import (CompassModel, CompassParams,
                                           GnssModel, GnssParams, get_profile)


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class SensorSim(Node):
    def __init__(self):
        super().__init__("sensor_sim")
        self.declare_parameter("profile", "m8n")
        self.declare_parameter("seed", 1)
        self.declare_parameter("repo", DEFAULT_REPO)
        overrides = {}
        for f in list(GnssParams.__dataclass_fields__):
            self.declare_parameter(f, -1.0)
            v = self.get_parameter(f).value
            if v is not None and v >= 0:
                overrides[f] = float(v)
        for f in list(CompassParams.__dataclass_fields__):
            self.declare_parameter("compass_" + f, -1.0)
            v = self.get_parameter("compass_" + f).value
            if v is not None and v >= 0:
                overrides["compass_" + f] = float(v)
        prof = self.get_parameter("profile").value
        seed = int(self.get_parameter("seed").value)
        repo = self.get_parameter("repo").value
        tyaml = os.path.join(repo, "config", "turbines.yaml")
        self.frame = LocalFrame(*read_origin(tyaml))
        towers = [(t["x"], t["y"]) for t in load_turbines(tyaml)]
        gp, cp = get_profile(prof, **overrides)
        self.gnss = GnssModel(gp, towers, seed)
        self.compass = CompassModel(cp, towers, seed)
        self.gt = None

        self.fix_pub = self.create_publisher(NavSatFix, "/rover/gnss/fix", 10)
        self.cmp_pub = self.create_publisher(Imu, "/rover/compass", 10)
        self.err_pub = self.create_publisher(Float64, "/rover/gnss/true_error", 10)
        self.create_subscription(Odometry, "/rover/ground_truth", self.on_gt, 20)
        self.gnss_dt = 1.0 / gp.rate_hz
        self.create_timer(self.gnss_dt, self.tick_gnss)
        self.create_timer(1.0 / cp.rate_hz, self.tick_compass)
        self.get_logger().info(
            f"sensor_sim: profile {prof}, seed {seed}; GNSS white {gp.white_sigma} m, "
            f"bias {gp.bias_sigma} m / {gp.bias_tau:.0f} s, outage every "
            f"~{gp.outage_every_s:.0f} s; compass bias sigma {cp.bias_sigma_deg} deg; "
            f"{len(towers)} towers")

    def on_gt(self, msg):
        self.gt = msg

    def tick_gnss(self):
        if self.gt is None:
            return
        p = self.gt.pose.pose.position
        m = self.gnss.step(self.gnss_dt, p.x, p.y, p.z)
        if m is None:
            return
        x, y, z, sh, sv = m
        lat, lon, alt = self.frame.to_lla(x, y, z)
        fix = NavSatFix()
        fix.header.stamp = self.get_clock().now().to_msg()
        fix.header.frame_id = "rover/gnss_link"
        fix.status.status = NavSatStatus.STATUS_FIX
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude, fix.longitude, fix.altitude = lat, lon, alt
        fix.position_covariance = [sh * sh, 0.0, 0.0, 0.0, sh * sh, 0.0, 0.0, 0.0, sv * sv]
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self.fix_pub.publish(fix)
        self.err_pub.publish(Float64(data=math.hypot(x - p.x, y - p.y)))

    def tick_compass(self):
        if self.gt is None:
            return
        p = self.gt.pose.pose
        h, sig = self.compass.step(p.position.x, p.position.y, yaw_from_quat(p.orientation))
        m = Imu()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "rover/imu_link"
        m.orientation.z, m.orientation.w = math.sin(h / 2.0), math.cos(h / 2.0)
        big = 1e6
        m.orientation_covariance = [big, 0.0, 0.0, 0.0, big, 0.0, 0.0, 0.0, sig * sig]
        m.angular_velocity_covariance[0] = -1.0
        m.linear_acceleration_covariance[0] = -1.0
        self.cmp_pub.publish(m)


def main():
    rclpy.init()
    node = SensorSim()
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
