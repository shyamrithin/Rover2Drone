#!/usr/bin/env python3
# =============================================================================
# File:        src/rover2drone_nav/rover2drone_nav/world_overlay.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-24
# Depends:     rclpy, nav_msgs, geometry_msgs, visualization_msgs; route_io.py
# =============================================================================
"""
world_overlay
=============

Publishes the static mission context for RViz in the world frame:

  /viz/world   visualization_msgs/MarkerArray
               roads (line strips at road-surface height, coloured by class),
               turbines (tower, nacelle, rotor disc outline, label)
  /viz/route   nav_msgs/Path     the planned route to the selected turbine

Both use transient-local QoS and are also re-sent every 2 s, so RViz shows
them whenever it connects. Everything is read from the repo's config/
files, so it stays in sync after rebuilding the world.

Parameters:
  turbine      route to show, e.g. turbine_07 ('' = no route)
  route_file   explicit route JSON (overrides turbine)
  repo         repository root (default ~/Rover2Drone)
"""

import math
import os

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from rover2drone_nav.route_io import (DEFAULT_REPO, load_roads, load_route,
                                      load_turbines, read_yaml_scalars, route_path)

ROAD_COLOURS = {"secondary": (0.95, 0.75, 0.35), "unclassified": (0.85, 0.60, 0.30),
                "track": (0.70, 0.50, 0.30)}
ROTOR_R = 26.0   # S52-class rotor radius, m


class WorldOverlay(Node):
    def __init__(self):
        super().__init__("world_overlay")
        self.declare_parameter("turbine", "turbine_07")
        self.declare_parameter("route_file", "")
        self.declare_parameter("repo", DEFAULT_REPO)
        repo = self.get_parameter("repo").value
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.world_pub = self.create_publisher(MarkerArray, "/viz/world", qos)
        self.route_pub = self.create_publisher(Path, "/viz/route", qos)

        self.markers = self.build_world(repo)
        rf = self.get_parameter("route_file").value
        tb = self.get_parameter("turbine").value
        self.path = None
        if rf or tb:
            rf = rf or route_path(repo, tb)
            if os.path.exists(rf):
                self.path = self.build_path(load_route(rf))
            else:
                self.get_logger().warn(f"no route file {rf}")
        self.publish()
        self.create_timer(2.0, self.publish)
        self.get_logger().info(
            f"overlay: {len(self.markers.markers)} markers"
            + (f", route {tb or rf}" if self.path else ""))

    def marker(self, ns, mid, mtype, rgba):
        m = Marker()
        m.header.frame_id = "world"
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    def build_world(self, repo):
        arr = MarkerArray()
        mid = 0
        for rid, rtype, pts in load_roads(os.path.join(repo, "config", "roads.json")):
            col = ROAD_COLOURS.get(rtype, (0.8, 0.6, 0.3))
            m = self.marker("roads", mid, Marker.LINE_STRIP, (*col, 0.9))
            m.scale.x = 1.5
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]) + 0.05)
                        for p in pts[::2]] + [Point(x=float(pts[-1, 0]),
                                                    y=float(pts[-1, 1]),
                                                    z=float(pts[-1, 2]) + 0.05)]
            arr.markers.append(m)
            mid += 1
        tpath = os.path.join(repo, "config", "turbines.yaml")
        meta = read_yaml_scalars(tpath)
        rotor_r = meta.get("rotor_diameter_m", 2 * ROTOR_R) / 2.0
        for k, t in enumerate(load_turbines(tpath)):
            gz, hz = t["ground_z"], t["hub_z"]
            h = hz - gz
            tower = self.marker("turbines", 10 * k, Marker.CYLINDER, (0.92, 0.92, 0.95, 1.0))
            tower.pose.position = Point(x=t["x"], y=t["y"], z=gz + h / 2.0)
            tower.scale.x = tower.scale.y = 2.4
            tower.scale.z = h
            nac = self.marker("turbines", 10 * k + 1, Marker.SPHERE, (0.92, 0.92, 0.95, 1.0))
            nac.pose.position = Point(x=t["x"], y=t["y"], z=hz)
            nac.scale.x = nac.scale.y = nac.scale.z = 4.0
            disc = self.marker("turbines", 10 * k + 2, Marker.LINE_STRIP, (0.6, 0.8, 1.0, 0.8))
            disc.scale.x = 0.4
            yaw = float(t.get("yaw", meta.get("rotor_yaw_rad", 0.0)))
            ax = (-math.sin(yaw), math.cos(yaw))   # disc lies across the rotor axis
            disc.points = [Point(x=t["x"] + rotor_r * math.cos(a) * ax[0],
                                 y=t["y"] + rotor_r * math.cos(a) * ax[1],
                                 z=hz + rotor_r * math.sin(a))
                           for a in [i * 2 * math.pi / 48 for i in range(49)]]
            label = self.marker("labels", k, Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0, 1.0))
            label.pose.position = Point(x=t["x"], y=t["y"], z=hz + rotor_r + 8.0)
            label.scale.z = 8.0
            label.text = str(t["id"]).replace("turbine_", "T")
            arr.markers += [tower, nac, disc, label]
        return arr

    def build_path(self, route):
        path = Path()
        path.header.frame_id = "world"
        for x, y, z in route.p:
            ps = PoseStamped()
            ps.header.frame_id = "world"
            ps.pose.position = Point(x=float(x), y=float(y), z=float(z) + 0.3)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        return path

    def publish(self):
        stamp = self.get_clock().now().to_msg()
        for m in self.markers.markers:
            m.header.stamp = stamp
        self.world_pub.publish(self.markers)
        if self.path is not None:
            self.path.header.stamp = stamp
            self.route_pub.publish(self.path)


def main():
    rclpy.init()
    node = WorldOverlay()
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
