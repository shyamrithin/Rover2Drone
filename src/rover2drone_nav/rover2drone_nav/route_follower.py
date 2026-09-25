#!/usr/bin/env python3
# =============================================================================
# File:        src/rover2drone_nav/rover2drone_nav/route_follower.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-24
# Updated:     2026-09-24  clean shutdown on Ctrl+C (no publish on a dead context)
#              2026-09-25  logs rover-drone slant range per step
#              2026-09-25  step 4: ground-truth columns (true CTE, loc error),
#              run_tag, exit_on_arrival, timeout_s
# Depends:     rclpy, nav_msgs, geometry_msgs, std_msgs, std_srvs,
#              visualization_msgs; pure_pursuit.py, route_io.py
# =============================================================================
"""
route_follower
==============

Drives the rover along a planned road route (config/routes/route_<t>.json,
from tools/road_planner.py) with the pure-pursuit controller in
pure_pursuit.py.

Pose source: /rover/ground_truth (nav_msgs/Odometry in the world frame,
from Gazebo's OdometryPublisher on the rover). Step 4 swaps this for the
EKF estimate; only the topic name changes (parameter pose_topic).

Subscribes   <pose_topic>          nav_msgs/Odometry   (default /rover/ground_truth)
Publishes    /rover/cmd_vel        geometry_msgs/Twist
             /nav/status           std_msgs/String     JSON: state, s, remaining, cte, ...
             /nav/cross_track_error std_msgs/Float64
             /viz/lookahead        visualization_msgs/Marker (world frame)
Also logs   /coordination/slant_range (std_msgs/Float32, from relative_state)
             into the run CSV, so the latched drone's seating is checked per run
Services     /nav/pause, /nav/resume  std_srvs/Trigger

Evaluation: the ground truth (gt_topic) is logged beside the pose the
follower drives on, with the TRUE cross-track error (gt vs route) and the
localisation error (pose vs gt). With exit_on_arrival the node exits after
ARRIVED (or after timeout_s of sim time) so batch runs can chain; run_tag
is added to the log file name.

Safety: if the pose is older than pose_timeout (sim time), the rover is
stopped until poses resume. On arrival (or Ctrl+C) it publishes zero
velocity. Every run is logged to <repo>/logs/route_<turbine>_<stamp>.csv
(one row per control step) for tools/plot_route_run.py.

Parameters (ros2 run ... --ros-args -p name:=value):
  turbine        turbine id, e.g. turbine_07 (selects the route file)
  route_file     explicit route JSON (overrides turbine)
  repo           repository root (default ~/Rover2Drone)
  pose_topic     default /rover/ground_truth
  rate_hz        control rate, default 20
  auto_start     start driving immediately (default true)
  gt_topic       ground truth for evaluation (default /rover/ground_truth)
  run_tag        label added to the log name, e.g. m8n_s1
  exit_on_arrival  exit after ARRIVED / timeout (default false)
  timeout_s      give up after this much sim time (0 = never)
  v_max ... goal_tol   controller tuning, see pure_pursuit.PPParams
                       (v_max defaults to 0.9 x rover.yaml max_speed_mps)
"""

import csv
import dataclasses
import json
import math
import os
import time

import rclpy
import rclpy.task
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32, Float64, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker

from rover2drone_nav.pure_pursuit import PPParams, PurePursuit
from rover2drone_nav.route_io import (DEFAULT_REPO, load_route, read_rover_yaml,
                                      route_path)


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def rpy_from_quat(q):
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    return roll, pitch, yaw_from_quat(q)


class RouteFollower(Node):
    def __init__(self):
        super().__init__("route_follower")
        self.declare_parameter("turbine", "turbine_07")
        self.declare_parameter("route_file", "")
        self.declare_parameter("repo", DEFAULT_REPO)
        self.declare_parameter("pose_topic", "/rover/ground_truth")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("pose_timeout", 0.5)
        self.declare_parameter("auto_start", True)
        self.declare_parameter("gt_topic", "/rover/ground_truth")
        self.declare_parameter("run_tag", "")
        self.declare_parameter("exit_on_arrival", False)
        self.declare_parameter("timeout_s", 0.0)   # sim s, 0 = none
        repo = self.get_parameter("repo").value
        rover = read_rover_yaml(os.path.join(repo, "config", "rover.yaml"))
        defaults = PPParams(v_max=0.9 * rover.get("max_speed_mps", 0.6))
        for f in dataclasses.fields(PPParams):
            self.declare_parameter(f.name, float(getattr(defaults, f.name)))
        params = PPParams(**{f.name: float(self.get_parameter(f.name).value)
                             for f in dataclasses.fields(PPParams)})

        rf = self.get_parameter("route_file").value or \
            route_path(repo, self.get_parameter("turbine").value)
        self.route = load_route(rf)
        self.pp = PurePursuit(self.route, params)
        self.get_logger().info(
            f"route {self.route.name}: {self.route.length:.0f} m, {self.route.n} points "
            f"({rf}); v_max {params.v_max:.2f} m/s")

        self.pose = None
        self.pose_stamp = None
        self.running = bool(self.get_parameter("auto_start").value)
        self.finished = False
        self.last_t = None
        self.t0 = None

        self.cmd_pub = self.create_publisher(Twist, "/rover/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, "/nav/status", 10)
        self.cte_pub = self.create_publisher(Float64, "/nav/cross_track_error", 10)
        self.la_pub = self.create_publisher(Marker, "/viz/lookahead", 10)
        self.create_subscription(Odometry, self.get_parameter("pose_topic").value,
                                 self.on_pose, 20)
        # Rover-drone distance from relative_state, logged alongside so each
        # run shows whether the latched drone stayed seated (NaN if absent).
        self.slant = None
        self.slant_stamp = None
        # Ground truth is always logged next to the pose the follower drives
        # on, so runs on an estimate (EKF) report the TRUE tracking error.
        self.gt = None
        gt_topic = self.get_parameter("gt_topic").value
        self.gt_same = not gt_topic or gt_topic == self.get_parameter("pose_topic").value
        if not self.gt_same:
            self.create_subscription(Odometry, gt_topic, self.on_gt, 20)
        self.last_gt_xy = (float("nan"), float("nan"))
        self.done_future = rclpy.task.Future()
        self.create_subscription(Float32, "/coordination/slant_range", self.on_slant, 10)
        self.create_service(Trigger, "/nav/pause", self.on_pause)
        self.create_service(Trigger, "/nav/resume", self.on_resume)

        os.makedirs(os.path.join(repo, "logs"), exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        tag = self.get_parameter("run_tag").value
        tag = f"_{tag}" if tag else ""
        self.log_path = os.path.join(repo, "logs", f"route_{self.route.name}{tag}_{stamp}.csv")
        self.log_f = open(self.log_path, "w", newline="")
        self.log = csv.writer(self.log_f)
        self.log.writerow(["t", "x", "y", "z", "roll", "pitch", "yaw", "v_meas", "w_meas",
                           "v_cmd", "w_cmd", "state", "s", "remaining", "cte", "alpha",
                           "grade", "kappa", "target_x", "target_y", "slant",
                           "gt_x", "gt_y", "gt_z", "gt_yaw", "true_cte", "loc_err"])
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self.tick)
        self.create_timer(5.0, self.report)
        self.last_cmd = None

    # ------------------------------------------------------------ callbacks
    def on_pose(self, msg):
        self.pose = msg
        self.pose_stamp = self.get_clock().now()

    def on_gt(self, msg):
        self.gt = msg

    def on_slant(self, msg):
        self.slant = float(msg.data)
        self.slant_stamp = self.get_clock().now()

    def slant_now(self, now):
        if self.slant is None or (now - self.slant_stamp).nanoseconds * 1e-9 > 1.0:
            return float("nan")
        return self.slant

    def on_pause(self, req, res):
        self.running = False
        self.publish_stop()
        res.success, res.message = True, "paused"
        return res

    def on_resume(self, req, res):
        self.running = True
        res.success, res.message = True, "resumed"
        return res

    # --------------------------------------------------------------- control
    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def tick(self):
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        if self.pose is None:
            return
        age = (now - self.pose_stamp).nanoseconds * 1e-9
        if age > float(self.get_parameter("pose_timeout").value):
            self.publish_stop()
            self.get_logger().warn("pose stale, stopping", throttle_duration_sec=2.0)
            return
        if self.finished or not self.running:
            self.publish_stop()
            return
        dt = 0.05 if self.last_t is None else max(min(t - self.last_t, 0.2), 1e-3)
        self.last_t = t
        if self.t0 is None:
            self.t0 = t

        p = self.pose.pose.pose
        roll, pitch, yaw = rpy_from_quat(p.orientation)
        c = self.pp.step(p.position.x, p.position.y, yaw, dt)
        self.last_cmd = c

        tw = Twist()
        tw.linear.x, tw.angular.z = c.v, c.omega
        self.cmd_pub.publish(tw)
        self.cte_pub.publish(Float64(data=c.cte))
        self.status_pub.publish(String(data=json.dumps({
            "state": c.state, "route": self.route.name, "s": round(c.s, 2),
            "remaining": round(c.remaining, 2), "cte": round(c.cte, 3),
            "v_cmd": round(c.v, 3), "grade_pct": round(100 * c.grade, 1)})))
        self.publish_lookahead(c, p.position.z)

        tw_meas = self.pose.twist.twist
        g = self.pose if self.gt_same else self.gt
        if g is not None:
            gp = g.pose.pose
            gx, gy, gz = gp.position.x, gp.position.y, gp.position.z
            gyaw = yaw_from_quat(gp.orientation)
            seg = self.pp.seg or 0
            _, _, tcte, _ = self.route.project(gx, gy, max(seg - 30, 0), seg + 60)
            lerr = math.hypot(p.position.x - gx, p.position.y - gy)
        else:
            gx = gy = gz = gyaw = tcte = lerr = float("nan")
        self.log.writerow([f"{t - self.t0:.3f}", f"{p.position.x:.3f}", f"{p.position.y:.3f}",
                           f"{p.position.z:.3f}", f"{roll:.4f}", f"{pitch:.4f}", f"{yaw:.4f}",
                           f"{tw_meas.linear.x:.3f}", f"{tw_meas.angular.z:.3f}",
                           f"{c.v:.3f}", f"{c.omega:.3f}", c.state, f"{c.s:.2f}",
                           f"{c.remaining:.2f}", f"{c.cte:.3f}", f"{c.alpha:.3f}",
                           f"{c.grade:.4f}", f"{c.kappa:.4f}",
                           f"{c.target[0]:.3f}", f"{c.target[1]:.3f}",
                           f"{self.slant_now(now):.4f}",
                           f"{gx:.3f}", f"{gy:.3f}", f"{gz:.3f}", f"{gyaw:.4f}",
                           f"{tcte:.3f}", f"{lerr:.3f}"])
        self.last_gt_xy = (gx, gy)

        tmax = float(self.get_parameter("timeout_s").value)
        if tmax > 0 and t - self.t0 > tmax and c.state != "arrived":
            self.finished = True
            self.publish_stop()
            self.log_f.flush()
            self.get_logger().warn(f"TIMEOUT after {tmax:.0f} s sim time; log {self.log_path}")
            if self.get_parameter("exit_on_arrival").value:
                self.done_future.set_result(False)
            return

        if c.state == "arrived":
            self.finished = True
            self.publish_stop()
            self.log_f.flush()
            gx_, gy_ = self.last_gt_xy
            ex, ey = self.route.p[-1, 0], self.route.p[-1, 1]
            self.get_logger().info(
                f"ARRIVED at {self.route.name} after {t - self.t0:.0f} s sim time; "
                f"true distance to goal {math.hypot(gx_ - ex, gy_ - ey):.2f} m; "
                f"log {self.log_path}")
            if self.get_parameter("exit_on_arrival").value:
                self.done_future.set_result(True)

    def publish_lookahead(self, c, z):
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = "lookahead", 0, Marker.SPHERE, Marker.ADD
        m.pose.position = Point(x=c.target[0], y=c.target[1], z=z)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.85, 0.0, 1.0
        self.la_pub.publish(m)

    def report(self):
        c = self.last_cmd
        if c is not None and not self.finished:
            self.get_logger().info(
                f"{c.state}: {c.s:.0f}/{self.route.length:.0f} m, {c.remaining:.0f} m left, "
                f"cte {c.cte:+.2f} m, v {c.v:.2f} m/s, grade {100 * c.grade:+.1f}%")
        self.log_f.flush()

    def destroy_node(self):
        # On Ctrl+C rclpy may already have shut the context down; publishing
        # then raises, so the final stop is best-effort (the rover's drive
        # plugin also times out on missing commands).
        try:
            if rclpy.ok():
                self.publish_stop()
        except Exception:
            pass
        try:
            self.log_f.close()
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = RouteFollower()
    try:
        rclpy.spin_until_future_complete(node, node.done_future)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
