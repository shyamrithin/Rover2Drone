# =============================================================================
# File:        src/rover2drone_nav/rover2drone_nav/pure_pursuit.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-24
# Depends:     route_io.py (no ROS: unit-testable offline)
# =============================================================================
"""
pure_pursuit.py
===============

Route-tracking controller for the skid-steer rover, independent of ROS so
it can be tested with a kinematic model (test/test_pure_pursuit.py).

Algorithm, each step, given the planar pose (x, y, yaw) in the route frame:

  1. Projection. Closest point on the route, searching only a window ahead
     of the last projection (routes double back on switchbacks, so a global
     search could jump to the wrong leg); the first call searches the whole
     route. Gives arc length s and the signed cross-track error (left +).
  2. Lookahead. Ld = clamp(Ld0 + k * v, Ld_min, Ld_max); the target is the
     route point at s + Ld.
  3. Steering. alpha = bearing of the target in the rover frame; curvature
     kappa = 2 sin(alpha) / d, d = distance to the target. If |alpha|
     exceeds rotate_threshold, turn in place first (skid steer can), until
     it drops below rotate_exit: handles the spawn heading and big errors.
  4. Speed. v = min(v_max,
                    sqrt(a_lat_max / |kappa|)          bend limit,
                    v_max * grade factor               slow on steep road,
                    sqrt(2 a_dec (remaining - stop)))  stop at the goal
     rate-limited by accel_max, floored at v_min while driving;
     omega = v * kappa clamped to omega_max.
  5. Arrival when the remaining arc length drops below goal_tol.
"""

import math
from dataclasses import dataclass, field


@dataclass
class PPParams:
    v_max: float = 0.55
    v_min: float = 0.08
    accel_max: float = 0.4
    omega_max: float = 0.8
    omega_align: float = 0.6
    lookahead_base: float = 1.0
    lookahead_gain: float = 1.5
    lookahead_min: float = 1.0
    lookahead_max: float = 3.0
    a_lat_max: float = 0.25
    a_dec: float = 0.25
    grade_slow_start: float = 0.08
    grade_slow_full: float = 0.20
    grade_speed_factor: float = 0.6
    rotate_threshold_deg: float = 60.0
    rotate_exit_deg: float = 20.0
    goal_tol: float = 1.0
    search_window_m: float = 25.0


@dataclass
class Command:
    v: float = 0.0
    omega: float = 0.0
    state: str = "idle"
    s: float = 0.0
    remaining: float = 0.0
    cte: float = 0.0
    alpha: float = 0.0
    lookahead: float = 0.0
    target: tuple = (0.0, 0.0)
    grade: float = 0.0
    kappa: float = 0.0
    limits: dict = field(default_factory=dict)


class PurePursuit:
    def __init__(self, route, params=None):
        self.route = route
        self.p = params or PPParams()
        self.reset()

    def reset(self):
        self.seg = None
        self.v_prev = 0.0
        self.aligning = False
        self.done = False

    def _window(self):
        spacing = max(self.route.length / max(self.route.n - 1, 1), 0.5)
        n_ahead = int(self.p.search_window_m / spacing) + 2
        return max(self.seg - 3, 0), self.seg + n_ahead

    def step(self, x, y, yaw, dt):
        r, p = self.route, self.p
        if self.seg is None:
            seg, s, cte, _ = r.project(x, y)
        else:
            seg, s, cte, _ = r.project(x, y, *self._window())
        self.seg = seg
        remaining = r.length - s
        cmd = Command(s=s, remaining=remaining, cte=cte, grade=r.grade_at(s))

        if self.done or remaining < p.goal_tol:
            self.done = True
            self.v_prev = 0.0
            cmd.state = "arrived"
            return cmd

        ld = min(max(p.lookahead_base + p.lookahead_gain * self.v_prev,
                     p.lookahead_min), p.lookahead_max)
        tx, ty, _, _ = r.point_at(s + ld)
        dx, dy = tx - x, ty - y
        c, sn = math.cos(yaw), math.sin(yaw)
        xr, yr = c * dx + sn * dy, -sn * dx + c * dy
        dist = max(math.hypot(xr, yr), 1e-3)
        alpha = math.atan2(yr, xr)
        kappa = 2.0 * math.sin(alpha) / dist
        cmd.alpha, cmd.lookahead, cmd.target, cmd.kappa = alpha, ld, (tx, ty), kappa

        a_deg = abs(math.degrees(alpha))
        self.aligning = (a_deg > p.rotate_exit_deg) if self.aligning \
            else (a_deg > p.rotate_threshold_deg)
        if self.aligning:
            self.v_prev = 0.0
            cmd.omega = math.copysign(p.omega_align, alpha)
            cmd.state = "align"
            return cmd

        g = abs(cmd.grade)
        u = min(max((g - p.grade_slow_start)
                    / max(p.grade_slow_full - p.grade_slow_start, 1e-6), 0.0), 1.0)
        v_grade = p.v_max * (1.0 - u * (1.0 - p.grade_speed_factor))
        v_bend = math.sqrt(p.a_lat_max / abs(kappa)) if abs(kappa) > 1e-6 else p.v_max
        v_goal = math.sqrt(2.0 * p.a_dec * max(remaining - 0.5 * p.goal_tol, 0.0))
        v = max(min(p.v_max, v_grade, v_bend, v_goal), p.v_min)
        v = min(v, self.v_prev + p.accel_max * dt)
        self.v_prev = v
        cmd.v = v
        cmd.omega = max(-p.omega_max, min(p.omega_max, v * kappa))
        cmd.state = "track"
        cmd.limits = {"grade": v_grade, "bend": v_bend, "goal": v_goal}
        return cmd
