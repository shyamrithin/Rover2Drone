# =============================================================================
# File:        src/rover2drone_nav/rover2drone_nav/pure_pursuit.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-24
# Updated:     2026-09-24  align: locked turn direction (no rocking at
#              +/-180 deg), forward creep, stuck detection + back-up recovery
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
     exceeds rotate_threshold, turn first (slow forward creep, turn
     direction locked) until it drops below rotate_exit: handles the spawn
     heading and big errors. If the turn makes no progress for stuck_time,
     back up for recover_time and try again.
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
    align_creep_v: float = 0.06      # small forward speed while turning
    stuck_time: float = 8.0          # s aligning without enough rotation...
    stuck_min_rot_deg: float = 15.0  # ...counts as stuck
    recover_v: float = -0.15         # back up at this speed...
    recover_time: float = 2.5        # ...for this long, then realign
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
        self.align_dir = 0.0
        self.align_t = 0.0
        self.align_rot = 0.0
        self.last_yaw = None
        self.recover_left = 0.0
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

        # Rotation actually achieved since the last step (for stuck detection).
        dyaw = 0.0 if self.last_yaw is None else abs(
            math.atan2(math.sin(yaw - self.last_yaw), math.cos(yaw - self.last_yaw)))
        self.last_yaw = yaw

        if self.recover_left > 0.0:
            # Back up briefly, straight, then try aligning again.
            self.recover_left -= dt
            self.v_prev = 0.0
            cmd.v, cmd.state = p.recover_v, "recover"
            return cmd

        a_deg = abs(math.degrees(alpha))
        if not self.aligning and a_deg > p.rotate_threshold_deg:
            # Enter align. Lock the turn direction for the whole manoeuvre:
            # near +/-180 deg the sign of alpha flips with tiny motions, which
            # made the rover rock left-right forever.
            self.aligning = True
            self.align_dir = math.copysign(1.0, alpha)
            self.align_t = self.align_rot = 0.0
        elif self.aligning and a_deg < p.rotate_exit_deg:
            self.aligning = False
        if self.aligning:
            self.align_t += dt
            self.align_rot += dyaw
            if (self.align_t > p.stuck_time
                    and math.degrees(self.align_rot) < p.stuck_min_rot_deg):
                self.aligning = False
                self.recover_left = p.recover_time
                cmd.state = "recover"
                cmd.v = p.recover_v
                return cmd
            # Turn with a slight forward creep: skid steer scrubs less on an
            # arc than when spinning on the spot, especially on slopes.
            self.v_prev = 0.0
            cmd.v = p.align_creep_v
            cmd.omega = self.align_dir * p.omega_align
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
