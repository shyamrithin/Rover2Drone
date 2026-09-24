# =============================================================================
# File:        src/rover2drone_nav/rover2drone_nav/route_io.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-24
# Depends:     numpy (no ROS; also imported by tools/ scripts)
# =============================================================================
"""
route_io.py
===========

Loaders for the planning data the navigation nodes share, with no ROS
dependency so the tools/ scripts and offline tests can import them too:

  load_route(path)         config/routes/route_<turbine>.json -> Route
  load_roads(path)         config/roads.json -> list of (id, type, (N, 3))
  load_turbines(path)      config/turbines.yaml -> list of dicts
  read_rover_yaml(path)    config/rover.yaml -> dict of numeric fields
  read_yaml_scalars(path)  top-level numeric fields of a simple YAML file

Route geometry is kept in the world ENU frame (Gazebo world frame, metres),
the same frame the ground-truth pose is published in.
"""

import json
import math
import os

import numpy as np

DEFAULT_REPO = os.path.expanduser("~/Rover2Drone")


class Route:
    """A planned route as arrays, with arc-length interpolation helpers."""

    def __init__(self, xyz, name="route", summary=None):
        self.name = name
        self.summary = summary or {}
        pts = np.asarray(xyz, dtype=float)
        # Drop zero-length steps (junction stubs can repeat a point).
        keep = [0]
        for i in range(1, len(pts)):
            if np.hypot(*(pts[i, :2] - pts[keep[-1], :2])) > 1e-3:
                keep.append(i)
        self.p = pts[keep]
        d = np.diff(self.p[:, :2], axis=0)
        self.seg_len = np.hypot(d[:, 0], d[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.s[-1])
        dz = np.diff(self.p[:, 2])
        g = dz / np.maximum(self.seg_len, 1e-6)
        # Short stubs carry artefact grades; inherit the previous segment's.
        for i in np.nonzero(self.seg_len < 3.0)[0]:
            g[i] = g[i - 1] if i > 0 else 0.0
        self.seg_grade = g
        self.seg_yaw = np.arctan2(d[:, 1], d[:, 0])

    @property
    def n(self):
        return len(self.p)

    def _seg_at(self, s):
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        return min(max(i, 0), len(self.seg_len) - 1)

    def point_at(self, s):
        """(x, y, z, yaw) at arc length s (clamped to the route)."""
        s = min(max(s, 0.0), self.length)
        i = self._seg_at(s)
        t = (s - self.s[i]) / max(self.seg_len[i], 1e-9)
        q = self.p[i] + (self.p[i + 1] - self.p[i]) * t
        return float(q[0]), float(q[1]), float(q[2]), float(self.seg_yaw[i])

    def grade_at(self, s):
        return float(self.seg_grade[self._seg_at(min(max(s, 0.0), self.length))])

    def project(self, x, y, i0=0, i1=None):
        """
        Closest point on segments i0..i1-1 to (x, y).
        Returns (seg index, s, signed cross-track error [left +], distance).
        """
        i1 = len(self.seg_len) if i1 is None else min(i1, len(self.seg_len))
        i0 = max(0, min(i0, i1 - 1))
        a = self.p[i0:i1, :2]
        b = self.p[i0 + 1:i1 + 1, :2]
        ab = b - a
        L2 = np.maximum((ab * ab).sum(1), 1e-12)
        t = np.clip(((np.array([x, y]) - a) * ab).sum(1) / L2, 0.0, 1.0)
        q = a + ab * t[:, None]
        d = np.hypot(q[:, 0] - x, q[:, 1] - y)
        k = int(d.argmin())
        seg = i0 + k
        cross = ab[k, 0] * (y - a[k, 1]) - ab[k, 1] * (x - a[k, 0])
        cte = math.copysign(float(d[k]), cross)
        s = float(self.s[seg] + t[k] * self.seg_len[seg])
        return seg, s, cte, float(d[k])


def route_path(repo, turbine):
    return os.path.join(repo, "config", "routes", f"route_{turbine}.json")


def load_route(path):
    with open(path) as f:
        d = json.load(f)
    xyz = [(w["x"], w["y"], w["z"]) for w in d["waypoints"]]
    name = d.get("summary", {}).get("turbine", os.path.basename(path))
    return Route(xyz, name=name, summary=d.get("summary", {}))


def load_roads(path):
    with open(path) as f:
        d = json.load(f)
    return [(r["id"], r.get("type", ""), np.asarray(r["points"], dtype=float))
            for r in d["roads"]]


def load_turbines(path):
    """Parse the flow-style turbine lines of turbines.yaml (no PyYAML)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("- {"):
                continue
            rec = {}
            for kv in line[3:].rstrip("}").split(","):
                k, v = kv.split(":", 1)
                k, v = k.strip(), v.strip()
                try:
                    rec[k] = float(v)
                except ValueError:
                    rec[k] = v
            out.append(rec)
    return out


def read_yaml_scalars(path):
    """Top-level 'key: number' lines of a simple YAML file (no PyYAML)."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                if line[:1] in (" ", "-", "#") or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                try:
                    out[k.strip()] = float(v.split("#")[0])
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def read_rover_yaml(path):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                if ":" not in line or line.startswith("#"):
                    continue
                k, v = line.split(":", 1)
                try:
                    out[k.strip()] = float(v.split("#")[0])
                except ValueError:
                    pass
    except OSError:
        pass
    return out
