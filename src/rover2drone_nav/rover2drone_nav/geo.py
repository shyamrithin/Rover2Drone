# =============================================================================
# File:        src/rover2drone_nav/rover2drone_nav/geo.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-25
# Depends:     python3 standard library
# =============================================================================
"""
geo.py
======

Conversion between the Gazebo world frame (ENU metres about the site
origin) and WGS84 latitude / longitude / ellipsoid-free altitude, using a
local tangent-plane approximation with the WGS84 meridional (M) and prime
vertical (N) radii of curvature at the origin latitude.

Over the 3 km site the approximation error is a few millimetres, far below
any GNSS error modelled here. The same functions are used in both
directions (sensor simulation and EKF), so they are exactly consistent.

The origin is read from config/turbines.yaml ('origin: {lat, lon,
elevation_m}'), written by tools/place_turbines.py.
"""

import math

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


class LocalFrame:
    def __init__(self, lat0, lon0, alt0=0.0):
        self.lat0, self.lon0, self.alt0 = lat0, lon0, alt0
        phi = math.radians(lat0)
        s2 = math.sin(phi) ** 2
        self.N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * s2)
        self.M = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * s2) ** 1.5
        self.cos0 = math.cos(phi)

    def to_lla(self, e, n, u):
        lat = self.lat0 + math.degrees(n / self.M)
        lon = self.lon0 + math.degrees(e / (self.N * self.cos0))
        return lat, lon, self.alt0 + u

    def to_enu(self, lat, lon, alt):
        n = math.radians(lat - self.lat0) * self.M
        e = math.radians(lon - self.lon0) * self.N * self.cos0
        return e, n, alt - self.alt0


def read_origin(turbines_yaml):
    """(lat, lon, elevation_m) from the 'origin: {...}' line of turbines.yaml."""
    with open(turbines_yaml) as f:
        for line in f:
            if line.startswith("origin:"):
                body = line.split("{", 1)[1].rsplit("}", 1)[0]
                kv = dict(p.split(":") for p in body.split(","))
                kv = {k.strip(): float(v) for k, v in kv.items()}
                return kv["lat"], kv["lon"], kv.get("elevation_m", 0.0)
    raise ValueError(f"no 'origin:' line in {turbines_yaml}")
