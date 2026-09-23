#!/usr/bin/env python3
"""
place_turbines.py
=================

Generates a complete wind-farm Gazebo world from real turbine coordinates:
georeferenced terrain, turbines standing on the actual ground at real scale,
and the ground rover positioned at a spawn point on the terrain.

This is the bridge between the terrain pipeline (make_terrain_model.py) and
the autonomy stack. It also writes config/turbines.yaml, which is the single
source of truth for turbine positions that the mission manager, the rover
waypoint follower and the inspection planner all read.

How ground height is found
--------------------------
Each turbine's WGS84 coordinate is projected into the same site-centred
transverse Mercator used by make_terrain_model.py (true north, matching
Gazebo's GPS frame), then its elevation is bilinearly sampled from a DEM
grid built with the identical extraction and void-filling routine (imported
directly from make_terrain_model.py). This guarantees the turbine sits on
the surface Gazebo actually renders, not on a subtly different resample.

World z of a terrain point is (elevation - z_centre), because the terrain
model is included with its centre at world z = 0.

Turbine geometry
----------------
Reuses the tower and blade meshes from gen_turbine_meshes.py, scaled
non-uniformly to real dimensions via SDF <scale>. The meshes were generated
at 12.5 m tower height and 6 m blade length; scaling vertical and radial
axes independently lets tower height, tower girth, blade length and blade
chord each be set without regenerating geometry.

Defaults match a Suzlon S52-600 class machine (52 m rotor, ~75 m hub). This
is inferred from site evidence (turbine spacing of ~3 rotor diameters, and
lattice towers visible in site photographs) rather than from a published
turbine list for the site; override with --hub-height and --rotor-diameter.
The real S52 uses a lattice tower. It is modelled here as a tapered tube,
which is a visual and obstacle-geometry simplification.

Rotors face into a configurable wind direction and each turbine is parked
at a different rotor azimuth, as a real farm at standstill would be.

Inputs
  --turbines    CSV of "lat, lon" pairs, one turbine per line
  --dem         the same DEM GeoTIFF used to build the terrain model
  --lat/--lon   terrain window centre, identical to make_terrain_model.py
  --extent/--grid  identical to make_terrain_model.py
  --rover-src   world file containing <model name="rover">, from which the
                rover model is extracted once into models/r2d_rover/

Outputs
  worlds/<world>.sdf           complete world
  models/r2d_rover/model.sdf   standalone rover model (created if missing)
  config/turbines.yaml         turbine positions for the autonomy stack

Dependencies
  pip install rasterio numpy pillow

Usage
  python3 tools/place_turbines.py \
      --turbines config/attappadi_turbines.csv \
      --dem "$HOME/Downloads/Map data/rasters_COP30/output_hh.tif" \
      --lat 11.0813198 --lon 76.713563 --extent 3000 --grid 513 \
      --terrain-model attappadi --world attappadi_windfarm \
      --rover-src worlds/turbine_site.sdf
"""

import argparse
import math
import os
import re
import sys

import numpy as np
from rasterio.warp import Resampling

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_terrain_model import (extract_window, fill_voids,  # noqa: E402
                                latlon_to_local, site_crs)

# PX4 x500 geometry, from PX4-gazebo-models x500_base/model.sdf. The model
# lifts its base_link 0.24 m above the spawn point, and the landing skids sit
# 0.227 m below base_link. So a drone resting on a surface has its spawn
# origin 0.013 m below that surface and its base_link 0.227 m above it.
X500_MODEL_Z_OFFSET = 0.24
X500_SKID_BELOW_BASE = 0.227
# Spawn with the skids slightly clear of the deck; latch_manager lets the
# drone settle onto the pad before locking it.
DRONE_SPAWN_GAP = 0.04

# Native dimensions of the meshes produced by gen_turbine_meshes.py.
MESH_TOWER_HEIGHT = 12.5
MESH_TOWER_BASE_R = 0.95
MESH_BLADE_LENGTH = 6.0
MESH_BLADE_CHORD = 0.78


def load_rover_dims(repo):
    """
    Rover geometry from config/rover.yaml, written by gen_rover.py.
    Minimal key: value parser so no YAML dependency is needed. Falls back
    to the original small rover's numbers if the file does not exist.
    """
    dims = {"base_height_m": 0.22, "deck_height_m": 0.225,
            "pad_center_x_m": 0.0, "pad_center_y_m": 0.0}
    path = os.path.join(repo, "config", "rover.yaml")
    if not os.path.exists(path):
        print("  config/rover.yaml not found, using legacy rover dimensions")
        return dims
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            k, v = (t.strip() for t in line.split(":", 1))
            try:
                dims[k] = float(v)
            except ValueError:
                pass
    print(f"  rover: base {dims['base_height_m']:.3f} m, deck +"
          f"{dims['deck_height_m']:.3f} m (from config/rover.yaml)")
    return dims


def read_turbines(path):
    """Parse 'lat, lon' lines, skipping blanks and # comments."""
    pts = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = [p for p in re.split(r'[,\s]+', line) if p]
            if len(parts) < 2:
                sys.exit(f"{path}:{ln}: expected 'lat, lon', got {line!r}")
            pts.append((float(parts[0]), float(parts[1])))
    if not pts:
        sys.exit(f"{path}: no turbines found")
    return pts


def bilinear(grid, col, row):
    """Bilinear sample of a 2D array at fractional (col, row)."""
    h, w = grid.shape
    if not (0 <= col <= w - 1 and 0 <= row <= h - 1):
        return None
    c0, r0 = int(math.floor(col)), int(math.floor(row))
    c1, r1 = min(c0 + 1, w - 1), min(r0 + 1, h - 1)
    fc, fr = col - c0, row - r0
    top = grid[r0, c0] * (1 - fc) + grid[r0, c1] * fc
    bot = grid[r1, c0] * (1 - fc) + grid[r1, c1] * fc
    return float(top * (1 - fr) + bot * fr)


def wind_to_yaw(wind_from_deg):
    """
    ENU yaw that points the rotor into the wind.

    The turbine model's rotor faces its local +x. A compass bearing B
    (clockwise from north) corresponds to the ENU direction
    (east = sin B, north = cos B), whose yaw is 90 deg - B.
    """
    return math.radians(90.0 - wind_from_deg)


def turbine_model_sdf(name, x, y, z, yaw, azimuth, hub_h, rotor_d):
    """One static turbine, fully specified inline."""
    blade_len = rotor_d / 2.0 - 1.3          # rotor radius minus hub radius
    blade_chord = rotor_d * 0.042            # ~2.2 m max chord for 52 m
    tower_top = hub_h - 1.6                  # nacelle centre sits on top

    tz = tower_top / MESH_TOWER_HEIGHT
    tr = (hub_h * 0.034) / MESH_TOWER_BASE_R  # ~2.5 m base radius at 75 m
    bz = blade_len / MESH_BLADE_LENGTH
    bc = blade_chord / MESH_BLADE_CHORD

    nac_len = rotor_d * 0.13
    nac_r = rotor_d * 0.027
    hub_r = 1.3
    hub_x = nac_len * 0.5 + 0.4

    blades = []
    for k in range(3):
        roll = azimuth + k * 2.0 * math.pi / 3.0
        blades.append(f"""
      <link name="blade_{k + 1}">
        <pose relative_to="hub">0 0 0 {roll:.5f} 0 0</pose>
        <visual name="visual">
          <geometry><mesh>
            <uri>model://turbine/meshes/blade.obj</uri>
            <scale>{bc:.4f} {bc:.4f} {bz:.4f}</scale>
          </mesh></geometry>
          <material><ambient>0.84 0.84 0.85 1</ambient><diffuse>0.96 0.96 0.97 1</diffuse><specular>0.4 0.4 0.4 1</specular></material>
        </visual>
        <collision name="collision">
          <pose>0 0 {blade_len / 2 + 0.3:.3f} 0 0 0</pose>
          <geometry><box><size>{blade_chord * 0.35:.3f} {blade_chord * 0.8:.3f} {blade_len:.3f}</size></box></geometry>
        </collision>
      </link>""")

    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.5f}</pose>

      <link name="tower">
        <visual name="visual">
          <geometry><mesh>
            <uri>model://turbine/meshes/tower.obj</uri>
            <scale>{tr:.4f} {tr:.4f} {tz:.4f}</scale>
          </mesh></geometry>
          <material><ambient>0.80 0.80 0.82 1</ambient><diffuse>0.93 0.93 0.94 1</diffuse><specular>0.3 0.3 0.3 1</specular></material>
        </visual>
        <collision name="collision">
          <pose>0 0 {tower_top / 2:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>{hub_h * 0.025:.3f}</radius><length>{tower_top:.3f}</length></cylinder></geometry>
        </collision>
      </link>

      <link name="nacelle">
        <pose>0 0 {hub_h:.3f} 0 0 0</pose>
        <visual name="body">
          <pose>{-nac_len * 0.15:.3f} 0 0 0 1.5708 0</pose>
          <geometry><cylinder><radius>{nac_r:.3f}</radius><length>{nac_len:.3f}</length></cylinder></geometry>
          <material><ambient>0.82 0.82 0.84 1</ambient><diffuse>0.93 0.93 0.95 1</diffuse></material>
        </visual>
        <visual name="rear">
          <pose>{-nac_len * 0.65:.3f} 0 0 0 0 0</pose>
          <geometry><sphere><radius>{nac_r:.3f}</radius></sphere></geometry>
          <material><ambient>0.82 0.82 0.84 1</ambient><diffuse>0.93 0.93 0.95 1</diffuse></material>
        </visual>
        <collision name="collision">
          <pose>{-nac_len * 0.15:.3f} 0 0 0 1.5708 0</pose>
          <geometry><cylinder><radius>{nac_r:.3f}</radius><length>{nac_len:.3f}</length></cylinder></geometry>
        </collision>
      </link>

      <link name="hub">
        <pose>{hub_x:.3f} 0 {hub_h:.3f} 0 0 0</pose>
        <visual name="spinner">
          <pose>0.3 0 0 0 1.5708 0</pose>
          <geometry><cylinder><radius>{hub_r:.3f}</radius><length>1.6</length></cylinder></geometry>
          <material><ambient>0.85 0.85 0.86 1</ambient><diffuse>0.95 0.95 0.96 1</diffuse></material>
        </visual>
        <visual name="nose">
          <pose>1.1 0 0 0 0 0</pose>
          <geometry><sphere><radius>{hub_r:.3f}</radius></sphere></geometry>
          <material><ambient>0.85 0.85 0.86 1</ambient><diffuse>0.95 0.95 0.96 1</diffuse></material>
        </visual>
        <collision name="collision">
          <geometry><sphere><radius>{hub_r:.3f}</radius></sphere></geometry>
        </collision>
      </link>
{''.join(blades)}
    </model>"""


def extract_rover(src_world, dst_dir):
    """
    Pull <model name="rover">...</model> out of an existing world into a
    standalone model, dropping its world pose so <include> controls it.
    Only runs if the standalone model does not already exist.
    """
    dst = os.path.join(dst_dir, "model.sdf")
    if os.path.exists(dst):
        print(f"  rover model already exists: {dst}")
        return
    text = open(src_world).read()
    m = re.search(r'<model name="rover">.*?\n    </model>', text, re.S)
    if not m:
        sys.exit(f"Could not find <model name=\"rover\"> in {src_world}")
    block = m.group(0)
    block = re.sub(r'\n\s*<pose>[^<]*</pose>', '', block, count=1)
    os.makedirs(dst_dir, exist_ok=True)
    with open(dst, "w") as f:
        f.write('<?xml version="1.0" ?>\n'
                '<!--\n'
                '  r2d_rover/model.sdf\n'
                '  Four-wheel skid-steer ground rover with a landing deck,\n'
                '  extracted from worlds/turbine_site.sdf by place_turbines.py.\n'
                '  Deck surface is 0.225 m above the model origin.\n'
                '  Topics: /rover/cmd_vel (in), /rover/odometry (out).\n'
                '-->\n<sdf version="1.9">\n' + block + '\n</sdf>\n')
    with open(os.path.join(dst_dir, "model.config"), "w") as f:
        f.write('<?xml version="1.0"?>\n<model>\n  <name>r2d_rover</name>\n'
                '  <version>1.0</version>\n  <sdf version="1.9">model.sdf</sdf>\n'
                '  <description>Rover2Drone skid-steer ground rover with landing '
                'deck.</description>\n</model>\n')
    print(f"  extracted rover model to {dst}")


WORLD = """<?xml version="1.0" ?>
<!--
  {world}.sdf
  Generated by tools/place_turbines.py. Do not edit by hand; regenerate.

  Real-terrain wind farm world. Terrain centre {lat:.6f} N, {lon:.6f} E,
  {extent:.0f} m square. Ground level at the terrain centre is world z = 0.
  {n} turbines at surveyed positions, {hub_h:.0f} m hub, {rotor_d:.0f} m rotor,
  rotors facing wind from {wind:.0f} deg.

  The rover spawns at ({sx:.1f}, {sy:.1f}). PX4 attaches in standalone mode
  and spawns the drone itself; see the command printed by place_turbines.py.
-->
<sdf version="1.9">
  <world name="{world}">

    <physics name="4ms" type="ignored">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer"/>
    <plugin filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <gui fullscreen="0">
      <plugin filename="MinimalScene" name="3D View">
        <gz-gui>
          <title>3D View</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="string" key="state">docked</property>
        </gz-gui>
        <engine>ogre2</engine>
        <scene>scene</scene>
        <ambient_light>0.7 0.7 0.72</ambient_light>
        <background_color>0.75 0.84 0.93</background_color>
        <camera_pose>{cam_x:.1f} {cam_y:.1f} 180 0 0.42 {cam_yaw:.3f}</camera_pose>
        <camera_clip><near>0.5</near><far>20000</far></camera_clip>
        <sky>true</sky>
      </plugin>
      <plugin filename="GzSceneManager" name="Scene Manager">
        <gz-gui>
          <property key="resizable" type="bool">false</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="state" type="string">floating</property>
          <property key="showTitleBar" type="bool">false</property>
        </gz-gui>
      </plugin>
      <plugin filename="InteractiveViewControl" name="Interactive view control">
        <gz-gui>
          <property key="resizable" type="bool">false</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="state" type="string">floating</property>
          <property key="showTitleBar" type="bool">false</property>
        </gz-gui>
      </plugin>
      <plugin filename="CameraTracking" name="Camera Tracking">
        <gz-gui>
          <property key="resizable" type="bool">false</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="state" type="string">floating</property>
          <property key="showTitleBar" type="bool">false</property>
        </gz-gui>
      </plugin>
      <plugin filename="EntityContextMenuPlugin" name="Entity context menu">
        <gz-gui>
          <property key="resizable" type="bool">false</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="state" type="string">floating</property>
          <property key="showTitleBar" type="bool">false</property>
        </gz-gui>
      </plugin>
      <plugin filename="WorldControl" name="World control">
        <gz-gui>
          <property type="bool" key="resizable">false</property>
          <property type="double" key="height">72</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View"><line own="left" target="left"/><line own="bottom" target="bottom"/></anchors>
        </gz-gui>
        <play_pause>true</play_pause>
        <step>true</step>
        <start_paused>true</start_paused>
      </plugin>
      <plugin filename="WorldStats" name="World stats">
        <gz-gui>
          <property type="bool" key="resizable">false</property>
          <property type="double" key="height">110</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View"><line own="right" target="right"/><line own="bottom" target="bottom"/></anchors>
        </gz-gui>
        <sim_time>true</sim_time>
        <real_time>true</real_time>
        <real_time_factor>true</real_time_factor>
      </plugin>
      <plugin filename="EntityTree" name="Entity tree"/>
      <plugin filename="ComponentInspector" name="Component inspector"/>
    </gui>

    <scene>
      <ambient>0.7 0.7 0.72 1</ambient>
      <background>0.75 0.84 0.93 1</background>
      <sky/>
    </scene>

    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>{lat:.7f}</latitude_deg>
      <longitude_deg>{lon:.7f}</longitude_deg>
      <elevation>{z_centre:.1f}</elevation>
    </spherical_coordinates>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 800 0 0 0</pose>
      <diffuse>0.95 0.95 0.92 1</diffuse>
      <specular>0.25 0.25 0.25 1</specular>
      <direction>-0.5 0.35 -0.78</direction>
    </light>

    <!-- Terrain positions itself via its heightmap <pos>; Gazebo ignores
         include poses for heightmap geometry, so this stays at the origin. -->
    <include>
      <uri>model://{terrain}</uri>
      <pose>0 0 0 0 0 0</pose>
    </include>

{foliage}
    <include>
      <uri>model://r2d_rover</uri>
      <name>rover</name>
      <pose>{sx:.3f} {sy:.3f} {sz:.3f} 0 0 {syaw:.5f}</pose>
    </include>
{turbines}
  </world>
</sdf>
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turbines", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--extent", type=float, default=3000.0)
    ap.add_argument("--grid", type=int, default=513)
    ap.add_argument("--terrain-model", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--rover-src", default="worlds/turbine_site.sdf")
    ap.add_argument("--hub-height", type=float, default=75.0)
    ap.add_argument("--rotor-diameter", type=float, default=52.0)
    ap.add_argument("--wind-from", type=float, default=250.0,
                    help="Compass bearing the wind blows FROM, degrees")
    ap.add_argument("--spawn-lat", type=float,
                    help="Rover spawn latitude (default: terrain centre)")
    ap.add_argument("--spawn-lon", type=float)
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    turbines = read_turbines(os.path.join(repo, args.turbines)
                             if not os.path.isabs(args.turbines) else args.turbines)

    crs = site_crs(args.lat, args.lon)
    half = args.extent / 2.0
    res = args.extent / (args.grid - 1)

    print("Rebuilding DEM grid (identical to make_terrain_model.py)...")
    dem = extract_window(os.path.expanduser(args.dem), args.lat, args.lon,
                         args.extent, args.grid, crs,
                         Resampling.bilinear, band_count=1)[0]
    dem = fill_voids(dem)
    zmin = float(dem.min())
    z_centre = float(dem[args.grid // 2, args.grid // 2])
    z_offset = -(z_centre - zmin)
    cx, cy = latlon_to_local(args.lat, args.lon, crs)

    def ground(lat, lon):
        """World (x, y, z) of the terrain surface at a WGS84 point."""
        ux, uy = latlon_to_local(lat, lon, crs)
        e, n = ux - cx, uy - cy
        # Row 0 of the grid is the north edge (see from_origin in
        # make_terrain_model.py), so row increases southward.
        col = (e + half) / res
        row = (half - n) / res
        elev = bilinear(dem, col, row)
        if elev is None:
            return e, n, None
        return e, n, elev - z_centre

    # Sample the exact surface Gazebo renders, from the terrain metadata.
    from terrain_io import Terrain
    terrain = Terrain(os.path.join(repo, "models", args.terrain_model))
    z_centre = terrain.elev0

    def ground(lat, lon):
        x, y = terrain.latlon_to_world(lat, lon)
        return x, y, terrain.ground_z(x, y, "visual")

    yaw = wind_to_yaw(args.wind_from)
    blocks, yaml_rows = [], []
    print(f"\n{'id':<4}{'lat':>12}{'lon':>12}{'x':>9}{'y':>9}"
          f"{'ground z':>10}{'hub z':>9}")
    for i, (la, lo) in enumerate(turbines, 1):
        x, y, z = ground(la, lo)
        if z is None:
            print(f"T{i:<3}{la:>12.6f}{lo:>12.6f}  OUTSIDE TERRAIN, skipped")
            continue
        azimuth = math.radians((i * 37) % 120)
        name = f"turbine_{i:02d}"
        blocks.append(turbine_model_sdf(name, x, y, z, yaw, azimuth,
                                        args.hub_height, args.rotor_diameter))
        hub_z = z + args.hub_height
        yaml_rows.append((name, la, lo, x, y, z, hub_z))
        print(f"T{i:<3}{la:>12.6f}{lo:>12.6f}{x:>9.1f}{y:>9.1f}"
              f"{z:>10.2f}{hub_z:>9.2f}")

    if not yaml_rows:
        sys.exit("No turbines fell inside the terrain window.")

    slat = args.spawn_lat if args.spawn_lat is not None else args.lat
    slon = args.spawn_lon if args.spawn_lon is not None else args.lon
    sx, sy, sz_ground = ground(slat, slon)
    if sz_ground is None:
        sys.exit("Rover spawn point is outside the terrain window.")
    # Spawn the rover slightly above its resting height and let it settle
    # onto the heightmap.
    rover = load_rover_dims(repo)
    base_h, deck_h = rover["base_height_m"], rover["deck_height_m"]
    sz = sz_ground + base_h + 0.08
    # Point the rover at the nearest turbine.
    near = min(yaml_rows, key=lambda r: math.hypot(r[3] - sx, r[4] - sy))
    syaw = math.atan2(near[4] - sy, near[3] - sx)

    extract_rover(os.path.join(repo, args.rover_src),
                  os.path.join(repo, "models", "r2d_rover"))

    # Frame the GUI camera on the farm from the southwest, looking in.
    fx = sum(r[3] for r in yaml_rows) / len(yaml_rows)
    fy = sum(r[4] for r in yaml_rows) / len(yaml_rows)
    cam_x, cam_y = fx - 450.0, fy - 450.0
    cam_yaw = math.atan2(fy - cam_y, fx - cam_x)

    # Optional scenery generated by gen_foliage.py / gen_buildings.py.
    foliage_xml = ""
    for extra in ("foliage", "buildings"):
        mname = f"{args.terrain_model}_{extra}"
        if os.path.isdir(os.path.join(repo, "models", mname)):
            foliage_xml += (f"    <include>\n      <uri>model://{mname}</uri>\n"
                            f"      <pose>0 0 0 0 0 0</pose>\n    </include>\n")
            print(f"  including {extra} model {mname}")

    world_path = os.path.join(repo, "worlds", f"{args.world}.sdf")
    with open(world_path, "w") as f:
        f.write(WORLD.format(
            world=args.world, lat=args.lat, lon=args.lon, extent=args.extent,
            n=len(yaml_rows), hub_h=args.hub_height, rotor_d=args.rotor_diameter,
            wind=args.wind_from, z_centre=z_centre, z_offset=z_offset,
            terrain=args.terrain_model, sx=sx, sy=sy, sz=sz, syaw=syaw,
            cam_x=cam_x, cam_y=cam_y, cam_yaw=cam_yaw,
            turbines="".join(blocks), foliage=foliage_xml))
    print(f"\nWrote {world_path}")

    cfg_dir = os.path.join(repo, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    yaml_path = os.path.join(cfg_dir, "turbines.yaml")
    with open(yaml_path, "w") as f:
        f.write("# turbines.yaml\n"
                "# Generated by tools/place_turbines.py. Do not edit by hand.\n"
                "# World-frame (ENU) positions of every turbine, for the mission\n"
                "# manager, rover waypoint follower and inspection planner.\n"
                "# ground_z is the terrain surface at the tower base; hub_z is\n"
                "# the rotor centre. Both in world metres.\n"
                f"world: {args.world}\n"
                f"origin: {{lat: {args.lat}, lon: {args.lon}, "
                f"elevation_m: {z_centre:.2f}}}\n"
                f"hub_height_m: {args.hub_height}\n"
                f"rotor_diameter_m: {args.rotor_diameter}\n"
                f"rotor_yaw_rad: {yaw:.5f}\n"
                "turbines:\n")
        for name, la, lo, x, y, z, hz in yaml_rows:
            f.write(f"  - {{id: {name}, lat: {la}, lon: {lo}, "
                    f"x: {x:.3f}, y: {y:.3f}, ground_z: {z:.3f}, "
                    f"hub_z: {hz:.3f}}}\n")
    print(f"Wrote {yaml_path}")

    deck_z = sz_ground + base_h + deck_h
    # Pad centre in world: rover-frame offset rotated by the rover heading.
    px = sx + rover["pad_center_x_m"] * math.cos(syaw) \
        - rover["pad_center_y_m"] * math.sin(syaw)
    py = sy + rover["pad_center_x_m"] * math.sin(syaw) \
        + rover["pad_center_y_m"] * math.cos(syaw)
    dz = deck_z - X500_MODEL_Z_OFFSET + X500_SKID_BELOW_BASE + DRONE_SPAWN_GAP
    # Where the drone's base_link (and so PX4's EKF origin) rests on the pad.
    drone_rest_z = deck_z + X500_SKID_BELOW_BASE
    print(f"""
Rover spawn:   ({sx:.2f}, {sy:.2f}), ground z {sz_ground:.2f}, heading to {near[0]}
Nearest turbine is {math.hypot(near[3] - sx, near[4] - sy):.0f} m away.

PX4 (drone on the rover deck):
  cd ~/PX4-Autopilot
  PX4_GZ_STANDALONE=1 PX4_GZ_WORLD={args.world} \\
  PX4_SIM_MODEL=gz_x500_gimbal \\
  PX4_GZ_MODEL_POSE="{px:.3f},{py:.3f},{dz:.3f},0,0,{syaw:.4f}" \\
  ./build/px4_sitl_default/bin/px4

relative_state origins for this world:
  ros2 run rover2drone_coordination relative_state --ros-args \\
    -p use_sim_time:=true \\
    -p rover_origin_x:={sx:.3f} -p rover_origin_y:={sy:.3f} -p rover_origin_z:={sz_ground + base_h:.3f} \\
    -p drone_origin_x:={px:.3f} -p drone_origin_y:={py:.3f} -p drone_origin_z:={drone_rest_z:.3f}
""")


if __name__ == "__main__":
    main()
