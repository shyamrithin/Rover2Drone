# Rover2Drone: project handoff

Marsupial UGV–UAV system for wind turbine inspection, simulated at the Suzlon
wind farm, Attappadi (Palakkad, Kerala). Repo: `github.com/shyamrithin/Rover2Drone`.
State as of 2026-09-23.

---

## 1. What the project is

A four-wheel ground rover carries a PX4 quadrotor on a latched landing deck with an
inductive charging pad. The rover drives the site's access roads to a turbine, the
drone launches, inspects the tower and blades, returns, lands, is re-latched and
recharged. The rover is the mobile base station and energy reservoir.

**Research direction.** Energy-aware marsupial planning: one drone flight cannot
cover all seven turbines, so the rover must decide where along the road network to
stop so that flights are short and recharge downtime is minimised, while keeping a
reliable rover–drone link. The terrain finding that motivates this: turbines sit
35–108 m above the rover's start at 12–16% average grade, so the rover is confined
to access roads and the drone covers the remaining climb and the 75 m tower.

---

## 2. Machine and software

Native Ubuntu 22.04 (Acer Nitro ANV15-41, RTX 4050 6 GB).

| Component | Version / source | Notes |
|---|---|---|
| ROS 2 | Humble (apt) | |
| Gazebo | Harmonic 8.14 (apt, `apt-mark hold` on all `gz-*`, `libgz-*`) | |
| ros_gz | `ros-humble-ros-gzharmonic` (apt binary) | Not built from source on this machine |
| PX4 | `~/PX4-Autopilot`, tag `v1.17.0` | `make px4_sitl` built |
| px4_msgs | `release/1.17`, via `setup/deps.repos` + vcstool | |
| Micro-XRCE-DDS Agent | v2.4.3 built from source, `/usr/local/bin/MicroXRCEAgent` | |
| RMW | `rmw_fastrtps_cpp`, set in `setup/env.local.sh` | `.bashrc` sets Cyclone for other projects |
| Python tools | `/usr/bin/python3` + `pip install --user rasterio pillow requests` | Do not install into conda base |

Environment is loaded by `~/.bashrc` → `source ~/Rover2Drone/setup/env.sh` (must be the
last line, after the Cyclone exports). Every shell prints
`[env] ROS_DOMAIN_ID=42 GZ_VERSION=harmonic PX4_DIR=...`.

---

## 3. Repository layout

```
Rover2Drone/
├── setup/            env.sh, env.local.sh (untracked), deps.repos, px4_sitl_params.txt
├── tools/            generators and validators (all runnable with /usr/bin/python3)
│   ├── terrain_io.py            shared terrain access; every tool reads geometry here
│   ├── fetch_terrain.py         download imagery (Esri) + reference DEM (AWS), no key
│   ├── make_terrain_model.py    DEM + imagery -> Gazebo terrain model + terrain.yaml
│   ├── place_turbines.py        turbines + rover + foliage -> worlds/<world>.sdf
│   ├── gen_rover.py             rover model + config/rover.yaml
│   ├── gen_foliage.py           trees/shrubs from imagery vegetation index
│   ├── gen_turbine_meshes.py    NACA blade + tapered tower OBJ meshes
│   ├── validate_environment.py  offline validation report
│   └── validate_live.py         live GPS probe test against running Gazebo
├── models/           attappadi (terrain), attappadi_foliage, r2d_rover, turbine
├── worlds/           attappadi_windfarm.sdf (main), turbine_site.sdf (old flat demo)
├── config/           attappadi_turbines.csv (input), turbines.yaml, rover.yaml (generated)
├── src/
│   ├── rover2drone_bringup/       launch/demo.launch.py
│   └── rover2drone_coordination/  relative_state.py, latch_manager.py
└── docs/             versions.md, validation/ (generated reports)
```

`src/ros_gz` and `src/px4_msgs` are vcstool-managed and gitignored. Meshes and PNGs
are in Git LFS (`*.obj *.dae *.stl *.png`).

---

## 4. Data sources

| Data | Source | Local path |
|---|---|---|
| Elevation (primary) | Copernicus GLO-30 via OpenTopography | `~/Downloads/Map data/rasters_COP30/output_hh.tif` |
| Elevation (reference) | AWS Terrain Tiles (SRTM-derived), via `fetch_terrain.py` | `~/Downloads/attappadi_dem.tif` |
| Imagery | Esri World Imagery tiles, via `fetch_terrain.py` | `~/Downloads/attappadi_rgb.tif` |
| Turbine positions | Read from satellite imagery | `config/attappadi_turbines.csv` |

Site centre (Wind Farm View Point): **11.0813198 N, 76.713563 E**, 852.7 m (EGM2008).
Terrain window 3000 m square.

Copernicus GLO-30 spec: surface model (canopy/buildings included), EGM2008 geoid,
absolute vertical < 4 m LE90, relative < 2 m (slope ≤ 20%) / < 4 m (> 20%),
horizontal < 6 m CE90. Citation: European Space Agency (2024), *Copernicus Global
Digital Elevation Model*, distributed by OpenTopography, doi:10.5069/G9028PQB.

---

## 5. Rebuilding the world (full pipeline)

Run from `~/Rover2Drone`. Steps 3–6 must run in this order.

```bash
# 1. Imagery at zoom 18 (~0.6 m/px, ~440 tiles) and the reference DEM.
#    If the texture shows grey "map data not available" squares, rerun with --img-zoom 17.
/usr/bin/python3 tools/fetch_terrain.py --lat 11.0813198 --lon 76.713563 \
  --extent 3000 --name attappadi --out ~/Downloads --img-zoom 18

# 2. Terrain: visual grid 2049 (1.46 m), collision 1025 (2.93 m), texture 4096 px.
/usr/bin/python3 tools/make_terrain_model.py \
  --dem "$HOME/Downloads/Map data/rasters_COP30/output_hh.tif" \
  --texture ~/Downloads/attappadi_rgb.tif \
  --lat 11.0813198 --lon 76.713563 --extent 3000 \
  --name attappadi --out models

# 3. Rover model (only if gen_rover.py changed).
/usr/bin/python3 tools/gen_rover.py

# 4. World, first pass (writes config/turbines.yaml, which foliage needs).
/usr/bin/python3 tools/place_turbines.py \
  --turbines config/attappadi_turbines.csv \
  --dem "$HOME/Downloads/Map data/rasters_COP30/output_hh.tif" \
  --lat 11.0813198 --lon 76.713563 --extent 3000 --grid 513 \
  --terrain-model attappadi --world attappadi_windfarm \
  --rover-src worlds/turbine_site.sdf

# 5. Foliage.
/usr/bin/python3 tools/gen_foliage.py --terrain-model attappadi

# 6. World again, now including foliage. Note the PX4 and relative_state
#    commands it prints.
#    (same command as step 4)
```

If real-time factor drops below ~0.8: `--visual-grid 1025 --collision-grid 513` on
step 2, and/or `--max-trees 2000 --max-shrubs 4000` on step 5.

---

## 6. Running the simulation

Clean up first, every time:

```bash
simclean; pkill -f MicroXRCEAgent; pkill -f px4
```

Five terminals, in order:

```bash
# 1 Agent
MicroXRCEAgent udp4 -p 8888

# 2 Gazebo + bridges (camera, rover drive, latch, clock) + camera viewer
cd ~/Rover2Drone && source install/setup.bash
ros2 launch rover2drone_bringup demo.launch.py

# 3 PX4 (use the pose printed by place_turbines.py; this is the current one)
cd ~/PX4-Autopilot
PX4_GZ_STANDALONE=1 PX4_GZ_WORLD=attappadi_windfarm \
PX4_SIM_MODEL=gz_x500_gimbal \
PX4_GZ_MODEL_POSE="-0.022,0.033,0.647,0,0,-0.9857" \
./build/px4_sitl_default/bin/px4

# 4 Rover–drone relative state (origins printed by place_turbines.py)
cd ~/Rover2Drone && source install/setup.bash
ros2 run rover2drone_coordination relative_state --ros-args -p use_sim_time:=true \
  -p rover_origin_x:=0.000 -p rover_origin_y:=0.000 -p rover_origin_z:=0.460 \
  -p drone_origin_x:=-0.022 -p drone_origin_y:=0.033 -p drone_origin_z:=0.847

# 5 Latch manager
cd ~/Rover2Drone && source install/setup.bash
ros2 run rover2drone_coordination latch_manager --ros-args -p use_sim_time:=true
```

PX4 shell, first run on a machine (params persist after `param save`):

```
param set NAV_RCL_ACT 0
param set NAV_DLL_ACT 0
param set COM_RCL_EXCEPT 4
param set MIS_TAKEOFF_ALT 15
param save
```

Useful:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/rover/cmd_vel
ros2 service call /rover/latch/release std_srvs/srv/Trigger
ros2 service call /rover/latch/engage  std_srvs/srv/Trigger
ros2 topic echo /coordination/slant_range
```

In Gazebo, Entity Tree → right-click `rover` → **Move to** to find the rover.

After changing anything in `src/`: `colcon build --packages-select <pkg>` then
`source install/setup.bash`.

---

## 7. Validation

```bash
# Offline report -> docs/validation/report.md + figures
cd ~/Rover2Drone/tools
/usr/bin/python3 validate_environment.py --terrain-model attappadi \
  --ref-dem ~/Downloads/attappadi_dem.tif --repo ..

# Live GPS test: world must be running (launch file running, not paused). No PX4 needed.
/usr/bin/python3 validate_live.py --terrain-model attappadi \
  --world attappadi_windfarm --repo ..
```

| Check | What it shows |
|---|---|
| V1 Georeferencing | World coords vs Gazebo's own GPS conversion across the terrain (0.1 cm in testing) |
| V2 Heightmap fidelity | Rendered heightmaps vs source DEM (≤ one 16-bit step); visual = collision |
| V3 DEM cross-check | Copernicus vs independent SRTM-derived DEM: bias, RMSE, LE90, per turbine |
| V4 Turbine placement | Surveyed vs world coords, elevations, overlay on imagery |
| V5 Layout | Spacing in rotor diameters, supports S52 inference |
| V6 Profiles | Grades from spawn to each turbine |
| Live | Gazebo GPS reading at each turbine vs surveyed coordinates |

Compass: Gazebo Harmonic's magnetometer recomputes the field each step from the
sensor's lat/lon using built-in declination/inclination/strength tables, so the
drone's compass matches the local field PX4's estimator expects. No bias.

---

## 8. Hard-won facts (read before debugging)

- **PX4 v1.17 versions some topics**: `/fmu/out/vehicle_local_position_v1`,
  `vehicle_status_v1`, `home_position_v1`. `vehicle_attitude`, `vehicle_odometry`,
  `vehicle_land_detected` are not. Subscribing to the wrong name fails silently:
  the topic appears in `ros2 topic list` (the subscriber creates it) with publisher
  count 0. Check `ros2 topic info <topic> --verbose` and the agent's writer log.
- **RMW must be Fast DDS** for this project (agent is a Fast DDS app).
- **Gazebo ignores include/model poses for heightmaps.** Vertical offset lives in
  the heightmap `<pos>`.
- **Heightmap vertices sit on the terrain edges.** Resampling must shift by half a
  pixel (fixed in `extract_window(vertex=True)`); textures use area pixels.
- **Projection**: site-centred transverse Mercator scaled by 1 + h/R. UTM zone 43
  is rotated 0.33° from true north here (up to 12 m error at the terrain corners).
- **x500 geometry**: PX4's model lifts `base_link` 0.24 m above the spawn point;
  skids are 0.227 m below `base_link`. Drone model name is `x500_gimbal_0`.
- **Latch**: `DetachableJoint` on the rover auto-attaches when the drone appears;
  `latch_manager` releases, settles 2 s, re-engages. Releases on arm, re-engages on
  landing if on the pad (needs `relative_state` running).
- **PX4 `ubuntu.sh` installs CMake 4 via Kitware** which breaks the gz build
  (jsoncpp `cmake_minimum_required < 3.5`). Fix: downgrade to Jammy CMake 3.22 and
  hold it, or `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`. Use `--no-nuttx` on SITL-only machines.
- **QGroundControl** worked poorly under WSL2 mirrored networking; untested on native
  Linux. Not needed: flight from `pxh>` or ROS 2.
- Heredoc pastes in the terminal sometimes truncate. Prefer downloading files or VS Code.

---

## 9. Status

Verified working:
- Full toolchain on native Ubuntu; PX4 takeoff/land (flat world)
- Attappadi terrain with imagery, 7 turbines on their real pads at S52 scale
- New rover (106 kg, charging deck, fiducials), drives via teleop
- Drone spawns on the deck; latch seating sequence (release → 2 s → engage)
- `relative_state` publishing (after the `_v1` fix and rover origin offset)

Written and tested on synthetic data tonight, **not yet run on the real site**:
- Terrain v3 (2049/1025 grids, 4096 texture, half-pixel and scale fixes)
- Foliage generator
- Both validation tools

Implemented, **not yet exercised in the Attappadi world**:
- Latch release on arming and re-engage on landing
- Takeoff from and landing on the moving/sloped rover

---

## 10. Next steps, in order

1. Run the pipeline and both validations on real data; commit the reports.
2. **Access roads**: get centrelines (check OSM first with the Overpass query below,
   otherwise trace from the texture), carve them into the heightmap so they are
   flat and drivable, and paint them into the texture.
3. RViz + TF: turbines from `turbines.yaml`, drone from PX4 (`ref_lat/ref_lon`
   removes the fragile origin parameters in `relative_state`), rover, terrain.
4. Rover waypoint follower (pure pursuit) along the road.
5. Drone offboard control (`OffboardControlMode` + `TrajectorySetpoint`).
6. Mission manager: drive → stop → release → launch → inspect → return → land →
   latch → recharge, with a battery/charging model (charge rate scaled by pad
   alignment).
7. Inspection viewpoint planning around the blades; later, synthetic-defect
   imagery for detection.

Overpass (overpass-turbo.eu) for access roads and mapped turbines:

```
[out:json][timeout:90];
(
  way["highway"](11.066,76.699,11.096,76.728);
  node["power"="generator"]["generator:source"="wind"](11.066,76.699,11.096,76.728);
);
out geom;
```

---

## 11. Known limitations (state these honestly)

- Elevation is 30 m resolution; finer grids interpolate. Road cuts are not in the DEM.
- Copernicus is a surface model (canopy included); foliage on top is decorative.
- Heights are EGM2008 orthometric; real GNSS reports ellipsoidal height.
- Imagery capture date unknown (Esri World Imagery); check licence before publishing
  figures, or substitute Sentinel-2 from the Copernicus Browser.
- Turbine model (Suzlon S52 class, 75 m hub, 52 m rotor) is inferred from spacing and
  lattice towers in site photos, not from a published turbine list. Real S52 has a
  lattice tower; modelled as a tube.
- Rover suspension is rigid (visual brackets only); foliage has no collision.
- Attappadi site has a documented land dispute with tribal communities; relevant if
  naming the site in publications. Kanjikode (KSEB, flat, industrial) is the
  alternative site.
