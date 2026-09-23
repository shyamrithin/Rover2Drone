# Rover2Drone: project handoff

Marsupial UGV–UAV system for wind turbine inspection, simulated on the real terrain
of the Suzlon wind farm at Attappadi (Palakkad, Kerala).
Repo: `github.com/shyamrithin/Rover2Drone`. State as of 2026-09-23 (late).

---

## 1. The project

A four-wheel ground rover carries a PX4 quadrotor on a latched landing deck with an
inductive charging pad. The rover drives the site's access roads to a turbine, the
drone launches, inspects the tower and blades, returns, lands, is re-latched and
recharged. The rover is the mobile base station and energy reservoir.

**Research direction: energy-aware marsupial planning.** One drone flight cannot cover
all seven turbines, so the rover must decide where along the road network to stop so
flights are short and recharge downtime is minimised, while keeping a reliable
rover–drone link. Motivating terrain fact: turbines sit 35–108 m above the rover's
start, with straight-line grades of 12–16% and steeper, so the rover is confined to
access roads and the drone covers the remaining climb and the 75 m tower.

---

## 2. Machine and software

Native Ubuntu 22.04 (Acer Nitro ANV15-41, RTX 4050 6 GB).

| Component | Version / source | Notes |
|---|---|---|
| ROS 2 | Humble (apt) | |
| Gazebo | Harmonic 8.14 (apt; `apt-mark hold` on all `gz-*`, `libgz-*`) | DART physics, default ODE collision detector |
| ros_gz | `ros-humble-ros-gzharmonic` (apt binary) | |
| PX4 | `~/PX4-Autopilot`, tag `v1.17.0` | `make px4_sitl` built |
| px4_msgs | `release/1.17` via `setup/deps.repos` + vcstool | |
| Micro-XRCE-DDS Agent | v2.4.3 from source, `/usr/local/bin/MicroXRCEAgent` | |
| RMW | `rmw_fastrtps_cpp`, set in `setup/env.local.sh` | `.bashrc` sets Cyclone for other projects |
| Python tools | `/usr/bin/python3` + `pip install --user rasterio pillow requests` | matplotlib from system. Never install into conda base |

`~/.bashrc` ends with `source ~/Rover2Drone/setup/env.sh` (must be after the Cyclone
exports). Every shell prints `[env] ROS_DOMAIN_ID=42 GZ_VERSION=harmonic PX4_DIR=...`.

---

## 3. Repository layout

```
Rover2Drone/
├── setup/          env.sh, env.local.sh (untracked), deps.repos, px4_sitl_params.txt
├── tools/
│   ├── terrain_io.py              shared terrain access (projection, ground_z); all tools use it
│   ├── fetch_terrain.py           Esri imagery + AWS reference DEM, no API key
│   ├── make_terrain_model.py      DEM + imagery -> terrain model + terrain.yaml
│   ├── gen_roads.py               carve OSM roads, tint texture, write config/roads.json
│   ├── gen_terrain_collision.py   triangle-mesh physics collision for the terrain
│   ├── place_turbines.py          turbines + rover + scenery -> worlds/<world>.sdf
│   ├── gen_rover.py               rover model + config/rover.yaml
│   ├── gen_foliage.py             trees/shrubs from imagery vegetation index
│   ├── gen_buildings.py           OSM building extrusion (unused here: OSM has 3 buildings)
│   ├── gen_turbine_meshes.py      NACA blade + tapered tower meshes
│   ├── validate_environment.py    offline validation report
│   └── validate_live.py           live GPS probe test against running Gazebo
├── models/         attappadi (terrain), attappadi_foliage (gitignored), r2d_rover, turbine
├── worlds/         attappadi_windfarm.sdf (main), rover_test.sdf (flat), turbine_site.sdf (old)
├── config/         attappadi_turbines.csv (input); turbines.yaml, rover.yaml, roads.json (generated)
├── src/
│   ├── rover2drone_bringup/       launch/demo.launch.py
│   └── rover2drone_coordination/  relative_state.py, latch_manager.py
└── docs/           HANDOFF.md, versions.md, validation/
```

Git LFS tracks `*.obj *.dae *.stl *.png`. Gitignored: `src/ros_gz`, `src/px4_msgs`,
`models/*_foliage/`, `models/*/materials/textures/*_raw.png`, build dirs.

---

## 4. Data

| Data | Source | Local path |
|---|---|---|
| Elevation | Copernicus GLO-30 via OpenTopography | `~/Downloads/Map data/rasters_COP30/output_hh.tif` |
| Reference elevation | AWS Terrain Tiles (SRTM-derived) via `fetch_terrain.py` | `~/Downloads/attappadi_dem.tif` |
| Imagery | Esri World Imagery, zoom 18 (0.6 m/px) via `fetch_terrain.py` | `~/Downloads/attappadi_rgb.tif` |
| Roads (and buildings) | OpenStreetMap, overpass-turbo GeoJSON export | `~/Downloads/attappadi_osm.geojson` |
| Turbine positions | Read from satellite imagery | `config/attappadi_turbines.csv` |

Site centre (Wind Farm View Point): **11.0813198 N, 76.713563 E**, 852.3 m (EGM2008).
Terrain window 3000 m square.

Copernicus GLO-30 spec: surface model (canopy/buildings included), EGM2008 geoid,
absolute vertical < 4 m LE90, relative < 2 m (slope ≤ 20%) / < 4 m (> 20%), horizontal
< 6 m CE90. Cite: European Space Agency (2024), *Copernicus Global Digital Elevation
Model*, distributed by OpenTopography, doi:10.5069/G9028PQB.

Overpass query used for roads (run at overpass-turbo.eu, Export → GeoJSON):

```
[out:json][timeout:120];
(
  way["building"](11.066,76.699,11.096,76.728);
  relation["building"](11.066,76.699,11.096,76.728);
  way["highway"](11.066,76.699,11.096,76.728);
);
out geom;
```

---

## 5. Rebuilding the world

Run from `~/Rover2Drone`, in this exact order. Always start from `make_terrain_model`
when rebuilding: it resets the pristine copies the road tool carves from, and rewrites
the terrain collision that `gen_terrain_collision` replaces.

```bash
# Helper for the world step (quotes the path with a space properly).
pt() { /usr/bin/python3 tools/place_turbines.py --turbines config/attappadi_turbines.csv \
  --dem "$HOME/Downloads/Map data/rasters_COP30/output_hh.tif" \
  --lat 11.0813198 --lon 76.713563 --extent 3000 --grid 513 \
  --terrain-model attappadi --world attappadi_windfarm --rover-src worlds/turbine_site.sdf; }

# 0. (Only if imagery/reference DEM missing.) ~440 tiles.
/usr/bin/python3 tools/fetch_terrain.py --lat 11.0813198 --lon 76.713563 \
  --extent 3000 --name attappadi --out ~/Downloads --img-zoom 18

# 1. Terrain: visual grid 2049 (1.46 m), collision 1025 (2.93 m), texture 4096 px (0.73 m)
/usr/bin/python3 tools/make_terrain_model.py \
  --dem "$HOME/Downloads/Map data/rasters_COP30/output_hh.tif" \
  --texture ~/Downloads/attappadi_rgb.tif \
  --lat 11.0813198 --lon 76.713563 --extent 3000 --name attappadi --out models

# 2. Carve roads (18.9 km: secondary, unclassified, track), write config/roads.json
/usr/bin/python3 tools/gen_roads.py --terrain-model attappadi \
  --geojson ~/Downloads/attappadi_osm.geojson

# 3. Mesh collision from the carved surface
/usr/bin/python3 tools/gen_terrain_collision.py --terrain-model attappadi

# 4. World (writes config/turbines.yaml, which foliage needs)
pt

# 5. Foliage, then world again to include it
/usr/bin/python3 tools/gen_foliage.py --terrain-model attappadi
pt

# 6. Validation
cd tools && /usr/bin/python3 validate_environment.py --terrain-model attappadi \
  --ref-dem ~/Downloads/attappadi_dem.tif --repo .. && cd ..
```

Rover model only needs regenerating if `gen_rover.py` changes: `/usr/bin/python3 tools/gen_rover.py`.
If the GUI is sluggish: `gen_foliage.py --max-trees 1500 --max-shrubs 3000`.

---

## 6. Running the simulation

Clean up first, every time:

```bash
simclean; pkill -f MicroXRCEAgent; pkill -f px4
```

Five terminals:

```bash
# 1 Agent
MicroXRCEAgent udp4 -p 8888

# 2 Gazebo + bridges (camera, rover drive, latch, clock) + camera viewer
cd ~/Rover2Drone && source install/setup.bash
ros2 launch rover2drone_bringup demo.launch.py

# 3 PX4 (pose printed by place_turbines.py; this is the current one)
cd ~/PX4-Autopilot
PX4_GZ_STANDALONE=1 PX4_GZ_WORLD=attappadi_windfarm \
PX4_SIM_MODEL=gz_x500_gimbal \
PX4_GZ_MODEL_POSE="-0.022,0.033,0.644,0,0,-0.9857" \
./build/px4_sitl_default/bin/px4

# 4 Rover–drone relative state (origins printed by place_turbines.py)
cd ~/Rover2Drone && source install/setup.bash
ros2 run rover2drone_coordination relative_state --ros-args -p use_sim_time:=true \
  -p rover_origin_x:=0.000 -p rover_origin_y:=0.000 -p rover_origin_z:=0.457 \
  -p drone_origin_x:=-0.022 -p drone_origin_y:=0.033 -p drone_origin_z:=0.844

# 5 Latch manager
cd ~/Rover2Drone && source install/setup.bash
ros2 run rover2drone_coordination latch_manager --ros-args -p use_sim_time:=true
```

PX4 shell, first run on a machine (persist with `param save`):

```
param set NAV_RCL_ACT 0
param set NAV_DLL_ACT 0
param set COM_RCL_EXCEPT 4
param set MIS_TAKEOFF_ALT 15
param save
```

World only (no PX4), for looking or driving:

```bash
simclean
gz sim -v4 -r worlds/attappadi_windfarm.sdf
# second terminal:
cd ~/Rover2Drone && source install/setup.bash
ros2 run ros_gz_bridge parameter_bridge /rover/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist &
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/rover/cmd_vel
```

Useful:

```bash
ros2 service call /rover/latch/release std_srvs/srv/Trigger
ros2 service call /rover/latch/engage  std_srvs/srv/Trigger
ros2 topic echo /coordination/slant_range
gz topic -e -n 1 -t /world/attappadi_windfarm/pose/info | grep -A8 'name: "rover"' | grep -A3 position
```

Rover at rest on its wheels reads z ≈ 0.46 at the spawn. In Gazebo, Entity Tree →
right-click `rover` → **Move to** to find it. After changing `src/`:
`colcon build --packages-select <pkg>` then `source install/setup.bash`.

---

## 7. Validation

```bash
cd ~/Rover2Drone/tools
/usr/bin/python3 validate_environment.py --terrain-model attappadi \
  --ref-dem ~/Downloads/attappadi_dem.tif --repo ..
# live: world must be running and playing, no PX4 needed
/usr/bin/python3 validate_live.py --terrain-model attappadi --world attappadi_windfarm --repo ..
```

Outputs `docs/validation/report.md`, `live_gps.md`, figures.

| Check | Current result |
|---|---|
| V1 Georeferencing vs Gazebo GPS model | PASS (≤ 0.1 cm over the terrain in testing) |
| V2 Heightmaps vs DEM, visual vs collision | REVIEW: 17.5 m DEM coverage strip at one edge (see §11) |
| V3 Copernicus vs independent DEM | PASS |
| V4 Turbine placement, overlay on imagery | PASS |
| V5 Layout spacing (rotor diameters) | PASS |
| V6 Terrain profiles | PASS |
| Live GPS probes at each turbine | not yet run |

V2 compares the pre-carving heightmaps with the DEM and reports road carving separately
(deliberate departure: max cut 7.2 m, fill 4.4 m along the roads).

Compass: Gazebo Harmonic's magnetometer recomputes the field each step from the
sensor's lat/lon (built-in declination/inclination/strength tables), so it matches
the local field PX4's estimator expects. No heading bias.

---

## 8. Hard-won facts (read before debugging)

Physics
- **Heightmap collision does not work for vehicles in Gazebo Harmonic's DART.** With the
  default ODE detector the rover floated with no traction (spun "like a beyblade");
  with the Bullet detector the heightmap is not collided at all (rover fell to z −957).
  Fix: `gen_terrain_collision.py` replaces it with a triangle mesh (fine 2.93 m over the
  operating area, coarse 23 m elsewhere; 327k triangles, 16 MB STL). Keep the default
  ODE detector. Heightmap is still used for rendering.
- **Rover wheels use sphere collisions**, not cylinders, for robust terrain contact.
  Visual tyres unchanged. The rover drives well on flat ground (`worlds/rover_test.sdf`)
  and on the terrain mesh.

Terrain and geodesy
- Gazebo ignores include/model poses for heightmaps; vertical offset lives in the
  heightmap `<pos>`.
- Heightmap vertices sit exactly on the terrain edges, so resampling shifts by half a
  pixel (`extract_window(vertex=True)`); textures use area pixels. This bug existed
  until 2026-09-23 and put the terrain ~4 m north-west of true.
- Projection is a site-centred transverse Mercator scaled by 1 + h/R, matching
  Gazebo's GPS tangent plane at site elevation. UTM zone 43 is rotated 0.33° from true
  north here (up to 12 m error at the terrain corners).
- `gen_roads.py` carves from `*_raw.png` pristine copies (idempotent);
  `make_terrain_model.py` deletes them on rebuild.

PX4 and ROS
- **PX4 v1.17 versions some topics**: `/fmu/out/vehicle_local_position_v1`,
  `vehicle_status_v1`, `home_position_v1`. `vehicle_attitude`, `vehicle_odometry`,
  `vehicle_land_detected` are not. A wrong name fails silently (topic listed, publisher
  count 0). Check `ros2 topic info <topic> --verbose` and the agent's writer log.
- RMW must be Fast DDS for this project (the agent is a Fast DDS application).
- x500: PX4's model lifts `base_link` 0.24 m above the spawn point; skids are 0.227 m
  below `base_link`. Model name `x500_gimbal_0`.
- Latch: `DetachableJoint` on the rover auto-attaches when the drone appears;
  `latch_manager` releases, settles 2 s, re-engages. Releases on arm, re-engages on
  landing if on the pad (needs `relative_state` running).

Tooling
- Paths containing spaces (`Map data`) break `eval`-built commands; use the `pt()`
  function or quote paths.
- PX4 `Tools/setup/ubuntu.sh` installs CMake 4 (Kitware) which breaks the gz build;
  downgrade to Jammy CMake 3.22 and hold, or `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.
  Use `--no-nuttx` on SITL-only machines.
- Terminal heredoc pastes sometimes truncate; prefer downloaded files or VS Code.
- QGroundControl: failed under WSL2, untested on native Linux, not needed.

---

## 9. Status

Verified working:
- Full toolchain on native Ubuntu; PX4 takeoff/land (flat world)
- Attappadi terrain (4096 px texture, 2049/1025 grids), 7 turbines on their pads at
  S52 scale, even clumpy foliage, 18.9 km of carved OSM roads
- Rover drives on the terrain and roads (mesh collision, sphere wheels)
- Drone spawns on the deck; latch seating sequence
- `relative_state` publishing (after the `_v1` fix and rover origin offset)
- Offline validation V1, V3–V6 pass

Not yet exercised:
- Live GPS validation
- Latch release on arm and re-engage on landing, in the Attappadi world
- Takeoff from and landing on the rover on sloped terrain
- Drone flight over the new terrain (terrain now has mesh collision; drone contact
  should be fine but is untested)

---

## 10. Next steps, in order

1. **Road graph and route planning.** `config/roads.json` has 15 separate OSM polylines
   (world coords, road-surface z, points every 4 m). Join them at junctions into a graph,
   then plan spawn → road point nearest a chosen turbine (A*).
2. **Rover waypoint follower**, pure pursuit on `/rover/odometry` (note: DiffDrive odom
   is 2D and relative to spawn; use ground-truth pose or add GNSS/IMU for 3D).
3. **RViz + TF**: roads, turbines, rover, drone. Use PX4 `ref_lat/ref_lon/ref_alt` with
   `terrain_io` to place the drone in world coordinates, which also removes the fragile
   origin parameters in `relative_state`.
4. **Drone offboard control** (`OffboardControlMode` + `TrajectorySetpoint`).
5. **Mission manager**: drive → stop → release → launch → inspect → return → land →
   latch → recharge, with a battery/charging model (charge rate scaled by pad alignment).
6. Inspection viewpoint planning around blades; later, synthetic-defect imagery.

---

## 11. Open issues and known limitations

Open:
- **DEM coverage strip (V2 REVIEW).** The Copernicus download is 17.5 m short on one
  edge (void rows at every grid size). Fix: re-download at OpenTopography with
  `Xmin 76.6975, Ymin 11.0655, Xmax 76.7300, Ymax 11.0970`, extract to
  `~/Downloads/Map data/cop30_v2/`, point both `make_terrain_model` and `pt()` at it,
  rerun §5.
- OSM road alignment against imagery not yet checked along all 18.9 km.
- `relative_state` origin parameters must be updated by hand after any spawn change.

Limitations to state honestly:
- Elevation is 30 m resolution; finer grids interpolate. Road cuts come from carving
  OSM centrelines, not from measured data.
- Copernicus is a surface model (canopy included); foliage is decorative, visual only.
- Heights are EGM2008 orthometric; real GNSS reports ellipsoidal height.
- Imagery capture date unknown (Esri World Imagery); check licence before publishing
  figures, or substitute Sentinel-2.
- Turbine model (Suzlon S52 class: 75 m hub, 52 m rotor) inferred from spacing and
  lattice towers in site photos; modelled with a tubular tower.
- Rover suspension rigid (visual brackets only).
- Attappadi has a documented land dispute with tribal communities, relevant if naming
  the site in publications. Kanjikode (KSEB, flat, industrial) is the alternative site.
