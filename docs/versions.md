# Environment baseline

Verified working: 2026-08-15

| Component | Version |
|---|---|
| Host | Windows 11 (build 26200), WSL2 |
| GPU | RTX 4050 Laptop, 6GB, via MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA |
| Distro | Ubuntu 22.04 Jammy |
| ROS 2 | Humble (apt, ros-humble-desktop) |
| Gazebo | Harmonic (apt, gz-harmonic, apt-mark hold applied) |
| ros_gz | source build, branch humble, GZ_VERSION=harmonic, ~/ws_ros_gz |

## Verified
- glxinfo reports D3D12 (NVIDIA GeForce RTX 4050 Laptop GPU)
- gz sim shapes.sdf: RTF ~100%
- Offscreen camera sensor renders, /camera topic advertised
- ros_gz_bridge relays /clock to ROS 2 with incrementing timestamps

## Gotchas
- Every shell must show ROS_DOMAIN_ID=42. A shell without it sees an empty
  node list and reports no error. Banner in .bashrc guards against this.
- Never build or clone under /mnt/c. 9p filesystem, ~10x slower.
- Do not install NVIDIA drivers inside WSL. Windows driver only.
- ros-humble-sdformat-urdf pulls libsdformat12/ignition-utils1. Expected,
  coexists with Harmonic's libsdformat14, not contamination.

## PX4 SITL verified: 2026-08-16
- PX4-Autopilot v1.17.0, Gazebo Sim 8.15.0 (Harmonic)
- `make px4_sitl gz_x500`: full takeoff and land cycle, RTF ~99.9%
- uXRCE-DDS client picked up ROS_DOMAIN_ID=42 from setup/env.sh
- Params needed in SITL to arm without a GCS or RC link:
    NAV_RCL_ACT 0, NAV_DLL_ACT 0, COM_RCL_EXCEPT 4
- Known-benign noise: `NodeShared::Publish() Interrupted system call`
  from gz-transport, and vehicle_imu timestamp errors on sim hitches.

## Demo startup order

Four processes, in this order. Order matters: the agent must exist before
PX4's uXRCE-DDS client tries to connect, though the client does retry
roughly every 30 s.

1. MicroXRCEAgent udp4 -p 8888
2. ros2 launch rover2drone_bringup demo.launch.py
3. PX4 SITL in its own terminal (keeps the interactive pxh> shell):
     cd ~/PX4-Autopilot
     PX4_GZ_STANDALONE=1 PX4_GZ_WORLD=turbine_site \
     PX4_SIM_MODEL=gz_x500_gimbal \
     PX4_GZ_MODEL_POSE="8.05,0,0.55,0,0,3.14159" \
     ./build/px4_sitl_default/bin/px4
   Then at pxh>:
     param set NAV_RCL_ACT 0
     param set NAV_DLL_ACT 0
     param set COM_RCL_EXCEPT 4
     param save
     gimbal start
     (wait ~20 s for EKF convergence)
     commander takeoff
4. ros2 run rover2drone_coordination relative_state --ros-args -p use_sim_time:=true

## Open issues at end of 2026-08-16

- /coordination/* topics silent on WSL. Agent reports datawriters created,
  but publisher count on /fmu/out/vehicle_local_position not yet confirmed
  non-zero. Prime suspect: ROS_LOCALHOST_ONLY=1 in setup/env.local.sh.
  MicroXRCEAgent is a raw Fast DDS app and ignores that RMW-level variable,
  so agent and ROS 2 nodes may not discover each other. Fix: drop
  ROS_LOCALHOST_ONLY (deprecated in Humble, and everything is on one host).
  Diagnostic: ros2 topic info /fmu/out/vehicle_local_position --verbose

- QGroundControl does not connect under WSL2 mirrored networking. QGC shows
  connected, PX4 shows rx 0 B/s, so the GCS arming check fails and
  NAV_DLL_ACT 0 is needed to arm. Expected to work unconfigured on native
  Linux. Being replaced by a Python GUI regardless.

## Linux machine setup (2026-08-17)

Reproduce with:
  git clone git@github.com:shyamrithin/Rover2Drone.git
  cd Rover2Drone
  vcs import src < setup/deps.repos
  cp setup/env.local.sh.example setup/env.local.sh   # then strip WSL-only lines
  colcon build --cmake-args -DBUILD_TESTING=OFF

Still needed separately on the new machine:
  - ROS 2 Humble, Gazebo Harmonic (apt-mark hold the gz-* packages)
  - PX4-Autopilot at tag v1.17.0, recursive submodules, bash Tools/setup/ubuntu.sh
  - Micro-XRCE-DDS-Agent v2.4.3 built from source
  - python3 tools/gen_turbine_meshes.py models/turbine/meshes  (LFS should
    fetch the .obj files, but regenerating is free)
