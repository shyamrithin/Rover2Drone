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
