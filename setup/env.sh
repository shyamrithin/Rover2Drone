#!/usr/bin/env bash
# env.sh
# Portable environment for the Rover2Drone marsupial inspection stack.
# Sourced on every machine. Machine-specific overrides belong in
# setup/env.local.sh, which is gitignored (see env.local.sh.example).

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

source /opt/ros/humble/setup.bash
export GZ_VERSION=harmonic
export ROS_DOMAIN_ID=42

[ -f "$REPO_ROOT/install/setup.bash" ] && source "$REPO_ROOT/install/setup.bash"
export PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
# Rebuild from scratch each time so repeated sourcing does not accumulate paths.
export GZ_SIM_RESOURCE_PATH="$REPO_ROOT/worlds:$REPO_ROOT/models:$PX4_DIR/Tools/simulation/gz/models:$PX4_DIR/Tools/simulation/gz/worlds"



[ -f "$REPO_ROOT/setup/env.local.sh" ] && source "$REPO_ROOT/setup/env.local.sh"

echo "[env] ROS_DOMAIN_ID=$ROS_DOMAIN_ID GZ_VERSION=$GZ_VERSION PX4_DIR=$PX4_DIR"
