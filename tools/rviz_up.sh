#!/usr/bin/env bash
# =============================================================================
# File:        tools/rviz_up.sh
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-24
# Updated:     2026-09-24  RobotModel: URDF + joint states replace the
#              static sensor transforms
# Depends:     ROS 2 Humble, ros_gz_bridge, tf2_ros, rviz2; Gazebo running
#              (demo.launch.py) before this is started
# =============================================================================
#
# rviz_up.sh
# ==========
# One command for RViz on the running simulation. Starts:
#   1. a ros_gz_bridge for the rover's drive-plugin TF (odom -> rover/base_link)
#      on /tf
#   2. a bridge for the rover's wheel joint states, and robot_state_publisher
#      with models/r2d_rover/rover.urdf (written by gen_rover.py): TF for the
#      wheels and the lidar/IMU/GNSS frames, plus /robot_description
#   3. rviz2 with config/rover2drone.rviz, on sim time
# Closing RViz (or Ctrl+C) stops the helpers too.
#
# Usage:  cd ~/Rover2Drone && ./tools/rviz_up.sh [world_name]
# =============================================================================
set -e
cd "$(dirname "$0")/.."
WORLD="${1:-attappadi_windfarm}"
[ -f models/r2d_rover/rover.urdf ] || { echo "run tools/gen_rover.py first (writes rover.urdf)"; exit 1; }
source install/setup.bash

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

ros2 run ros_gz_bridge parameter_bridge \
  '/rover/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V' \
  --ros-args -r /rover/tf:=/tf -p use_sim_time:=true >/dev/null 2>&1 &
pids+=($!)
# Rover URDF -> TF for every link (wheels spin with the joint states) and
# the /robot_description topic RViz's RobotModel reads. frame_prefix makes
# base_link -> rover/base_link, matching the drive plugin's TF.
ros2 run ros_gz_bridge parameter_bridge \
  "/world/${WORLD}/model/rover/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model" \
  --ros-args -r "/world/${WORLD}/model/rover/joint_state:=/rover/joint_states" \
  -p use_sim_time:=true >/dev/null 2>&1 &
pids+=($!)
ros2 run robot_state_publisher robot_state_publisher --ros-args \
  -p robot_description:="$(cat models/r2d_rover/rover.urdf)" \
  -p frame_prefix:=rover/ -p use_sim_time:=true \
  -r joint_states:=/rover/joint_states >/dev/null 2>&1 &
pids+=($!)

rviz2 -d config/rover2drone.rviz --ros-args -p use_sim_time:=true
