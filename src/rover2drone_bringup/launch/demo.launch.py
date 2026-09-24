#!/usr/bin/env python3
"""
demo.launch.py
==============

Brings up everything for the Rover2Drone simulation EXCEPT PX4.

Starts:
  1. Gazebo Harmonic with the chosen world
  2. ros_gz_bridge for the drone camera, rover cmd_vel, rover odometry,
     rover IMU / GNSS / 2D lidar, the drone latch and the simulation clock,
     with topics remapped to short stable names
  3. rqt_image_view on the drone camera feed (optional)

World-agnostic: Gazebo embeds the world name in sensor topic paths
(/world/<name>/model/x500_gimbal_0/...), so the camera bridge topic is
derived from the world file at launch time rather than hardcoded. Pass a
different world and the video keeps working.

PX4 is deliberately NOT launched here, to keep its interactive pxh> shell.
Start it in its own terminal with PX4_GZ_WORLD matching the world name;
tools/place_turbines.py prints the exact command for generated worlds.

Launch arguments:
  world        world SDF filename under worlds/ (default attappadi_windfarm.sdf)
  drone_model  Gazebo model instance name of the drone (default x500_gimbal_0)
  image_view   open rqt_image_view on the drone camera (default true)

Usage:
  ros2 launch rover2drone_bringup demo.launch.py
  ros2 launch rover2drone_bringup demo.launch.py world:=turbine_site.sdf
"""

import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            OpaqueFunction, TimerAction)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HOME = os.path.expanduser('~')
REPO = os.path.join(HOME, 'Rover2Drone')


def launch_setup(context, *args, **kwargs):
    world_file = LaunchConfiguration('world').perform(context)
    drone = LaunchConfiguration('drone_model').perform(context)

    world_path = (world_file if os.path.isabs(world_file)
                  else os.path.join(REPO, 'worlds', world_file))
    world_name = os.path.splitext(os.path.basename(world_path))[0]
    gz_cam = (f'/world/{world_name}/model/{drone}/link/camera_link'
              f'/sensor/camera')

    # -r starts the simulation running, so no need to press play.
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v', '3', world_path],
        output='screen',
        name='gazebo')

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        output='screen',
        arguments=[
            f'{gz_cam}/image@sensor_msgs/msg/Image[gz.msgs.Image',
            f'{gz_cam}/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/rover/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/rover/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            # Rover sensors (gen_rover.py v3).
            '/rover/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/rover/navsat@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat',
            '/rover/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # Drone latch on the rover deck (DetachableJoint).
            '/rover/latch/attach@std_msgs/msg/Empty]gz.msgs.Empty',
            '/rover/latch/detach@std_msgs/msg/Empty]gz.msgs.Empty',
            '/rover/latch/state@std_msgs/msg/String[gz.msgs.StringMsg',
        ],
        remappings=[
            (f'{gz_cam}/image', '/drone/camera/image_raw'),
            (f'{gz_cam}/camera_info', '/drone/camera/camera_info'),
        ],
        parameters=[{'use_sim_time': True}])

    viewer = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='rqt_image_view',
                executable='rqt_image_view',
                name='drone_camera_view',
                arguments=['/drone/camera/image_raw'],
                condition=IfCondition(LaunchConfiguration('image_view')),
                parameters=[{'use_sim_time': True}]),
        ])

    print(f'[demo.launch] world "{world_name}", camera topic {gz_cam}/image')
    return [gazebo, bridge, viewer]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value='attappadi_windfarm.sdf',
            description='World SDF filename under worlds/, or an absolute path'),
        DeclareLaunchArgument(
            'drone_model', default_value='x500_gimbal_0',
            description='Gazebo model instance name of the drone'),
        DeclareLaunchArgument(
            'image_view', default_value='true',
            description='Open rqt_image_view on the drone camera'),
        OpaqueFunction(function=launch_setup),
    ])
