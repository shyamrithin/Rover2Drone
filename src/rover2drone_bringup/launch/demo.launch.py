#!/usr/bin/env python3
"""
demo.launch.py
==============

Brings up everything for the Rover2Drone demonstration EXCEPT PX4.

Starts:
  1. Gazebo Harmonic with worlds/turbine_site.sdf (turbine + rover)
  2. ros_gz_bridge for the drone camera, rover cmd_vel, rover odometry
     and the simulation clock, with topics remapped to short stable names
  3. rqt_image_view on the drone camera feed

PX4 is deliberately NOT launched here. Running it separately keeps its
interactive pxh> shell available, which is needed for `gimbal start`,
`commander takeoff` and parameter changes. Start it in its own terminal
once Gazebo is up:

  cd ~/PX4-Autopilot
  PX4_GZ_STANDALONE=1 \
  PX4_GZ_WORLD=turbine_site \
  PX4_SIM_MODEL=gz_x500_gimbal \
  PX4_GZ_MODEL_POSE="8.05,0,0.55,0,0,3.14159" \
  ./build/px4_sitl_default/bin/px4

The camera bridge is started lazily: the Gazebo camera topic does not
exist until PX4 spawns the drone, so the bridge is configured with
lazy subscription and will connect when the topic appears.

Launch arguments:
  world        path to the world SDF
  image_view   whether to open rqt_image_view (default true)
  teleop       whether to open a rover teleop terminal (default false)

Usage:
  ros2 launch rover2drone_bringup demo.launch.py
  ros2 launch rover2drone_bringup demo.launch.py image_view:=false
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Gazebo derives this topic path from the model and link names. If you
# change the PX4 model from x500_gimbal, this prefix changes with it.
GZ_CAM = ('/world/turbine_site/model/x500_gimbal_0/link/camera_link'
          '/sensor/camera')

HOME = os.path.expanduser('~')
REPO = os.path.join(HOME, 'Rover2Drone')


def generate_launch_description():
    world = LaunchConfiguration('world')
    image_view = LaunchConfiguration('image_view')

    args = [
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(REPO, 'worlds', 'turbine_site.sdf'),
            description='Path to the Gazebo world SDF'),
        DeclareLaunchArgument(
            'image_view', default_value='true',
            description='Open rqt_image_view on the drone camera'),
    ]

    # -r starts the world already running, so no need to press play.
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v', '3', world],
        output='screen',
        name='gazebo')

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        output='screen',
        arguments=[
            f'{GZ_CAM}/image@sensor_msgs/msg/Image[gz.msgs.Image',
            f'{GZ_CAM}/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/rover/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/rover/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        remappings=[
            (f'{GZ_CAM}/image', '/drone/camera/image_raw'),
            (f'{GZ_CAM}/camera_info', '/drone/camera/camera_info'),
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
                condition=IfCondition(image_view),
                parameters=[{'use_sim_time': True}]),
        ])

    return LaunchDescription(args + [gazebo, bridge, viewer])