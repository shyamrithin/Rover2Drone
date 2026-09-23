#!/usr/bin/env python3
"""
latch_manager.py
================

Owns the drone latch on the rover deck: seats the drone when it first
appears, holds it during transit, releases it for flight, and re-engages it
after landing.

The latch itself is Gazebo's DetachableJoint system on the rover model
(see tools/gen_rover.py), driven over three bridged topics:

    /rover/latch/attach   std_msgs/Empty   -> engage
    /rover/latch/detach   std_msgs/Empty   -> release
    /rover/latch/state    std_msgs/String  <- "attached" / "detached"

Why a manager is needed
-----------------------
DetachableJoint latches automatically the instant the drone model appears,
and the weld freezes whatever relative pose exists at that moment. PX4
spawns the drone level and slightly above the deck, while the rover on a
slope sits tilted, so an unmanaged latch would lock the drone hovering at a
gap or an angle. This node waits for PX4 to come up, releases the latch,
lets the drone settle onto the pad under gravity, then re-engages it, so
the drone is always locked resting flat on the deck.

The Gazebo state topic is published on change only and is not latched, so a
node that starts late misses the initial "attached". The seating sequence
therefore issues release-then-engage unconditionally; both commands are
idempotent inside DetachableJoint.

Behaviour
---------
  startup      wait for PX4 land-detector data (proves the drone exists),
               release, wait settle_time_s, engage.
  on arming    if release_on_arm, release immediately so the motors are not
               fighting a weld. In a real system the latch would instead
               inhibit arming until released; this is a simulation
               convenience so manual `commander takeoff` works.
  on landing   if auto_latch_on_landing and the drone is within
               pad_tolerance_m of the pad centre (from relative_state's
               /coordination/relative_position), engage.

Services
--------
  /rover/latch/engage   std_srvs/Trigger   engage; refused unless landed
  /rover/latch/release  std_srvs/Trigger   release

Published
---------
  /rover/latch/latched  std_msgs/Bool, transient-local so late subscribers
                        get the current state immediately.

PX4 v1.17 topic names: vehicle_status is versioned (_v1) on the wire,
vehicle_land_detected is not. Both are parameters.

Usage
  ros2 run rover2drone_coordination latch_manager --ros-args -p use_sim_time:=true
  ros2 service call /rover/latch/release std_srvs/srv/Trigger
  ros2 service call /rover/latch/engage  std_srvs/srv/Trigger
"""

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import Bool, Empty, String
from std_srvs.srv import Trigger

from px4_msgs.msg import VehicleLandDetected, VehicleStatus

WAITING, SETTLING, READY = "waiting_for_drone", "settling", "ready"


class LatchManager(Node):
    def __init__(self):
        super().__init__('latch_manager')

        self.declare_parameter('settle_time_s', 2.0)
        self.declare_parameter('seat_delay_s', 1.0)
        self.declare_parameter('release_on_arm', True)
        self.declare_parameter('auto_latch_on_landing', True)
        # Pad centre in the rover base_link frame (config/rover.yaml).
        self.declare_parameter('pad_x', -0.04)
        self.declare_parameter('pad_y', 0.0)
        self.declare_parameter('pad_tolerance_m', 0.25)
        # Drone base_link height above rover base_link when resting on the
        # pad: deck 0.16 + skids 0.227. Generous band either side.
        self.declare_parameter('rest_z_min_m', 0.20)
        self.declare_parameter('rest_z_max_m', 0.70)
        self.declare_parameter('land_topic', '/fmu/out/vehicle_land_detected')
        self.declare_parameter('status_topic', '/fmu/out/vehicle_status_v1')

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.settle_time = p('settle_time_s')
        self.seat_delay = p('seat_delay_s')
        self.release_on_arm = p('release_on_arm')
        self.auto_latch = p('auto_latch_on_landing')
        self.pad = (p('pad_x'), p('pad_y'))
        self.pad_tol = p('pad_tolerance_m')
        self.rest_z = (p('rest_z_min_m'), p('rest_z_max_m'))

        px4_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        latched_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)

        self.pub_attach = self.create_publisher(Empty, '/rover/latch/attach', 10)
        self.pub_detach = self.create_publisher(Empty, '/rover/latch/detach', 10)
        self.pub_latched = self.create_publisher(Bool, '/rover/latch/latched',
                                                 latched_qos)

        self.create_subscription(String, '/rover/latch/state',
                                 self._on_state, 10)
        self.create_subscription(VehicleLandDetected, p('land_topic'),
                                 self._on_land, px4_qos)
        self.create_subscription(VehicleStatus, p('status_topic'),
                                 self._on_status, px4_qos)
        self.create_subscription(PointStamped,
                                 '/coordination/relative_position',
                                 self._on_rel, 10)

        self.create_service(Trigger, '/rover/latch/engage', self._srv_engage)
        self.create_service(Trigger, '/rover/latch/release', self._srv_release)

        self.phase = WAITING
        self.phase_t = None
        self.latched = None
        self.landed = None
        self.armed = False
        self.rel = None
        self.warned_no_rel = False
        self.armed_value = getattr(VehicleStatus, 'ARMING_STATE_ARMED', 2)

        self.create_timer(0.1, self._tick)
        self.get_logger().info(
            'latch_manager up, waiting for PX4 before seating the drone.')

    # ------------------------------------------------------------------ io
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _set_latched(self, value):
        if value != self.latched:
            self.latched = value
            self.pub_latched.publish(Bool(data=value))

    def _engage(self, why):
        self.pub_attach.publish(Empty())
        self._set_latched(True)
        self.get_logger().info(f'latch ENGAGED ({why})')

    def _release(self, why):
        self.pub_detach.publish(Empty())
        self._set_latched(False)
        self.get_logger().info(f'latch RELEASED ({why})')

    def _on_state(self, msg):
        # Ground truth from Gazebo overrides our commanded assumption.
        self._set_latched(msg.data == 'attached')

    def _on_rel(self, msg):
        self.rel = msg.point

    def _drone_on_pad(self):
        if self.rel is None:
            if not self.warned_no_rel:
                self.get_logger().warn(
                    'No /coordination/relative_position; cannot confirm the '
                    'drone is on the pad. Is relative_state running?')
                self.warned_no_rel = True
            return False
        dx, dy = self.rel.x - self.pad[0], self.rel.y - self.pad[1]
        horiz_ok = (dx * dx + dy * dy) ** 0.5 <= self.pad_tol
        vert_ok = self.rest_z[0] <= self.rel.z <= self.rest_z[1]
        return horiz_ok and vert_ok

    # ------------------------------------------------------------- vehicle
    def _on_land(self, msg):
        was = self.landed
        self.landed = bool(msg.landed)
        if (self.phase == READY and self.auto_latch and was is False
                and self.landed and not self.latched):
            if self._drone_on_pad():
                self._engage('drone landed on pad')
            else:
                self.get_logger().warn(
                    'Drone landed but not on the pad; latch not engaged.')

    def _on_status(self, msg):
        was = self.armed
        self.armed = (msg.arming_state == self.armed_value)
        if (self.armed and not was and self.release_on_arm
                and self.latched is not False):
            self._release('drone armed')

    # ---------------------------------------------------------------- tick
    def _tick(self):
        now = self._now()
        if self.phase == WAITING and self.landed is not None:
            # PX4 is publishing, so the drone model exists and the joint
            # has almost certainly auto-attached at its spawn pose. Give it
            # a moment, then unseat it.
            if self.phase_t is None:
                self.phase_t = now
            elif now - self.phase_t >= self.seat_delay:
                self._release('seating: letting drone settle onto pad')
                self.phase, self.phase_t = SETTLING, now
        elif self.phase == SETTLING and now - self.phase_t >= self.settle_time:
            self._engage('seating complete, drone resting on pad')
            self.phase = READY

    # ------------------------------------------------------------ services
    def _srv_engage(self, _req, res):
        if self.landed is False:
            res.success, res.message = False, 'refused: drone is airborne'
        elif self.armed:
            res.success, res.message = False, 'refused: drone is armed'
        else:
            self._engage('service request')
            res.success, res.message = True, 'latch engaged'
        return res

    def _srv_release(self, _req, res):
        self._release('service request')
        res.success, res.message = True, 'latch released'
        return res


def main(args=None):
    rclpy.init(args=args)
    node = LatchManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
