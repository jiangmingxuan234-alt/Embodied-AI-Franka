import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="ROS2 Direct Joint Bridge (World + Franka)")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=False)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)
ext_manager.set_extension_enabled_immediate("omni.isaac.core", True)
ext_manager.set_extension_enabled_immediate("omni.isaac.franka", True)

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty

from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.types import ArticulationAction


JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]

JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0], dtype=np.float32)
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04], dtype=np.float32)
GOAL_POSITION = np.array([0.40, 0.0, 0.40], dtype=np.float32)


def _as_vec(x, n):
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    out = np.zeros((n,), dtype=np.float32)
    m = min(n, arr.shape[0])
    out[:m] = arr[:m]
    return out


class DirectBridgeNode(Node):
    def __init__(self):
        super().__init__("isaac_lab_direct_joint_bridge")

        self.target_joint_pos = None
        self.assist_mode = False
        self.need_controller_reset = False

        self.sub_joint_cmd = self.create_subscription(JointState, "/joint_command", self._on_joint_command, 10)
        self.sub_assist_start = self.create_subscription(Empty, "/assist_start", self._on_assist_start, 10)
        self.sub_assist_stop = self.create_subscription(Empty, "/assist_stop", self._on_assist_stop, 10)

        self.pub_joint_states = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_ee_pose = self.create_publisher(PoseStamped, "/franka/ee_pose", 10)
        self.pub_obj_pose = self.create_publisher(PoseStamped, "/udisk/pose", 10)

    def _on_joint_command(self, msg: JointState):
        if self.assist_mode:
            return  # assist 模式下忽略策略节点的命令
        target = np.zeros((9,), dtype=np.float32)
        if msg.name and len(msg.name) == len(msg.position):
            idx = {name: i for i, name in enumerate(msg.name)}
            for i, name in enumerate(JOINT_NAMES):
                if name in idx:
                    target[i] = float(msg.position[idx[name]])
        else:
            n = min(len(msg.position), 9)
            if n > 0:
                target[:n] = np.asarray(msg.position[:n], dtype=np.float32)
        self.target_joint_pos = np.clip(target, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

    def _on_assist_start(self, msg):
        self.assist_mode = True
        self.need_controller_reset = True
        self.get_logger().info("Assist mode ON — bridge takes over control")

    def _on_assist_stop(self, msg):
        self.assist_mode = False
        self.get_logger().info("Assist mode OFF — returning to /joint_command")

    def publish_state(self, franka: Franka, u_disk: DynamicCuboid):
        stamp = self.get_clock().now().to_msg()

        j_pos = _as_vec(franka.get_joint_positions(), 9)
        j_vel = _as_vec(franka.get_joint_velocities(), 9)

        ee_pos, ee_quat_wxyz = franka.end_effector.get_world_pose()
        ee_pos = _as_vec(ee_pos, 3)
        ee_quat_wxyz = _as_vec(ee_quat_wxyz, 4)

        obj_pos, obj_quat_wxyz = u_disk.get_world_pose()
        obj_pos = _as_vec(obj_pos, 3)
        obj_quat_wxyz = _as_vec(obj_quat_wxyz, 4)

        js = JointState()
        js.header.stamp = stamp
        js.name = JOINT_NAMES
        js.position = j_pos.tolist()
        js.velocity = j_vel.tolist()
        self.pub_joint_states.publish(js)

        ee = PoseStamped()
        ee.header.stamp = stamp
        ee.header.frame_id = "world"
        ee.pose.position.x = float(ee_pos[0])
        ee.pose.position.y = float(ee_pos[1])
        ee.pose.position.z = float(ee_pos[2])
        ee.pose.orientation.w = float(ee_quat_wxyz[0])
        ee.pose.orientation.x = float(ee_quat_wxyz[1])
        ee.pose.orientation.y = float(ee_quat_wxyz[2])
        ee.pose.orientation.z = float(ee_quat_wxyz[3])
        self.pub_ee_pose.publish(ee)

        obj = PoseStamped()
        obj.header.stamp = stamp
        obj.header.frame_id = "world"
        obj.pose.position.x = float(obj_pos[0])
        obj.pose.position.y = float(obj_pos[1])
        obj.pose.position.z = float(obj_pos[2])
        obj.pose.orientation.w = float(obj_quat_wxyz[0])
        obj.pose.orientation.x = float(obj_quat_wxyz[1])
        obj.pose.orientation.y = float(obj_quat_wxyz[2])
        obj.pose.orientation.z = float(obj_quat_wxyz[3])
        self.pub_obj_pose.publish(obj)


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 30.0)
    world.scene.add_default_ground_plane()

    franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))

    u_disk = world.scene.add(
        DynamicCuboid(
            prim_path="/World/u_disk",
            name="u_disk",
            position=np.array([0.5, 0.0, 0.02]),
            scale=np.array([0.05, 0.02, 0.01]),
            color=np.array([1.0, 0.0, 0.0]),
            mass=0.02,
        )
    )

    world.reset()
    init_joints = np.array([0.0, -1.1, 0.0, -2.3, 0.0, 2.4, 0.8, 0.04, 0.04], dtype=np.float32)
    franka.set_joint_positions(init_joints)
    franka.apply_action(ArticulationAction(joint_positions=init_joints))
    expert_controller = PickPlaceController(
        name="ros2_assist_pick_place_controller",
        gripper=franka.gripper,
        robot_articulation=franka,
    )

    rclpy.init()
    ros_node = DirectBridgeNode()

    print("\n ROS2 direct bridge started (World + Franka)")
    print("Listening: /joint_command, /assist_start, /assist_stop")
    print("Publishing: /joint_states, /franka/ee_pose, /udisk/pose\n")

    while simulation_app.is_running():
        world.step(render=not args_cli.headless)
        rclpy.spin_once(ros_node, timeout_sec=0.0)

        if ros_node.assist_mode:
            if ros_node.need_controller_reset:
                expert_controller.reset()
                ros_node.need_controller_reset = False
            j_pos = _as_vec(franka.get_joint_positions(), 9)
            obj_pos, _ = u_disk.get_world_pose()
            obj_pos = _as_vec(obj_pos, 3)
            safe_pick = np.clip(obj_pos, a_min=[0.2, -0.5, 0.0], a_max=[0.8, 0.5, 0.6])
            expert_actions = expert_controller.forward(
                picking_position=safe_pick,
                placing_position=GOAL_POSITION,
                current_joint_positions=j_pos,
            )
            franka.apply_action(expert_actions)
        else:
            target = ros_node.target_joint_pos if ros_node.target_joint_pos is not None else _as_vec(franka.get_joint_positions(), 9)
            target = np.clip(target, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
            franka.apply_action(ArticulationAction(joint_positions=target))

        ros_node.publish_state(franka, u_disk)

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
