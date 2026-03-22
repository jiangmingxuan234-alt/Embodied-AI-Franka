#!/usr/bin/env python3
import os
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty


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

JOINT_LIMITS_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0], dtype=np.float32
)
JOINT_LIMITS_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04], dtype=np.float32
)


def sanitize(x: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    return np.nan_to_num(x, nan=fallback, posinf=fallback, neginf=fallback).astype(np.float32, copy=False)


class CustomPlayRosNode(Node):
    """Pure BC deployment node with anti-stall shaping.

    Main path remains BC-only:
    observation -> BC delta -> shaping -> EMA -> absolute joint target.
    """

    def __init__(self):
        super().__init__("franka_customplay_ros_node")

        self.declare_parameter("topic_joint_states", "/joint_states")
        self.declare_parameter("topic_ee_pose", "/franka/ee_pose")
        self.declare_parameter("topic_object_pose", "/udisk/pose")
        self.declare_parameter("topic_joint_command", "/joint_command")

        self.declare_parameter("control_rate_hz", 15.0)
        self.declare_parameter("goal_position", [0.40, 0.0, 0.40])

        self.declare_parameter(
            "checkpoint_path",
            "/home/jmx001/my_program/my_robot_project/logs/robomimic/Isaac-UDisk-Grasp-v0/bc/20260320182155/models/model_epoch_100.pth",
        )
        self.declare_parameter("dataset_path", "/home/jmx001/my_program/my_robot_project/logs/robomimic/rmpflow_expert.hdf5")

        # BC output shaping
        self.declare_parameter("arm_delta_limit", [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        self.declare_parameter("arm_delta_gain", 3.2)
        self.declare_parameter("ema_alpha", 0.7)

        # Fine-zone anti-wobble (applies only when aligned enough)
        self.declare_parameter("near_object_dist", 0.40)
        self.declare_parameter("near_object_gain_scale", 0.60)
        self.declare_parameter("fine_xy_thresh", 0.105)
        self.declare_parameter("fine_relz_thresh", -0.14)

        self.declare_parameter("delta_deadband", 2.5e-4)
        self.declare_parameter("delta_deadband_far", 1.0e-4)
        self.declare_parameter("flip_suppress_threshold", 1.5e-3)
        self.declare_parameter("arm_slew_limit", [0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004])

        self.declare_parameter("nodding_damp_joints", [3, 5])
        self.declare_parameter("nodding_damp_scale", 0.5)
        self.declare_parameter("nodding_xy_thresh", 0.11)
        self.declare_parameter("nodding_abs_relz_max", 0.22)

        # Stall recovery pulse (still BC-only; no external expert / state machine)
        self.declare_parameter("stall_best_margin", 0.0020)
        self.declare_parameter("stall_steps", 26)
        self.declare_parameter("unstick_trigger_dist", 0.20)
        self.declare_parameter("unstick_steps", 18)
        self.declare_parameter("unstick_gain", 2.6)
        self.declare_parameter("unstick_min_arm", 0.0015)

        # Keep the same gripper closure heuristic from custom_play.
        self.declare_parameter("grasp_xy", 0.09)
        self.declare_parameter("grasp_z", 0.10)

        # Optional lightweight logging.
        self.declare_parameter("log_every_n_steps", 30)

        self.topic_joint_states = self.get_parameter("topic_joint_states").value
        self.topic_ee_pose = self.get_parameter("topic_ee_pose").value
        self.topic_object_pose = self.get_parameter("topic_object_pose").value
        self.topic_joint_command = self.get_parameter("topic_joint_command").value

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.goal_position = np.array(self.get_parameter("goal_position").value, dtype=np.float32)

        self.checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self.dataset_path = str(self.get_parameter("dataset_path").value)

        self.arm_delta_limit = np.array(self.get_parameter("arm_delta_limit").value, dtype=np.float32)
        self.arm_delta_gain = float(self.get_parameter("arm_delta_gain").value)
        self.ema_alpha = float(self.get_parameter("ema_alpha").value)

        self.near_object_dist = float(self.get_parameter("near_object_dist").value)
        self.near_object_gain_scale = float(self.get_parameter("near_object_gain_scale").value)
        self.fine_xy_thresh = float(self.get_parameter("fine_xy_thresh").value)
        self.fine_relz_thresh = float(self.get_parameter("fine_relz_thresh").value)

        self.delta_deadband = float(self.get_parameter("delta_deadband").value)
        self.delta_deadband_far = float(self.get_parameter("delta_deadband_far").value)
        self.flip_suppress_threshold = float(self.get_parameter("flip_suppress_threshold").value)
        self.arm_slew_limit = np.array(self.get_parameter("arm_slew_limit").value, dtype=np.float32)

        self.nodding_damp_joints = np.array(self.get_parameter("nodding_damp_joints").value, dtype=np.int64)
        self.nodding_damp_scale = float(self.get_parameter("nodding_damp_scale").value)
        self.nodding_xy_thresh = float(self.get_parameter("nodding_xy_thresh").value)
        self.nodding_abs_relz_max = float(self.get_parameter("nodding_abs_relz_max").value)

        self.stall_best_margin = float(self.get_parameter("stall_best_margin").value)
        self.stall_steps = int(self.get_parameter("stall_steps").value)
        self.unstick_trigger_dist = float(self.get_parameter("unstick_trigger_dist").value)
        self.unstick_steps = int(self.get_parameter("unstick_steps").value)
        self.unstick_gain = float(self.get_parameter("unstick_gain").value)
        self.unstick_min_arm = float(self.get_parameter("unstick_min_arm").value)

        self.grasp_xy = float(self.get_parameter("grasp_xy").value)
        self.grasp_z = float(self.get_parameter("grasp_z").value)
        self.log_every_n_steps = max(1, int(self.get_parameter("log_every_n_steps").value))

        if self.arm_slew_limit.shape[0] != 7:
            self.get_logger().warn("arm_slew_limit must have 7 entries; fallback to 0.004 for all joints")
            self.arm_slew_limit = np.array([0.004] * 7, dtype=np.float32)

        # Runtime state
        self.joint_pos: Optional[np.ndarray] = None
        self.joint_vel: Optional[np.ndarray] = None
        self.ee_pos: Optional[np.ndarray] = None
        self.ee_quat_wxyz: Optional[np.ndarray] = None
        self.obj_pos: Optional[np.ndarray] = None

        self.policy = None
        self.has_action_stats = False
        self.expected_obs_dim = 37
        self.action_min: Optional[np.ndarray] = None
        self.action_max: Optional[np.ndarray] = None

        self.smoothed_delta: Optional[np.ndarray] = None
        self.prev_arm_delta: Optional[np.ndarray] = None
        self.prev_target: Optional[np.ndarray] = None
        self.prev_dist: Optional[float] = None
        self.best_dist = np.inf
        self.stall_counter = 0
        self.unstick_remaining = 0
        self.step_count = 0

        # Assist state
        self.assist_active = False
        self.assist_steps = 0
        self.assist_cooldown = 0
        self.task_done = False
        self.ASSIST_MAX_STEPS = 1400

        self.sub_joint = self.create_subscription(JointState, self.topic_joint_states, self._on_joint_state, 10)
        self.sub_ee = self.create_subscription(PoseStamped, self.topic_ee_pose, self._on_ee_pose, 10)
        self.sub_obj = self.create_subscription(PoseStamped, self.topic_object_pose, self._on_obj_pose, 10)
        self.pub_cmd = self.create_publisher(JointState, self.topic_joint_command, 10)
        self.pub_assist_start = self.create_publisher(Empty, "/assist_start", 10)
        self.pub_assist_stop = self.create_publisher(Empty, "/assist_stop", 10)

        self._load_dataset_stats()
        self._load_policy()

        self.timer = self.create_timer(1.0 / max(self.control_rate_hz, 1.0), self._control_step)
        self.get_logger().info("BC+assist ROS node started.")

    @staticmethod
    def _extract_joint_vector(msg: JointState) -> np.ndarray:
        target = np.zeros(9, dtype=np.float32)

        if msg.name and len(msg.name) == len(msg.position):
            idx = {n: i for i, n in enumerate(msg.name)}
            if all(n in idx for n in JOINT_NAMES):
                for i, n in enumerate(JOINT_NAMES):
                    target[i] = float(msg.position[idx[n]])
                return target

        n = min(9, len(msg.position))
        if n > 0:
            target[:n] = np.asarray(msg.position[:n], dtype=np.float32)
        return target

    def _load_dataset_stats(self):
        try:
            import h5py
        except Exception as exc:
            self.get_logger().warn(f"h5py unavailable, skip dataset stats: {exc}")
            return

        if not os.path.exists(self.dataset_path):
            self.get_logger().warn(f"Dataset not found: {self.dataset_path}")
            return

        act_min = np.full((9,), np.inf, dtype=np.float32)
        act_max = np.full((9,), -np.inf, dtype=np.float32)

        with h5py.File(self.dataset_path, "r") as f:
            data = f.get("data", None)
            if data is None or len(data.keys()) == 0:
                self.get_logger().warn("Dataset has no demos under /data")
                return

            for demo_name in data.keys():
                actions = data[demo_name]["actions"][:]
                act_min = np.minimum(act_min, np.min(actions, axis=0))
                act_max = np.maximum(act_max, np.max(actions, axis=0))

            first_demo = next(iter(data.keys()))
            self.expected_obs_dim = int(data[first_demo]["obs"]["policy"].shape[1])

        self.action_min = act_min
        self.action_max = act_max
        self.get_logger().info(f"Dataset loaded: obs_dim={self.expected_obs_dim}")

    def _load_policy(self):
        try:
            import robomimic.utils.file_utils as file_utils
            import robomimic.utils.torch_utils as torch_utils
        except Exception as exc:
            self.get_logger().error(f"robomimic unavailable, BC policy cannot run: {exc}")
            self.policy = None
            return

        if not os.path.exists(self.checkpoint_path):
            self.get_logger().error(f"Checkpoint not found: {self.checkpoint_path}")
            self.policy = None
            return

        device = torch_utils.get_torch_device(try_to_use_cuda=True)
        self.policy, ckpt = file_utils.policy_from_checkpoint(ckpt_path=self.checkpoint_path, device=device, verbose=False)
        self.policy.start_episode()
        self.has_action_stats = ckpt.get("action_normalization_stats", None) is not None
        self.expected_obs_dim = int(ckpt["shape_metadata"]["all_shapes"]["policy"][0])

        self.get_logger().info(
            f"Policy loaded: obs_dim={self.expected_obs_dim}, has_action_stats={self.has_action_stats}"
        )

    def _on_joint_state(self, msg: JointState):
        if len(msg.position) == 0:
            return

        pos = self._extract_joint_vector(msg)
        vel = np.zeros(9, dtype=np.float32)

        if msg.name and len(msg.name) == len(msg.velocity):
            idx = {n: i for i, n in enumerate(msg.name)}
            if all(n in idx for n in JOINT_NAMES):
                for i, n in enumerate(JOINT_NAMES):
                    vel[i] = float(msg.velocity[idx[n]])
            else:
                n = min(9, len(msg.velocity))
                vel[:n] = np.asarray(msg.velocity[:n], dtype=np.float32)
        elif len(msg.velocity) > 0:
            n = min(9, len(msg.velocity))
            vel[:n] = np.asarray(msg.velocity[:n], dtype=np.float32)

        self.joint_pos = pos
        self.joint_vel = vel

    def _on_ee_pose(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        self.ee_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        self.ee_quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)

    def _on_obj_pose(self, msg: PoseStamped):
        p = msg.pose.position
        self.obj_pos = np.array([p.x, p.y, p.z], dtype=np.float32)

    def _compose_obs(self, rel: np.ndarray) -> np.ndarray:
        assert self.joint_pos is not None and self.joint_vel is not None and self.obj_pos is not None and self.ee_pos is not None

        if self.obj_pos[2] <= 0.09:
            phase = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif np.linalg.norm(self.obj_pos - self.goal_position) > 0.06:
            phase = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            phase = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        goal_rel = self.goal_position - self.obj_pos

        if self.expected_obs_dim == 24:
            obs = np.concatenate([self.joint_pos, self.joint_vel, rel, self.obj_pos]).astype(np.float32)
        else:
            q = self.ee_quat_wxyz if self.ee_quat_wxyz is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            obs = np.concatenate([self.joint_pos, self.joint_vel, rel, self.obj_pos, self.ee_pos, q, goal_rel, phase]).astype(np.float32)

        return sanitize(obs)

    def _policy_delta(self, obs: np.ndarray) -> np.ndarray:
        if self.policy is None:
            return np.zeros(9, dtype=np.float32)

        normalized = sanitize(self.policy({"policy": obs}))
        if self.has_action_stats:
            return normalized

        if self.action_min is not None and self.action_max is not None:
            return 0.5 * (normalized + 1.0) * (self.action_max - self.action_min) + self.action_min
        return normalized

    def _publish_joint_target(self, target: np.ndarray):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = target.tolist()
        self.pub_cmd.publish(msg)

    def _control_step(self):
        if self.joint_pos is None or self.joint_vel is None or self.ee_pos is None or self.obj_pos is None:
            return
        if self.task_done:
            return

        rel = self.obj_pos - self.ee_pos
        dist = float(np.linalg.norm(rel))
        dist_xy = float(np.linalg.norm(rel[:2]))

        # Assist trigger: descend mode (aligned above object) or stall
        descend_mode = dist_xy < 0.07 and rel[2] < -0.10
        hovering_above = rel[2] < -0.16 and dist_xy < 0.18
        if self.prev_dist is not None and hovering_above and abs(self.prev_dist - dist) < 6e-4:
            self.stall_counter += 1
        else:
            if not (dist + self.stall_best_margin < self.best_dist):
                self.stall_counter += 1
            else:
                self.best_dist = dist
                self.stall_counter = 0
        self.prev_dist = dist

        if self.assist_cooldown > 0:
            self.assist_cooldown -= 1

        trigger_assist = (descend_mode or self.stall_counter > 25) and self.assist_cooldown == 0
        if trigger_assist and not self.assist_active:
            self.assist_active = True
            self.assist_steps = 0
            self.stall_counter = 0
            self.smoothed_delta = None
            self.pub_assist_start.publish(Empty())
            self.get_logger().warn(f"Assist LATCH ({'descend' if descend_mode else 'stall'}): dist={dist:.3f} rel={np.round(rel, 3)}")

        if self.assist_active:
            self.assist_steps += 1
            obj_z = float(self.obj_pos[2])
            reached_goal = (obj_z > 0.10) and (np.linalg.norm(self.obj_pos - self.goal_position) < 0.08)
            timed_out = self.assist_steps > self.ASSIST_MAX_STEPS

            if reached_goal:
                self.task_done = True
                self.get_logger().info("Task DONE — holding position. Bridge keeps control.")
                return

            if timed_out:
                self.assist_active = False
                self.assist_steps = 0
                self.best_dist = np.inf
                self.prev_dist = None
                self.stall_counter = 0
                self.assist_cooldown = 150
                self.pub_assist_stop.publish(Empty())
                self.get_logger().info("Assist released by timeout")

            self.step_count += 1
            if self.step_count % self.log_every_n_steps == 0:
                self.get_logger().info(f"ASSIST RUN steps={self.assist_steps} dist={dist:.3f} obj_z={self.obj_pos[2]:.3f}")
            return

        obs = self._compose_obs(rel)
        delta = self._policy_delta(obs)

        raw_arm = delta[:7].copy()
        arm = self.arm_delta_gain * delta[:7]

        in_fine_zone = (
            dist < self.near_object_dist
            and dist_xy < self.fine_xy_thresh
            and rel[2] > self.fine_relz_thresh
        )

        near_scale = 1.0
        deadband = self.delta_deadband_far
        if in_fine_zone:
            near_scale = self.near_object_gain_scale
            deadband = self.delta_deadband
            arm = near_scale * arm

        if (
            in_fine_zone
            and dist_xy < self.nodding_xy_thresh
            and abs(rel[2]) < self.nodding_abs_relz_max
        ):
            for j in self.nodding_damp_joints:
                ji = int(j)
                if 0 <= ji < 7:
                    arm[ji] = self.nodding_damp_scale * arm[ji]

        arm = np.clip(arm, -self.arm_delta_limit, self.arm_delta_limit)
        arm[np.abs(arm) < deadband] = 0.0

        if self.prev_arm_delta is not None:
            flip = (arm * self.prev_arm_delta < 0.0) & (
                np.abs(arm) < self.flip_suppress_threshold
            ) & (np.abs(self.prev_arm_delta) < self.flip_suppress_threshold)
            arm[flip] = 0.0
        self.prev_arm_delta = arm.copy()

        delta[:7] = arm
        delta = sanitize(delta)

        if self.smoothed_delta is None:
            self.smoothed_delta = delta.copy()
        else:
            self.smoothed_delta = self.ema_alpha * delta + (1.0 - self.ema_alpha) * self.smoothed_delta
        self.smoothed_delta = sanitize(self.smoothed_delta)

        if abs(rel[2]) < self.grasp_z and dist_xy < self.grasp_xy:
            self.smoothed_delta[7] = -0.04
            self.smoothed_delta[8] = -0.04

        target = self.joint_pos.copy()
        target[:9] = target[:9] + self.smoothed_delta[:9]

        if self.prev_target is not None:
            step = target[:7] - self.prev_target[:7]
            step = np.clip(step, -self.arm_slew_limit, self.arm_slew_limit)
            target[:7] = self.prev_target[:7] + step

        target = np.clip(target, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER).astype(np.float32)
        self.prev_target = target.copy()
        self._publish_joint_target(target)

        self.step_count += 1
        if self.step_count % self.log_every_n_steps == 0:
            self.get_logger().info(
                f"BC RUN dist={dist:.3f} rel={np.round(rel, 3)} fine={int(in_fine_zone)} "
                f"scale={near_scale:.2f} stall={self.stall_counter} best={self.best_dist:.3f} "
                f"mean|arm|={np.mean(np.abs(self.smoothed_delta[:7])):.5f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = CustomPlayRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
