#!/usr/bin/env python3
import math
import os
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped


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


class HybridPolicyNode(Node):
    """ROS2 deployment node: BC policy + assist state machine (route-1 engineering)."""

    def __init__(self):
        super().__init__("franka_hybrid_policy_node")

        # ---------------- parameters ----------------
        self.declare_parameter("topic_joint_states", "/joint_states")
        self.declare_parameter("topic_ee_pose", "/franka/ee_pose")
        self.declare_parameter("topic_object_pose", "/udisk/pose")
        self.declare_parameter("topic_joint_command", "/joint_command")

        self.declare_parameter("control_rate_hz", 60.0)
        self.declare_parameter("goal_position", [0.40, 0.0, 0.40])

        self.declare_parameter(
            "checkpoint_path",
            "/home/jmx001/my_program/my_robot_project/logs/robomimic/Isaac-UDisk-Grasp-v0/bc/20260320182155/models/model_epoch_100.pth",
        )
        self.declare_parameter("dataset_path", "/home/jmx001/my_program/my_robot_project/logs/robomimic/rmpflow_expert.hdf5")

        self.declare_parameter("enable_assist", True)
        self.declare_parameter("assist_descend_xy", 0.11)
        self.declare_parameter("assist_descend_z", -0.10)
        self.declare_parameter("assist_hover_xy", 0.18)
        self.declare_parameter("assist_hover_z", -0.16)
        self.declare_parameter("assist_stuck_eps", 6e-4)
        self.declare_parameter("assist_stuck_steps", 25)
        self.declare_parameter("assist_max_steps", 1400)
        self.declare_parameter("assist_trigger_max_dist", 0.30)
        self.declare_parameter("assist_mix_policy", 0.15)
        self.declare_parameter("assist_knn_k", 96)
        self.declare_parameter("assist_requires_policy", True)

        self.declare_parameter("grasp_xy", 0.09)
        self.declare_parameter("grasp_z", 0.10)
        self.declare_parameter("lift_success_z", 0.10)
        self.declare_parameter("goal_tolerance", 0.08)

        self.declare_parameter("arm_delta_limit", [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        self.declare_parameter("soft_limit_margin", 0.05)
        self.declare_parameter("soft_limit_pullback", 0.003)

        self.declare_parameter("velocity_damping", 0.06)
        self.declare_parameter("near_object_dist", 0.24)
        self.declare_parameter("near_object_gain", 0.45)
        self.declare_parameter("delta_deadband", 4e-4)
        self.declare_parameter("target_slew_limit", [0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006])
        self.declare_parameter("assist_relock_cooldown_steps", 45)
        self.declare_parameter("assist_use_descend_trigger", False)
        self.declare_parameter("assist_descend_policy_weight", 0.7)
        self.declare_parameter("assist_descend_expert_weight", 0.3)
        self.declare_parameter("descent_progress_eps", 5e-4)
        self.declare_parameter("descent_stall_steps", 50)
        self.declare_parameter("anti_wobble_xy", 0.12)
        self.declare_parameter("anti_wobble_relz", -0.14)
        self.declare_parameter("descent_stall_min_assist_steps", 20)
        self.declare_parameter("descent_rescue_steps", 18)
        self.declare_parameter("descent_rescue_policy_weight", 0.1)
        self.declare_parameter("descent_rescue_expert_weight", 0.9)
        self.declare_parameter("descent_rescue_xy_damp", 0.15)
        self.declare_parameter("descend_lateral_scale", 0.05)
        self.declare_parameter("descend_lock_xy_thresh", 0.10)
        self.declare_parameter("descend_pitch_boost", 1.50)
        self.declare_parameter("descent_rescue_max_cycles", 3)
        self.declare_parameter("descent_rescue_suspend_steps", 180)
        self.declare_parameter("descend_force_xy", 0.03)
        self.declare_parameter("descend_force_relz", -0.12)
        self.declare_parameter("descend_force_stall_steps", 20)
        self.declare_parameter("descend_min_pulse", [0.0025, 0.0012, 0.0020])
        self.declare_parameter("descend_burst_enable", True)
        self.declare_parameter("descend_burst_softlimit_bypass", True)
        self.declare_parameter("descend_burst_pulse", [0.010, 0.008, 0.010])
        self.declare_parameter("descend_burst_slew", [0.012, 0.012, 0.012])
        self.declare_parameter("joint_target_fallback_enable", True)
        self.declare_parameter("joint_target_fallback_weight", 0.75)
        self.declare_parameter("joint_target_fallback_stall_steps", 24)
        self.declare_parameter("joint_target_knn_k", 64)
        self.declare_parameter("no_policy_fallback_weight", 0.45)
        self.declare_parameter("force_joint_fallback", False)

        self.topic_joint_states = self.get_parameter("topic_joint_states").value
        self.topic_ee_pose = self.get_parameter("topic_ee_pose").value
        self.topic_object_pose = self.get_parameter("topic_object_pose").value
        self.topic_joint_command = self.get_parameter("topic_joint_command").value

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.goal_position = np.array(self.get_parameter("goal_position").value, dtype=np.float32)

        self.checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self.dataset_path = str(self.get_parameter("dataset_path").value)

        self.enable_assist = bool(self.get_parameter("enable_assist").value)
        self.assist_descend_xy = float(self.get_parameter("assist_descend_xy").value)
        self.assist_descend_z = float(self.get_parameter("assist_descend_z").value)
        self.assist_hover_xy = float(self.get_parameter("assist_hover_xy").value)
        self.assist_hover_z = float(self.get_parameter("assist_hover_z").value)
        self.assist_stuck_eps = float(self.get_parameter("assist_stuck_eps").value)
        self.assist_stuck_steps = int(self.get_parameter("assist_stuck_steps").value)
        self.assist_max_steps = int(self.get_parameter("assist_max_steps").value)
        self.assist_trigger_max_dist = float(self.get_parameter("assist_trigger_max_dist").value)
        self.assist_mix_policy = float(self.get_parameter("assist_mix_policy").value)
        self.assist_knn_k = int(self.get_parameter("assist_knn_k").value)
        self.assist_requires_policy = bool(self.get_parameter("assist_requires_policy").value)

        self.grasp_xy = float(self.get_parameter("grasp_xy").value)
        self.grasp_z = float(self.get_parameter("grasp_z").value)
        self.lift_success_z = float(self.get_parameter("lift_success_z").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)

        self.arm_delta_limit = np.array(self.get_parameter("arm_delta_limit").value, dtype=np.float32)
        self.soft_limit_margin = float(self.get_parameter("soft_limit_margin").value)
        self.soft_limit_pullback = float(self.get_parameter("soft_limit_pullback").value)

        self.velocity_damping = float(self.get_parameter("velocity_damping").value)
        self.near_object_dist = float(self.get_parameter("near_object_dist").value)
        self.near_object_gain = float(self.get_parameter("near_object_gain").value)
        self.delta_deadband = float(self.get_parameter("delta_deadband").value)
        self.target_slew_limit = np.array(self.get_parameter("target_slew_limit").value, dtype=np.float32)
        self.assist_relock_cooldown_steps = int(self.get_parameter("assist_relock_cooldown_steps").value)
        self.assist_use_descend_trigger = bool(self.get_parameter("assist_use_descend_trigger").value)
        self.assist_descend_policy_weight = float(self.get_parameter("assist_descend_policy_weight").value)
        self.assist_descend_expert_weight = float(self.get_parameter("assist_descend_expert_weight").value)
        self.descent_progress_eps = float(self.get_parameter("descent_progress_eps").value)
        self.descent_stall_steps = int(self.get_parameter("descent_stall_steps").value)
        self.anti_wobble_xy = float(self.get_parameter("anti_wobble_xy").value)
        self.anti_wobble_relz = float(self.get_parameter("anti_wobble_relz").value)
        self.descent_stall_min_assist_steps = int(self.get_parameter("descent_stall_min_assist_steps").value)
        self.descent_rescue_steps = int(self.get_parameter("descent_rescue_steps").value)
        self.descent_rescue_policy_weight = float(self.get_parameter("descent_rescue_policy_weight").value)
        self.descent_rescue_expert_weight = float(self.get_parameter("descent_rescue_expert_weight").value)
        self.descent_rescue_xy_damp = float(self.get_parameter("descent_rescue_xy_damp").value)
        self.descend_lateral_scale = float(self.get_parameter("descend_lateral_scale").value)
        self.descend_lock_xy_thresh = float(self.get_parameter("descend_lock_xy_thresh").value)
        self.descend_pitch_boost = float(self.get_parameter("descend_pitch_boost").value)
        self.descent_rescue_max_cycles = int(self.get_parameter("descent_rescue_max_cycles").value)
        self.descent_rescue_suspend_steps = int(self.get_parameter("descent_rescue_suspend_steps").value)
        self.descend_force_xy = float(self.get_parameter("descend_force_xy").value)
        self.descend_force_relz = float(self.get_parameter("descend_force_relz").value)
        self.descend_force_stall_steps = int(self.get_parameter("descend_force_stall_steps").value)
        self.descend_min_pulse = np.array(self.get_parameter("descend_min_pulse").value, dtype=np.float32)
        if self.descend_min_pulse.shape[0] < 3:
            self.descend_min_pulse = np.pad(self.descend_min_pulse, (0, 3 - self.descend_min_pulse.shape[0]), constant_values=0.0015)
        elif self.descend_min_pulse.shape[0] > 3:
            self.descend_min_pulse = self.descend_min_pulse[:3]

        self.descend_burst_enable = bool(self.get_parameter("descend_burst_enable").value)
        self.descend_burst_softlimit_bypass = bool(self.get_parameter("descend_burst_softlimit_bypass").value)
        self.descend_burst_pulse = np.array(self.get_parameter("descend_burst_pulse").value, dtype=np.float32)
        self.descend_burst_slew = np.array(self.get_parameter("descend_burst_slew").value, dtype=np.float32)
        if self.descend_burst_pulse.shape[0] < 3:
            self.descend_burst_pulse = np.pad(self.descend_burst_pulse, (0, 3 - self.descend_burst_pulse.shape[0]), constant_values=0.008)
        elif self.descend_burst_pulse.shape[0] > 3:
            self.descend_burst_pulse = self.descend_burst_pulse[:3]
        if self.descend_burst_slew.shape[0] < 3:
            self.descend_burst_slew = np.pad(self.descend_burst_slew, (0, 3 - self.descend_burst_slew.shape[0]), constant_values=0.012)
        elif self.descend_burst_slew.shape[0] > 3:
            self.descend_burst_slew = self.descend_burst_slew[:3]

        self.joint_target_fallback_enable = bool(self.get_parameter("joint_target_fallback_enable").value)
        self.joint_target_fallback_weight = float(self.get_parameter("joint_target_fallback_weight").value)
        self.joint_target_fallback_stall_steps = int(self.get_parameter("joint_target_fallback_stall_steps").value)
        self.joint_target_knn_k = int(self.get_parameter("joint_target_knn_k").value)
        self.no_policy_fallback_weight = float(self.get_parameter("no_policy_fallback_weight").value)
        self.force_joint_fallback = bool(self.get_parameter("force_joint_fallback").value)

        # ---------------- runtime state ----------------
        self.joint_pos: Optional[np.ndarray] = None
        self.joint_vel: Optional[np.ndarray] = None
        self.ee_pos: Optional[np.ndarray] = None
        self.ee_quat: Optional[np.ndarray] = None
        self.obj_pos: Optional[np.ndarray] = None

        self.policy = None
        self.has_action_stats = False
        self.expected_obs_dim = 37
        self.action_min = None
        self.action_max = None

        self.rel_bank = None
        self.act_bank = None
        self.joint_bank = None
        self.descent_template = np.zeros(7, dtype=np.float32)

        self.assist_active = False
        self.assist_steps = 0
        self.stuck_counter = 0
        self.prev_dist_to_obj: Optional[float] = None
        self.smoothed_delta_arm: Optional[np.ndarray] = None
        self.assist_relock_cooldown = 0
        self.prev_rel_z: Optional[float] = None
        self.no_descent_steps = 0
        self.descent_rescue_remaining = 0
        self.descent_rescue_cycles = 0

        self.last_target = np.array([0.0, -1.1, 0.0, -2.3, 0.0, 2.4, 0.8, 0.04, 0.04], dtype=np.float32)

        # ---------------- io ----------------
        self.sub_joint = self.create_subscription(JointState, self.topic_joint_states, self._on_joint_state, 10)
        self.sub_ee = self.create_subscription(PoseStamped, self.topic_ee_pose, self._on_ee_pose, 10)
        self.sub_obj = self.create_subscription(PoseStamped, self.topic_object_pose, self._on_obj_pose, 10)
        self.pub_cmd = self.create_publisher(JointState, self.topic_joint_command, 10)

        # ---------------- load assets ----------------
        self._load_dataset_bank()
        self._load_policy()

        self.assist_runtime_enabled = self.enable_assist and (not self.force_joint_fallback) and (not self.assist_requires_policy or self.policy is not None)
        if self.force_joint_fallback and self.enable_assist:
            self.get_logger().warn("Assist disabled at runtime because force_joint_fallback is enabled.")
        elif self.enable_assist and not self.assist_runtime_enabled:
            self.get_logger().warn("Assist disabled at runtime because policy is unavailable.")

        self.timer = self.create_timer(1.0 / max(self.control_rate_hz, 1.0), self._control_step)
        self.get_logger().info("Hybrid policy node started. Waiting for /joint_states, /franka/ee_pose, /udisk/pose")

    @staticmethod
    def _sanitize(x: np.ndarray, fallback: float = 0.0) -> np.ndarray:
        return np.nan_to_num(x, nan=fallback, posinf=fallback, neginf=fallback).astype(np.float32, copy=False)

    def _load_dataset_bank(self):
        try:
            import h5py
        except Exception as exc:
            self.get_logger().error(f"h5py import failed: {exc}")
            return

        if not os.path.exists(self.dataset_path):
            self.get_logger().warn(f"Dataset not found: {self.dataset_path}. KNN rescue disabled.")
            return

        rel_list = []
        act_list = []
        joint_list = []
        descent_act_list = []
        act_min = np.full((9,), np.inf, dtype=np.float32)
        act_max = np.full((9,), -np.inf, dtype=np.float32)

        with h5py.File(self.dataset_path, "r") as f:
            for demo_name in f["data"].keys():
                obs = f["data"][demo_name]["obs"]["policy"][:]
                actions = f["data"][demo_name]["actions"][:]
                rel = obs[:, 18:21].astype(np.float32)
                act = actions[:, :7].astype(np.float32)
                joint = obs[:, :7].astype(np.float32)
                rel_list.append(rel)
                act_list.append(act)
                joint_list.append(joint)
                if rel.shape[0] > 1:
                    rel_t = rel[:-1]
                    rel_next = rel[1:]
                    d_rel_z = rel_next[:, 2] - rel_t[:, 2]
                    m_desc = (np.linalg.norm(rel_t[:, :2], axis=1) < 0.12) & (rel_t[:, 2] < -0.10) & (d_rel_z > 5e-4)
                    if np.any(m_desc):
                        descent_act_list.append(act[:-1][m_desc])
                act_min = np.minimum(act_min, np.min(actions, axis=0))
                act_max = np.maximum(act_max, np.max(actions, axis=0))

            all_obs_dim = f["data"][next(iter(f["data"].keys()))]["obs"]["policy"].shape[1]
            self.expected_obs_dim = int(all_obs_dim)

        self.rel_bank = np.concatenate(rel_list, axis=0)
        self.act_bank = np.concatenate(act_list, axis=0)
        self.joint_bank = np.concatenate(joint_list, axis=0)
        self.action_min = act_min
        self.action_max = act_max

        # descent prior from data manifold (prefer transitions with improving rel_z)
        if len(descent_act_list) > 0:
            desc_bank = np.concatenate(descent_act_list, axis=0)
            self.descent_template = np.mean(desc_bank, axis=0).astype(np.float32)
        else:
            dxy = np.linalg.norm(self.rel_bank[:, :2], axis=1)
            m = (dxy < 0.10) & (self.rel_bank[:, 2] < -0.12)
            if np.any(m):
                self.descent_template = np.mean(self.act_bank[m], axis=0).astype(np.float32)

        self.get_logger().info(
            f"Dataset bank loaded: rel={self.rel_bank.shape}, act={self.act_bank.shape}, obs_dim={self.expected_obs_dim}"
        )

    def _load_policy(self):
        try:
            import robomimic.utils.file_utils as file_utils
            import robomimic.utils.torch_utils as torch_utils
        except Exception as exc:
            self.get_logger().warn(f"robomimic not available, running assist-only mode: {exc}")
            self.policy = None
            return

        if not os.path.exists(self.checkpoint_path):
            self.get_logger().warn(f"Checkpoint not found: {self.checkpoint_path}. Running assist-only mode.")
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

        pos = np.zeros(9, dtype=np.float32)
        vel = np.zeros(9, dtype=np.float32)

        if msg.name and len(msg.name) == len(msg.position):
            idx = {n: i for i, n in enumerate(msg.name)}
            ok = all(n in idx for n in JOINT_NAMES)
            if ok:
                for i, n in enumerate(JOINT_NAMES):
                    j = idx[n]
                    pos[i] = msg.position[j]
                    if j < len(msg.velocity):
                        vel[i] = msg.velocity[j]
            else:
                n = min(9, len(msg.position))
                pos[:n] = np.array(msg.position[:n], dtype=np.float32)
                if len(msg.velocity) > 0:
                    vel[: min(9, len(msg.velocity))] = np.array(msg.velocity[: min(9, len(msg.velocity))], dtype=np.float32)
        else:
            n = min(9, len(msg.position))
            pos[:n] = np.array(msg.position[:n], dtype=np.float32)
            if len(msg.velocity) > 0:
                vel[: min(9, len(msg.velocity))] = np.array(msg.velocity[: min(9, len(msg.velocity))], dtype=np.float32)

        self.joint_pos = pos
        self.joint_vel = vel

    def _on_ee_pose(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        self.ee_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        self.ee_quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)

    def _on_obj_pose(self, msg: PoseStamped):
        p = msg.pose.position
        self.obj_pos = np.array([p.x, p.y, p.z], dtype=np.float32)

    def _knn_delta(self, rel_xyz: np.ndarray) -> Optional[np.ndarray]:
        if self.rel_bank is None or self.act_bank is None or self.rel_bank.shape[0] == 0:
            return None
        d = np.linalg.norm(self.rel_bank - rel_xyz[None, :], axis=1)
        k = min(self.assist_knn_k, d.size)
        idx = np.argpartition(d, k - 1)[:k]
        d_sel = d[idx]
        if np.min(d_sel) > 0.25:
            return None
        w = 1.0 / (d_sel + 1e-4)
        delta = np.sum(self.act_bank[idx] * w[:, None], axis=0) / np.sum(w)
        return delta.astype(np.float32)

    def _knn_joint_target(self, rel_xyz: np.ndarray) -> Optional[np.ndarray]:
        if self.rel_bank is None or self.joint_bank is None or self.rel_bank.shape[0] == 0:
            return None
        d = np.linalg.norm(self.rel_bank - rel_xyz[None, :], axis=1)
        if d.size == 0:
            return None
        k = min(self.joint_target_knn_k, d.size)
        idx = np.argpartition(d, k - 1)[:k]
        d_sel = d[idx]
        w = 1.0 / (d_sel + 1e-4)
        jt = np.sum(self.joint_bank[idx] * w[:, None], axis=0) / np.sum(w)
        return jt.astype(np.float32)

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
            ee_quat = self.ee_quat if self.ee_quat is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            obs = np.concatenate([self.joint_pos, self.joint_vel, rel, self.obj_pos, self.ee_pos, ee_quat, goal_rel, phase]).astype(np.float32)

        return self._sanitize(obs, 0.0)

    def _policy_delta(self, obs: np.ndarray) -> np.ndarray:
        if self.policy is None:
            return np.zeros(9, dtype=np.float32)

        a = self._sanitize(self.policy({"policy": obs}), 0.0)
        if self.has_action_stats:
            return a
        if self.action_min is not None and self.action_max is not None:
            return 0.5 * (a + 1.0) * (self.action_max - self.action_min) + self.action_min
        return a

    def _publish_joint_target(self, target: np.ndarray):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = target.tolist()
        self.pub_cmd.publish(msg)

    def _control_step(self):
        if self.joint_pos is None or self.joint_vel is None or self.ee_pos is None or self.obj_pos is None:
            return

        rel = self.obj_pos - self.ee_pos
        dist = float(np.linalg.norm(rel))
        dist_xy = float(np.linalg.norm(rel[:2]))

        if self.assist_relock_cooldown > 0:
            self.assist_relock_cooldown -= 1

        obs = self._compose_obs(rel)
        if self.force_joint_fallback:
            delta = np.zeros(9, dtype=np.float32)
        else:
            delta = self._policy_delta(obs)

        hovering = rel[2] < self.assist_hover_z and dist_xy < self.assist_hover_xy
        if self.prev_dist_to_obj is not None and hovering and abs(self.prev_dist_to_obj - dist) < self.assist_stuck_eps:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.prev_dist_to_obj = dist

        descend_mode = dist_xy < self.assist_descend_xy and rel[2] < self.assist_descend_z
        if descend_mode:
            if self.prev_rel_z is not None and (rel[2] - self.prev_rel_z) > self.descent_progress_eps:
                self.no_descent_steps = 0
                self.descent_rescue_cycles = 0
            else:
                self.no_descent_steps += 1
        else:
            self.no_descent_steps = 0
        self.prev_rel_z = float(rel[2])
        assist_trigger_core = self.stuck_counter > self.assist_stuck_steps
        if self.assist_use_descend_trigger:
            assist_trigger_core = assist_trigger_core or descend_mode
        trigger_assist = self.assist_runtime_enabled and self.assist_relock_cooldown == 0 and assist_trigger_core and dist < self.assist_trigger_max_dist

        if trigger_assist and not self.assist_active:
            self.assist_active = True
            self.assist_steps = 0
            self.no_descent_steps = 0
            self.descent_rescue_cycles = 0
            self.prev_rel_z = float(rel[2])
            self.get_logger().info(f"Assist latched: rel={np.round(rel, 3)}, dist={dist:.3f}")

        delta_arm = delta[:7].copy()

        if (self.policy is None or self.force_joint_fallback) and self.joint_target_fallback_enable and not self.assist_active:
            jt = self._knn_joint_target(rel.astype(np.float32))
            if jt is not None:
                w0 = float(np.clip(self.no_policy_fallback_weight, 0.0, 1.0))
                delta_arm = w0 * (jt - self.joint_pos[:7])
                delta_arm[[0, 2, 4, 6]] *= 0.6
                if descend_mode:
                    delta_arm[[0, 2, 4, 6]] *= 0.4
                    delta_arm[[1, 3, 5]] *= 1.25

        if self.assist_active:
            self.assist_steps += 1

            expert = self.descent_template.copy()
            knn = self._knn_delta(rel.astype(np.float32))
            if knn is not None:
                expert = 0.3 * expert + 0.7 * knn

            if descend_mode:
                delta_arm = self.assist_descend_policy_weight * delta_arm + self.assist_descend_expert_weight * expert
            else:
                delta_arm = self.assist_mix_policy * delta_arm + (1.0 - self.assist_mix_policy) * expert

            reached_goal = (self.obj_pos[2] > self.lift_success_z) and (np.linalg.norm(self.obj_pos - self.goal_position) < self.goal_tolerance)
            timed_out = self.assist_steps > self.assist_max_steps
            if reached_goal or timed_out:
                self.assist_active = False
                self.assist_steps = 0
                self.stuck_counter = 0
                if timed_out:
                    self.assist_relock_cooldown = self.assist_relock_cooldown_steps
                    self.smoothed_delta_arm = np.zeros(7, dtype=np.float32)
                reason = "goal" if reached_goal else "timeout"
                self.get_logger().info(f"Assist released by {reason}")

            if descend_mode and self.no_descent_steps > self.descent_stall_steps and self.assist_steps > self.descent_stall_min_assist_steps:
                self.no_descent_steps = 0
                self.descent_rescue_cycles += 1
                if self.descent_rescue_cycles > self.descent_rescue_max_cycles:
                    self.assist_active = False
                    self.assist_steps = 0
                    self.descent_rescue_remaining = 0
                    self.assist_relock_cooldown = max(self.assist_relock_cooldown_steps, self.descent_rescue_suspend_steps)
                    self.smoothed_delta_arm = np.zeros(7, dtype=np.float32)
                    self.descent_rescue_cycles = 0
                    self.get_logger().info("Assist suspended by repeated descent-stall")
                else:
                    self.descent_rescue_remaining = self.descent_rescue_steps
                    self.get_logger().info(f"Descent rescue pulse armed ({self.descent_rescue_cycles}/{self.descent_rescue_max_cycles})")

        if self.descent_rescue_remaining > 0:
            rescue = self.descent_template.copy()
            rescue[[0, 2, 4, 6]] *= self.descent_rescue_xy_damp
            delta_arm = self.descent_rescue_policy_weight * delta_arm + self.descent_rescue_expert_weight * rescue
            self.descent_rescue_remaining -= 1

        delta_arm = self._sanitize(delta_arm, 0.0)

        delta_arm = delta_arm - self.velocity_damping * self.joint_vel[:7]
        if dist < self.near_object_dist:
            delta_arm = self.near_object_gain * delta_arm
            delta_arm[[0, 6]] *= 0.5
        delta_arm[np.abs(delta_arm) < self.delta_deadband] = 0.0
        if dist_xy < self.anti_wobble_xy and rel[2] < self.anti_wobble_relz:
            delta_arm[[0, 4, 6]] *= 0.25

        if self.assist_active and rel[2] < self.anti_wobble_relz:
            delta_arm[[0, 2, 4, 6]] *= 0.40

        if descend_mode:
            delta_arm[[0, 2, 4, 6]] *= self.descend_lateral_scale
            if dist_xy < self.descend_lock_xy_thresh:
                delta_arm[[0, 2, 4, 6]] = 0.0
            delta_arm[[1, 3, 5]] *= self.descend_pitch_boost

        forced_descend = self.assist_active and descend_mode and (dist_xy < self.descend_force_xy) and (rel[2] < self.descend_force_relz) and (self.no_descent_steps > self.descend_force_stall_steps)
        forced_descend_burst = False
        if forced_descend:
            delta_arm[[0, 2, 4, 6]] = 0.0
            pulse_sign = np.sign(self.descent_template[[1, 3, 5]])
            pulse_sign[pulse_sign == 0.0] = 1.0
            min_pulse = self.descend_min_pulse * pulse_sign
            if self.descend_burst_enable:
                burst_pulse = np.maximum(np.abs(self.descend_burst_pulse), np.abs(min_pulse)) * pulse_sign
                delta_arm[[1, 3, 5]] = burst_pulse
                forced_descend_burst = True
            else:
                for i, joint_idx in enumerate([1, 3, 5]):
                    if abs(delta_arm[joint_idx]) < abs(min_pulse[i]):
                        delta_arm[joint_idx] = float(min_pulse[i])

        fallback_active = (
            self.joint_target_fallback_enable
            and self.assist_active
            and descend_mode
            and self.no_descent_steps > self.joint_target_fallback_stall_steps
        )
        if fallback_active:
            jt = self._knn_joint_target(rel.astype(np.float32))
            if jt is not None:
                delta_from_jt = jt - self.joint_pos[:7]
                w = float(np.clip(self.joint_target_fallback_weight, 0.0, 1.0))
                delta_arm = (1.0 - w) * delta_arm + w * delta_from_jt
                delta_arm[[0, 2, 4, 6]] *= 0.5

        # soft-limit protection: if already near limits, do not push further outward
        current_arm = self.joint_pos[:7]
        near_lower = current_arm < (JOINT_LIMITS_LOWER[:7] + self.soft_limit_margin)
        near_upper = current_arm > (JOINT_LIMITS_UPPER[:7] - self.soft_limit_margin)
        if forced_descend_burst and self.descend_burst_softlimit_bypass:
            near_lower[[1, 3, 5]] = False
            near_upper[[1, 3, 5]] = False
        delta_arm = np.where(near_lower & (delta_arm < 0.0), self.soft_limit_pullback, delta_arm)
        delta_arm = np.where(near_upper & (delta_arm > 0.0), -self.soft_limit_pullback, delta_arm)

        delta_arm = np.clip(delta_arm, -self.arm_delta_limit, self.arm_delta_limit)

        if self.smoothed_delta_arm is None or forced_descend_burst:
            self.smoothed_delta_arm = delta_arm.copy()
        else:
            self.smoothed_delta_arm = 0.7 * delta_arm + 0.3 * self.smoothed_delta_arm
        self.smoothed_delta_arm = self._sanitize(self.smoothed_delta_arm, 0.0)
        self.smoothed_delta_arm = np.clip(self.smoothed_delta_arm, -self.arm_delta_limit, self.arm_delta_limit)

        target = self.joint_pos.copy()
        desired_arm = target[:7] + self.smoothed_delta_arm
        prev_arm = self.last_target[:7]
        arm_step = np.clip(desired_arm - prev_arm, -self.target_slew_limit, self.target_slew_limit)
        if forced_descend_burst:
            arm_step[[0, 2, 4, 6]] = 0.0
            arm_step[[1, 3, 5]] = np.clip(
                desired_arm[[1, 3, 5]] - prev_arm[[1, 3, 5]],
                -self.descend_burst_slew,
                self.descend_burst_slew,
            )
        target[:7] = prev_arm + arm_step

        close_gripper = (dist_xy < self.grasp_xy) and (abs(rel[2]) < self.grasp_z)
        target[7] = 0.0 if close_gripper else 0.04
        target[8] = 0.0 if close_gripper else 0.04

        target = np.clip(target, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER).astype(np.float32)
        self.last_target = target.copy()

        self._publish_joint_target(target)

        if self.assist_active:
            self.get_logger().info(
                f"Assist RUN steps={self.assist_steps} dist={dist:.3f} rel={np.round(rel,3)} obj_z={self.obj_pos[2]:.3f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = HybridPolicyNode()
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
