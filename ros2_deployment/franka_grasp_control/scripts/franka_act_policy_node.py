#!/usr/bin/env python3
"""ROS2 deployment node for ACT (Action Chunking with Transformers) policy.

Subscribes to /joint_states, /franka/ee_pose, /udisk/pose.
Publishes /joint_command (absolute joint targets).
Publishes /assist_start, /assist_stop when stuck detection triggers.
"""
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty


# ======================== ACT Model (from train_act.py) ========================
class ACTPolicy(nn.Module):
    def __init__(self, obs_dim=37, action_dim=9, chunk_size=50,
                 latent_dim=32, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        enc_input_dim = obs_dim + chunk_size * action_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_input_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.latent_proj = nn.Linear(latent_dim, d_model)

        self.action_queries = nn.Embedding(chunk_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def decode(self, obs, z):
        B = obs.shape[0]
        obs_token = self.obs_encoder(obs).unsqueeze(1)
        z_token = self.latent_proj(z).unsqueeze(1)
        memory = torch.cat([obs_token, z_token], dim=1)
        queries = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1)
        decoded = self.transformer_decoder(tgt=queries, memory=memory)
        return self.action_head(decoded)

    def inference(self, obs):
        B = obs.shape[0]
        z = torch.zeros(B, self.latent_dim, device=obs.device)
        return self.decode(obs, z)


# ======================== Constants ========================
JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
    "panda_finger_joint1", "panda_finger_joint2",
]
J_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0], dtype=np.float32)
J_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04], dtype=np.float32)


class ACTPolicyNode(Node):
    def __init__(self):
        super().__init__("franka_act_policy_node")

        # --- parameters ---
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("norm_stats_path", "")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("goal_position", [0.40, 0.0, 0.40])
        self.declare_parameter("temporal_decay", 0.1)
        self.declare_parameter("target_slew_limit", [0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006])
        self.declare_parameter("grasp_xy", 0.09)
        self.declare_parameter("grasp_z", 0.10)
        self.declare_parameter("stuck_eps", 6e-4)
        self.declare_parameter("stuck_steps", 25)
        self.declare_parameter("assist_max_steps", 1400)

        self.checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self.norm_stats_path = str(self.get_parameter("norm_stats_path").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.goal = np.array(self.get_parameter("goal_position").value, dtype=np.float32)
        self.temporal_decay = float(self.get_parameter("temporal_decay").value)
        self.slew_limit = np.array(self.get_parameter("target_slew_limit").value, dtype=np.float32)
        self.grasp_xy = float(self.get_parameter("grasp_xy").value)
        self.grasp_z = float(self.get_parameter("grasp_z").value)
        self.stuck_eps = float(self.get_parameter("stuck_eps").value)
        self.stuck_steps_thresh = int(self.get_parameter("stuck_steps").value)
        self.assist_max_steps = int(self.get_parameter("assist_max_steps").value)

        # --- state ---
        self.joint_pos: Optional[np.ndarray] = None
        self.joint_vel: Optional[np.ndarray] = None
        self.ee_pos: Optional[np.ndarray] = None
        self.ee_quat: Optional[np.ndarray] = None
        self.obj_pos: Optional[np.ndarray] = None

        self.last_target = np.array([0.0, -1.1, 0.0, -2.3, 0.0, 2.4, 0.8, 0.04, 0.04], dtype=np.float32)
        self.action_chunks = []  # list of (chunk_array, start_step)
        self.step = 0
        self.stuck_counter = 0
        self.prev_dist: Optional[float] = None
        self.assist_active = False
        self.assist_steps = 0

        # --- model ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.act_min = self.act_max = self.act_range = None
        self.obs_min = self.obs_max = self.obs_range = None
        self._load_model()

        # --- ROS2 io ---
        self.sub_joint = self.create_subscription(JointState, "/joint_states", self._on_joint, 10)
        self.sub_ee = self.create_subscription(PoseStamped, "/franka/ee_pose", self._on_ee, 10)
        self.sub_obj = self.create_subscription(PoseStamped, "/udisk/pose", self._on_obj, 10)
        self.pub_cmd = self.create_publisher(JointState, "/joint_command", 10)
        self.pub_assist_start = self.create_publisher(Empty, "/assist_start", 10)
        self.pub_assist_stop = self.create_publisher(Empty, "/assist_stop", 10)

        self.timer = self.create_timer(1.0 / max(self.control_rate_hz, 1.0), self._control_step)
        self.get_logger().info("ACT policy node started.")

    def _load_model(self):
        if not os.path.exists(self.checkpoint_path):
            self.get_logger().error(f"Checkpoint not found: {self.checkpoint_path}")
            return
        if not os.path.exists(self.norm_stats_path):
            self.get_logger().error(f"Norm stats not found: {self.norm_stats_path}")
            return

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        self.model = ACTPolicy(
            obs_dim=cfg["obs_dim"], action_dim=cfg["action_dim"],
            chunk_size=cfg["chunk_size"], latent_dim=cfg["latent_dim"],
            d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        stats = np.load(self.norm_stats_path)
        self.act_min, self.act_max = stats["act_min"], stats["act_max"]
        self.obs_min, self.obs_max = stats["obs_min"], stats["obs_max"]
        self.act_range = self.act_max - self.act_min
        self.act_range[self.act_range < 1e-6] = 1.0
        self.obs_range = self.obs_max - self.obs_min
        self.obs_range[self.obs_range < 1e-6] = 1.0

        self.get_logger().info(f"ACT model loaded: chunk_size={cfg['chunk_size']}, device={self.device}")

    # --- callbacks ---
    def _on_joint(self, msg: JointState):
        if len(msg.position) == 0:
            return
        pos = np.zeros(9, dtype=np.float32)
        vel = np.zeros(9, dtype=np.float32)
        if msg.name and len(msg.name) == len(msg.position):
            idx = {n: i for i, n in enumerate(msg.name)}
            for i, n in enumerate(JOINT_NAMES):
                if n in idx:
                    j = idx[n]
                    pos[i] = msg.position[j]
                    if j < len(msg.velocity):
                        vel[i] = msg.velocity[j]
        else:
            n = min(9, len(msg.position))
            pos[:n] = np.array(msg.position[:n], dtype=np.float32)
            if len(msg.velocity) > 0:
                vel[:min(9, len(msg.velocity))] = np.array(msg.velocity[:min(9, len(msg.velocity))], dtype=np.float32)
        self.joint_pos = pos
        self.joint_vel = vel

    def _on_ee(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        self.ee_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        self.ee_quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)

    def _on_obj(self, msg: PoseStamped):
        p = msg.pose.position
        self.obj_pos = np.array([p.x, p.y, p.z], dtype=np.float32)

    # --- control ---
    def _build_obs(self) -> np.ndarray:
        rel = self.obj_pos - self.ee_pos
        goal_rel = self.goal - self.obj_pos
        if self.obj_pos[2] <= 0.09:
            phase = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif np.linalg.norm(self.obj_pos - self.goal) > 0.06:
            phase = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            phase = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.concatenate([
            self.joint_pos, self.joint_vel, rel, self.obj_pos,
            self.ee_pos, self.ee_quat, goal_rel, phase,
        ]).astype(np.float32)

    def _control_step(self):
        if self.joint_pos is None or self.ee_pos is None or self.obj_pos is None or self.model is None:
            return

        # assist mode: let bridge handle control
        if self.assist_active:
            self.assist_steps += 1
            if self.assist_steps > self.assist_max_steps:
                self.assist_active = False
                self.assist_steps = 0
                self.pub_assist_stop.publish(Empty())
                self.get_logger().info("Assist timeout, returning to ACT.")
            return

        self.step += 1
        rel = self.obj_pos - self.ee_pos
        dist = float(np.linalg.norm(rel))
        dist_xy = float(np.linalg.norm(rel[:2]))

        # stuck detection
        if self.prev_dist is not None and abs(self.prev_dist - dist) < self.stuck_eps:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.prev_dist = dist

        if self.stuck_counter > self.stuck_steps_thresh:
            self.assist_active = True
            self.assist_steps = 0
            self.stuck_counter = 0
            self.action_chunks.clear()
            self.pub_assist_start.publish(Empty())
            self.get_logger().info(f"Stuck detected, assist ON. dist={dist:.3f}")
            return

        # build obs and normalize
        obs = self._build_obs()
        obs_norm = (2.0 * (obs - self.obs_min) / self.obs_range - 1.0).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_norm).unsqueeze(0).to(self.device)

        # inference → new chunk
        with torch.no_grad():
            pred_norm = self.model.inference(obs_tensor)
        pred_np = pred_norm.squeeze(0).cpu().numpy()
        pred_actions = 0.5 * (pred_np + 1.0) * self.act_range + self.act_min
        self.action_chunks.append((pred_actions, self.step))

        # temporal ensemble
        weights = []
        actions = []
        for chunk, start in self.action_chunks:
            idx = self.step - start
            if 0 <= idx < len(chunk):
                weights.append(np.exp(-self.temporal_decay * idx))
                actions.append(chunk[idx])

        target = np.average(actions, axis=0, weights=weights).astype(np.float32)

        # gripper: latest prediction, binarized
        grip_val = 0.04 if pred_actions[0, 7] > 0.02 else 0.0
        target[7] = grip_val
        target[8] = grip_val

        # rule-based gripper override: close only when very close
        if dist_xy < self.grasp_xy and abs(rel[2]) < self.grasp_z:
            target[7] = 0.0
            target[8] = 0.0
        # prevent premature closure: force open if still far
        elif dist > 0.08:
            target[7] = 0.04
            target[8] = 0.04

        # expire old chunks
        self.action_chunks = [(c, s) for c, s in self.action_chunks if self.step - s < len(c)]

        # slew rate limiting
        arm_desired = target[:7]
        arm_prev = self.last_target[:7]
        arm_step = np.clip(arm_desired - arm_prev, -self.slew_limit, self.slew_limit)
        target[:7] = arm_prev + arm_step

        # joint limits
        target = np.clip(target, J_LOW, J_HIGH).astype(np.float32)
        self.last_target = target.copy()

        # publish
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = target.tolist()
        self.pub_cmd.publish(msg)

        if self.step % 15 == 0:
            self.get_logger().info(
                f"step={self.step} dist={dist:.3f} rel={np.round(rel, 3)} "
                f"obj_z={self.obj_pos[2]:.3f} chunks={len(self.action_chunks)}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ACTPolicyNode()
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
