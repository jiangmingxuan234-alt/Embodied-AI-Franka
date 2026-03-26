"""
ACT Policy 验证脚本 — Isaac Sim 中运行
用法: python scripts/diffusion_policy/play_act.py
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str,
    default="/home/jmx001/my_program/my_robot_project/logs/act_policy/act_epoch_300.pt")
parser.add_argument("--norm_stats", type=str,
    default="/home/jmx001/my_program/my_robot_project/logs/act_policy/norm_stats.npz")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.core", True)
ext_manager.set_extension_enabled_immediate("omni.isaac.franka", True)

import numpy as np
import torch

from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.types import ArticulationAction

import sys
sys.path.insert(0, "/home/jmx001/my_program/my_robot_project/scripts/diffusion_policy")
from train_act import ACTPolicy

GOAL = np.array([0.40, 0.0, 0.40], dtype=np.float32)
INIT_JOINTS = np.array([0.0, -1.1, 0.0, -2.3, 0.0, 2.4, 0.8, 0.04, 0.04], dtype=np.float32)
J_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0])
J_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPORAL_M = 0.1  # temporal ensemble 衰减系数（越大越跟随最新预测）


def build_obs(j_pos, j_vel, ee_pos, ee_quat, obj_pos):
    rel = obj_pos - ee_pos
    goal_rel = GOAL - obj_pos
    if obj_pos[2] <= 0.09:
        phase = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif np.linalg.norm(obj_pos - GOAL) > 0.06:
        phase = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        phase = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.concatenate([j_pos, j_vel, rel, obj_pos, ee_pos, ee_quat, goal_rel, phase]).astype(np.float32)


def main():
    ckpt = torch.load(args_cli.checkpoint, map_location=DEVICE)
    cfg = ckpt["config"]
    model = ACTPolicy(
        obs_dim=cfg["obs_dim"], action_dim=cfg["action_dim"],
        chunk_size=cfg["chunk_size"], latent_dim=cfg["latent_dim"],
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stats = np.load(args_cli.norm_stats)
    act_min, act_max = stats["act_min"], stats["act_max"]
    obs_min, obs_max = stats["obs_min"], stats["obs_max"]
    act_range = act_max - act_min
    act_range[act_range < 1e-6] = 1.0
    obs_range = obs_max - obs_min
    obs_range[obs_range < 1e-6] = 1.0

    # Isaac Sim setup
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/120.0, rendering_dt=1.0/30.0)
    world.scene.add_default_ground_plane()
    franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
    u_disk = world.scene.add(DynamicCuboid(
        prim_path="/World/u_disk", name="u_disk",
        position=np.array([0.5, 0.0, 0.02]),
        scale=np.array([0.05, 0.02, 0.01]),
        color=np.array([1.0, 0.0, 0.0]), mass=0.02,
    ))

    world.reset()
    franka.set_joint_positions(INIT_JOINTS)
    franka.apply_action(ArticulationAction(joint_positions=INIT_JOINTS))

    chunk_size = cfg["chunk_size"]
    action_chunks = []  # list of (chunk_array, start_step)
    step = 0

    print(f"Loaded: {args_cli.checkpoint}")
    print(f"ACT Policy: chunk_size={chunk_size}, temporal_m={TEMPORAL_M}")

    while simulation_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            continue

        step += 1
        j_pos = np.array(franka.get_joint_positions(), dtype=np.float32).reshape(-1)[:9]
        j_vel = np.array(franka.get_joint_velocities(), dtype=np.float32).reshape(-1)[:9]
        ee_pos, ee_quat = franka.end_effector.get_world_pose()
        ee_pos = np.array(ee_pos, dtype=np.float32).reshape(-1)[:3]
        ee_quat = np.array(ee_quat, dtype=np.float32).reshape(-1)[:4]
        obj_pos, _ = u_disk.get_world_pose()
        obj_pos = np.array(obj_pos, dtype=np.float32).reshape(-1)[:3]

        obs = build_obs(j_pos, j_vel, ee_pos, ee_quat, obj_pos)
        obs_norm = (2.0 * (obs - obs_min) / obs_range - 1.0).astype(np.float32)
        rel = obj_pos - ee_pos
        dist = float(np.linalg.norm(rel))

        # 每步推理一次，获取新 chunk
        obs_tensor = torch.from_numpy(obs_norm).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_norm = model.inference(obs_tensor)  # (1, chunk_size, action_dim)
        pred_np = pred_norm.squeeze(0).cpu().numpy()
        pred_actions = 0.5 * (pred_np + 1.0) * act_range + act_min
        action_chunks.append((pred_actions, step))

        # Temporal ensemble: 手臂关节加权平均，夹爪用最新预测二值化
        weights = []
        actions = []
        for chunk, start in action_chunks:
            idx = step - start
            if 0 <= idx < len(chunk):
                w = np.exp(-TEMPORAL_M * idx)
                weights.append(w)
                actions.append(chunk[idx])

        # 手臂：temporal ensemble 加权平均
        target = np.average(actions, axis=0, weights=weights).astype(np.float32)
        # 夹爪：只用最新推理结果（不参与 ensemble），二值化
        latest_grip = pred_actions[0, 7]  # 当前步最新推理的第 0 步
        grip_val = 0.04 if latest_grip > 0.02 else 0.0
        target[7] = grip_val
        target[8] = grip_val

        # 清理过期 chunk
        action_chunks = [(c, s) for c, s in action_chunks if step - s < len(c)]

        target = np.clip(target, J_LOW, J_HIGH).astype(np.float32)
        franka.apply_action(ArticulationAction(joint_positions=target))

        if step % 15 == 0:
            n_chunks = len(action_chunks)
            print(f"[ACT] step={step} dist={dist:.3f} rel={np.round(rel, 3)} "
                  f"obj_z={obj_pos[2]:.3f} grip={j_pos[7]:.3f} chunks={n_chunks}")


if __name__ == "__main__":
    main()
    simulation_app.close()
