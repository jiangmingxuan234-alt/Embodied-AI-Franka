"""
Diffusion Policy 验证脚本 — Isaac Sim 中运行
用法: python scripts/diffusion_policy/play_diffusion.py
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str,
    default="/home/jmx001/my_program/my_robot_project/logs/diffusion_policy/model_epoch_200.pt")
parser.add_argument("--norm_stats", type=str,
    default="/home/jmx001/my_program/my_robot_project/logs/diffusion_policy/norm_stats.npz")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.core", True)
ext_manager.set_extension_enabled_immediate("omni.isaac.franka", True)

import numpy as np
import torch
from collections import deque

from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.types import ArticulationAction

# Import model definition
import sys
sys.path.insert(0, "/home/jmx001/my_program/my_robot_project/scripts/diffusion_policy")
from train_diffusion import DiffusionPolicy

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

GOAL = np.array([0.40, 0.0, 0.40], dtype=np.float32)
INIT_JOINTS = np.array([0.0, -1.1, 0.0, -2.3, 0.0, 2.4, 0.8, 0.04, 0.04], dtype=np.float32)
J_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0])
J_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    # Load model
    ckpt = torch.load(args_cli.checkpoint, map_location=DEVICE)
    cfg = ckpt["config"]
    model = DiffusionPolicy(
        obs_dim=cfg["obs_dim"], action_dim=cfg["action_dim"],
        obs_horizon=cfg["obs_horizon"], pred_horizon=cfg["pred_horizon"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load normalization stats
    stats = np.load(args_cli.norm_stats)
    act_min, act_max = stats["act_min"], stats["act_max"]
    obs_min, obs_max = stats["obs_min"], stats["obs_max"]
    act_range = act_max - act_min
    act_range[act_range < 1e-6] = 1.0
    obs_range = obs_max - obs_min
    obs_range[obs_range < 1e-6] = 1.0

    # DDIM scheduler for fast inference
    scheduler = DDIMScheduler(
        num_train_timesteps=cfg["diffusion_steps"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(10)  # 10 denoising steps

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

    obs_horizon = cfg["obs_horizon"]
    pred_horizon = cfg["pred_horizon"]
    action_horizon = 4   # 执行前 4 步，频繁重规划
    action_dim = cfg["action_dim"]
    is_delta = cfg.get("action_type", "absolute") == "delta"

    obs_deque = deque(maxlen=obs_horizon)
    action_queue = []
    step = 0
    prev_target = INIT_JOINTS.copy()

    print(f"Loaded: {args_cli.checkpoint}")
    print(f"Diffusion Policy: obs_h={obs_horizon}, pred_h={pred_horizon}, act_h={action_horizon}, delta={is_delta}")

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
        obs_deque.append(obs_norm)
        while len(obs_deque) < obs_horizon:
            obs_deque.append(obs_norm)

        rel = obj_pos - ee_pos
        dist = float(np.linalg.norm(rel))
        dist_xy = float(np.linalg.norm(rel[:2]))
        dist_z = float(abs(rel[2]))

        if len(action_queue) == 0:
            obs_seq = np.stack(list(obs_deque), axis=0)
            obs_tensor = torch.from_numpy(obs_seq).unsqueeze(0).to(DEVICE)
            noisy = torch.randn((1, pred_horizon, action_dim), device=DEVICE)
            with torch.no_grad():
                for t in scheduler.timesteps:
                    noise_pred = model(noisy, torch.tensor([t], device=DEVICE), obs_tensor)
                    noisy = scheduler.step(noise_pred, t, noisy).prev_sample
            actions_norm = noisy.squeeze(0).cpu().numpy()
            actions = 0.5 * (actions_norm + 1.0) * act_range + act_min
            action_queue = list(actions[:action_horizon])

        target = np.array(action_queue.pop(0), dtype=np.float32)
        if is_delta:
            target[:7] = np.clip(target[:7], -0.05, 0.05)
            target = j_pos + target

        # EMA 平滑
        target = 0.7 * target + 0.3 * prev_target

        # 规则夹爪 + 强制下压
        if dist_xy < 0.09 and dist_z < 0.12:
            target[7] = 0.0
            target[8] = 0.0
            # 强制下压：如果 xy 已对齐但 z 还没到位，给关节 1,3,5 一个微小下压增量
            if dist_z > 0.04:
                target[1] += 0.005  # 肩关节前倾
                target[3] += 0.005  # 肘关节下压
        else:
            target[7] = 0.04
            target[8] = 0.04

        target = np.clip(target, J_LOW, J_HIGH).astype(np.float32)
        prev_target = target.copy()
        franka.apply_action(ArticulationAction(joint_positions=target))

        if step % 15 == 0:
            print(f"[DP] step={step} dist={dist:.3f} rel={np.round(rel, 3)} obj_z={obj_pos[2]:.3f} grip={j_pos[7]:.3f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
