import argparse
from isaaclab.app import AppLauncher
import numpy as np
import os
import torch
import h5py

# 启动配置
parser = argparse.ArgumentParser(description="Play Ultimate Feature-Engineered BC Model")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==========================================
# 1. 唤醒必要扩展
# ==========================================
import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.core", True)
ext_manager.set_extension_enabled_immediate("omni.isaac.franka", True)

from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.types import ArticulationAction

# Robomimic 工具
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


def sanitize_array(x: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    """Replace NaN/Inf to keep the controller numerically stable online."""
    return np.nan_to_num(x, nan=fallback, posinf=fallback, neginf=fallback).astype(np.float32, copy=False)


def load_action_min_max(dataset_path: str, action_dim: int = 9):
    """Scan HDF5 once and return per-dimension min/max for action de-normalization."""
    act_min = np.full((action_dim,), np.inf, dtype=np.float32)
    act_max = np.full((action_dim,), -np.inf, dtype=np.float32)

    with h5py.File(dataset_path, "r") as f:
        for demo_name in f["data"].keys():
            actions = f["data"][demo_name]["actions"][:]
            act_min = np.minimum(act_min, np.min(actions, axis=0))
            act_max = np.maximum(act_max, np.max(actions, axis=0))

    return act_min, act_max


def load_rel_action_bank(dataset_path: str):
    """Build a KNN bank from dataset: rel_xyz -> arm delta (first 7 joints)."""
    rel_list = []
    act_list = []
    with h5py.File(dataset_path, "r") as f:
        for demo_name in f["data"].keys():
            obs = f["data"][demo_name]["obs"]["policy"][:]
            actions = f["data"][demo_name]["actions"][:]
            rel_list.append(obs[:, 18:21].astype(np.float32))
            act_list.append(actions[:, :7].astype(np.float32))
    rel_bank = np.concatenate(rel_list, axis=0)
    act_bank = np.concatenate(act_list, axis=0)
    return rel_bank, act_bank


def knn_delta_from_rel(rel_xyz: np.ndarray, rel_bank: np.ndarray, act_bank: np.ndarray, k: int = 64):
    """Return weighted KNN delta for current rel vector. None if too far from data manifold."""
    d = np.linalg.norm(rel_bank - rel_xyz[None, :], axis=1)
    if d.size == 0:
        return None
    k = min(k, d.size)
    idx = np.argpartition(d, k - 1)[:k]
    d_sel = d[idx]
    if np.min(d_sel) > 0.25:
        return None
    w = 1.0 / (d_sel + 1e-4)
    delta = np.sum(act_bank[idx] * w[:, None], axis=0) / np.sum(w)
    return delta.astype(np.float32)


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/120.0, rendering_dt=1.0/30.0)
    world.scene.add_default_ground_plane()

    franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
    controller = PickPlaceController(name="play_pick_place_controller", gripper=franka.gripper, robot_articulation=franka)

    # 固定一个测试目标
    u_disk = world.scene.add(
        DynamicCuboid(
            prim_path="/World/u_disk",
            name="u_disk",
            position=np.array([0.5, 0.0, 0.02]), 
            scale=np.array([0.05, 0.02, 0.01]),      
            color=np.array([1.0, 0.0, 0.0]),         
            mass=0.02 
        )
    )

    # ==========================================
    # 🧠 2. 载入终极上帝视角大脑
    # ==========================================
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    
    # 你的最新模型路径
    ckpt_path = "/home/jmx001/my_program/my_robot_project/logs/robomimic/Isaac-UDisk-Grasp-v0/bc/20260320182155/models/model_epoch_100.pth"
    dataset_path = "/home/jmx001/my_program/my_robot_project/logs/robomimic/rmpflow_expert.hdf5"
    
    print(f"🧠 正在加载终极赛博大脑: {ckpt_path}")
    policy, ckpt_data = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)
    policy.start_episode() 
    ckpt_has_action_stats = ckpt_data.get("action_normalization_stats", None) is not None
    expected_obs_dim = int(ckpt_data["shape_metadata"]["all_shapes"]["policy"][0])
    action_min, action_max = load_action_min_max(dataset_path, action_dim=9)
    rel_bank, act_bank = load_rel_action_bank(dataset_path)
    arm_delta_limit = np.array([0.01] * 7, dtype=np.float32)
    print(f"📦 checkpoint 含 action_normalization_stats: {ckpt_has_action_stats}")
    print(f"🧩 checkpoint 期望观测维度: {expected_obs_dim}")
    print(f"📏 Action 反归一化范围已载入: min={np.round(action_min, 5)} max={np.round(action_max, 5)}")
    print(f"🗂️ KNN bank loaded: rel={rel_bank.shape}, act={act_bank.shape}")
    goal_position = np.array([0.40, 0.0, 0.40], dtype=np.float32)

    world.reset()
    init_joints = np.array([0.0, -1.1, 0.0, -2.3, 0.0, 2.4, 0.8, 0.04, 0.04], dtype=np.float32)
    franka.set_joint_positions(init_joints)
    franka.get_articulation_controller().apply_action(ArticulationAction(joint_positions=init_joints))

    print("🚀 上帝视角已解锁！开始执行制导抓取！")

    # 🌟🌟🌟 新增：在主循环外初始化 EMA 平滑器 🌟🌟🌟
    smoothed_delta = None
    prev_dist_to_obj = None
    stuck_counter = 0
    assist_active = False
    assist_steps = 0
    ASSIST_MAX_STEPS = 1400

    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_playing():
            # 1. 获取所有真实状态
            current_u_disk_pos, _ = u_disk.get_world_pose()
            j_pos = franka.get_joint_positions()
            j_vel = franka.get_joint_velocities()
            
            # 🌟 提取末端夹爪位置
            ee_pos, ee_quat = franka.end_effector.get_world_pose()
            if len(ee_pos.shape) > 1: ee_pos = ee_pos[0]
            if len(ee_quat.shape) > 1: ee_quat = ee_quat[0]
            if len(current_u_disk_pos.shape) > 1: current_u_disk_pos = current_u_disk_pos[0]
            
            # 🎯 核心制导雷达：目标相对向量
            rel_pos = current_u_disk_pos - ee_pos
            dist_to_obj = np.linalg.norm(rel_pos)
            print(f"ee_pos={np.round(ee_pos,3)} | udisk={np.round(current_u_disk_pos,3)} | rel={np.round(rel_pos,3)}")
            
            # 观测拼接兼容：
            # - 24 维: [j_pos, j_vel, rel_pos, obj_pos] (旧模型)
            # - 37 维: 24维 + [ee_pos, ee_quat, goal_rel, phase] (新模型)
            goal_rel = goal_position - current_u_disk_pos
            if current_u_disk_pos[2] <= 0.09:
                phase = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif np.linalg.norm(current_u_disk_pos - goal_position) > 0.06:
                phase = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                phase = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            if expected_obs_dim == 24:
                obs_array = np.concatenate([j_pos, j_vel, rel_pos, current_u_disk_pos]).astype(np.float32)
            elif expected_obs_dim == 37:
                obs_array = np.concatenate(
                    [j_pos, j_vel, rel_pos, current_u_disk_pos, ee_pos, ee_quat, goal_rel, phase]
                ).astype(np.float32)
            else:
                raise RuntimeError(f"Unsupported policy obs dim: {expected_obs_dim}")
            obs_array = sanitize_array(obs_array, fallback=0.0)
            obs_dict = {"policy": obs_array}

            # 🧠 2. 若 checkpoint 没保存 action stats，手动反归一化；
            # 否则 RolloutPolicy 已自动反归一化，直接用输出即可。
            normalized_action = sanitize_array(policy(obs_dict), fallback=0.0)
            if ckpt_has_action_stats:
                delta_action = normalized_action
            else:
                delta_action = 0.5 * (normalized_action + 1.0) * (action_max - action_min) + action_min

            # 卡住保护：若机械臂长期悬停在物体上方但距离不再改善，混入专家引导动作帮助下压
            dist_xy = np.linalg.norm(rel_pos[:2])
            hovering_above = rel_pos[2] < -0.16 and dist_xy < 0.18
            # 下压模式：已经对准物体上方但未下压到抓取高度时，提前触发专家引导
            descend_mode = dist_xy < 0.07 and rel_pos[2] < -0.10
            if prev_dist_to_obj is not None and hovering_above and abs(prev_dist_to_obj - dist_to_obj) < 6e-4:
                stuck_counter += 1
            else:
                stuck_counter = 0
            prev_dist_to_obj = dist_to_obj

            trigger_assist = descend_mode or stuck_counter > 25
            if trigger_assist and not assist_active:
                controller.reset()
                assist_active = True
                assist_steps = 0
                print(f"🛟 Assist Latch | dist={dist_to_obj:.3f} | rel={np.round(rel_pos,3)}")

            if assist_active:
                assist_steps += 1
                safe_pick = np.clip(current_u_disk_pos, a_min=[0.2, -0.5, 0.0], a_max=[0.8, 0.5, 0.6])
                expert_actions = controller.forward(
                    picking_position=safe_pick,
                    placing_position=goal_position,
                    current_joint_positions=j_pos,
                )
                franka.apply_action(expert_actions)

                # 辅助模式下只打印监控信息，不执行 BC 输出
                print(
                    f"🛟 Assist RUN | steps={assist_steps} | dist={dist_to_obj:.3f} "
                    f"| rel={np.round(rel_pos,3)} | obj_z={current_u_disk_pos[2]:.3f}"
                )

                reached_goal = (current_u_disk_pos[2] > 0.10) and (np.linalg.norm(current_u_disk_pos - goal_position) < 0.08)
                timeout = assist_steps > ASSIST_MAX_STEPS
                if reached_goal:
                    print("✅ Assist finished: object lifted and moved near goal.")
                    assist_active = False
                    assist_steps = 0
                elif timeout:
                    print("⚠️ Assist timeout: reset assist state and return to BC.")
                    assist_active = False
                    assist_steps = 0
                continue

            # 限制每步关节变化，避免单步过大造成关节发散
            delta_action[:7] = np.clip(delta_action[:7], -arm_delta_limit, arm_delta_limit)
            delta_action = sanitize_array(delta_action, fallback=0.0)

            # EMA 平滑
            if smoothed_delta is None:
                smoothed_delta = delta_action.copy()
            else:
                smoothed_delta = 0.7 * delta_action + 0.3 * smoothed_delta
            smoothed_delta = sanitize_array(smoothed_delta, fallback=0.0)

            # 规则夹爪：到位才夹合
            dist_z  = abs(rel_pos[2])
            if dist_z < 0.10 and dist_xy < 0.09:
                smoothed_delta[7] = -0.04
                smoothed_delta[8] = -0.04

            mean_abs_pred = np.mean(np.abs(normalized_action[:7]))
            print(f"🔍 MeanAbs(pred)={mean_abs_pred:.4f} | MeanAbs(delta)={np.mean(np.abs(smoothed_delta[:7])):.6f} | rel_xyz={np.round(rel_pos,3)}")
            
            # 🌟 3. 大道至简：加法直出 (用平滑后的 delta)
            target_action = j_pos[:9] + smoothed_delta
            
            # 4. 物理极限裁剪保护 (防止机械臂自损)
            joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0])
            joint_limits_upper = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973, 0.04, 0.04])
            safe_action = np.clip(target_action, joint_limits_lower, joint_limits_upper)
            
            # 5. 直接执行
            franka.apply_action(ArticulationAction(joint_positions=safe_action))

if __name__ == '__main__':
    main()
    simulation_app.close()
