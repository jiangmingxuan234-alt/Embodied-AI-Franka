import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

# ==========================================================
# 阶段 1：基础空间引导 (找位置、夹紧、抬起)
# ==========================================================
def reaching_reward(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """引导机械臂靠近 U 盘"""
    robot = env.scene["robot"]
    hand_idx = robot.find_bodies("panda_hand")[0]
    tcp_pos = robot.data.body_state_w[:, hand_idx, :3].view(env.num_envs, 3)
    object_pos = env.scene[object_cfg.name].data.root_pos_w[:, :3].view(env.num_envs, 3)
    
    dist = torch.norm(tcp_pos - object_pos, dim=-1)
    return 1.0 / (1.0 + 10.0 * dist)

def grasping_reward(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """只有当两根手指夹住 U 盘，并且离开桌面时才给高分"""
    robot = env.scene["robot"]
    lf_idx = robot.find_bodies("panda_leftfinger")[0]
    rf_idx = robot.find_bodies("panda_rightfinger")[0]
    
    lf_pos = robot.data.body_state_w[:, lf_idx, :3].view(env.num_envs, 3)
    rf_pos = robot.data.body_state_w[:, rf_idx, :3].view(env.num_envs, 3)
    object_pos = env.scene[object_cfg.name].data.root_pos_w[:, :3].view(env.num_envs, 3)
    
    finger_midpoint = (lf_pos + rf_pos) / 2.0
    dist_to_obj = torch.norm(finger_midpoint - object_pos, dim=-1)
    
    object_z = env.scene[object_cfg.name].data.root_pos_w[:, 2].view(-1)
    is_lifted = object_z > 0.04
    
    return ((dist_to_obj < 0.05) & is_lifted).float()

def lift_to_target_reward(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, target_height: float) -> torch.Tensor:
    """鼓励将 U 盘平稳保持在目标高度"""
    object_z = env.scene[object_cfg.name].data.root_pos_w[:, 2].view(-1)
    height_diff = torch.abs(target_height - object_z)
    is_lifted = object_z > 0.04
    return (1.0 / (1.0 + 10.0 * height_diff)) * is_lifted.float()


# ==========================================================
# 阶段 2：规范动作姿态 (抓娃娃机逻辑 - RMPflow 模仿)
# ==========================================================
def top_down_posture_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """🌟 强制夹爪垂直朝下 (从天而降)"""
    robot = env.scene["robot"]
    hand_idx = robot.find_bodies("panda_hand")[0]
    
    # 获取手部的四元数 (w, x, y, z)
    hand_quat = robot.data.body_state_w[:, hand_idx, 3:7].view(env.num_envs, 4)
    
    # 将四元数转化为局部 Z 轴在世界坐标系下的方向
    # 我们只关心它的 Z 分量：1 - 2*(x^2 + y^2)
    # 索引说明：0=w, 1=x, 2=y, 3=z
    z_dir_z = 1.0 - 2.0 * (hand_quat[:, 1]**2 + hand_quat[:, 2]**2)
    
    # 我们希望局部 Z 轴 (手指伸出的方向) 指向地面的世界 -Z 轴，即 z_dir_z 趋近于 -1
    # 取负号后变成 1，作为最大奖励
    return torch.clamp(-z_dir_z, min=0.0).view(-1)

def hover_above_reward(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """🌟 鼓励先移动到 U 盘正上方 (XY 轴对齐)"""
    robot = env.scene["robot"]
    hand_idx = robot.find_bodies("panda_hand")[0]
    
    tcp_pos = robot.data.body_state_w[:, hand_idx, :3].view(env.num_envs, 3)
    object_pos = env.scene[object_cfg.name].data.root_pos_w[:, :3].view(env.num_envs, 3)
    
    # 仅提取 X 和 Y 坐标计算平面距离
    xy_dist = torch.norm(tcp_pos[:, :2] - object_pos[:, :2], dim=-1)
    
    # XY 距离越近，奖励越大 (促使它在下降前先对准)
    return (1.0 / (1.0 + 20.0 * xy_dist)).view(-1)


# ==========================================================
# 阶段 3：严厉惩罚 (防作弊、防抽搐)
# ==========================================================
def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """惩罚动作突变，专治帕金森式抖动"""
    action_diff = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(action_diff), dim=-1).view(-1)

def drop_penalty(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚 U 盘掉落桌面下方"""
    object_z = env.scene[object_cfg.name].data.root_pos_w[:, 2].view(-1)
    return (object_z < 0.0).float()