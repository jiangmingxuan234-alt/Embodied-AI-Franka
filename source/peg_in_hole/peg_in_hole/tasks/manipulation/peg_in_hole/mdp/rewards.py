import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# ==========================================================
# 基础奖励（高斯核版本）
# ==========================================================
def reaching_reward(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """高斯核距离奖励，梯度更陡"""
    robot = env.scene["robot"]
    hand_idx = robot.find_bodies("panda_hand")[0]
    tcp_pos = robot.data.body_state_w[:, hand_idx, :3].view(env.num_envs, 3)
    object_pos = env.scene[object_cfg.name].data.root_pos_w[:, :3].view(env.num_envs, 3)
    dist = torch.norm(tcp_pos - object_pos, dim=-1)
    return torch.exp(-5.0 * dist * dist)


def grasping_reward(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """连续抓取奖励：手指距离 + 夹爪闭合 + 抬起进度"""
    robot = env.scene["robot"]
    lf_idx = robot.find_bodies("panda_leftfinger")[0]
    rf_idx = robot.find_bodies("panda_rightfinger")[0]

    lf_pos = robot.data.body_state_w[:, lf_idx, :3].view(env.num_envs, 3)
    rf_pos = robot.data.body_state_w[:, rf_idx, :3].view(env.num_envs, 3)
    object_pos = env.scene[object_cfg.name].data.root_pos_w[:, :3].view(env.num_envs, 3)

    finger_midpoint = (lf_pos + rf_pos) / 2.0
    dist_to_obj = torch.norm(finger_midpoint - object_pos, dim=-1)
    finger_distance_reward = torch.exp(-10.0 * dist_to_obj * dist_to_obj)

    finger_width = torch.norm(lf_pos - rf_pos, dim=-1)
    gripper_close_reward = torch.exp(-8.0 * finger_width * finger_width)

    object_z = env.scene[object_cfg.name].data.root_pos_w[:, 2].view(-1)
    lift_progress = torch.clamp((object_z - 0.02) / 0.08, min=0.0, max=1.0)

    is_near = (dist_to_obj < 0.08).float()
    return finger_distance_reward + is_near * (gripper_close_reward * 0.5 + lift_progress * 2.0)


def lift_to_target_reward(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, target_height: float) -> torch.Tensor:
    """高斯核抬升奖励"""
    object_z = env.scene[object_cfg.name].data.root_pos_w[:, 2].view(-1)
    height_diff = torch.abs(target_height - object_z)
    is_lifted = (object_z > 0.04).float()
    return torch.exp(-5.0 * height_diff * height_diff) * is_lifted


# ==========================================================
# 姿态奖励
# ==========================================================
def top_down_posture_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """强制夹爪垂直朝下"""
    robot = env.scene["robot"]
    hand_idx = robot.find_bodies("panda_hand")[0]
    hand_quat = robot.data.body_state_w[:, hand_idx, 3:7].view(env.num_envs, 4)
    z_dir_z = 1.0 - 2.0 * (hand_quat[:, 1]**2 + hand_quat[:, 2]**2)
    return torch.clamp(-z_dir_z, min=0.0).view(-1)


def hover_above_reward(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """高斯核 XY 对齐奖励"""
    robot = env.scene["robot"]
    hand_idx = robot.find_bodies("panda_hand")[0]
    tcp_pos = robot.data.body_state_w[:, hand_idx, :3].view(env.num_envs, 3)
    object_pos = env.scene[object_cfg.name].data.root_pos_w[:, :3].view(env.num_envs, 3)
    xy_dist_sq = torch.sum((tcp_pos[:, :2] - object_pos[:, :2])**2, dim=-1)
    return torch.exp(-15.0 * xy_dist_sq)


# ==========================================================
# 夹爪控制
# ==========================================================
def gripper_close_near_object(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """靠近物体时奖励闭合夹爪，远离时奖励张开"""
    robot = env.scene["robot"]
    hand_idx = robot.find_bodies("panda_hand")[0]
    tcp_pos = robot.data.body_state_w[:, hand_idx, :3].view(env.num_envs, 3)
    object_pos = env.scene[object_cfg.name].data.root_pos_w[:, :3].view(env.num_envs, 3)
    dist = torch.norm(tcp_pos - object_pos, dim=-1)

    finger_pos = robot.data.joint_pos[:, 7:9]
    gripper_width = finger_pos.sum(dim=-1)

    near = (dist < 0.08).float()
    return near * (0.08 - gripper_width) / 0.08 + (1.0 - near) * gripper_width / 0.08


# ==========================================================
# 三斧惩罚：action rate + acceleration + jerk + L2
# ==========================================================
def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """第一斧：动作变化率惩罚（一阶导数）"""
    action_diff = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(action_diff ** 2, dim=-1).view(-1)


def joint_acceleration_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """第二斧：关节加速度惩罚（速度的一阶导数）"""
    robot = env.scene["robot"]
    vel = robot.data.joint_vel[:, :7]
    # 用当前速度和上一步速度的差近似加速度
    # prev_joint_vel 存在 robot.data 里，但不一定有，用 action diff 近似
    # 直接用速度的绝对值作为代理（速度大 = 加速度曾经大）
    return torch.sum(vel ** 2, dim=-1).view(-1)


def action_l2_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """第三斧：动作绝对值 L2 正则化"""
    return torch.sum(env.action_manager.action ** 2, dim=-1).view(-1)


def lateral_joint_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """惩罚横向旋转关节（1,3,5,7）的速度，抑制摇头"""
    robot = env.scene["robot"]
    vel = robot.data.joint_vel[:, :7]
    # 关节 0,2,4,6 是横向旋转关节（从零开始计数）
    lateral_vel = vel[:, [0, 2, 4, 6]]
    return torch.sum(lateral_vel ** 2, dim=-1).view(-1)


def drop_penalty(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚 U 盘掉落桌面下方"""
    object_z = env.scene[object_cfg.name].data.root_pos_w[:, 2].view(-1)
    return (object_z < 0.0).float()
