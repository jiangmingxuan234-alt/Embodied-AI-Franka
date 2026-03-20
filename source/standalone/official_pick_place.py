import argparse
from isaaclab.app import AppLauncher
import numpy as np
import os
import h5py

# 启动配置
parser = argparse.ArgumentParser(description="Record Ultimate Expert Data for Imitation Learning")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.core", True)
ext_manager.set_extension_enabled_immediate("omni.isaac.franka", True)

from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.core.utils.types import ArticulationAction

def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/120.0, rendering_dt=1.0/30.0)
    world.scene.add_default_ground_plane()

    franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))

    OBJ_X_RANGE = (0.42, 0.58)
    OBJ_Y_RANGE = (-0.30, 0.30)

    random_x = np.random.uniform(*OBJ_X_RANGE)
    random_y = np.random.uniform(*OBJ_Y_RANGE)
    u_disk = world.scene.add(
        DynamicCuboid(
            prim_path="/World/u_disk",
            name="u_disk",
            position=np.array([random_x, random_y, 0.02]), 
            scale=np.array([0.05, 0.02, 0.01]),      
            color=np.array([1.0, 0.0, 0.0]),         
            mass=0.02 
        )
    )

    world.reset()
    init_joints = np.array([0.0, -1.1, 0.0, -2.3, 0.0, 2.4, 0.8, 0.04, 0.04])
    franka.set_joint_positions(init_joints)
    franka.get_articulation_controller().apply_action(ArticulationAction(joint_positions=init_joints))

    controller = PickPlaceController(name="pick_place_controller", gripper=franka.gripper, robot_articulation=franka)
    goal_position = np.array([0.40, 0.0, 0.40])
    
    MAX_DEMOS = 180
    current_demo = 0
    ep_obs, ep_actions = [], []
    last_valid_action = np.zeros(9, dtype=np.float32)

    # 采集质量控制参数：减少长时间发呆帧，增加有效动作密度
    MAX_STEPS_PER_DEMO = 900
    MAX_POST_FRAMES = 12
    RECORD_STRIDE = 2
    DELTA_NORM_THRESHOLD = 0.0012
    NEAR_OBJECT_DIST = 0.12
    MIN_DEMO_FRAMES = 120
    ALWAYS_KEEP_WARMUP_STEPS = 80

    # 成功后追录计数器
    post_success_frames = 0
    episode_step = 0

    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/robomimic"))
    os.makedirs(save_dir, exist_ok=True)
    hdf5_path = os.path.join(save_dir, "rmpflow_expert.hdf5")
    
    print(f"\n🎥 开始录制最强特征工程版数据集！目标: {MAX_DEMOS} 条")
    f = h5py.File(hdf5_path, 'w')
    data_grp = f.create_group('data')

    while simulation_app.is_running():
        world.step(render=True)
        if world.is_playing():
            episode_step += 1
            # ==========================================
            # 1. 终极特征工程 (Feature Engineering)
            # ==========================================
            current_u_disk_pos, current_u_disk_quat = u_disk.get_world_pose()
            j_pos = franka.get_joint_positions()
            j_vel = franka.get_joint_velocities()
            
            # 获取末端执行器(夹爪)的真实位置
            ee_pos, ee_quat = franka.end_effector.get_world_pose()
            if len(ee_pos.shape) > 1: ee_pos = ee_pos[0]
            if len(ee_quat.shape) > 1: ee_quat = ee_quat[0]
            if len(current_u_disk_pos.shape) > 1: current_u_disk_pos = current_u_disk_pos[0]
            
            # 🎯 核心指南针：目标方向向量 (U盘位置 - 夹爪位置)
            rel_pos = current_u_disk_pos - ee_pos
            goal_rel = goal_position - current_u_disk_pos
            # 三阶段提示：接近 -> 搬运 -> 放置
            if current_u_disk_pos[2] <= 0.09:
                phase = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif np.linalg.norm(current_u_disk_pos - goal_position) > 0.06:
                phase = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                phase = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            # 37 维 Observation:
            # 9 j_pos + 9 j_vel + 3 rel_pos + 3 obj_pos + 3 ee_pos + 4 ee_quat + 3 goal_rel + 3 phase
            obs_array = np.concatenate(
                [j_pos, j_vel, rel_pos, current_u_disk_pos, ee_pos, ee_quat, goal_rel, phase]
            ).astype(np.float32)

            # ==========================================
            # 2. RMPflow 专家动作 & 提取 Delta
            # ==========================================
            safe_picking_pos = np.clip(current_u_disk_pos, a_min=[0.2, -0.5, 0.0], a_max=[0.8, 0.5, 0.6])
            actions = controller.forward(picking_position=safe_picking_pos, placing_position=goal_position, current_joint_positions=j_pos)
            franka.apply_action(actions) # 让机械臂动起来
            
            action_array = np.array(j_pos, dtype=np.float32)
            if actions.joint_positions is not None:
                l = len(actions.joint_positions)
                if actions.joint_indices is not None:
                    action_array[actions.joint_indices] = actions.joint_positions
                else:
                    action_array[:l] = actions.joint_positions

            # 解毒 NaN
            if len(ep_actions) == 0: last_valid_action = np.array(j_pos, dtype=np.float32)
            nan_mask = np.isnan(action_array) | np.isinf(action_array)
            action_array[nan_mask] = last_valid_action[nan_mask]
            last_valid_action = action_array.copy()

            # 🌟🌟🌟 核心治本修改：提取纯净 Delta，并挂载 100 倍放大器！ 🌟🌟🌟
            raw_delta = action_array - j_pos[:9]
            scaled_delta_action = raw_delta  # 存原始物理量，归一化由训练时处理

            # ==========================================
            # 3. 垃圾帧过滤 & 成功后追录逻辑
            # ==========================================
            # 严苛的成功定义：必须抬起 10cm 以上，且靠近目标点
            lifted = current_u_disk_pos[2] > 0.09
            at_goal = np.linalg.norm(current_u_disk_pos - goal_position) < 0.06

            if lifted and at_goal and post_success_frames == 0:
                print("🎯 达到目标点！开启成功后追录 (Hold) 阶段...")
                post_success_frames = 1 

            # 过滤逻辑：
            # 1) 动作幅度足够大才保存
            # 2) 靠近目标时保留更多细动作
            # 3) 成功后追录强制保留
            # 4) 用 stride 下采样，减少强相关重复帧
            arm_delta_norm = np.linalg.norm(raw_delta[:7])
            dist_to_obj = np.linalg.norm(rel_pos)
            dist_xy = np.linalg.norm(rel_pos[:2])
            # 抓取关键窗口：对齐后向下接近 + 夹爪开始闭合
            grasp_window = dist_xy < 0.10 and rel_pos[2] < -0.05
            gripper_closing = raw_delta[7] < -0.001 and raw_delta[8] < -0.001
            keep_frame = (
                episode_step <= ALWAYS_KEEP_WARMUP_STEPS
                or
                post_success_frames > 0
                or arm_delta_norm > DELTA_NORM_THRESHOLD
                or dist_to_obj < NEAR_OBJECT_DIST
                or grasp_window
                or gripper_closing
            )
            record_now = (
                post_success_frames > 0
                or grasp_window
                or gripper_closing
                or episode_step % RECORD_STRIDE == 0
            )
            if keep_frame and record_now:
                ep_obs.append(obs_array)
                ep_actions.append(scaled_delta_action)

            # 超时重置：避免失败回合拖成几百帧“慢动作噪声”
            if post_success_frames == 0 and episode_step >= MAX_STEPS_PER_DEMO:
                print(f"⏱️ Demo {current_demo} 超时 ({episode_step} steps)，丢弃本回合并重置。")
                ep_obs, ep_actions = [], []
                post_success_frames = 0
                episode_step = 0
                new_x = np.random.uniform(*OBJ_X_RANGE)
                new_y = np.random.uniform(*OBJ_Y_RANGE)
                u_disk.set_world_pose(position=np.array([new_x, new_y, 0.02]))
                franka.set_joint_positions(init_joints)
                controller.reset()
                continue
            
            # ==========================================
            # 4. 追录结束，保存数据并重置
            # ==========================================
            if post_success_frames > 0:
                post_success_frames += 1
                
                if post_success_frames > MAX_POST_FRAMES:
                    if len(ep_actions) >= MIN_DEMO_FRAMES:
                        print(
                            f"✅ Demo {current_demo} 录制完成 | frames={len(ep_actions)} "
                            f"| steps={episode_step}"
                        )
                        demo_grp = data_grp.create_group(f'demo_{current_demo}')
                        demo_grp.create_group('obs').create_dataset('policy', data=np.array(ep_obs))
                        demo_grp.create_dataset('actions', data=np.array(ep_actions))
                        rewards = np.zeros(len(ep_actions), dtype=np.float32)
                        dones = np.zeros(len(ep_actions), dtype=np.float32)
                        dones[-1] = 1.0
                        demo_grp.create_dataset('rewards', data=rewards)
                        demo_grp.create_dataset('dones', data=dones)
                        demo_grp.attrs['num_samples'] = len(ep_actions)
                        current_demo += 1
                    else:
                        print(
                            f"⚠️ Demo 丢弃：有效帧不足 ({len(ep_actions)} < {MIN_DEMO_FRAMES}) "
                            f"| steps={episode_step}"
                        )

                    ep_obs, ep_actions = [], []
                    post_success_frames = 0
                    episode_step = 0

                    if current_demo >= MAX_DEMOS:
                        print(f"\n🚀 绝世武功秘籍已大功告成！文件保存在:\n{hdf5_path}\n")
                        f.close()
                        break
                    
                    # 重新随机摆放物体，复位机械臂
                    new_x = np.random.uniform(*OBJ_X_RANGE)
                    new_y = np.random.uniform(*OBJ_Y_RANGE)
                    u_disk.set_world_pose(position=np.array([new_x, new_y, 0.02]))
                    franka.set_joint_positions(init_joints)
                    controller.reset()

if __name__ == '__main__':
    main()
    simulation_app.close()
