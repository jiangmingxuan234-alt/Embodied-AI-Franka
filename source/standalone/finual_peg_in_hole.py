import argparse
from isaaclab.app import AppLauncher
import time
import math
import torch

parser = argparse.ArgumentParser(description="Final Peg-in-Hole Autonomous Controller")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=False)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from peg_in_hole.tasks.manipulation.peg_in_hole.env_cfg import PegInHoleEnvCfg

try:
    from isaaclab.controllers.differential_ik import DifferentialIKController, DifferentialIKControllerCfg
except ImportError:
    from isaaclab.controllers.differential_ik import DifferentialIKController
    from isaaclab.controllers import DifferentialIKControllerCfg

# 平滑插值函数 (让机械臂像太极一样温柔移动)
def smooth_step(start, end, progress):
    if progress <= 0.0: return start
    if progress >= 1.0: return end
    return start + (end - start) * (0.5 * (1.0 - math.cos(math.pi * progress)))

def main():
    # 1. 加载你专属的 U 盘和插座场景！
    env_cfg = PegInHoleEnvCfg()
    env_cfg.scene.num_envs = 1 
    env_cfg.episode_length_s = 10000.0 # 🌟 终极破解：打破强制重置，让咱们有足够时间完成插入！
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 2. 获取机械臂和夹爪的索引
    robot = env.scene["robot"]
    ee_idx = robot.data.body_names.index("panda_hand")
    action = torch.zeros((1, 8), device=env.device)

    # 3. 召唤底层的空间逆运动学大脑 (IK Controller)
    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=1, device=env.device)

    env.reset()
    print("\n🎬 终极大戏开场：专属场景加载完毕！准备执行 U 盘抓取与插入！")

    start_time = time.time()
    
    # 🌟 定义动作的关键坐标 (请根据你场景里 U 盘和插座的真实位置微调！)
    # 假设 U 盘在 X=0.26, Y=0.0，插座在 X=0.40, Y=0.20
    u_disk_x, u_disk_y = 0.26, -0.003
    port_x, port_y = 0.40, 0.20   # 🎯 假设插座在这个位置
    
    # 锁定夹爪始终朝下
    target_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32, device=env.device)
    
    # 初始化目标变量
    target_x, target_y, target_z = u_disk_x, u_disk_y, 0.50
    gripper_cmd = 1.0 # 1.0 张开, -1.0 闭合

    while simulation_app.is_running():
        t = time.time() - start_time
        
        # ==========================================
        # 🧠 核心大脑：全自动时序状态机 (State Machine)
        # ==========================================
        if t < 2.0:
            # 阶段 1：高空对准 U 盘
            target_x, target_y, target_z = u_disk_x, u_disk_y, 0.40
            gripper_cmd = 1.0
            
        elif t < 5.0:
            # 阶段 2：温柔下潜抓取 (Z轴下降到 0.145)
            progress = (t - 2.0) / 3.0
            target_z = smooth_step(0.40, 0.145, progress)
            
        elif t < 6.0:
            # 阶段 3：捏紧 U 盘！
            gripper_cmd = -1.0 
            
        elif t < 9.0:
            # 阶段 4：拔起 U 盘，悬停在空中
            progress = (t - 6.0) / 3.0
            target_z = smooth_step(0.145, 0.40, progress)
            
        elif t < 13.0:
            # 🌟 阶段 5：平移！带着 U 盘飞向插座的正上方
            progress = (t - 9.0) / 4.0
            target_x = smooth_step(u_disk_x, port_x, progress)
            target_y = smooth_step(u_disk_y, port_y, progress)
            
        elif t < 16.0:
            # 🌟 阶段 6：终极对准，下插！(假设插座高度是 0.16)
            progress = (t - 13.0) / 3.0
            target_z = smooth_step(0.40, 0.16, progress)
            
        else:
            # 阶段 7：大功告成，松开手，机械臂退回高空！
            gripper_cmd = 1.0
            if t > 17.0:
                target_z = smooth_step(0.16, 0.40, (t - 17.0) / 2.0)
                if t > 19.0: target_z = 0.40

        # ==========================================
        # 🦾 身体执行层：把计算出的坐标交给物理引擎
        # ==========================================
        target_pos = torch.tensor([[target_x, target_y, target_z]], dtype=torch.float32, device=env.device)
        
        # 获取当前机械臂的真实物理状态
        ee_pos = robot.data.body_pos_w[:, ee_idx]
        ee_quat = robot.data.body_quat_w[:, ee_idx]
        current_joint_pos = robot.data.joint_pos[:, 0:7]
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_idx, :, 0:7]

        # IK 控制器算出这帧需要转动多少电机角度
        ik_controller.set_command(torch.cat([target_pos, target_quat], dim=-1))
        action[:, 0:7] = ik_controller.compute(ee_pos, ee_quat, jacobian, current_joint_pos)
        action[:, 7] = gripper_cmd
        
        env.step(action)

if __name__ == "__main__":
    main()
    simulation_app.close()