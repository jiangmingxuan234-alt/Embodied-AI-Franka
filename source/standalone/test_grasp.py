import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Direct Math Mode + IK Grasping")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.utils.math as math_utils  
from isaaclab.envs import ManagerBasedRLEnv
from peg_in_hole.tasks.manipulation.peg_in_hole.env_cfg import PegInHoleEnvCfg

# =========================================================================
# 🌟 核心修复：找回版本兼容的 IK 导入语法！
# =========================================================================
try:
    from isaaclab.controllers.differential_ik import DifferentialIKController, DifferentialIKControllerCfg
except ImportError:
    from isaaclab.controllers.differential_ik import DifferentialIKController
    from isaaclab.controllers import DifferentialIKControllerCfg

def main():
    env_cfg = PegInHoleEnvCfg()
    env_cfg.scene.num_envs = 1 
    env = ManagerBasedRLEnv(cfg=env_cfg)

    action = torch.zeros((env.num_envs, 8), device=env.device)
    
    # 获取机器人的身体数据 
    robot = env.scene["robot"]
    ee_idx = robot.data.body_names.index("panda_hand")

    # 🌟 初始化 IK 翻译器
    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=1, device=env.device)
    
    # 夹爪永远保持“垂直朝下”的固定姿态 (四元数)
    target_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32, device=env.device)

    print("\n🎉 成功模仿纯数学模式！开启张量驱动的抓取流水线...\n")
    
    # 直接沿用你写的完美坐标作为目标点
    stick_pos_w = torch.tensor([[0.26, -0.003, 0.35]], dtype=torch.float32, device=env.device)
    port_pos_w = torch.tensor([[0.293, -0.0023, 0.352]], dtype=torch.float32, device=env.device)
    
    obs, _ = env.reset()
    step_count = 0
    
    while simulation_app.is_running():
        # 获取机器人手臂当前的状态
        ee_pos = robot.data.body_pos_w[:, ee_idx]
        ee_quat = robot.data.body_quat_w[:, ee_idx]
        current_joint_pos = robot.data.joint_pos[:, 0:7]
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_idx, :, 0:7]

        # 默认目标定在 U 盘的位置
        target_pos = stick_pos_w.clone()
        
        # --- 🎬 开始按照时间轴分配动作 ---
        if step_count < 150:
            target_pos[:, 2] += 0.15 # 阶段0：飞到 U 盘上方 15cm
            action[:, 7] = 1.0  
            
        elif step_count < 300:
            target_pos[:, 2] += 0.115 # 阶段1：压低高度，准备套住
            action[:, 7] = 1.0  
            
        elif step_count < 400:
            target_pos[:, 2] += 0.115
            action[:, 7] = -1.0 # 阶段2：保持高度，狠狠夹紧！
            
        elif step_count < 550:
            target_pos[:, 2] += 0.25# 阶段3：保持夹紧，垂直往上拔起！
            action[:, 7] = -1.0 
            
        # ... 前面 0 到 800 步的代码不变 ...
        
        elif step_count < 800:
            target_pos = port_pos_w.clone()
            target_pos[:, 2] += 0.15 # 阶段4：横移飞到插座上方 15cm 处
            action[:, 7] = -1.0 
            
        else:
            # 🌟 终极绝杀：我不重置了！我就这样死死夹着 U 盘悬停在插座上方！
            target_pos = port_pos_w.clone()
            target_pos[:, 2] += 0.15 
            action[:, 7] = -1.0 
            if step_count == 800:
                print("🎉 抓取测试完美结束！机械臂已锁定战利品，不再重置环境。")

        # 🌟 IK 控制器算角度 -> 执行动作
        ik_controller.set_command(torch.cat([target_pos, target_quat], dim=-1))
        action[:, 0:7] = ik_controller.compute(ee_pos, ee_quat, jacobian, current_joint_pos)
            
        env.step(action)
        
        # 只要不大于 801，就一直加，大于了就永远停在 801，防止溢出
        if step_count <= 800:
            step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()