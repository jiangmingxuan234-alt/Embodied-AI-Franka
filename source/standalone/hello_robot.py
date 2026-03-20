import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Direct Math Mode")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.utils.math as math_utils  
from isaaclab.envs import ManagerBasedRLEnv
from peg_in_hole.tasks.manipulation.peg_in_hole.env_cfg import PegInHoleEnvCfg

def main():
    env_cfg = PegInHoleEnvCfg()
    env_cfg.scene.num_envs = 1 
    env = ManagerBasedRLEnv(cfg=env_cfg)

    action = torch.zeros((env.num_envs, 8), device=env.device)
    default_joints = torch.tensor([[0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785]], device=env.device)
    
    # =========================================================================
    # 🌟 降维打击：直接读取你在 env_cfg.py 里写好的完美坐标！
    # =========================================================================
    print("\n🎉 启动纯数学降维打击！正在直接计算【终极相对位姿矩阵】...\n")
    
    # U 盘的完美坐标 (来自你的配置)
    stick_pos_w = torch.tensor([[0.26, -0.003, 0.35]], dtype=torch.float32, device=env.device)
    stick_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=env.device)
    
    # 插座的完美坐标 (来自你的配置)
    port_pos_w = torch.tensor([[0.293, -0.0023, 0.352]], dtype=torch.float32, device=env.device)
    port_quat_w = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32, device=env.device)
    
    # --- 开始纯数学运算 ---
    pos_error_w = stick_pos_w - port_pos_w
    rel_pos = math_utils.quat_rotate_inverse(port_quat_w, pos_error_w)
    
    port_quat_inv = math_utils.quat_inv(port_quat_w)
    rel_quat = math_utils.quat_mul(port_quat_inv, stick_quat_w)
    rel_euler = math_utils.euler_xyz_from_quat(rel_quat)
    
    print(f"==========================================")
    print(f"📏 【纯算力目标常数 Ground Truth】出炉：")
    print(f"👉 相对平移 (Δx, Δy, Δz) : [{rel_pos[0,0]:.5f},  {rel_pos[0,1]:.5f},  {rel_pos[0,2]:.5f}]")
    print(f"👉 相对旋转 四元数 (w,x,y,z): [{rel_quat[0,0]:.5f},  {rel_quat[0,1]:.5f},  {rel_quat[0,2]:.5f},  {rel_quat[0,3]:.5f}]")
    print(f"👉 相对旋转 欧拉角 (Roll, Pitch, Yaw): [{rel_euler[0][0]:.5f},  {rel_euler[1][0]:.5f},  {rel_euler[2][0]:.5f}]")
    print(f"==========================================\n")
    
    obs, _ = env.reset()
    step_count = 0
    
    while simulation_app.is_running():
        action[:, 0:7] = default_joints
        
        if step_count < 200:
            action[:, 7] = 1.0  
        else:
            action[:, 7] = -1.0 # 保持夹紧
            
        env.step(action)
        step_count += 1
        
        if step_count >= 500:
            step_count = 0
            env.reset()

if __name__ == "__main__":
    main()
    simulation_app.close()