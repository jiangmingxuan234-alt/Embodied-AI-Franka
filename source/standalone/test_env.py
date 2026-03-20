import argparse
from isaaclab.app import AppLauncher

# 1. 启动仿真器
parser = argparse.ArgumentParser(description="Test 300 Parallel RL Environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)

import sys
import os

# 🌟 路径修复：精确瞄准外层 peg_in_hole 目录
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
extension_root_dir = os.path.join(source_dir, "peg_in_hole") 

sys.path.insert(0, extension_root_dir)

import gymnasium as gym
import torch

try:
    import peg_in_hole.tasks.manipulation.peg_in_hole
    print("\n✅ 成功击破套娃！找到 tasks 文件夹并导入任务包！")
except ModuleNotFoundError as e:
    print(f"\n❌ 导入依然失败！当前搜索路径: {extension_root_dir}")
    raise e

# 🌟 引入 Isaac Lab 专用的图纸解析器
from isaaclab_tasks.utils import parse_env_cfg

def main():
    # 🌟 核心修复 1：先解析出我们在 env_cfg.py 里写的配置类
    env_cfg = parse_env_cfg("Isaac-UDisk-Grasp-v0")
    
    # 🌟 核心修复 2：把解析好的 cfg 塞进 make 函数里！
    env = gym.make("Isaac-UDisk-Grasp-v0", cfg=env_cfg)
    
    print(f"\n🚀 成功启动！当前并行环境数量: {env.unwrapped.num_envs}")
    
    obs, info = env.reset()

    while app_launcher.app.is_running():
        random_actions = 2.0 * torch.rand((env.unwrapped.num_envs, 9), device=env.unwrapped.device) - 1.0
        obs, rewards, terminated, truncated, info = env.step(random_actions)

if __name__ == "__main__":
    main()
    app_launcher.app.close()