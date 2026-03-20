import argparse
import sys
import os

print("\n🚀 [DEBUG] play.py 成功被唤醒，准备连接引擎...\n")

from isaaclab.app import AppLauncher
import cli_args

# 1. 基础参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=32) # 默认 32 个环境
parser.add_argument("--task", type=str, default="Isaac-UDisk-Grasp-v0")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 2. 启动仿真引擎
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3. 强行锁定你的项目模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
extension_root_dir = os.path.join(project_root, "source", "peg_in_hole")
sys.path.insert(0, extension_root_dir)

try:
    # 🌟 核心突破口：直接导入我们手写的配置类，跳过所有中间商！
    from peg_in_hole.tasks.manipulation.peg_in_hole.env_cfg import GraspEnvCfg
    from peg_in_hole.tasks.manipulation.peg_in_hole.agents import udisk_grasp_ppo_cfg
    print("✅ 成功手动载入环境说明书与 PPO 大脑配置！")
except Exception as e:
    print(f"❌ 模块导入失败，请检查代码: {e}")
    simulation_app.close()
    sys.exit(1)

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

def main():
    print("\n[INFO] 开始实例化配置...")
    env_cfg = GraspEnvCfg()
    agent_cfg = udisk_grasp_ppo_cfg()

    # 覆盖环境数量为 32
    env_cfg.scene.num_envs = args_cli.num_envs

    # 锁定权重所在的日志文件夹
    log_root_path = os.path.join(project_root, "logs", "rsl_rl", agent_cfg.experiment_name)
    print(f"[INFO] 正在前往日志库提取大脑: {log_root_path}")

    print("[INFO] 正在构建物理世界...")
    env = gym.make(args_cli.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    print("[INFO] 正在注入 PPO 算法...")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_root_path, device=agent_cfg.device)

    # 自动搜索最新的权重文件 (.pt)
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"🌟 成功找到满级神装权重: {resume_path}")
    runner.load(resume_path)

    # 开启推理模式
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs, _ = env.get_observations()
    
    print("\n🎬 Action！开始播放抓取动画！\n")
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

if __name__ == "__main__":
    main()
    simulation_app.close()