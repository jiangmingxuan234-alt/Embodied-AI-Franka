import argparse
import sys
import os

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default="Isaac-UDisk-Grasp-Finetune-v0")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
extension_root_dir = os.path.join(project_root, "source", "peg_in_hole")
sys.path.insert(0, extension_root_dir)

try:
    from peg_in_hole.tasks.manipulation.peg_in_hole.env_cfg import GraspEnvCfg, GraspEnvFinetuneCfg
    from peg_in_hole.tasks.manipulation.peg_in_hole.agents import udisk_grasp_ppo_cfg, udisk_grasp_ppo_finetune_cfg
    print("✅ 成功载入环境配置！")
except Exception as e:
    print(f"❌ 模块导入失败: {e}")
    simulation_app.close()
    sys.exit(1)

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path


def main():
    # 根据 task 自动选择正确的配置
    if "Finetune" in args_cli.task:
        env_cfg = GraspEnvFinetuneCfg()
        agent_cfg = udisk_grasp_ppo_finetune_cfg()
        print("📋 Fine-tune 配置 (37 维观测, actor=[256,256])")
    else:
        env_cfg = GraspEnvCfg()
        agent_cfg = udisk_grasp_ppo_cfg()
        print("📋 标准 RL 配置 (25 维观测, actor=[256,128,64])")

    env_cfg.scene.num_envs = args_cli.num_envs

    log_root_path = os.path.join(project_root, "logs", "rsl_rl", agent_cfg.experiment_name)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_root_path, device=agent_cfg.device)

    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"🧠 加载权重: {resume_path}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs, _ = env.get_observations()

    prev_actions = None

    print("\n🎬 开始推理！\n")
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            raw = policy(obs)
        actions = raw.detach().clone()

        # EMA 平滑 (alpha=0.7: 70% 新动作, 30% 旧动作)
        if prev_actions is not None:
            actions = 0.7 * actions + 0.3 * prev_actions

        # 规则夹爪：从环境读取状态
        robot = env.unwrapped.scene["robot"]
        obj = env.unwrapped.scene["u_disk"]
        hand_idx = robot.find_bodies("panda_hand")[0][0]
        ee_pos = robot.data.body_state_w[:, hand_idx, :3]
        obj_pos = obj.data.root_pos_w[:, :3]
        rel = obj_pos - ee_pos
        dist_xy = torch.norm(rel[:, :2], dim=-1)
        dist_z = torch.abs(rel[:, 2])

        # 靠近时强制闭合夹爪，远离时保持张开
        close_mask = (dist_z < 0.10) & (dist_xy < 0.09)
        actions[close_mask, 7] = -0.04  # 闭合
        actions[close_mask, 8] = -0.04

        prev_actions = actions.clone()
        obs, _, _, _ = env.step(actions)

        step += 1
        if step % 100 == 0:
            d = dist_xy[0].item()
            z = dist_z[0].item()
            print(f"step={step} dist_xy={d:.3f} dist_z={z:.3f} gripper={'close' if close_mask[0] else 'open'}")


if __name__ == "__main__":
    main()
    simulation_app.close()
