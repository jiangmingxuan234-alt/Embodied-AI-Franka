# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic.

This script loads a robomimic policy and plays it in an Isaac Lab environment.
"""

import argparse
import os
import sys

# =========================================================================
# 🌟 击破“套娃路径”：强制让 robomimic 认识我们的 300 臂环境
# =========================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
extension_root_dir = os.path.join(project_root, "source", "peg_in_hole")
sys.path.insert(0, extension_root_dir)

try:
    import peg_in_hole.tasks.manipulation.peg_in_hole
    print("✅ 成功将自定义环境载入回放脚本！")
except ModuleNotFoundError:
    print(f"❌ 找不到环境模块，请检查路径: {extension_root_dir}")
# =========================================================================

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import gymnasium as gym
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


def rollout(policy, env, success_term, horizon, device):
    """Perform a single rollout of the policy in the environment."""
    policy.start_episode()
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[])

    for i in range(horizon):
        # Prepare observations - 🌟 因为我们训练时直接用的一维 policy 数组
        # 所以这里的 obs 就不拆分了，直接当成一整个输入传给网络
        obs = {"policy": torch.squeeze(obs_dict["policy"]).cpu().numpy()} 

        traj["obs"].append(obs)

        # Compute actions
        actions = policy(obs)
        
        # 👇 加一行这个打印：
        print(f"🤖 神经网络思考出的动作是: {actions}")
        # Unnormalize actions
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        # 这里的 action 是 9 维的，我们把它转成 tensor 喂给环境
        actions_tensor = torch.from_numpy(actions).to(device=device).view(1, env.unwrapped.action_space.shape[1])

        # Apply actions
        obs_dict, _, terminated, truncated, _ = env.step(actions_tensor)
        
        # 记录纯数据
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs_dict["policy"].cpu().numpy())

        # 🌟 糊弄大法生效：如果成功条件不存在，那我们就只靠超时来结束
        if success_term is not None:
            if bool(success_term.func(env, **success_term.params)[0]):
                return True, traj
                
        if terminated or truncated:
            return False, traj

    return False, traj


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # 🌟 因为我们在训练时，是把低维数据直接打包成了一个叫 "policy" 的 25 维数组
    # 所以在测试时，我们也要告诉环境，保持打包状态，不要把它切成字典！
    env_cfg.observations.policy.concatenate_terms = True

    # 🌟 提取 success_term（如果没有就给 None）
    success_term = getattr(env_cfg.terminations, "success", None)
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    # Disable recorder
    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Load policy
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)

    # Run policy
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        terminated, traj = rollout(policy, env, success_term, args_cli.horizon, device)
        results.append(terminated)
        print(f"[INFO] Trial {trial}: {terminated}\n")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()