# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
ACT → RL Fine-tune 训练脚本

从预训练的 ACT 模型初始化 PPO Actor，然后进行 RL fine-tune。

使用方法:
    # 1. 先提取 ACT 权重
    python scripts/act_policy/extract_actor_weights.py --act_checkpoint logs/act_policy/act_epoch_300.pt

    # 2. 开始 fine-tune
    python scripts/rsl_rl/train_finetune.py \\
        --task Isaac-UDisk-Grasp-Finetune-v0 \\
        --act_checkpoint logs/act_policy/act_epoch_300_for_ppo.pt \\
        --init_from_act

    # 3. 对比纯 RL（不使用 ACT 初始化）
    python scripts/rsl_rl/train_finetune.py \\
        --task Isaac-UDisk-Grasp-Finetune-v0
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Fine-tune RL agent with ACT initialization.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-UDisk-Grasp-Finetune-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# ACT 初始化相关参数
parser.add_argument(
    "--act_checkpoint", 
    type=str, 
    default=None,
    help="Path to ACT checkpoint (extracted format with _for_ppo.pt suffix)"
)
parser.add_argument(
    "--init_from_act",
    action="store_true",
    help="Whether to initialize PPO actor from ACT weights"
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# =========================================================================
# 核心修改区：击破"套娃路径"，强制让 train.py 认识我们的环境
# =========================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
extension_root_dir = os.path.join(project_root, "source", "peg_in_hole")
sys.path.insert(0, extension_root_dir)

try:
    import peg_in_hole.tasks.manipulation.peg_in_hole
    print("\n✅ 成功将自定义环境和 PPO 算法载入训练脚本！\n")
except ModuleNotFoundError as e:
    print(f"\n❌ 找不到环境模块，请检查路径: {extension_root_dir}\n")
    raise e
# =========================================================================

import gymnasium as gym
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def load_act_weights_for_ppo(act_checkpoint_path: str, runner: OnPolicyRunner):
    """
    从 ACT checkpoint 加载权重到 PPO Actor
    
    Args:
        act_checkpoint_path: ACT 权重文件路径（_for_ppo.pt 格式）
        runner: RSL-RL runner 实例
    """
    print(f"\n🔄 从 ACT checkpoint 加载权重: {act_checkpoint_path}")
    
    act_data = torch.load(act_checkpoint_path, map_location="cpu", weights_only=False)
    act_state_dict = act_data["state_dict"]

    # runner.alg.policy is the ActorCritic module
    ppo_state_dict = runner.alg.policy.state_dict()
    
    # 统计映射信息
    mapped_keys = []
    skipped_keys = []
    
    # 逐层映射权重
    for ppo_key, ppo_param in ppo_state_dict.items():
        # 只映射 actor 相关的权重（跳过 critic）
        if not ppo_key.startswith("actor."):
            skipped_keys.append(ppo_key)
            continue
        
        # 检查 ACT 权重中是否有对应的 key
        if ppo_key in act_state_dict:
            act_param = act_state_dict[ppo_key]
            
            # 检查形状是否匹配
            if ppo_param.shape == act_param.shape:
                ppo_state_dict[ppo_key] = act_param
                mapped_keys.append(ppo_key)
                print(f"  ✅ {ppo_key}: {act_param.shape}")
            else:
                print(f"  ⚠️  {ppo_key}: 形状不匹配 PPO={ppo_param.shape}, ACT={act_param.shape}")
                skipped_keys.append(ppo_key)
        else:
            skipped_keys.append(ppo_key)
    
    # 加载映射后的权重
    runner.alg.policy.load_state_dict(ppo_state_dict)
    
    print(f"\n📊 权重映射统计:")
    print(f"  ✅ 成功映射: {len(mapped_keys)} 层")
    print(f"  ⏭️  跳过 (critic/不匹配): {len(skipped_keys)} 层")
    
    if len(mapped_keys) == 0:
        print("⚠️  警告: 没有成功映射任何权重！请检查 ACT checkpoint 格式。")
    
    return mapped_keys


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Fine-tune with ACT initialization."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # 添加 ACT 初始化标记到日志目录名
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.init_from_act:
        log_dir += "_act_init"
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # 🌟 ACT 权重初始化
    if args_cli.init_from_act:
        if args_cli.act_checkpoint is None:
            raise ValueError("--act_checkpoint 必须指定当使用 --init_from_act 时")
        
        if not os.path.exists(args_cli.act_checkpoint):
            raise ValueError(f"找不到 ACT checkpoint: {args_cli.act_checkpoint}")
        
        # 加载 ACT 权重到 PPO Actor
        mapped_keys = load_act_weights_for_ppo(args_cli.act_checkpoint, runner)
        
        # 保存初始化信息
        init_info = {
            "act_checkpoint": args_cli.act_checkpoint,
            "mapped_layers": mapped_keys,
            "init_method": "act_pretraining"
        }
        dump_pickle(os.path.join(log_dir, "params", "init_info.pkl"), init_info)
        print(f"\n💾 初始化信息已保存到: {log_dir}/params/init_info.pkl")
    else:
        print("\n📝 纯 RL 训练（不使用 ACT 初始化）")
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    print("\n" + "="*60)
    print("🚀 开始 Fine-tune 训练")
    print("="*60 + "\n")
    
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
