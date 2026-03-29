"""
ACT 权重提取脚本
从训练好的 ACT 模型中提取 obs_encoder 和 action_head 权重，
转换为 PPO Actor 可用的格式。

ACT 结构:
  obs_encoder: Linear(37, 256) -> ReLU -> Linear(256, 256)
  action_head: Linear(256, 9)

PPO Actor 目标结构:
  MLP: Linear(37, 256) -> ReLU -> Linear(256, 256) -> action_head(256, 9)
"""
import os
import torch
import argparse
import numpy as np


def extract_act_weights(act_checkpoint_path: str, output_path: str = None):
    """
    从 ACT checkpoint 提取可用于 PPO 初始化的权重

    Args:
        act_checkpoint_path: ACT 模型路径 (如 act_epoch_300.pt)
        output_path: 输出路径，默认与 checkpoint 同目录
    """
    print(f"📂 加载 ACT checkpoint: {act_checkpoint_path}")
    ckpt = torch.load(act_checkpoint_path, map_location="cpu")

    if "model" not in ckpt:
        raise ValueError("Checkpoint 格式错误，找不到 'model' key")

    act_state_dict = ckpt["model"]
    config = ckpt.get("config", {})

    print(f"📋 ACT 配置: obs_dim={config.get('obs_dim')}, action_dim={config.get('action_dim')}")

    # 构建映射后的 state_dict
    # rsl_rl ActorCritic 用 nn.Sequential 构建 actor:
    #   actor.0 = Linear(obs_dim, 256)  + actor.1 = ELU
    #   actor.2 = Linear(256, 256)      + actor.3 = ELU
    #   actor.4 = Linear(256, action_dim)  (输出层)

    mapped_weights = {}

    # ACT obs_encoder.0 (37 -> 256) -> PPO actor.0
    if "obs_encoder.0.weight" in act_state_dict:
        mapped_weights["actor.0.weight"] = act_state_dict["obs_encoder.0.weight"]
        mapped_weights["actor.0.bias"] = act_state_dict["obs_encoder.0.bias"]
        print("✅ 映射 obs_encoder.0 -> actor.0")
    else:
        raise ValueError("找不到 ACT obs_encoder.0 权重")

    # ACT obs_encoder.2 (256 -> 256) -> PPO actor.2
    if "obs_encoder.2.weight" in act_state_dict:
        mapped_weights["actor.2.weight"] = act_state_dict["obs_encoder.2.weight"]
        mapped_weights["actor.2.bias"] = act_state_dict["obs_encoder.2.bias"]
        print("✅ 映射 obs_encoder.2 -> actor.2")
    else:
        raise ValueError("找不到 ACT obs_encoder.2 权重")

    # ACT action_head (256 -> 9) -> PPO actor.4
    if "action_head.weight" in act_state_dict:
        mapped_weights["actor.4.weight"] = act_state_dict["action_head.weight"]
        mapped_weights["actor.4.bias"] = act_state_dict["action_head.bias"]
        print("✅ 映射 action_head -> actor.4")
    else:
        raise ValueError("找不到 ACT action_head 权重")

    # 保存映射后的权重
    if output_path is None:
        output_path = act_checkpoint_path.replace(".pt", "_for_ppo.pt")

    torch.save({
        "state_dict": mapped_weights,
        "source_checkpoint": act_checkpoint_path,
        "source_config": config,
        "target_actor_hidden_dims": [256, 256],
    }, output_path)

    print(f"\n💾 权重已保存到: {output_path}")
    print(f"📊 映射了 {len(mapped_weights)} 个权重张量")

    # 打印权重形状验证
    print("\n🔍 权重形状验证:")
    for name, param in mapped_weights.items():
        print(f"  {name}: {param.shape}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="从 ACT checkpoint 提取 PPO Actor 初始化权重")
    parser.add_argument(
        "--act_checkpoint",
        type=str,
        default="/home/jmx001/my_program/my_robot_project/logs/act_policy/act_epoch_300.pt",
        help="ACT 模型 checkpoint 路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出路径（默认与 checkpoint 同目录）"
    )

    args = parser.parse_args()

    if not os.path.exists(args.act_checkpoint):
        print(f"❌ 错误: 找不到 checkpoint 文件: {args.act_checkpoint}")
        return

    extract_act_weights(args.act_checkpoint, args.output)


if __name__ == "__main__":
    main()
