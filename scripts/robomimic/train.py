# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""
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
    print("✅ 成功将自定义环境载入模仿学习脚本！")
except ModuleNotFoundError:
    print(f"❌ 找不到环境模块，请检查路径: {extension_root_dir}")
# =========================================================================

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Standard library imports
import json
import numpy as np
import shutil
import time
import torch
import traceback
from collections import OrderedDict
from torch.utils.data import DataLoader
import h5py

import psutil

# Robomimic imports
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import algo_factory
from robomimic.config import Config, config_factory
from robomimic.utils.log_utils import DataLogger, PrintLogger

# Isaac Lab imports (needed so that environment is registered)
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401


def normalize_hdf5_actions(config: Config, log_dir: str) -> str:
    # 此处不需要对我们的动作进行归一化
    pass


def train(config: Config, device: str, log_dir: str, ckpt_dir: str, video_dir: str):
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")

    if config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    ObsUtils.initialize_obs_utils_with_config(config)

    # 🌟 提取绝对真实的字符串路径
    if isinstance(config.train.data, list):
        check_path = config.train.data[0]["path"] if isinstance(config.train.data[0], dict) else config.train.data[0]
    else:
        check_path = config.train.data

    dataset_path = os.path.expanduser(check_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset at provided path {dataset_path} not found!")

    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(check_path)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        {"path": check_path},
        all_obs_keys=config.all_obs_keys, 
        action_keys=config.train.action_keys,
        verbose=True
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env

    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        env_names = [env_meta["env_name"]]
        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)
        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
            )
            envs[env.name] = env

    data_logger = DataLogger(log_dir, config=config, log_tb=config.experiment.logging.log_tb)
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)
    print("")

    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()
    # 训练时如果用了 action normalization，这里会计算并缓存统计量；
    # 保存到 checkpoint 后，推理端 RolloutPolicy 才能自动反归一化。
    action_normalization_stats = trainset.get_action_normalization_stats()

    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )

    if config.experiment.validate:
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        valid_loader = None

    best_valid_loss = None
    last_ckpt_time = time.time()
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(1, config.train.num_epochs + 1):
        step_log = TrainUtils.run_epoch(model=model, data_loader=train_loader, epoch=epoch, num_steps=train_num_steps)
        model.on_epoch_end(epoch)

        epoch_ckpt_name = f"model_epoch_{epoch}"
        should_save_ckpt = False
        
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and (
                time.time() - last_ckpt_time > config.experiment.save.every_n_seconds
            )
            epoch_check = (
                (config.experiment.save.every_n_epochs is not None)
                and (epoch > 0)
                and (epoch % config.experiment.save.every_n_epochs == 0)
            )
            epoch_list_check = epoch in config.experiment.save.epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check
            
        print(f"Train Epoch {epoch}")
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record(f"Timing_Stats/Train_{k[5:]}", v, epoch)
            else:
                data_logger.record(f"Train/{k}", v, epoch)

        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Valid/{k}", v, epoch)

            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += f"_best_validation_{best_valid_loss}"
                    should_save_ckpt = True

        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print(f"Epoch {epoch} Memory Usage: {mem_usage} MB")

    data_logger.close()


def main(args: argparse.Namespace):
    if args.task is not None:
        cfg_entry_point_key = f"robomimic_{args.algo}_cfg_entry_point"
        print(f"Loading configuration for task: {args.task}")
        import yaml
        
        try:
            import gymnasium as gym
            cfg_entry_point_file = gym.spec(args.task).kwargs.get(cfg_entry_point_key)
            if cfg_entry_point_file is None:
                raise KeyError
        except KeyError:
            print(f"\n⚠️ 没找到你的专属 BC 配置，直接强行注入官方 Franka 满级配置！")
            cfg_entry_point_file = "/home/jmx001/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents/robomimic/bc.json"
            print(f"✅ 成功装备官方满级配置: {cfg_entry_point_file}\n")
            
        with open(cfg_entry_point_file) as f:
            if cfg_entry_point_file.endswith(".yaml"):
                ext_cfg = yaml.safe_load(f)
            else:
                ext_cfg = json.load(f)
            config = config_factory(ext_cfg["algo_name"])
            
        with config.values_unlocked():
            config.update(ext_cfg)
            config.observation.modalities.obs.low_dim = ["policy"]

            # 1) 观测不做自动归一化（避免零方差维度导致不稳定）
            config.train.hdf5_normalize_obs = False

            # 2) 策略骨干网络（部署优先：保持无状态 MLP，便于 ROS2 实时推理）
            config.algo.actor_layer_dims = [256, 256, 256]

            # 3) RNN 设置（通过命令行 --rnn 开启）
            config.algo.rnn.enabled = False
            config.train.seq_length = 1
            config.train.pad_seq_length = True
            config.train.frame_stack = 1
            config.train.pad_frame_stack = True

            # 4) 训练轮次与步数
            config.train.num_epochs = 100
            config.experiment.epoch_every_n_steps = 500
            config.experiment.validation_epoch_every_n_steps = 10
            config.train.batch_size = 100

            # 5) 数据键与动作归一化
            config.train.dataset_keys = ["actions"]
            config.train.action_config = {"actions": {"normalization": "min_max"}}

            # 6) 优化稳定性
            config.train.max_grad_norm = 10.0
            config.algo.optim_params.policy.learning_rate.initial = 1e-4

            # RNN 覆盖（--rnn 时启用）
            if args.rnn:
                config.algo.rnn.enabled = True
                config.algo.rnn.horizon = 10
                config.algo.rnn.hidden_dim = 400
                config.algo.rnn.rnn_type = "LSTM"
                config.algo.rnn.num_layers = 2
                config.train.seq_length = 10
                config.train.pad_seq_length = True
                config.train.batch_size = 64
                config.train.num_epochs = 200
    if args.dataset is not None:
        # 🌟 喂给它梦寐以求的“列表套字典”格式
        config.train.data = [{"path": args.dataset}]

    if args.name is not None:
        config.experiment.name = args.name

    config.train.output_dir = os.path.abspath(os.path.join("./logs", args.log_dir, args.task))

    exp_dirs = TrainUtils.get_exp_dir(config)
    log_dir, ckpt_dir = exp_dirs[0], exp_dirs[1]
    video_dir = exp_dirs[-1]

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    
    # 🌟 关键补丁：生成严格不重叠的数据集名单
    if args.dataset:
        with h5py.File(args.dataset, "a") as f:
            if "env_args" not in f["data"].attrs:
                f["data"].attrs["env_args"] = json.dumps({"env_name": args.task, "type": 1, "env_kwargs": {}})
            
            if "mask" in f:
                del f["mask"]
                
            print("⚠️ 正在生成严格不重叠的 train/valid 分割名单...")
            mask_grp = f.create_group("mask")
            demos = list(f["data"].keys())
            demo_encoded = [d.encode("utf-8") for d in demos]
            
            # 80% 训练，20% 验证（互不重叠）
            split_idx = int(len(demo_encoded) * 0.8)
            train_demos = demo_encoded[:split_idx]
            valid_demos = demo_encoded[split_idx:]
            
            mask_grp.create_dataset("train", data=train_demos)
            mask_grp.create_dataset("valid", data=valid_demos)
            print(f"✅ 名单划分完成！训练集: {len(train_demos)}条，验证集: {len(valid_demos)}条")
            
    config.lock()

    res_str = "finished run successfully!"
    try:
        train(config, device, log_dir, ckpt_dir, video_dir)
    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
    print(res_str)
    import sys; sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="robomimic")
    parser.add_argument("--normalize_training_actions", action="store_true", default=False)
    parser.add_argument("--rnn", action="store_true", default=False, help="启用 RNN-BC")
    
    args = parser.parse_args()
    main(args)
    simulation_app.close()
