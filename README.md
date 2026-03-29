# Franka Panda 具身智能抓取系统

基于 Isaac Lab 仿真平台的 Franka Panda 机械臂 U 盘抓取项目，集成多种模仿学习与强化学习算法，支持 ROS2 实机部署。

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)

### BC + 状态机辅助抓取效果

<p align="center">
  <img src="demo.gif" width="600" alt="BC + 状态机辅助抓取演示">
</p>

### ACT 算法训练效果

<p align="center">
  <img src="output.gif" width="600" alt="ACT 算法抓取演示">
</p>

---

## 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [环境依赖](#环境依赖)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [算法详解](#算法详解)
- [ROS2 部署](#ros2-部署)
- [数据工具链](#数据工具链)
- [Docker](#docker)

---

## 项目概述

本项目实现了一个完整的机器人抓取 pipeline：

1. **仿真环境** — 基于 Isaac Lab 构建 Franka Panda + U 盘抓取场景
2. **专家数据采集** — 使用 RMPflow PickPlaceController 自动采集示教轨迹
3. **策略训练** — 支持 BC、ACT、Diffusion Policy、PPO 四种算法
4. **仿真验证** — 在 Isaac Sim 中回放策略并统计成功率
5. **实机部署** — 通过 ROS2 节点将训练好的策略部署到真实 Franka 机械臂

### 任务描述

机械臂从桌面抓取一个 U 盘（红色长方体，12cm x 3cm x 2.5cm），抬升至目标位置 `[0.40, 0.0, 0.40]`。

| 空间 | 维度 | 内容 |
|------|------|------|
| 观测 | 37 维 | 关节位置(9) + 关节速度(9) + 物体位姿(7) + 末端位姿(7) + 目标相对位置(3) + 阶段编码(3) |
| 动作 | 9 维 | 7 个手臂关节 + 2 个夹爪关节（绝对关节位置） |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Isaac Lab 仿真环境                     │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐ │
│  │ Franka   │  │  U-Disk  │  │  PickPlaceController   │ │
│  │ Panda    │  │ (Cuboid) │  │  (RMPflow 专家策略)     │ │
│  └──────────┘  └──────────┘  └────────────────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       │ HDF5 数据集
                       ▼
┌─────────────────────────────────────────────────────────┐
│                      策略训练                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ BC (MLP) │  │   ACT    │  │ Diffusion│  │   PPO   │ │
│  │ robomimic│  │ CVAE+TF  │  │  Policy  │  │  rsl_rl │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       │ .pth checkpoint
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    ROS2 实机部署                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ franka_hybrid_policy_node                          │ │
│  │ BC 推理 + KNN 辅助 + 卡住检测 + EMA 平滑           │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 环境依赖

| 组件 | 版本 |
|------|------|
| Isaac Sim | 4.5.0 |
| Isaac Lab | 2.1.0 |
| Python | 3.10 |
| PyTorch | 2.x (CUDA) |
| robomimic | 0.3+ |
| ROS2 | Humble |

---

## 项目结构

```
my_robot_project/
│
├── source/                             # 核心源码
│   ├── peg_in_hole/                    # Isaac Lab 自定义扩展
│   │   └── peg_in_hole/tasks/manipulation/peg_in_hole/
│   │       ├── env_cfg.py              #   场景配置（Franka + U 盘 + 地面）
│   │       ├── agents.py               #   PPO 超参数配置
│   │       └── mdp/
│   │           ├── rewards.py          #   奖励函数（靠近 → 夹取 → 抬升）
│   │           ├── observations.py     #   观测定义
│   │           └── terminations.py     #   终止条件
│   └── standalone/                     # 独立脚本
│       ├── collect_full_grasp.py       #   专家数据采集（RMPflow）
│       ├── ros2_env_bridge.py          #   Isaac Lab ↔ ROS2 桥接
│       └── hello_robot.py              #   环境测试
│
├── scripts/                            # 训练与评估脚本
│   ├── robomimic/                      # Behavior Cloning (BC)
│   │   ├── train.py                    #   BC 训练
│   │   └── play.py                     #   通用评估（Isaac Lab 封装）
│   ├── act_policy/                     # ACT (Action Chunking with Transformers)
│   │   ├── train_act.py                #   ACT 训练（CVAE + Transformer）
│   │   ├── play_act.py                 #   ACT 评估
│   │   ├── clean_data.py              #   数据清洗
│   │   └── augment_descent.py          #   数据增强
│   └── rsl_rl/                         # 强化学习 (PPO)
│       ├── train.py                    #   PPO 训练
│       └── play.py                     #   RL 评估
│
├── ros2_deployment/                    # ROS2 部署包
│   └── franka_grasp_control/
│       ├── config/
│       │   ├── hybrid_policy.yaml      #   BC 混合策略参数
│       │   └── act_policy.yaml         #   ACT 策略参数
│       ├── scripts/
│       │   ├── franka_hybrid_policy_node.py  # BC 混合策略节点（BC + KNN 辅助）
│       │   ├── franka_act_policy_node.py     # ACT 策略节点（Temporal Ensemble）
│       │   └── franka_joint_publisher.py     # 关节状态发布
│       └── src/
│           └── grasp_node.cpp          #   C++ 控制节点
│
├── experiments/                        # 实验性基线（已归档）
│   ├── diffusion_policy_baseline/      #   Diffusion Policy
│   └── ik_grasp_baseline/              #   纯 IK 抓取
│
├── assets/                             # 3D 模型资产
│   ├── usb_stick/                      #   U 盘模型（STEP/USD/OBJ）
│   └── usb_port/                       #   USB 接口模型
│
├── docker/                             # Docker 容器化
│   ├── Dockerfile
│   └── docker-compose.yaml
│
├── custom_play.py                      # 自定义验证脚本（推荐）
├── add_noise.py                        # 数据加噪
├── cure_dataset.py                     # 动作前瞻偏移
├── check_data.py                       # 数据集检查
├── heal_data.py                        # 数据修复
└── logs/                               # 训练输出与数据集
    ├── robomimic/                      #   BC 模型 + 专家数据集
    ├── act_policy/                     #   ACT 模型
    └── rsl_rl/                         #   RL 模型
```

---

## 快速开始

### 1. 安装

```bash
# 前提：已安装 Isaac Lab 2.1.0
# 安装自定义扩展
cd source/peg_in_hole && pip install -e .

# 安装 robomimic
pip install robomimic
```

### 2. 数据采集

使用 RMPflow 专家控制器自动采集示教轨迹：

```bash
python source/standalone/collect_full_grasp.py \
    --n_demos 300 \
    --output logs/robomimic/full_grasp_abs_v5.hdf5 \
    --headless
```

### 3. 训练

**Behavior Cloning (BC)**

```bash
python scripts/robomimic/train.py \
    --task Isaac-UDisk-Grasp-v0 \
    --dataset logs/robomimic/full_grasp_abs_v5.hdf5 \
    --algo bc \
    --name bc_run
```

可选：`--rnn` 启用 RNN-BC（LSTM）

**ACT (Action Chunking with Transformers)**

```bash
python scripts/act_policy/train_act.py
```

**强化学习 (PPO)**

```bash
# 纯 RL 从零训练
python scripts/rsl_rl/train.py --task Isaac-UDisk-Grasp-v0
```

**ACT → RL Fine-tune**

```bash
# 1. 提取 ACT 权重为 PPO 格式
python scripts/act_policy/extract_actor_weights.py \
    --act_checkpoint logs/act_policy/act_epoch_300.pt

# 2. 用 ACT 权重初始化 PPO Actor 进行 fine-tune
python scripts/rsl_rl/train_finetune.py \
    --task Isaac-UDisk-Grasp-Finetune-v0 \
    --act_checkpoint logs/act_policy/act_epoch_300_for_ppo.pt \
    --init_from_act
```

详细说明见 [scripts/rsl_rl/README.md](scripts/rsl_rl/README.md)

### 4. 验证

推荐使用 `custom_play.py`（含 KNN 辅助、EMA 平滑、卡住保护）：

```bash
python custom_play.py
```

**ACT 验证：**

```bash
python scripts/act_policy/play_act.py
```

**BC 通用评估（批量 rollout）：**

```bash
python scripts/robomimic/play.py \
    --task Isaac-UDisk-Grasp-v0 \
    --checkpoint logs/robomimic/.../model_epoch_100.pth \
    --num_rollouts 10
```

**RL Fine-tune 验证：**

```bash
# 可视化推理（16 个并行环境）
python scripts/rsl_rl/play.py \
    --task Isaac-UDisk-Grasp-Finetune-v0 \
    --num_envs 16

# 批量评估成功率
python scripts/rsl_rl/eval_finetuned.py \
    --task Isaac-UDisk-Grasp-Finetune-v0 \
    --num_envs 50 \
    --num_episodes 5
```

---

## 算法详解

### Behavior Cloning (BC)

基于 robomimic 框架的标准 BC，3 层 MLP `[256, 256, 256]`。

| 参数 | 值 |
|------|-----|
| 动作归一化 | min-max → [-1, 1] |
| 训练/验证划分 | 80% / 20% |
| 学习率 | 1e-4 |
| Epochs | 100 |
| Batch Size | 100 |

### ACT (Action Chunking with Transformers)

CVAE 编码器 + Transformer 解码器，一次预测未来 50 步动作序列。

| 参数 | 值 |
|------|-----|
| Chunk Size | 50 |
| Latent Dim | 32 |
| Transformer Layers | 4 |
| Attention Heads | 4 |
| Hidden Dim | 256 |
| Epochs | 300 |
| KL Weight | 10.0 |

- 训练损失：L1 重建 + KL 散度
- 采样加权：对"悬停区"样本降权，避免策略在物体上方犹豫

### PPO (强化学习)

基于 rsl_rl 框架，多阶段奖励函数：

| 奖励 | 作用 |
|------|------|
| `reaching_reward` | 高斯核距离引导末端靠近 U 盘 |
| `grasping_reward` | 连续抓取奖励（手指距离 + 夹爪闭合 + 抬起进度） |
| `lift_to_target_reward` | 将物体抬升至目标高度 |
| `top_down_posture_reward` | 强制夹爪垂直朝下 |
| `gripper_close_near_object` | 靠近物体时奖励闭合夹爪 |
| `action_rate_penalty` | 惩罚动作突变，抑制抖动 |
| `joint_acceleration_penalty` | 惩罚关节加速度 |

### ACT → RL Fine-tune（实验性）

将 ACT 预训练权重初始化 PPO Actor，再用 RL 微调。

| 参数 | 值 |
|------|-----|
| Actor 结构 | [256, 256]（匹配 ACT obs_encoder） |
| 学习率 | 3e-5（保护预训练权重） |
| 初始噪声 | 0.1 |
| 训练轮数 | 800 |
| 推理后处理 | EMA 平滑 + 规则夹爪 |

### 算法效果对比

| 算法 | 成功率 | 动作平滑度 | 说明 |
|------|--------|-----------|------|
| BC + 状态机辅助 | 较高 | 高（EMA + KNN 辅助） | 需要复杂工程辅助机制 |
| ACT (Temporal Ensemble) | 较高 | 高（chunk 加权平均） | 端到端，部署简洁 |
| PPO (纯 RL) | 低 | 低（逐帧决策） | 探索效率低，夹爪控制难学 |
| ACT → RL Fine-tune | ~12% | 中（需后处理） | 实验性，策略能靠近但抓取不稳定 |

结论：对于精密抓取任务，模仿学习（BC/ACT）显著优于纯 RL。RL 的优势在于不需要示教数据，但在高维关节空间中探索效率低，尤其是夹爪开合时机难以通过奖励信号学习。

---

## ROS2 部署

部署架构采用两个独立进程：**桥接节点**（运行在 Isaac Sim 内）负责仿真环境交互，**策略节点**（纯 ROS2）负责推理决策。

### 话题接口

```
                    ┌──────────────────────────┐
/joint_states ──┐   │                          │
/franka/ee_pose ─┤──▶  策略节点（BC 或 ACT）   ──▶ /joint_command ──▶ 桥接节点
/udisk/pose ────┘   │                          │        │
                    └──────────────────────────┘        ▼
                         │                        Isaac Sim 仿真
                         ├── /assist_start ──▶ 桥接节点（专家接管）
                         └── /assist_stop  ──▶ 桥接节点（归还控制）
```

### 启动桥接节点

```bash
# 终端 1：启动 Isaac Sim + ROS2 桥接
python source/standalone/ros2_env_bridge.py
```

桥接节点发布 `/joint_states`、`/franka/ee_pose`、`/udisk/pose`，接收 `/joint_command` 控制机械臂。当收到 `/assist_start` 时切换到内置 PickPlaceController 专家控制。

### 方案一：BC 混合策略部署

```bash
# 终端 2：
python ros2_deployment/franka_grasp_control/scripts/franka_hybrid_policy_node.py \
    --ros-args --params-file \
    ros2_deployment/franka_grasp_control/config/hybrid_policy.yaml
```

BC 节点特点：
- robomimic BC 网络推理，输出 delta 动作（增量叠加到当前关节）
- KNN 专家引导：从数据集构建 KNN bank，卡住时混入专家动作
- EMA 平滑（alpha=0.7）+ 关节增量限幅 0.004 rad/step
- 复杂辅助状态机：卡住检测、下压救援、burst 脉冲（~60 个可调参数）
- 规则夹爪：xy < 10cm 且 z < 14cm 时闭合

配置文件：`ros2_deployment/franka_grasp_control/config/hybrid_policy.yaml`

### 方案二：ACT 策略部署

```bash
# 终端 2：
python ros2_deployment/franka_grasp_control/scripts/franka_act_policy_node.py \
    --ros-args --params-file \
    ros2_deployment/franka_grasp_control/config/act_policy.yaml
```

ACT 节点特点：
- ACT 模型推理（CVAE + Transformer Decoder），输出绝对关节位置（非增量）
- Temporal Ensemble：每步推理产生 50 步 chunk，多个 chunk 加权平均（`w = exp(-0.1 * idx)`），动作更平滑
- 观测归一化：输入前将 37 维观测归一化到 [-1, 1]
- 动作反归一化：`action = 0.5 * (norm + 1.0) * range + min`
- 夹爪控制：模型预测 + 规则保护（xy < 5cm 且 z < 6cm 才闭合，dist > 8cm 强制张开）
- 简化辅助：仅卡住检测 → 发 `/assist_start` 事件让桥接节点接管（~15 个参数）
- Slew rate limiting：步间关节变化限幅 0.006 rad/step

配置文件：`ros2_deployment/franka_grasp_control/config/act_policy.yaml`

### BC vs ACT 部署对比

| | BC 混合策略 | ACT 策略 |
|---|---|---|
| 动作类型 | delta（当前关节 + 增量） | 绝对关节位置（直接应用） |
| 推理方式 | 单步输出 | chunk(50步) + temporal ensemble |
| 观测处理 | 原始值直接输入 | 归一化到 [-1, 1] |
| 模型依赖 | robomimic + h5py | 纯 PyTorch（自包含） |
| 辅助机制 | KNN + 状态机 + burst 脉冲 | 简化卡住检测 → assist 事件 |
| 参数复杂度 | ~60 个 | ~15 个 |
| 适用场景 | 需要精细工程调优的场景 | 模型能力较强、追求简洁部署 |

---

## 数据工具链

| 脚本 | 功能 |
|------|------|
| `collect_full_grasp.py` | RMPflow 专家采集绝对关节位置轨迹 |
| `add_noise.py` | 注入微小噪声（std=1e-5），防止归一化零方差 |
| `cure_dataset.py` | 动作前瞻偏移（向前 15 帧），迫使网络预测未来 |
| `check_data.py` | 检查数据集中的 NaN / Inf |
| `heal_data.py` | 修复损坏数据 |

HDF5 数据格式：

```
data/
├── demo_0/
│   ├── obs/policy     # (T, 37) float32
│   └── actions        # (T, 9)  float32
├── demo_1/ ...
└── mask/
    ├── train          # 训练集 demo 列表
    └── valid          # 验证集 demo 列表
```

---

## Docker

```bash
# 构建镜像（需先构建 isaac-lab-base）
cd docker
docker compose --env-file .env.base build

# 启动容器
docker compose --env-file .env.base up -d

# 进入容器
docker exec -it isaac-lab-template /bin/bash
```
