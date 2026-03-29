# 强化学习训练与 ACT → RL Fine-tune

本目录包含基于 [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 框架的强化学习训练脚本，支持纯 PPO 从零训练和 ACT 预训练权重初始化后的 RL 微调。

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `train.py` | 纯 RL 训练 —— PPO 从零学习抓取策略 |
| `train_finetune.py` | ACT → RL Fine-tune —— 用 ACT 权重初始化 PPO Actor 后微调 |
| `play.py` | 可视化推理 —— 加载训练好的策略在仿真中回放 |
| `eval_finetuned.py` | 批量评估 —— 统计多回合成功率 |
| `cli_args.py` | 命令行参数工具 |

## 环境配置

项目注册了两个 Gymnasium 环境：

| 环境 ID | 观测维度 | 用途 |
|---------|---------|------|
| `Isaac-UDisk-Grasp-v0` | 25 维 | 纯 RL 训练（joint_pos + joint_vel + obj_pos + obj_quat） |
| `Isaac-UDisk-Grasp-Finetune-v0` | 37 维 | ACT → RL Fine-tune（与 ACT 训练时观测一致） |

37 维观测构成：

```
joint_pos(9) + joint_vel(9) + rel_pos(3) + obj_pos(3) + ee_pos(3) + ee_quat(4) + goal_rel(3) + phase(3)
```

## 方案一：纯 RL 从零训练

### 训练

```bash
python scripts/rsl_rl/train.py \
    --task Isaac-UDisk-Grasp-v0 \
    --num_envs 300 \
    --headless
```

### PPO 超参数

| 参数 | 值 |
|------|-----|
| Actor 网络 | MLP [256, 128, 64] |
| Critic 网络 | MLP [256, 128, 64] |
| 学习率 | 1e-3（自适应衰减） |
| 初始噪声 | 1.0 |
| 探索系数 | 0.006 |
| 训练轮数 | 1500 |
| 并行环境数 | 300 |
| 每步采样 | 24 steps/env |

### 验证

```bash
python scripts/rsl_rl/play.py \
    --task Isaac-UDisk-Grasp-v0 \
    --num_envs 32
```

Checkpoint 保存在 `logs/rsl_rl/Franka_UDisk_Grasp/` 目录下。

---

## 方案二：ACT → RL Fine-tune（推荐）

将训练好的 ACT 模型的 `obs_encoder` 和 `action_head` 权重映射到 PPO Actor 网络，作为初始化权重，再用 RL 奖励信号微调。

### 完整流程

#### 第 1 步：提取 ACT 权重

将 ACT checkpoint 中的权重转换为 PPO Actor 可用的格式：

```bash
python scripts/act_policy/extract_actor_weights.py \
    --act_checkpoint logs/act_policy/act_epoch_300.pt
```

输出文件：`logs/act_policy/act_epoch_300_for_ppo.pt`

权重映射关系：

| ACT 层 | PPO Actor 层 | 形状 |
|--------|-------------|------|
| `obs_encoder.0` (Linear) | `actor.0` | (256, 37) |
| `obs_encoder.2` (Linear) | `actor.2` | (256, 256) |
| `action_head` (Linear) | `actor.4` | (9, 256) |

#### 第 2 步：Fine-tune 训练

```bash
python scripts/rsl_rl/train_finetune.py \
    --task Isaac-UDisk-Grasp-Finetune-v0 \
    --act_checkpoint logs/act_policy/act_epoch_300_for_ppo.pt \
    --init_from_act \
    --headless
```

训练时间约 1 小时 45 分钟（800 iterations，400 并行环境）。

### Fine-tune PPO 超参数

| 参数 | 值 | 与纯 RL 的区别 |
|------|-----|---------------|
| Actor 网络 | MLP [256, 256] | 匹配 ACT obs_encoder 结构 |
| Critic 网络 | MLP [256, 128, 64] | 不变 |
| 学习率 | 3e-5 | 降低 33 倍，保护预训练权重 |
| 初始噪声 | 0.1 | 降低 10 倍，信任 ACT 策略 |
| 探索系数 | 0.002 | 降低 3 倍，减少随机探索 |
| 训练轮数 | 800 | 减少，收敛更快 |
| 并行环境数 | 400 | 增加，更稳定梯度 |
| 每步采样 | 48 steps/env | 增加，更长 horizon |

#### 第 3 步：可视化验证

```bash
python scripts/rsl_rl/play.py \
    --task Isaac-UDisk-Grasp-Finetune-v0 \
    --num_envs 16
```

`play.py` 会自动根据 task 名选择正确的环境配置和网络架构。推理时包含以下后处理：

- EMA 平滑（alpha=0.7）：滤除高频抖动
- 规则夹爪：末端距物体 xy < 9cm 且 z < 10cm 时强制闭合

#### 第 4 步：批量评估

```bash
python scripts/rsl_rl/eval_finetuned.py \
    --task Isaac-UDisk-Grasp-Finetune-v0 \
    --num_envs 50 \
    --num_episodes 5
```

Checkpoint 保存在 `logs/rsl_rl/Franka_UDisk_Grasp_Finetune/` 目录下。

---

## 奖励函数

### 纯 RL 奖励 (`RewardsCfg`)

| 奖励 | 权重 | 说明 |
|------|------|------|
| `reaching_reward` | +1.0 | 高斯核距离奖励 `exp(-5d²)` |
| `grasping_reward` | +10.0 | 手指距离 + 夹爪闭合 + 抬起进度 |
| `lift_to_target_reward` | +20.0 | 高斯核抬升奖励 |
| `top_down_posture_reward` | +5.0 | 强制夹爪垂直朝下 |
| `hover_above_reward` | +5.0 | XY 轴对齐奖励 |
| `action_rate_penalty` | -0.05 | 惩罚动作突变 |
| `drop_penalty` | -10.0 | 惩罚物体掉落 |

### Fine-tune 奖励 (`RewardsFinetuneCfg`)

在纯 RL 基础上增加了三斧惩罚和夹爪控制：

| 奖励 | 权重 | 说明 |
|------|------|------|
| `reaching_reward` | +1.5 | 高斯核距离奖励 |
| `grasping_reward` | +5.0 | 连续抓取奖励 |
| `lift_to_target_reward` | +25.0 | 高斯核抬升奖励（主导） |
| `top_down_posture_reward` | +3.0 | 姿态约束 |
| `hover_above_reward` | +3.0 | XY 对齐 |
| `gripper_close_near_object` | +5.0 | 靠近时奖励闭合夹爪 |
| `action_rate_penalty` | -0.5 | 严惩动作突变（第一斧） |
| `joint_acceleration_penalty` | -0.01 | 惩罚关节加速度（第二斧） |
| `action_l2_penalty` | -0.005 | 动作 L2 正则化（第三斧） |
| `drop_penalty` | -10.0 | 惩罚物体掉落 |

---

## 实验结果

### 算法对比

| 算法 | 成功率 | 动作平滑度 | 说明 |
|------|--------|-----------|------|
| BC + 状态机辅助 | 较高 | 高 | 需要 KNN + EMA + 卡住检测等工程辅助 |
| ACT (Temporal Ensemble) | 较高 | 高 | 端到端，chunk 加权平均天然平滑 |
| PPO (纯 RL) | 低 | 低 | 9 维关节空间探索效率低 |
| ACT → RL Fine-tune | ~12% | 中 | 策略能靠近目标，抓取不稳定 |

### 分析

ACT → RL Fine-tune 的主要瓶颈：

1. **夹爪控制** — RL 难以学习精确的夹爪开合时机，推理时需要规则夹爪覆盖
2. **动作平滑性** — PPO 逐帧决策，天然缺乏时序连续性，需要 EMA 后处理
3. **探索效率** — 9 维关节空间中，随机探索很难碰巧完成"靠近 → 对齐 → 下压 → 闭合 → 抬起"的完整序列
4. **奖励稀疏** — `lift_to_target` 只有在物体被成功抬起后才有信号，前期几乎为零

### 可能的改进方向

- 缩小动作空间：RL 输出末端 xyz + 夹爪（4 维），底层用 IK 控制器
- 课程学习：分阶段训练（reaching → grasping → lifting）
- 增加训练量：3000+ iterations + 1024 并行环境
- DAgger：训练时混入 ACT 专家动作在线纠正

---

## 目录结构

```
scripts/rsl_rl/
├── train.py                # 纯 RL 训练（PPO 从零）
├── train_finetune.py       # ACT → RL Fine-tune 训练
├── play.py                 # 可视化推理（自动选择环境配置）
├── eval_finetuned.py       # 批量评估成功率
├── cli_args.py             # 命令行参数工具
└── README.md               # 本文件

scripts/act_policy/
└── extract_actor_weights.py  # ACT 权重提取为 PPO 格式

source/peg_in_hole/.../
├── env_cfg.py              # 环境配置（GraspEnvCfg / GraspEnvFinetuneCfg）
├── agents.py               # PPO 超参数（纯 RL / Fine-tune）
└── mdp/rewards.py          # 奖励函数定义

logs/rsl_rl/
├── Franka_UDisk_Grasp/           # 纯 RL 训练输出
└── Franka_UDisk_Grasp_Finetune/  # Fine-tune 训练输出
```
