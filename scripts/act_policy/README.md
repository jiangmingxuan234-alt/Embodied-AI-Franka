# ACT (Action Chunking with Transformers) 训练与验证

基于 CVAE + Transformer Decoder 的模仿学习算法，一次预测未来 50 步动作序列（action chunk），通过 temporal ensemble 实现平滑控制。

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `train_act.py` | ACT 模型训练 |
| `play_act.py` | Isaac Sim 中可视化验证 |
| `clean_data.py` | 数据清洗（下采样悬停区冗余帧） |
| `augment_descent.py` | 数据增强（插入合成下压帧） |
| `extract_actor_weights.py` | 提取权重为 PPO 格式（用于 RL fine-tune） |

## 模型架构

```
观测 (37 维) ──▶ Obs Encoder (MLP 37→256→256)
                        │
                        ▼
              ┌─── Memory (2 tokens) ───┐
              │  obs_token   z_token    │
              └──────────┬──────────────┘
                         ▼
              Transformer Decoder (4 层, 4 头)
                         │
                    Action Queries (50 个)
                         │
                         ▼
              Action Head (256→9) × 50 步
```

| 参数 | 值 |
|------|-----|
| 观测维度 | 37 |
| 动作维度 | 9（7 手臂关节 + 2 夹爪） |
| Chunk Size | 50（一次预测 50 步） |
| Latent Dim | 32（CVAE 隐变量） |
| Hidden Dim (d_model) | 256 |
| Transformer Layers | 4 |
| Attention Heads | 4 |
| 总参数量 | ~3.5M |

## 观测空间（37 维）

```
[0:9]   关节位置（7 手臂 + 2 夹爪）
[9:18]  关节速度（7 手臂 + 2 夹爪）
[18:21] 相对位置（物体 - 末端执行器）
[21:24] 物体绝对位置
[24:27] 末端执行器位置
[27:31] 末端执行器四元数 (w,x,y,z)
[31:34] 目标相对位置（目标 - 物体）
[34:37] 阶段编码 [在桌面, 移动中, 到达目标]
```

## 训练

### 数据准备

```bash
# 1. 清洗数据（下采样悬停区，每 8 帧保留 1 帧）
python scripts/act_policy/clean_data.py

# 2. 数据增强（在夹爪闭合前插入 25 帧合成下压动作）
python scripts/act_policy/augment_descent.py
```

### 开始训练

```bash
python scripts/act_policy/train_act.py
```

训练配置（在脚本内修改）：

| 参数 | 值 |
|------|-----|
| 数据集 | `logs/robomimic/full_grasp_abs_v5.hdf5` |
| Batch Size | 256 |
| Epochs | 300 |
| 学习率 | 1e-4（AdamW + CosineAnnealing） |
| KL Weight | 10.0 |
| 梯度裁剪 | norm=1.0 |

训练损失：
- 重建损失：MSE（预测 chunk vs 真实 chunk）
- KL 散度：标准 VAE KL 项
- 总损失：`loss = recon_loss + 10.0 * kl_loss`

采样策略：对悬停区样本（`rel_z ∈ [-0.08, -0.05]`）降权至 0.15，避免策略在物体上方犹豫。

输出文件：
- `logs/act_policy/act_epoch_*.pt` — 模型 checkpoint（每 50 epoch 保存）
- `logs/act_policy/norm_stats.npz` — 归一化统计量（act_min/max, obs_min/max）

## 验证

```bash
python scripts/act_policy/play_act.py \
    --checkpoint logs/act_policy/act_epoch_300.pt \
    --norm_stats logs/act_policy/norm_stats.npz
```

### 推理流程

1. 构建 37 维观测 → 归一化到 [-1, 1]
2. `model.inference(obs)` → 输出 (1, 50, 9) 归一化 action chunk
3. 反归一化：`action = 0.5 * (norm + 1.0) * range + min`
4. **Temporal Ensemble**：多个 chunk 加权平均，`w = exp(-0.1 * idx)`
5. 夹爪：最新 chunk 的 step-0 预测，>0.02 → 张开(0.04)，否则闭合(0.0)
6. 关节限位裁剪

### Temporal Ensemble 原理

每步推理产生一个 50 步的 chunk。多个 chunk 在时间上重叠，对当前步取所有活跃 chunk 的加权平均：

```
chunk_1: [a1, a2, a3, a4, ...]     权重: [1.0, 0.90, 0.82, 0.74, ...]
chunk_2:     [b1, b2, b3, ...]     权重: [1.0, 0.90, 0.82, ...]
chunk_3:         [c1, c2, ...]     权重: [1.0, 0.90, ...]
                  ↓
当前步动作 = weighted_average(a3, b2, c1)
```

这使得动作输出天然平滑，无需额外的 EMA 或 slew limiting。

## 提取权重用于 RL Fine-tune

```bash
python scripts/act_policy/extract_actor_weights.py \
    --act_checkpoint logs/act_policy/act_epoch_300.pt
```

输出：`logs/act_policy/act_epoch_300_for_ppo.pt`

映射关系：`obs_encoder.0` → `actor.0`，`obs_encoder.2` → `actor.2`，`action_head` → `actor.4`

## 数据格式

HDF5 文件结构：

```
data/
├── demo_0/
│   ├── obs/policy     # (T, 37) float32 — 观测序列
│   └── actions        # (T, 9)  float32 — 绝对关节位置
├── demo_1/ ...
```

动作归一化到 [-1, 1]，观测同样归一化，统计量保存在 `norm_stats.npz`。
