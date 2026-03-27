# Robomimic BC 训练与验证

本目录包含基于 [robomimic](https://robomimic.github.io/) + [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 的**行为克隆（BC）训练**与策略验证脚本，任务为自定义 `peg_in_hole`。

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `train.py` | BC 训练 —— 读取 HDF5 演示数据集，训练 MLP（或 RNN-BC）策略，保存 checkpoint |
| `play.py` | 通用评估脚本（Isaac Lab 框架封装） |
| `custom_play.py` | **推荐验证脚本** —— 直接调用 omni.isaac API，含 KNN 辅助、EMA 平滑、卡住保护 |

## 训练

```bash
python scripts/robomimic/train.py \
    --task Isaac-PegInHole-v0 \
    --dataset /path/to/demos.hdf5 \
    --algo bc \
    --name my_bc_run \
    --log_dir robomimic
```

可选参数：
- `--rnn` — 启用 RNN-BC（LSTM，horizon=10），默认为 MLP
- `--normalize_training_actions` — 训练前对动作进行归一化

Checkpoint 保存在 `./logs/robomimic/<task>/` 目录下。

## 验证（推荐使用 custom_play.py）

```bash
python custom_play.py --headless
```

> 注意：`custom_play.py` 位于项目根目录，checkpoint 路径和数据集路径在脚本内直接配置。

## custom_play.py 与 play.py 对比

| | `custom_play.py`（推荐） | `play.py` |
|---|---|---|
| 框架层 | 直接调用 `omni.isaac` 底层 API | Isaac Lab 高层封装（`gym.make`） |
| 环境搭建 | 手动创建 World / Franka / Cuboid | 由 `parse_env_cfg` 自动配置 |
| 观测构建 | 手动拼接（支持 24 维 / 37 维自适应） | 依赖环境 `policy` 观测键 |
| 动作反归一化 | 自动检测 checkpoint 是否含 stats，否则手动 min-max 反归一化 | 通过 `--norm_factor_min/max` 手动传入 |
| 辅助机制 | KNN 专家引导 + 卡住检测 + EMA 平滑 + 规则夹爪 | 无，纯 BC 输出 |
| 适用场景 | 真实部署调试、精细控制验证 | 快速批量 rollout 成功率统计 |

## 说明

- 观测空间使用单一拼接的 `policy` 向量（低维，无图像）。
- 训练/验证集按 80/20 自动划分，写入 HDF5 的 `mask` 组。
- 默认网络：3 层 MLP `[256, 256, 256]`，lr=1e-4，训练 100 epoch。
