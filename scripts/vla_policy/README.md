# VLA (Vision-Language-Action) 训练与验证

基于 ViT-B/16 + 语言编码器 + Transformer Decoder 的视觉-语言-动作模型，实验性探索将视觉和语言条件引入机械臂抓取策略。

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `train_vla.py` | VLA 模型训练（Action Chunking 版本） |
| `play_vla.py` | Isaac Sim 中可视化验证 |

数据采集脚本位于 `source/standalone/collect_full_grasp_vision.py`。

## 模型架构

```
相机图像 (128×128) ──▶ ViT-B/16 (后 3 层解冻) ──▶ 768 维
                                                       │
语言指令 "pick up the red USB disk" ──▶ 词嵌入+MLP ──▶ 256 维 ──▶ Fusion MLP ──▶ Transformer Decoder ──▶ (16, 9)
                                                       │
低维状态 (37 维) ──────────────────────────────────────┘
```

| 组件 | 结构 |
|------|------|
| Vision Encoder | ViT-B/16, ImageNet 预训练, 最后 3 层解冻 |
| Language Encoder | nn.Embedding(64, 128) + Linear → 256 维 |
| Fusion | Linear(768+256+37, 256) + ReLU |
| Action Decoder | Transformer Decoder (4 层, 4 头, d_model=256) |
| Action Head | Linear(256, 9) × 16 步 (Action Chunking) |
| 总参数 | ~88M (其中 ~8M 可训练) |

## 完整流程

### 1. 采集带图像的数据

```bash
python source/standalone/collect_full_grasp_vision.py \
    --n_demos 300 \
    --output logs/robomimic/full_grasp_vision.hdf5
```

HDF5 数据结构：
```
data/demo_N/
├── obs/
│   ├── policy     # (T, 37) float32
│   └── images     # (T, 128, 128, 3) uint8 (gzip 压缩)
└── actions        # (T, 9) float32
```

### 2. 训练

```bash
python scripts/vla_policy/train_vla.py
```

| 参数 | 值 |
|------|-----|
| Batch Size | 32 |
| Epochs | 300 |
| 学习率 | 1e-4 (AdamW + CosineAnnealing) |
| Chunk Size | 16 |
| 梯度裁剪 | norm=1.0 |

支持断点续训：中断后重新运行会自动加载最新 checkpoint 继续。

Checkpoint 保存在 `logs/vla_policy/vla_chunk_epoch_*.pt`。

### 3. 验证

```bash
python scripts/vla_policy/play_vla.py \
    --checkpoint logs/vla_policy/vla_chunk_epoch_300.pt \
    --norm_stats logs/vla_policy/norm_stats.npz
```

推理流程：
1. 相机采集 128×128 RGB → resize 224×224 → ImageNet 归一化
2. 构建 37 维状态观测 → 归一化到 [-1, 1]
3. 语言指令分词 → 嵌入
4. 模型推理 → (1, 16, 9) action chunk
5. Temporal Ensemble：多 chunk 加权平均 `w = exp(-0.1 * idx)`
6. 规则夹爪 + 悬停检测辅助下压

## 实验结果

| 阶段 | 表现 |
|------|------|
| 靠近物体 | 成功（dist 从 0.5 降到 0.08） |
| 动作平滑 | 成功（Action Chunking + Temporal Ensemble 消除震荡） |
| 下压抓取 | 失败（卡在 ~5-7cm 高度悬停） |
| 整体成功率 | 低 |

### 失败分析

1. **悬停区问题** — 训练数据中物体上方的帧占比过高，模型学到"停留在上方"是安全的
2. **视觉特征贡献有限** — 300 个 demo 的图像多样性不足，模型主要依赖 37 维状态输入
3. **单任务场景** — VLA 的优势在于多任务泛化，单任务下不如专门训练的 ACT

### 可能的改进方向

- 更激进地清洗悬停区数据（`clean_data.py` 降采样）
- 增加数据多样性（随机光照、随机物体颜色、相机扰动）
- 换用 CLIP 做视觉-语言对齐编码器（`pip install open-clip-torch`）
- 增加训练数据量（1000+ demos）
- 多任务训练（不同物体 + 不同指令）才能发挥 VLA 优势
