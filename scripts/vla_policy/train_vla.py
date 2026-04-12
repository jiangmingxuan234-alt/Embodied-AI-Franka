"""
VLA (Vision-Language-Action) 训练脚本 — Action Chunking + ViT 特征预计算
ViT-B/16 (冻结，预计算) + 语言编码器 + 状态融合 + Transformer Decoder → action chunk
"""
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

# ======================== Config ========================
DATA_PATH = "/home/jmx001/my_program/my_robot_project/logs/robomimic/full_grasp_vision.hdf5"
SAVE_DIR = "/home/jmx001/my_program/my_robot_project/logs/vla_policy"
FEAT_CACHE = os.path.join(SAVE_DIR, "vit_features.npz")
OBS_DIM = 37
ACTION_DIM = 9
IMG_SIZE = 224
CHUNK_SIZE = 16
BATCH_SIZE = 256  # 不用跑 ViT 了，可以开大
NUM_EPOCHS = 300
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INSTRUCTION = "pick up the red USB disk"


# ======================== Language Tokenizer ========================
class SimpleTokenizer:
    def __init__(self, max_len=16):
        self.max_len = max_len
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self._idx = 2

    def fit(self, text):
        for w in text.lower().split():
            if w not in self.vocab:
                self.vocab[w] = self._idx
                self._idx += 1

    def encode(self, text):
        tokens = [self.vocab.get(w, 1) for w in text.lower().split()]
        tokens = tokens[:self.max_len]
        tokens += [0] * (self.max_len - len(tokens))
        return tokens

    @property
    def vocab_size(self):
        return max(self._idx, 64)


# ======================== ViT 特征预计算 ========================
def precompute_vit_features():
    """一次性跑完所有图像的 ViT 前向传播，保存到 npz"""
    print("预计算 ViT 特征...")
    vit = models.vit_b_16(weights="IMAGENET1K_V1")
    vit.heads = nn.Identity()
    vit.eval().to(DEVICE)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_feats = []
    with h5py.File(DATA_PATH, "r") as f:
        for k in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1])):
            images = f["data"][k]["obs"]["images"][:]  # (T, 128, 128, 3)
            # 分 batch 处理避免 OOM
            batch = []
            for img in images:
                batch.append(transform(img))
                if len(batch) == 32:
                    with torch.no_grad():
                        feats = vit(torch.stack(batch).to(DEVICE))
                    all_feats.append(feats.cpu().numpy())
                    batch = []
            if batch:
                with torch.no_grad():
                    feats = vit(torch.stack(batch).to(DEVICE))
                all_feats.append(feats.cpu().numpy())
                batch = []
            print(f"  {k}: {len(images)} frames")

    all_feats = np.concatenate(all_feats, axis=0).astype(np.float32)
    np.savez_compressed(FEAT_CACHE, features=all_feats)
    print(f"ViT 特征已保存: {FEAT_CACHE} ({all_feats.shape})")
    del vit
    torch.cuda.empty_cache()
    return all_feats


# ======================== Dataset (图像版，ViT 在训练循环中跑) ========================
class VLAChunkDataset(Dataset):
    def __init__(self, path=DATA_PATH, chunk_size=16, instruction="pick up the red USB disk"):
        self.path = path
        self.chunk_size = chunk_size
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.tokenizer = SimpleTokenizer()
        self.tokenizer.fit(instruction)
        self.instruction_tokens = torch.tensor(self.tokenizer.encode(instruction), dtype=torch.long)

        self.slices = []  # (demo_key, frame_idx, global_offset)
        all_obs, all_act = [], []
        offset = 0

        with h5py.File(path, "r") as f:
            for k in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1])):
                obs = f["data"][k]["obs"]["policy"][:]
                act = f["data"][k]["actions"][:]
                n = min(len(obs), len(act))
                for i in range(n - chunk_size + 1):
                    self.slices.append((k, i, offset + i))
                all_obs.append(obs[:n])
                all_act.append(act[:n])
                offset += n

        self.obs = np.concatenate(all_obs, axis=0).astype(np.float32)
        self.act = np.concatenate(all_act, axis=0).astype(np.float32)

        self.act_min = self.act.min(axis=0)
        self.act_max = self.act.max(axis=0)
        act_range = self.act_max - self.act_min
        act_range[act_range < 1e-6] = 1.0
        self.act_norm = 2.0 * (self.act - self.act_min) / act_range - 1.0

        self.obs_min = self.obs.min(axis=0)
        self.obs_max = self.obs.max(axis=0)
        obs_range = self.obs_max - self.obs_min
        obs_range[obs_range < 1e-6] = 1.0
        self.obs_norm = 2.0 * (self.obs - self.obs_min) / obs_range - 1.0

        self._file = None
        print(f"VLA Chunk Dataset: {len(self.slices)} samples, chunk_size={chunk_size}")

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        demo_key, frame_idx, global_idx = self.slices[idx]
        f = self._get_file()
        img = f["data"][demo_key]["obs"]["images"][frame_idx]
        img_tensor = self.transform(img)
        obs = torch.from_numpy(self.obs_norm[global_idx])
        act_chunk = torch.from_numpy(self.act_norm[global_idx: global_idx + self.chunk_size])
        return img_tensor, obs, self.instruction_tokens, act_chunk


# ======================== VLA Model (ViT 最后 3 层解冻) ========================
class VLAPolicy(nn.Module):
    def __init__(self, obs_dim=37, action_dim=9, chunk_size=16,
                 vocab_size=64, lang_dim=256, max_len=16, d_model=256, vision_dim=768,
                 unfreeze_vit_layers=3):
        super().__init__()
        self.chunk_size = chunk_size
        self.vision_dim = vision_dim

        # ViT: 冻结大部分，解冻最后 N 层
        self.vit = models.vit_b_16(weights="IMAGENET1K_V1")
        self.vit.heads = nn.Identity()
        # 先冻结全部
        for p in self.vit.parameters():
            p.requires_grad = False
        # 解冻最后 N 层 encoder block + layernorm
        total_blocks = len(self.vit.encoder.layers)
        for i in range(total_blocks - unfreeze_vit_layers, total_blocks):
            for p in self.vit.encoder.layers[i].parameters():
                p.requires_grad = True
        for p in self.vit.encoder.ln.parameters():
            p.requires_grad = True

        # Language encoder
        self.lang_embed = nn.Embedding(vocab_size, 128)
        self.lang_proj = nn.Linear(max_len * 128, lang_dim)

        # Fusion
        fusion_dim = vision_dim + lang_dim + obs_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.ReLU(),
        )

        # Action chunk decoder
        self.action_queries = nn.Embedding(chunk_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward_with_features(self, vis_feat, obs, tokens):
        """接收预计算特征（兼容旧接口）"""
        B = obs.shape[0]
        lang_emb = self.lang_embed(tokens).flatten(1)
        lang_feat = self.lang_proj(lang_emb)
        fused = torch.cat([vis_feat, lang_feat, obs], dim=-1)
        memory = self.fusion_proj(fused).unsqueeze(1)
        queries = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1)
        decoded = self.transformer_decoder(tgt=queries, memory=memory)
        return self.action_head(decoded)

    def load_vit(self):
        """play_vla.py 兼容接口（ViT 已在 __init__ 中加载）"""
        pass

    def forward(self, img, obs, tokens):
        """接收图像，ViT 解冻层会更新梯度"""
        vis_feat = self.vit(img)
        return self.forward_with_features(vis_feat, obs, tokens)

    def inference(self, img, obs, tokens):
        return self.forward(img, obs, tokens)


# ======================== Training ========================
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = VLAChunkDataset(chunk_size=CHUNK_SIZE, instruction=INSTRUCTION)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    model = VLAPolicy(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
        vocab_size=dataset.tokenizer.vocab_size,
    ).to(DEVICE)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-5)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 断点续训：自动找最新的 checkpoint
    start_epoch = 1
    for e in [250, 200, 150, 100, 50]:
        p = os.path.join(SAVE_DIR, f"vla_chunk_epoch_{e}.pt")
        if os.path.exists(p) and os.path.getsize(p) > 100_000_000:  # >100MB = 含 ViT 权重
            ckpt = torch.load(p, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model"])
            start_epoch = e + 1
            for _ in range(e):
                lr_sched.step()
            print(f"Resumed from epoch {e}, starting at epoch {start_epoch}")
            break

    np.savez(os.path.join(SAVE_DIR, "norm_stats.npz"),
             act_min=dataset.act_min, act_max=dataset.act_max,
             obs_min=dataset.obs_min, obs_max=dataset.obs_max)

    train_params = sum(p.numel() for p in trainable) / 1e6
    print(f"VLA (chunk={CHUNK_SIZE}, ViT last 3 layers unfrozen) on {DEVICE}: {train_params:.1f}M trainable")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        total_loss, n = 0, 0
        for img, obs, tokens, act_chunk in loader:
            img, obs, tokens, act_chunk = img.to(DEVICE), obs.to(DEVICE), tokens.to(DEVICE), act_chunk.to(DEVICE)

            pred = model(img, obs, tokens)
            loss = F.mse_loss(pred, act_chunk)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        lr_sched.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{NUM_EPOCHS} | loss={total_loss/n:.6f} | lr={lr_sched.get_last_lr()[0]:.2e}")

        if epoch % 50 == 0 or epoch == NUM_EPOCHS:
            path = os.path.join(SAVE_DIR, f"vla_chunk_epoch_{epoch}.pt")
            torch.save({"model": model.state_dict(), "config": {
                "obs_dim": OBS_DIM, "action_dim": ACTION_DIM,
                "chunk_size": CHUNK_SIZE, "vocab_size": dataset.tokenizer.vocab_size,
                "instruction": INSTRUCTION,
            }}, path)
            print(f"  Saved: {path}")

    print("Done.")


if __name__ == "__main__":
    train()
