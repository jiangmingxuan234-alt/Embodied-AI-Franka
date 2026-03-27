"""
Diffusion Policy 训练脚本
用 diffusers DDPMScheduler + 自定义 1D Conditional UNet
数据：full_grasp_delta.hdf5 (obs=37dim, action=9dim delta 关节增量)
"""
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# ======================== Config ========================
DATA_PATH = "/home/jmx001/my_program/my_robot_project/logs/robomimic/full_grasp_abs_v5.hdf5"
SAVE_DIR = "/home/jmx001/my_program/my_robot_project/logs/diffusion_policy"
OBS_DIM = 37
ACTION_DIM = 9
OBS_HORIZON = 4
PRED_HORIZON = 8
BATCH_SIZE = 256
NUM_EPOCHS = 300
LR = 1e-4
DIFFUSION_STEPS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTION_TYPE = "absolute"  # "delta" or "absolute"


# ======================== Dataset ========================
class GraspDataset(Dataset):
    def __init__(self, path, obs_horizon=2, pred_horizon=16):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.slices = []

        with h5py.File(path, "r") as f:
            all_obs, all_act = [], []
            offset = 0
            for k in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1])):
                obs = f["data"][k]["obs"]["policy"][:]
                act = f["data"][k]["actions"][:]
                n = min(len(obs), len(act))
                all_obs.append(obs[:n])
                all_act.append(act[:n])
                # 每个 demo 内的有效起始索引
                for i in range(obs_horizon - 1, n - pred_horizon + 1):
                    self.slices.append(offset + i)
                offset += n

        self.obs = np.concatenate(all_obs, axis=0).astype(np.float32)
        self.act = np.concatenate(all_act, axis=0).astype(np.float32)

        # 归一化 action 到 [-1, 1]
        self.act_min = self.act.min(axis=0)
        self.act_max = self.act.max(axis=0)
        act_range = self.act_max - self.act_min
        act_range[act_range < 1e-6] = 1.0
        self.act_norm = 2.0 * (self.act - self.act_min) / act_range - 1.0

        # 归一化 obs 到 [-1, 1]
        self.obs_min = self.obs.min(axis=0)
        self.obs_max = self.obs.max(axis=0)
        obs_range = self.obs_max - self.obs_min
        obs_range[obs_range < 1e-6] = 1.0
        self.obs_norm = 2.0 * (self.obs - self.obs_min) / obs_range - 1.0

        # 计算采样权重：对悬停区 rel_z ∈ [-0.08, -0.05] 下采样
        # rel_z = obs[:, 20] (obs 中第 20 维是 rel[2])
        self.sample_weights = self._compute_weights()

        print(f"Dataset: {len(self.slices)} samples, {len(self.obs)} total frames")

    def _compute_weights(self):
        """对悬停区帧降权，让训练分布更均匀"""
        weights = np.ones(len(self.slices), dtype=np.float32)
        for i, t in enumerate(self.slices):
            rel_z = self.obs[t, 20]  # obs dim 20 = rel[2]
            if -0.08 < rel_z < -0.05:
                weights[i] = 0.15  # 悬停区降到 15% 权重
            elif -0.10 < rel_z <= -0.08:
                weights[i] = 0.5   # 过渡区适度降权
        return weights

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        t = self.slices[idx]
        obs_seq = self.obs_norm[t - self.obs_horizon + 1: t + 1]  # (obs_horizon, obs_dim)
        act_seq = self.act_norm[t: t + self.pred_horizon]          # (pred_horizon, action_dim)
        return torch.from_numpy(obs_seq), torch.from_numpy(act_seq)


# ======================== 1D Conditional UNet ========================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = np.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.Mish(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.Mish(),
        )
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.blocks[0](x)
        h = self.blocks[1](h)
        h = self.blocks[2](h)
        # Add condition
        h = h + self.cond_proj(cond)[:, :, None]
        h = self.blocks[3](h)
        h = self.blocks[4](h)
        h = self.blocks[5](h)
        return h + self.residual(x)


class ConditionalUNet1D(nn.Module):
    def __init__(self, action_dim=9, cond_dim=128, channels=(256, 512, 1024)):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.input_proj = nn.Conv1d(action_dim, channels[0], 1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_blocks.append(ConditionalResBlock1D(channels[i], channels[i + 1], cond_dim))
            self.down_samples.append(nn.Conv1d(channels[i + 1], channels[i + 1], 3, stride=2, padding=1))

        # Middle
        self.mid_block = ConditionalResBlock1D(channels[-1], channels[-1], cond_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up_samples.append(nn.ConvTranspose1d(channels[i], channels[i], 4, stride=2, padding=1))
            self.up_blocks.append(ConditionalResBlock1D(channels[i] * 2, channels[i - 1], cond_dim))

        self.output_proj = nn.Conv1d(channels[0], action_dim, 1)

    def forward(self, x, timestep, cond):
        """
        x: (B, action_dim, pred_horizon) — noisy action sequence
        timestep: (B,) — diffusion timestep
        cond: (B, cond_dim) — observation condition
        """
        t_emb = self.time_emb(timestep)
        cond = cond + t_emb  # combine obs and time

        h = self.input_proj(x)
        skips = []
        for block, down in zip(self.down_blocks, self.down_samples):
            h = block(h, cond)
            skips.append(h)
            h = down(h)

        h = self.mid_block(h, cond)

        for up, block, skip in zip(self.up_samples, self.up_blocks, reversed(skips)):
            h = up(h)
            # Handle size mismatch from downsampling
            if h.shape[-1] != skip.shape[-1]:
                h = h[..., :skip.shape[-1]]
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond)

        return self.output_proj(h)


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim=37, action_dim=9, obs_horizon=2, pred_horizon=16, cond_dim=256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim * obs_horizon, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.noise_pred_net = ConditionalUNet1D(action_dim=action_dim, cond_dim=cond_dim)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

    def forward(self, noisy_actions, timestep, obs_seq):
        """
        noisy_actions: (B, pred_horizon, action_dim)
        timestep: (B,)
        obs_seq: (B, obs_horizon, obs_dim)
        """
        B = obs_seq.shape[0]
        obs_flat = obs_seq.reshape(B, -1)
        cond = self.obs_encoder(obs_flat)
        # UNet expects (B, C, T)
        x = noisy_actions.permute(0, 2, 1)
        noise_pred = self.noise_pred_net(x, timestep, cond)
        return noise_pred.permute(0, 2, 1)  # (B, pred_horizon, action_dim)


# ======================== Training ========================
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = GraspDataset(DATA_PATH, obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON)
    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)

    model = DiffusionPolicy(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
    ).to(DEVICE)

    scheduler = DDPMScheduler(
        num_train_timesteps=DIFFUSION_STEPS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Save normalization stats
    np.savez(os.path.join(SAVE_DIR, "norm_stats.npz"),
             act_min=dataset.act_min, act_max=dataset.act_max,
             obs_min=dataset.obs_min, obs_max=dataset.obs_max)

    print(f"Training on {DEVICE}, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        for obs_seq, act_seq in loader:
            obs_seq = obs_seq.to(DEVICE)
            act_seq = act_seq.to(DEVICE)

            noise = torch.randn_like(act_seq)
            timesteps = torch.randint(0, DIFFUSION_STEPS, (act_seq.shape[0],), device=DEVICE).long()
            noisy_actions = scheduler.add_noise(act_seq, noise, timesteps)

            noise_pred = model(noisy_actions, timesteps, obs_seq)
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        lr_sched.step()
        avg_loss = total_loss / max(n_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{NUM_EPOCHS} | loss={avg_loss:.6f} | lr={lr_sched.get_last_lr()[0]:.2e}")

        if epoch % 50 == 0 or epoch == NUM_EPOCHS:
            path = os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pt")
            torch.save({"model": model.state_dict(), "config": {
                "obs_dim": OBS_DIM, "action_dim": ACTION_DIM,
                "obs_horizon": OBS_HORIZON, "pred_horizon": PRED_HORIZON,
                "diffusion_steps": DIFFUSION_STEPS,
                "action_type": ACTION_TYPE,
            }}, path)
            print(f"  Saved: {path}")

    print("Done.")


if __name__ == "__main__":
    train()
