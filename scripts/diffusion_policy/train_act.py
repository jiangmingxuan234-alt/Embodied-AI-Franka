"""
ACT (Action Chunking with Transformers) 训练脚本
CVAE + Transformer Decoder，预测 action chunk
数据：full_grasp_abs_v5.hdf5 (obs=37dim, action=9dim 绝对关节位置)
"""
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ======================== Config ========================
DATA_PATH = "/home/jmx001/my_program/my_robot_project/logs/robomimic/full_grasp_abs_v5.hdf5"
SAVE_DIR = "/home/jmx001/my_program/my_robot_project/logs/act_policy"
OBS_DIM = 37
ACTION_DIM = 9
CHUNK_SIZE = 50
LATENT_DIM = 32
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 4
BATCH_SIZE = 256
NUM_EPOCHS = 300
LR = 1e-4
KL_WEIGHT = 10.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================== Dataset ========================
class GraspDataset(Dataset):
    def __init__(self, path, chunk_size=50):
        self.chunk_size = chunk_size
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
                for i in range(n - chunk_size + 1):
                    self.slices.append(offset + i)
                offset += n

        self.obs = np.concatenate(all_obs, axis=0).astype(np.float32)
        self.act = np.concatenate(all_act, axis=0).astype(np.float32)

        # 归一化到 [-1, 1]
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

        self.sample_weights = self._compute_weights()
        print(f"Dataset: {len(self.slices)} samples, {len(self.obs)} total frames")

    def _compute_weights(self):
        weights = np.ones(len(self.slices), dtype=np.float32)
        for i, t in enumerate(self.slices):
            rel_z = self.obs[t, 20]
            if -0.08 < rel_z < -0.05:
                weights[i] = 0.15
            elif -0.10 < rel_z <= -0.08:
                weights[i] = 0.5
        return weights

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        t = self.slices[idx]
        obs = self.obs_norm[t]                              # (obs_dim,)
        act_chunk = self.act_norm[t: t + self.chunk_size]   # (chunk_size, action_dim)
        return torch.from_numpy(obs), torch.from_numpy(act_chunk)


# ======================== ACT Policy ========================
class ACTPolicy(nn.Module):
    def __init__(self, obs_dim=37, action_dim=9, chunk_size=50,
                 latent_dim=32, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        # CVAE encoder (training only)
        enc_input_dim = obs_dim + chunk_size * action_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_input_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

        # Obs encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Latent projection
        self.latent_proj = nn.Linear(latent_dim, d_model)

        # Transformer decoder
        self.action_queries = nn.Embedding(chunk_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Action head
        self.action_head = nn.Linear(d_model, action_dim)

    def encode(self, obs, actions):
        """CVAE encoder: obs (B, obs_dim), actions (B, chunk_size, action_dim) -> mu, logvar"""
        x = torch.cat([obs, actions.flatten(1)], dim=1)
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, obs, z):
        """Transformer decoder: obs (B, obs_dim), z (B, latent_dim) -> (B, chunk_size, action_dim)"""
        B = obs.shape[0]
        obs_token = self.obs_encoder(obs).unsqueeze(1)      # (B, 1, d_model)
        z_token = self.latent_proj(z).unsqueeze(1)           # (B, 1, d_model)
        memory = torch.cat([obs_token, z_token], dim=1)      # (B, 2, d_model)

        queries = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, chunk_size, d_model)
        decoded = self.transformer_decoder(tgt=queries, memory=memory)       # (B, chunk_size, d_model)
        return self.action_head(decoded)                                     # (B, chunk_size, action_dim)

    def forward(self, obs, actions):
        """Training forward: returns (pred_actions, mu, logvar)"""
        mu, logvar = self.encode(obs, actions)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        pred_actions = self.decode(obs, z)
        return pred_actions, mu, logvar

    def inference(self, obs):
        """Inference: z=0, returns (B, chunk_size, action_dim)"""
        B = obs.shape[0]
        z = torch.zeros(B, self.latent_dim, device=obs.device)
        return self.decode(obs, z)


# ======================== Training ========================
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = GraspDataset(DATA_PATH, chunk_size=CHUNK_SIZE)
    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights, num_samples=len(dataset), replacement=True,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)

    model = ACTPolicy(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE,
        latent_dim=LATENT_DIM, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    np.savez(os.path.join(SAVE_DIR, "norm_stats.npz"),
             act_min=dataset.act_min, act_max=dataset.act_max,
             obs_min=dataset.obs_min, obs_max=dataset.obs_max)

    print(f"Training ACT on {DEVICE}, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_recon, total_kl, n_batches = 0, 0, 0
        for obs, act_chunk in loader:
            obs, act_chunk = obs.to(DEVICE), act_chunk.to(DEVICE)

            pred_actions, mu, logvar = model(obs, act_chunk)
            recon_loss = nn.functional.mse_loss(pred_actions, act_chunk)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + KL_WEIGHT * kl_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        lr_sched.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{NUM_EPOCHS} | recon={total_recon/n_batches:.6f} "
                  f"kl={total_kl/n_batches:.6f} | lr={lr_sched.get_last_lr()[0]:.2e}")

        if epoch % 50 == 0 or epoch == NUM_EPOCHS:
            path = os.path.join(SAVE_DIR, f"act_epoch_{epoch}.pt")
            torch.save({"model": model.state_dict(), "config": {
                "obs_dim": OBS_DIM, "action_dim": ACTION_DIM,
                "chunk_size": CHUNK_SIZE, "latent_dim": LATENT_DIM,
                "d_model": D_MODEL, "nhead": NHEAD, "num_layers": NUM_LAYERS,
            }}, path)
            print(f"  Saved: {path}")

    print("Done.")


if __name__ == "__main__":
    train()
