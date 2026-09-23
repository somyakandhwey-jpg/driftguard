"""
models/diffusion.py -- Phase 5: graph diffusion model for label-free fraud
detection.

Core idea: train a denoising model to reverse small amounts of Gaussian noise
added to NORMAL transaction node features (using only unlabeled/"licit"-ish
data -- no fraud labels required). A node's transaction pattern is graph-
contextualized via message passing before denoising, so the model learns
"what a normal transaction looks like given its neighborhood."

At inference time, a node whose true features are hard to reconstruct from a
noised version (i.e. don't fit the learned "normal" manifold) gets a high
denoising error -> high fraud score. This gives us the unsupervised fraud
detector described in the project pitch, and it slots directly into the
Phase 4 continual-learning loop (re-estimate the noise schedule / fine-tune
the denoiser per time chunk with EWC protecting old chunks).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.999)


class GraphDenoiser(nn.Module):
    """Predicts the noise added to node features, conditioned on graph
    structure and the diffusion timestep."""

    def __init__(self, feat_dim, hidden_dim=128, time_emb_dim=32, num_layers=3):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.in_proj = nn.Linear(feat_dim + time_emb_dim, hidden_dim)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden_dim, feat_dim)

    def forward(self, x_noisy, edge_index, t):
        # t: [N] long tensor of diffusion step per node (same value broadcast)
        t_norm = (t.float() / t.max().clamp(min=1)).unsqueeze(-1)
        t_emb = self.time_mlp(t_norm)
        h = torch.cat([x_noisy, t_emb], dim=-1)
        h = F.silu(self.in_proj(h))
        for conv in self.convs:
            h = F.silu(conv(h, edge_index))
        return self.out_proj(h)  # predicted noise, same shape as x


class GraphDiffusionModel:
    """Wraps a GraphDenoiser with a forward diffusion process and gives a
    train_step() and an anomaly_score() -- no fraud labels used in either."""

    def __init__(self, feat_dim, timesteps=200, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        self.betas = cosine_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.model = GraphDenoiser(feat_dim).to(device)

    def q_sample(self, x0, t, noise=None):
        """Forward process: add noise to clean features x0 at step t (per-node t)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.alphas_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_ac = (1 - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)
        return sqrt_ac * x0 + sqrt_one_minus_ac * noise, noise

    def train_step(self, x0, edge_index, normal_mask, optimizer, ewc=None):
        """One denoising-loss training step, using only rows where
        normal_mask is True (i.e. no fraud labels, just 'assumed normal').
        If `ewc` (models.continual.EWC) is provided, its penalty is added to
        the same loss/backward pass so old time-chunks aren't forgotten."""
        self.model.train()
        n = x0.size(0)
        t = torch.randint(0, self.timesteps, (n,), device=self.device)
        x_noisy, noise = self.q_sample(x0, t)
        pred_noise = self.model(x_noisy, edge_index, t)

        loss = F.mse_loss(pred_noise[normal_mask], noise[normal_mask])
        if ewc is not None:
            loss = loss + ewc.penalty()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def anomaly_score(self, x0, edge_index, t_probe=50, n_repeats=4):
        """Fraud score = average denoising error at a fixed, moderate noise
        level. Higher error -> feature pattern doesn't fit the learned
        'normal' manifold -> more likely fraud. Purely unsupervised."""
        self.model.eval()
        n = x0.size(0)
        errors = torch.zeros(n, device=self.device)
        t = torch.full((n,), t_probe, device=self.device, dtype=torch.long)
        for _ in range(n_repeats):
            x_noisy, noise = self.q_sample(x0, t)
            pred_noise = self.model(x_noisy, edge_index, t)
            errors += F.mse_loss(pred_noise, noise, reduction="none").mean(dim=-1)
        return (errors / n_repeats).cpu()

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)
