"""VQ-VAE-style motion tokenizer: encode a telemetry window, quantize, decode."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden, latent_dim)

    def forward(self, x):  # x: (B, T, C)
        h = self.net(x.transpose(1, 2)).squeeze(-1)  # (B, hidden)
        return self.proj(h)  # (B, latent_dim)


class _Decoder(nn.Module):
    def __init__(self, out_channels: int, latent_dim: int, window: int, hidden: int):
        super().__init__()
        self.window = window
        self.hidden = hidden
        self.fc = nn.Linear(latent_dim, hidden * window)
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z):  # z: (B, latent_dim)
        h = self.fc(z).view(-1, self.hidden, self.window)
        return self.net(h).transpose(1, 2)  # (B, T, C)


class _VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, latent_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, latent_dim)
        self.codebook.weight.data.uniform_(-1 / num_codes, 1 / num_codes)
        self.commitment_cost = commitment_cost

    def forward(self, z):  # z: (B, D)
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)
        z_q = self.codebook(indices)

        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        z_q = z + (z_q - z).detach()  # straight-through estimator
        return z_q, indices, vq_loss


class MotionTokenizer(nn.Module):
    def __init__(self, in_channels: int, window: int, latent_dim: int = 32, num_codes: int = 512, hidden: int = 64):
        super().__init__()
        self.encoder = _Encoder(in_channels, latent_dim, hidden)
        self.quantizer = _VectorQuantizer(num_codes, latent_dim)
        self.decoder = _Decoder(in_channels, latent_dim, window, hidden)

    def encode_tokens(self, x):
        z = self.encoder(x)
        z_q, indices, _ = self.quantizer(z)
        return z_q, indices

    def forward(self, x):
        z = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z)
        recon = self.decoder(z_q)
        recon_loss = F.mse_loss(recon, x)
        return {
            "recon": recon,
            "indices": indices,
            "loss": recon_loss + vq_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
        }
