from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class VAEArchitecture:
    input_dim: int
    latent_dim: int
    num_classes: int
    encoder_hidden_sizes: List[int]
    decoder_hidden_sizes: List[int]
    decoder_dropout: float = 0.3

class Encoder(nn.Module):
    def __init__(self, arch: VAEArchitecture):
        super().__init__()
        self.num_classes = arch.num_classes
        h = arch.encoder_hidden_sizes

        self.fc1 = nn.Linear(arch.input_dim + arch.num_classes, h[0])
        self.fc2 = nn.Linear(h[0], h[1])
        self.fc3 = nn.Linear(h[1], h[2])
        self.fc_mu = nn.Linear(h[2], arch.latent_dim)
        self.fc_logvar = nn.Linear(h[2], arch.latent_dim)
        self.fc_class = nn.Linear(arch.latent_dim, arch.num_classes)

        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(h[0])
        self.bn2 = nn.BatchNorm1d(h[1])

    def forward(self, x, labels):
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        x = torch.cat([x, labels_onehot], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        class_pred = self.fc_class(mu)
        return mu, logvar, class_pred

    @staticmethod
    def sample_latent(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class Decoder(nn.Module):
    def __init__(self, arch: VAEArchitecture):
        super().__init__()
        self.num_classes = arch.num_classes
        h = arch.decoder_hidden_sizes

        self.fc1 = nn.Linear(arch.latent_dim + arch.num_classes, h[0])
        self.fc2 = nn.Linear(h[0], h[1])
        self.fc3 = nn.Linear(h[1], h[2])
        self.fc4 = nn.Linear(h[2], arch.input_dim)

        self.dropout = nn.Dropout(arch.decoder_dropout)
        self.bn1 = nn.BatchNorm1d(h[0])
        self.bn2 = nn.BatchNorm1d(h[1])

    def forward(self, z_cond):
        x = F.relu(self.bn1(self.fc1(z_cond)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc4(x))
        return x

class ConditionalVAE(nn.Module):
    def __init__(self, arch: VAEArchitecture):
        super().__init__()
        self.arch = arch
        self.num_classes = arch.num_classes
        self.encoder = Encoder(arch)
        self.decoder = Decoder(arch)

    def forward(self, x, labels):
        mu, logvar, class_pred = self.encoder(x, labels)
        z = self.encoder.sample_latent(mu, logvar)
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        z_cond = torch.cat([z, labels_onehot], dim=1)
        recon_x = self.decoder(z_cond)
        return recon_x, mu, logvar, class_pred

    def sample(self, num_samples: int, label: int, device: torch.device) -> torch.Tensor:
        label_tensor = torch.full((num_samples,), label, dtype=torch.long, device=device)
        labels_onehot = F.one_hot(label_tensor, num_classes=self.num_classes).float()
        z = torch.randn(num_samples, self.arch.latent_dim, device=device)
        z_cond = torch.cat([z, labels_onehot], dim=1)
        with torch.no_grad():
            samples = self.decoder(z_cond)
        return samples

def vae_loss(
    recon_x,
    x,
    mu,
    logvar,
    class_pred,
    labels,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total VAE loss = recon + beta * KL + alpha * class CE.

    Matches the original script, except that the reduction is explicit.
    """
    mse = F.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ce = F.cross_entropy(class_pred, labels)

    total = mse + beta * kld + alpha * ce
    return total, mse, kld, ce
