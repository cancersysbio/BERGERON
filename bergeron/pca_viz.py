from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from .models import ConditionalVAE

def precompute_rescaled_real_pca(
    dataset,
    sample_size: int,
    global_min: float,
    global_max: float,
    save_dir: str | None = None,
) -> Tuple[PCA, np.ndarray, np.ndarray]:
    """Fit PCA on a balanced sample of real tiles in original data space.

    dataset is expected to have .data (tensor) and .labels (tensor),
    where data is in [-1, 1] and is rescaled back to [global_min, global_max].
    """
    print("Precomputing PCA on unnormalized real data...")
    data = dataset.data
    labels = dataset.labels

    print("precompute_rescaled_real_pca max(data):", data.max().item())
    print("precompute_rescaled_real_pca min(data):", data.min().item())

    # Sample sample_size per class (as in original script; classes 0 and 1)
    c0_idx = (labels == 0).nonzero(as_tuple=True)[0]
    c1_idx = (labels == 1).nonzero(as_tuple=True)[0]

    sample_size0 = min(sample_size, len(c0_idx))
    sample_size1 = min(sample_size, len(c1_idx))

    c0_sample = c0_idx[torch.randperm(len(c0_idx))[:sample_size0]]
    c1_sample = c1_idx[torch.randperm(len(c1_idx))[:sample_size1]]

    real_data = torch.cat([data[c0_sample], data[c1_sample]])
    real_data_rescaled = 0.5 * (real_data + 1) * (global_max - global_min) + global_min

    print("precompute_rescaled_real_pca max(rescaled):", real_data_rescaled.max().item())
    print("precompute_rescaled_real_pca min(rescaled):", real_data_rescaled.min().item())

    pca = PCA(n_components=2)
    pca.fit(real_data_rescaled)

    pcs_real = pca.transform(real_data_rescaled)
    pcs_real_0 = pcs_real[:sample_size0]
    pcs_real_1 = pcs_real[sample_size0:]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.scatter(pcs_real_0[:, 0], pcs_real_0[:, 1], c="green", label="Real Class 0", s=3)
        plt.scatter(pcs_real_1[:, 0], pcs_real_1[:, 1], c="orange", label="Real Class 1", s=3)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Precomputed PCA: Real Data Only (Original Scale)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "pca_precomputed_rescaled_real_only.png"))
        plt.close()

    return pca, pcs_real_0, pcs_real_1

def plot_rescaled_pca_with_generated(
    pca: PCA,
    pcs_real_0: np.ndarray,
    pcs_real_1: np.ndarray,
    vae: ConditionalVAE,
    epoch: int,
    save_dir: str,
    sample_size: int,
    global_min: float,
    global_max: float,
    device: torch.device,
) -> None:
    """Use precomputed PCA to project newly generated samples."""
    print(f"Plotting PCA with rescaled generated data for epoch {epoch}")

    vae.eval()
    with torch.no_grad():
        gen_0 = vae.sample(sample_size, label=0, device=device).cpu()
        gen_1 = vae.sample(sample_size, label=1, device=device).cpu()

    print("plot_rescaled_pca_with_generated max(gen_0):", gen_0.max().item())
    print("plot_rescaled_pca_with_generated min(gen_0):", gen_0.min().item())

    def rescale(data: torch.Tensor) -> torch.Tensor:
        return 0.5 * (data + 1) * (global_max - global_min) + global_min

    gen_0_rescaled = rescale(gen_0)
    gen_1_rescaled = rescale(gen_1)

    print("plot_rescaled_pca_with_generated max(rescaled gen_0):", gen_0_rescaled.max().item())
    print("plot_rescaled_pca_with_generated min(rescaled gen_0):", gen_0_rescaled.min().item())

    pcs_gen0 = pca.transform(gen_0_rescaled.numpy())
    pcs_gen1 = pca.transform(gen_1_rescaled.numpy())

    print(f"  Real PCS Class 0: min={pcs_real_0.min(axis=0)}, max={pcs_real_0.max(axis=0)}")
    print(f"  Real PCS Class 1: min={pcs_real_1.min(axis=0)}, max={pcs_real_1.max(axis=0)}")
    print(f"  Gen (rescaled) Class 0: min={gen_0_rescaled.min().item()}, max={gen_0_rescaled.max().item()}")
    print(f"  Gen (rescaled) Class 1: min={gen_1_rescaled.min().item()}, max={gen_1_rescaled.max().item()}")
    print(f"  Gen PCS Class 0: min={pcs_gen0.min(axis=0)}, max={pcs_gen0.max(axis=0)}")
    print(f"  Gen PCS Class 1: min={pcs_gen1.min(axis=0)}, max={pcs_gen1.max(axis=0)}")

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(pcs_real_0[:, 0], pcs_real_0[:, 1], c="green", label="Real Class 0", s=1)
    plt.scatter(pcs_real_1[:, 0], pcs_real_1[:, 1], c="orange", label="Real Class 1", s=1)
    plt.scatter(pcs_gen0[:, 0], pcs_gen0[:, 1], c="red", label="Gen Class 0 (rescaled)", s=1)
    plt.scatter(pcs_gen1[:, 0], pcs_gen1[:, 1], c="blue", label="Gen Class 1 (rescaled)", s=1)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA (Real Rescaled + Gen) - Epoch {epoch}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pca_rescaled_gen_epoch_{epoch}.png"))
    plt.close()
