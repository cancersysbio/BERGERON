from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from typing import List

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from .data import load_label_dict, parse_train_csv
from .models import VAEArchitecture, ConditionalVAE
from .utils import set_seed, get_device, ensure_dir


@dataclass
class SampleConfig:
    """Configuration for BERGERON pseudo-bag generation."""

    vae_ckpt_path: str
    h5_dir: str
    label_csv: str
    train_csv: str
    pseudo_bag_output_dir: str

    # Sampling hyperparameters
    num_bags: int = 10000          # pseudo-WSIs per class
    num_real: int = 1000           # real tiles per real WSI used to estimate latent mean/var
    num_synth: int = 1000          # synthetic tiles per pseudo-WSI
    num_classes: int = 2
    fold: int = 0
    iteration: str = "iteration1"
    prefix: str = "iter0"

    # VAE architecture (must match training)
    latent_dim: int = 64
    data_dim: int = 1536
    encoder_hidden_sizes: List[int] | None = None
    decoder_hidden_sizes: List[int] | None = None


def generate_pseudo_bags(cfg: SampleConfig) -> None:
    """Generate synthetic pseudo-bags using a trained BERGERON VAE.

    Workflow (per class):

      1. Load list of training samples and label dictionary.
      2. Filter samples belonging to the target class.
      3. For each pseudo-bag:
         a. Choose a real WSI from that class.
         b. Sample `num_real` tiles from it and normalize to [-1, 1].
         c. Encode them with the VAE encoder to get latent mus.
         d. Compute latent mean/variance across tiles and sample
            `num_synth` latent vectors from N(mean, var).
         e. Decode these into synthetic tiles (normalized space),
            then rescale to original data range using global min/max.
         f. Save each pseudo-bag as an HDF5 (`features`) and a Torch
            tensor, and append an entry to the label CSV.
    """
    set_seed(42)
    device = get_device()

    # ------------------------------------------------------------------
    # Global min/max from training
    # ------------------------------------------------------------------
    ckpt_dir = os.path.dirname(cfg.vae_ckpt_path)
    minmax_path = os.path.join(ckpt_dir, "global_minmax.npz")
    if not os.path.exists(minmax_path):
        raise FileNotFoundError(f"Global min/max file not found at {minmax_path}")
    minmax = np.load(minmax_path)
    global_min = float(minmax["global_min"])
    global_max = float(minmax["global_max"])
    print(f"Loaded global_min = {global_min}, global_max = {global_max}", flush=True)

    # ------------------------------------------------------------------
    # Labels and sample list
    # ------------------------------------------------------------------
    label_dict = load_label_dict(cfg.label_csv)  # {case_id: int_label}
    sample_list, _ = parse_train_csv(cfg.train_csv)  # train files with .h5 suffix

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if cfg.encoder_hidden_sizes is None or cfg.decoder_hidden_sizes is None:
        raise ValueError("encoder_hidden_sizes and decoder_hidden_sizes must be provided for sampling.")

    arch = VAEArchitecture(
        input_dim=cfg.data_dim,
        latent_dim=cfg.latent_dim,
        num_classes=cfg.num_classes,
        encoder_hidden_sizes=cfg.encoder_hidden_sizes,
        decoder_hidden_sizes=cfg.decoder_hidden_sizes,
    )
    vae = ConditionalVAE(arch).to(device)
    state = torch.load(cfg.vae_ckpt_path, map_location=device)
    vae.load_state_dict(state)
    vae.eval()

    # Output directories
    h5_out_dir = ensure_dir(os.path.join(cfg.pseudo_bag_output_dir, "h5_files"))
    pt_out_dir = ensure_dir(os.path.join(cfg.pseudo_bag_output_dir, "pt_files"))

    # ------------------------------------------------------------------
    # Per-class pseudo-bag generation
    # ------------------------------------------------------------------
    for class_to_gen in range(cfg.num_classes):
        print(f"=== Generating pseudo bags for class {class_to_gen} ===")

        # Filter for class-specific samples
        class_samples = [
            s for s in sample_list
            if label_dict.get(s.replace(".h5", ""), None) == class_to_gen
        ]
        if not class_samples:
            print(f"  Warning: no samples found for class {class_to_gen}, skipping.")
            continue

        selected_samples = np.random.choice(class_samples, size=cfg.num_bags, replace=True)
        print(f"  Using {len(class_samples)} real WSIs for class {class_to_gen}")
        print(f"  Will generate {cfg.num_bags} pseudo-bags for this class.")

        # Generate pseudo bags
        for i, fname in enumerate(selected_samples):
            h5_path = os.path.join(cfg.h5_dir, fname)
            with h5py.File(h5_path, "r") as f:
                feats = f["features"][:]
                n_tiles = len(feats)
                idx = np.random.choice(n_tiles, min(cfg.num_real, n_tiles), replace=False)
                selected = feats[idx]

            # Normalize using global min/max -> [-1, 1]
            feats_norm = 2 * (selected - global_min) / (global_max - global_min) - 1
            x = torch.tensor(feats_norm, dtype=torch.float32, device=device)
            labels = torch.full((x.size(0),), class_to_gen, dtype=torch.long, device=device)

            # Encode and compute latent mean/variance
            with torch.no_grad():
                mu, logvar, _ = vae.encoder(x, labels)
            mu_np = mu.cpu().numpy()
            latent_mean = mu_np.mean(axis=0)
            latent_var = mu_np.var(axis=0)

            # Sample from N(mean, var) in latent space
            z_samples = np.random.normal(
                latent_mean,
                np.sqrt(latent_var + 1e-8),
                size=(cfg.num_synth, cfg.latent_dim),
            )
            z_samples = torch.tensor(z_samples, dtype=torch.float32, device=device)

            # Decode from sampled latent vectors
            label_tensor = torch.full((cfg.num_synth,), class_to_gen, dtype=torch.long, device=device)
            z_cond = torch.cat(
                [z_samples, F.one_hot(label_tensor, cfg.num_classes).float()],
                dim=1,
            )
            with torch.no_grad():
                decoded = vae.decoder(z_cond).cpu().numpy()

            # Rescale back to original data range
            decoded_rescaled = 0.5 * (decoded + 1) * (global_max - global_min) + global_min

            # Save pseudo-bag
            base_name = f"{cfg.iteration}_fold{cfg.fold}_c{class_to_gen}_{cfg.prefix}_pseudowsi_{i}"
            save_path_h5 = os.path.join(h5_out_dir, f"{base_name}.h5")
            save_path_pt = os.path.join(pt_out_dir, f"{base_name}.pt")

            with h5py.File(save_path_h5, "w") as f_out:
                f_out.create_dataset("features", data=decoded_rescaled.astype(np.float32))

            torch.save(torch.tensor(decoded_rescaled.astype(np.float32)), save_path_pt)

        # Append to label CSV for this class
        with open(cfg.label_csv, "a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            for i in range(cfg.num_bags):
                name = f"{cfg.iteration}_fold{cfg.fold}_c{class_to_gen}_{cfg.prefix}_pseudowsi_{i}"
                csv_writer.writerow([i, name, "synth", class_to_gen])

    print(f"Done! Saved pseudo bags to: {cfg.pseudo_bag_output_dir}")
