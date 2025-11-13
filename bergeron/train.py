from __future__ import annotations

import os
import logging
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import (
    load_label_dict,
    parse_train_csv,
    compute_global_min_max,
    PerSampleSubsampledDataset,
)
from .models import VAEArchitecture, ConditionalVAE, vae_loss
from .pca_viz import precompute_rescaled_real_pca, plot_rescaled_pca_with_generated
from .utils import set_seed, get_device, ensure_dir

logger = logging.getLogger(__name__)

def _prepare_output_dirs(cfg: TrainConfig) -> Tuple[str, str]:
    pc_dir = os.path.join(cfg.output_dir, "PC_plots", cfg.iteration_name)
    model_dir = os.path.join(cfg.output_dir, "saved_models", cfg.iteration_name)
    ensure_dir(pc_dir)
    ensure_dir(model_dir)
    return pc_dir, model_dir

def train(cfg: TrainConfig) -> None:
    """Main BERGERON training loop."""
    set_seed(42)
    device = get_device()

    logger.info("=== BERGERON Conditional VAE Training ===")
    logger.info(f"Iteration name: {cfg.iteration_name}")
    logger.info(f"H5 directory: {cfg.h5_dir}")
    logger.info(f"Train CSV: {cfg.train_csv}")
    logger.info(f"Label CSV: {cfg.label_csv}")
    logger.info(f"Output dir: {cfg.output_dir}")
    logger.info(f"Num epochs: {cfg.num_epochs}, batch size: {cfg.batch_size}, LR: {cfg.learning_rate}")
    logger.info(f"Latent dim: {cfg.latent_dim}, num classes: {cfg.num_classes}")
    logger.info(f"Beta schedule: {cfg.beta_initial} -> {cfg.beta_final}")
    logger.info(f"Decoder dropout: {cfg.decoder_dropout}")
    logger.info(f"Tiles per sample per epoch: {cfg.tiles_per_sample}")

    pc_dir, model_dir = _prepare_output_dirs(cfg)

    # ------------------------------------------------------------------
    # Labels + file lists
    # ------------------------------------------------------------------
    label_dict = load_label_dict(cfg.label_csv)
    to_train, to_val = parse_train_csv(cfg.train_csv)
    logger.info(f"Using {len(to_train)} training files, {len(to_val)} validation files (val not used yet).")

    # ------------------------------------------------------------------
    # Global min/max for scaling
    # ------------------------------------------------------------------
    gmin, gmax = compute_global_min_max(cfg.h5_dir, to_train)
    logger.info(f"Global min: {gmin}, Global max: {gmax}")

    # Save global min/max next to model checkpoints for sampling
    import numpy as _np
    _np.savez(os.path.join(model_dir, "global_minmax.npz"), global_min=gmin, global_max=gmax)

    # ------------------------------------------------------------------
    # PCA precomputation (on a subsampled per-sample dataset)
    # ------------------------------------------------------------------
    pre_pca_dataset = PerSampleSubsampledDataset(
        h5_dir=cfg.h5_dir,
        file_list=to_train,
        label_dict=label_dict,
        global_min=gmin,
        global_max=gmax,
        vector_size=cfg.data_dim,
        samples_per_file=cfg.tiles_per_sample,
    )

    pre_pca, pcs_real0, pcs_real1 = precompute_rescaled_real_pca(
        dataset=pre_pca_dataset,
        sample_size=10000,
        global_min=gmin,
        global_max=gmax,
        save_dir=pc_dir,
    )

    # ------------------------------------------------------------------
    # Model + optimizer
    # ------------------------------------------------------------------
    if cfg.encoder_hidden_sizes is None:
        raise ValueError("encoder_hidden_sizes must be provided in the config.")
    if cfg.decoder_hidden_sizes is None:
        raise ValueError("decoder_hidden_sizes must be provided in the config.")

    arch = VAEArchitecture(
        input_dim=cfg.data_dim,
        latent_dim=cfg.latent_dim,
        num_classes=cfg.num_classes,
        encoder_hidden_sizes=cfg.encoder_hidden_sizes,
        decoder_hidden_sizes=cfg.decoder_hidden_sizes,
        decoder_dropout=cfg.decoder_dropout,
    )
    vae = ConditionalVAE(arch).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.learning_rate)

    beta_increment = (cfg.beta_final - cfg.beta_initial) / max(cfg.num_epochs - 1, 1)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(cfg.num_epochs):
        beta = cfg.beta_initial + beta_increment * epoch

        # Fresh subsampled dataset each epoch (as in original script)
        epoch_dataset = PerSampleSubsampledDataset(
            h5_dir=cfg.h5_dir,
            file_list=to_train,
            label_dict=label_dict,
            global_min=gmin,
            global_max=gmax,
            vector_size=cfg.data_dim,
            samples_per_file=cfg.tiles_per_sample,
        )
        loader = DataLoader(
            epoch_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        vae.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_class = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            recon, mu, logvar, class_pred = vae(x, y)
            loss, recon_loss, kl_loss, class_loss = vae_loss(
                recon, x, mu, logvar, class_pred, y, beta=beta, alpha=0.0
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_class += class_loss.item()

        n = len(loader.dataset)
        avg_loss = total_loss / n
        avg_recon = total_recon / n
        avg_kl = total_kl / n
        avg_class = total_class / n

        logger.info(
            f"Epoch {epoch+1}/{cfg.num_epochs} "
            f"Loss={avg_loss:.4f} Recon={avg_recon:.4f} KL={avg_kl:.4f} "
            f"Class={avg_class:.4f} Beta={beta:.4f}"
        )

        if epoch in (0, cfg.num_epochs - 1):
            plot_rescaled_pca_with_generated(
                pca=pre_pca,
                pcs_real_0=pcs_real0,
                pcs_real_1=pcs_real1,
                vae=vae,
                epoch=epoch,
                save_dir=pc_dir,
                sample_size=10000,
                global_min=gmin,
                global_max=gmax,
                device=device,
            )

            out_path = os.path.join(model_dir, f"vae_epoch_{epoch}.pth")
            torch.save(vae.state_dict(), out_path)
            logger.info(f"Saved model checkpoint: {out_path}")
