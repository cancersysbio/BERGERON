#!/usr/bin/env python
import argparse
import logging

from bergeron.sampling import SampleConfig, generate_pseudo_bags

def parse_args():
    parser = argparse.ArgumentParser(description="Generate BERGERON pseudo-bags from a trained VAE.")
    parser.add_argument("--vae_ckpt_path", type=str, required=True, help="Path to VAE checkpoint (.pth).")
    parser.add_argument("--h5_dir", type=str, required=True, help="Path to raw H5 data directory.")
    parser.add_argument("--label_csv", type=str, required=True, help="Path to label dictionary CSV (will be appended).")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV used for VAE training.")
    parser.add_argument("--pseudo_bag_output_dir", type=str, required=True, help="Directory for pseudo-bag outputs.")
    parser.add_argument("--num_bags", type=int, default=10000, help="Number of pseudo-WSIs to generate per class.")
    parser.add_argument("--num_real", type=int, default=1000, help="Number of real tiles per WSI to estimate latent mean/var.")
    parser.add_argument("--num_synth", type=int, default=1000, help="Number of synthetic tiles per pseudo-WSI.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (used in output naming).")
    parser.add_argument("--iteration", type=str, required=True, help="Iteration name (should match VAE training iteration).")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix for this sampling run.")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension size (must match VAE).")
    parser.add_argument("--data_dim", type=int, default=1536, help="Input/output feature dimension (must match VAE).")
    parser.add_argument("--encoder_hidden_sizes", nargs="+", type=int, required=True, help="Encoder hidden sizes (must match VAE).")
    parser.add_argument("--decoder_hidden_sizes", nargs="+", type=int, required=True, help="Decoder hidden sizes (must match VAE).")
    return parser.parse_args()

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    )

    cfg = SampleConfig(
        vae_ckpt_path=args.vae_ckpt_path,
        h5_dir=args.h5_dir,
        label_csv=args.label_csv,
        train_csv=args.train_csv,
        pseudo_bag_output_dir=args.pseudo_bag_output_dir,
        num_bags=args.num_bags,
        num_real=args.num_real,
        num_synth=args.num_synth,
        num_classes=args.num_classes,
        fold=args.fold,
        iteration=args.iteration,
        prefix=args.prefix,
        latent_dim=args.latent_dim,
        data_dim=args.data_dim,
        encoder_hidden_sizes=args.encoder_hidden_sizes,
        decoder_hidden_sizes=args.decoder_hidden_sizes,
    )

    generate_pseudo_bags(cfg)

if __name__ == "__main__":
    main()
