#!/usr/bin/env python
import argparse
import logging

from bergeron.config import TrainConfig
from bergeron.train import train

def parse_args():
    parser = argparse.ArgumentParser(description="Train the BERGERON Conditional VAE on H5 datasets.")
    parser.add_argument("--h5_dir", type=str, required=True, help="Path to the H5 directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to a file containing a list of H5 files.")
    parser.add_argument("--label_csv", type=str, required=True, help="Path to a file containing all sample labels.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--latent_dim", type=int, default=64, help="Size of the latent dimension.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes.")
    parser.add_argument("--beta_initial", type=float, default=0.01, help="Initial beta annealing value.")
    parser.add_argument("--beta_final", type=float, default=0.05, help="Final beta annealing value.")
    parser.add_argument("--decoder_dropout", type=float, default=0.3, help="Decoder dropout value.")
    parser.add_argument("--data_dim", type=int, default=1536, help="Dimension of input data.")
    parser.add_argument("--iteration_name", type=str, default="iteration1", help="Name of the training iteration.")
    parser.add_argument("--tiles_per_sample", type=int, default=2000, help="Number of tiles per sample to use.")
    parser.add_argument(
        "--encoder_hidden_sizes",
        nargs="+",
        type=int,
        required=True,
        help="Size of the hidden dimensions for the encoder.",
    )
    parser.add_argument(
        "--decoder_hidden_sizes",
        nargs="+",
        type=int,
        required=True,
        help="Size of the hidden dimensions for the decoder.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    )

    cfg = TrainConfig(
        h5_dir=args.h5_dir,
        output_dir=args.output_dir,
        train_csv=args.train_csv,
        label_csv=args.label_csv,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        num_classes=args.num_classes,
        beta_initial=args.beta_initial,
        beta_final=args.beta_final,
        decoder_dropout=args.decoder_dropout,
        data_dim=args.data_dim,
        iteration_name=args.iteration_name,
        tiles_per_sample=args.tiles_per_sample,
        encoder_hidden_sizes=args.encoder_hidden_sizes,
        decoder_hidden_sizes=args.decoder_hidden_sizes,
    )

    train(cfg)

if __name__ == "__main__":
    main()
