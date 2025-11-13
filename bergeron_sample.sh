#!/bin/bash

python scripts/sample_bergeron.py \
--vae_ckpt_path /path/to/checkpoint.pth \
--h5_dir /path/to/h5_files \
--train_csv ./examples/splits_0.csv \
--label_csv ./examples/meta_labels.csv \
--pseudo_bag_output_dir /path/to/pseudobag_output_directory \
--num_bags 10000 \
--num_real 1000 \
--num_synth 1000 \
--encoder_hidden_sizes 512 256 128 \
--decoder_hidden_sizes 256 512 1024 \
--latent_dim 64 \
--num_class 2 \
--fold 0 \
--iteration iteration1_fold0 \
--prefix run1
