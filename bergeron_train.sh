#!/bin/bash

python scripts/train_bergeron.py \
--h5_dir /path/to/h5_files \
--output_dir /path/to/output_directory \
--train_csv ./examples/splits_0.csv \
--label_csv ./examples/meta_labels.csv \
--num_epochs 50 \
--learning_rate 1e-4 \
--batch_size 64 \
--latent_dim 64 \
--num_classes 2 \
--beta_initial 0.01 \
--beta_final 0.05 \
--decoder_dropout 0.3 \
--data_dim 1536 \
--iteration_name iteration1_fold0 \
--tiles_per_sample 2000 \
--encoder_hidden_sizes 512 256 128 \
--decoder_hidden_sizes 256 512 1024
