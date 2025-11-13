from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TrainConfig:
    h5_dir: str
    output_dir: str
    train_csv: str
    label_csv: str

    num_epochs: int = 50
    learning_rate: float = 1e-4
    batch_size: int = 64
    latent_dim: int = 64
    num_classes: int = 2
    beta_initial: float = 0.01
    beta_final: float = 0.05
    decoder_dropout: float = 0.3
    data_dim: int = 1536
    iteration_name: str = "iteration1"
    tiles_per_sample: int = 2000

    encoder_hidden_sizes: Optional[List[int]] = None
    decoder_hidden_sizes: Optional[List[int]] = None


@dataclass
class SampleConfig:
    vae_ckpt_path: str
    h5_dir: str
    label_csv: str
    train_csv: str
    pseudo_bag_output_dir: str

    num_bags: int = 10000          # pseudo WSIs per class
    num_real: int = 1000           # tiles per real sample for mean/var
    num_synth: int = 1000          # synthetic tiles per pseudobag
    num_classes: int = 2
    fold: int = 0
    iteration: str = "iteration1"  # links to VAE
    prefix: str = "sample1"

    latent_dim: int = 64
    encoder_hidden_sizes: Optional[List[int]] = None
    decoder_hidden_sizes: Optional[List[int]] = None
    data_dim: int = 1536           # feature dimension (for sanity)
