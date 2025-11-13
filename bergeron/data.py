from __future__ import annotations

import os
import csv
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

def load_label_dict(label_csv: str) -> Dict[str, int]:
    """Load mapping from base H5 ID (e.g. TCGA-XX) to integer label.

    Expects a CSV with header containing "case_id" in the first column and
    the ID + label in columns 2 and 4 (as in the original BERGERON script).
    """
    label_dict: Dict[str, int] = {}
    with open(label_csv, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            if not line:
                continue
            if line[0] == "case_id":
                continue
            # original code: label_dict[line[1]] = int(line[3])
            key = line[1]
            label = int(line[3])
            label_dict[key] = label
    return label_dict

def parse_train_csv(train_csv_path: str) -> Tuple[List[str], List[str]]:
    """Return file lists (without .h5 extension added) for train and val.

    Original convention:
      - CSV has header "train" in first column
      - Column 0: train ID (no .h5 extension)
      - Column 1: val ID (no .h5 extension)
    """
    to_train: List[str] = []
    to_val: List[str] = []
    with open(train_csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0] == "train":
                continue
            if row[0]:
                to_train.append(row[0] + ".h5")
            if len(row) > 1 and row[1]:
                to_val.append(row[1] + ".h5")
    return to_train, to_val

def compute_global_min_max(h5_dir: str, file_list: List[str]) -> Tuple[float, float]:
    """Compute global min and max across all 'features' datasets."""
    global_min = float("inf")
    global_max = float("-inf")
    for fname in file_list:
        path = os.path.join(h5_dir, fname)
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            feats = f["features"][:]
            global_min = min(global_min, float(np.min(feats)))
            global_max = max(global_max, float(np.max(feats)))
    return global_min, global_max

class H5Dataset(Dataset):
    """Indexable dataset over all tiles in one or more H5 files.

    Differs from the per-sample subsampled dataset below; this one is mainly
    used in the original code to compute the global min and max.
    """

    def __init__(
        self,
        h5_dir: str,
        file_list: List[str],
        label_dict: Dict[str, int],
        data_dim: int,
        global_min: float,
        global_max: float,
    ) -> None:
        self.h5_dir = h5_dir
        self.file_list = file_list
        self.label_dict = label_dict
        self.data_dim = data_dim
        self.global_min = global_min
        self.global_max = global_max

        self.indices = self._create_indices()

    def _create_indices(self):
        indices = []
        for file_idx, fname in enumerate(self.file_list):
            path = os.path.join(self.h5_dir, fname)
            if not os.path.exists(path):
                continue
            with h5py.File(path, "r") as f:
                n = f["features"].shape[0]
                indices.extend([(file_idx, i) for i in range(n)])
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, feature_idx = self.indices[idx]
        fname = self.file_list[file_idx]
        path = os.path.join(self.h5_dir, fname)
        with h5py.File(path, "r") as f:
            feat = f["features"][feature_idx]

        # normalize to [-1, 1]
        feat = 2 * (feat - self.global_min) / (self.global_max - self.global_min) - 1

        base_id = fname.replace(".h5", "")
        label = int(self.label_dict.get(base_id, 0))

        x = torch.tensor(feat, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

class PerSampleSubsampledDataset(Dataset):
    """Dataset that subsamples a fixed number of tiles per H5 sample.

    This mirrors the original PerSampleSubsampledDataset used for training.
    """

    def __init__(
        self,
        h5_dir: str,
        file_list: List[str],
        label_dict: Dict[str, int],
        global_min: float,
        global_max: float,
        vector_size: int = 1536,
        samples_per_file: int = 1000,
    ) -> None:
        import random

        self.h5_dir = h5_dir
        self.file_list = file_list
        self.label_dict = label_dict
        self.vector_size = vector_size
        self.samples_per_file = samples_per_file
        self.global_min = global_min
        self.global_max = global_max

        self.data = []
        self.labels = []

        # Build in memory (same behavior as original script)
        print(f"Subsampling {self.samples_per_file} tiles per sample from {len(self.file_list)} H5 files...")
        for fname in self.file_list:
            h5_path = os.path.join(self.h5_dir, fname)
            if not os.path.exists(h5_path):
                continue
            try:
                with h5py.File(h5_path, "r") as f:
                    feats = f["features"][:]
                    n_tiles = feats.shape[0]
                    idxs = random.sample(range(n_tiles), min(self.samples_per_file, n_tiles))
                    subsampled_feats = feats[idxs]

                    subsampled_feats = 2 * (subsampled_feats - self.global_min) / (
                        self.global_max - self.global_min
                    ) - 1

                    self.data.extend(subsampled_feats)
                    base_id = fname.replace(".h5", "")
                    label = int(self.label_dict.get(base_id, 0))
                    self.labels.extend([label] * len(subsampled_feats))
            except Exception as e:  # pragma: no cover - logging only
                print(f"Error reading {fname}: {e}")

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
