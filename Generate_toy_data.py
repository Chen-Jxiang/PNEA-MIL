import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm



def make_one_slide(label, num_patches, feature_dim, signal_dim, signal_patches):
    """
    Create one toy slide.

    The output x has shape:
        [num_patches, feature_dim]

    The label is NOT saved inside the .pt file.
    The label is saved in the CSV file.

    Toy signal:
        label 0 slides have stronger signal in features [0:signal_dim]
        label 1 slides have stronger signal in features [signal_dim:2*signal_dim]

    Only a small subset of patches receives the signal, mimicking the MIL setting
    where only some patches are informative.
    """
    x = torch.randn(num_patches, feature_dim)

    k = min(signal_patches, num_patches)
    patch_idx = torch.randperm(num_patches)[:k]

    if label == 0:
        x[patch_idx, 0:signal_dim] += 1.
    else:
        x[patch_idx, signal_dim:2*signal_dim] += 1.

    return x


def make_split(
    split_name,
    output_dir,
    n_slides,
    feature_dim,
    min_patches,
    max_patches,
    signal_dim,
    signal_patches,
):
    """
    Create one data split.

    Files created:
        f"{output_dir}/{split_name}/slide_0000.pt"
        f"{output_dir}/{split_name}/slide_0001.pt"
        ...

    CSV created:
        f"{output_dir}/{split_name}.csv"

    CSV format:
        file,label
        slide_0000.pt,0
        slide_0001.pt,1
        ...

    Each .pt file stores only the patch feature tensor:
        x = torch.load("slide_0000.pt")
        x.shape == [num_patches, feature_dim]
    """
    split_dir = Path( f"{output_dir}/{split_name}" )
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for i in tqdm(range(n_slides), desc=f"Generating {split_name}"):
        label = i % 2
        num_patches = np.random.randint(min_patches, max_patches + 1)

        x = make_one_slide(
            label=label,
            num_patches=num_patches,
            feature_dim=feature_dim,
            signal_dim=signal_dim,
            signal_patches=signal_patches,
        )

        file_name = f"{split_dir}/slide_{i:04d}.pt"
        torch.save(x, file_name)

        rows.append({"file": file_name, "label": label})

    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/{split_name}.csv", index=False)





def main():
    make_split(
        split_name="train",
        output_dir = "toy_data",
        n_slides=100,
        feature_dim = 1024,
        min_patches = 10000,
        max_patches = 20000,
        signal_dim = 10,
        signal_patches = 1000,
    )

    make_split(
        split_name="test",
        output_dir = "toy_data",
        n_slides=20,
        feature_dim = 1024,
        min_patches = 10000,
        max_patches = 20000,
        signal_dim = 10,
        signal_patches = 1000,
    )


if __name__ == "__main__":
    main()