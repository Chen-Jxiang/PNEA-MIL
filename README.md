# PNEA-MIL

This repository provides the implementation of **PNEA-MIL**: **Positive–Negative Evidence Analysis Multiple-Instance Learning** for whole-slide image classification.

PNEA-MIL is designed for computational pathology tasks where each whole-slide image (WSI) is represented as a bag of image patches and only slide-level labels are available. The method aims to improve the interpretability of multiple-instance learning by explicitly modeling both evidence that supports the positive class and evidence that contradicts it.

The full methodological details are provided in our paper:

> **PNEA-MIL: Interpretable Multiple-Instance Learning for Whole-Slide Images through Positive-Negative Evidence Analysis**
> Paper link: **[Coming soon]**

## Overview

Whole-slide images are extremely large, so they are commonly analyzed by first dividing each slide into smaller image patches. In many pathology applications, only slide-level labels are available, while patch-level annotations are unavailable or expensive to obtain. Multiple-instance learning (MIL) provides a natural framework for this setting.

Most MIL methods focus on identifying image regions that support a positive prediction. However, in pathology, diagnostic or prognostic decisions often depend on balancing both supportive and opposing histopathological evidence.

PNEA-MIL addresses this problem by decomposing the slide-level prediction into two types of evidence:

* **Positive evidence**, which supports the positive class.
* **Negative evidence**, which supports the negative class or contradicts the positive prediction.

The model learns patch-level positive and negative evidence representations, aggregates them at the slide level, and compares the two aggregated evidence scores to produce the final prediction. Non-negative evidence weights are used so that each evidence component has a clear and consistent contribution to the prediction.

## Installation

Clone the repository:

```bash
git clone https://github.com/Chen-Jxiang/PNEA-MIL.git
cd PNEA-MIL
```

Create and activate a conda environment:

```bash
conda create -n pnea-mil python=3.10
conda activate pnea-mil
```

Install PyTorch first. For GPU support, install the CUDA-enabled PyTorch version that matches your system. For example, for CUDA 12.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only installation:

```bash
pip install torch torchvision torchaudio
```

Please check the official PyTorch installation selector if a different CUDA version is needed.

Then install the remaining required packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file intentionally does not include `torch`, because the correct PyTorch installation depends on the user’s CUDA and hardware configuration.

## Data Format

The input data should be organized at the slide level. Each slide is represented by one .pt file containing its patch-level feature matrix, and the slide-level label is provided separately in a CSV file.

Each .pt file should store a single PyTorch tensor:

x = torch.load(file_path)

The tensor should have shape:

[num_patches, feature_dim]

where num_patches is the number of patches extracted from the slide, and feature_dim is the dimension of the feature vector for each patch.

The number of patches does not need to be the same across slides. For example, one slide may contain 8,000 patches, while another slide may contain 20,000 patches. However, the feature dimension should be the same for all slides.

The slide-level labels are stored in a CSV file. The CSV file should contain at least two columns:

file,label

The file column gives the path to the .pt file for each slide. The label column gives the slide-level class label.

An example CSV file is shown below:

file,label
/path/to/slide_0001.pt,0
/path/to/slide_0002.pt,1
/path/to/slide_0003.pt,0

Each row corresponds to one slide. For example, the row

/path/to/slide_0001.pt,0

means that /path/to/slide_0001.pt stores the patch-level feature tensor for one slide, and the slide-level label for that slide is 0.

The label should not be saved inside the .pt file. The .pt file should only contain the patch feature tensor. This separation allows the same feature files to be reused with different label files or different train/test splits.

For training and testing, separate CSV files can be provided. For example:

train.csv
test.csv

Each CSV file should follow the same format, with one row per slide and columns specifying the feature file path and slide-level label.

This format is designed for multiple-instance learning. Each slide is treated as a bag of patch-level feature vectors, while supervision is provided only at the slide level.

## Toy Data Generating

We provide code for generating toy data:

```bash
python Generate_toy_data.py
```

## Training

A simple training command utilizing the toy data is:

```bash
python train_PNEA-MIL.py \
    --train_csv toy_data/train.csv \
    --test_csv toy_data/test.csv \
    --output_model_file ./model/model
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{chen2026pnea,
  title     = {PNEA-MIL: Interpretable Multiple-Instance Learning for Whole-Slide Images through Positive-Negative Evidence Analysis},
  author    = {Chen, Junxiang and Couetil, Justin and Maher, Nigel Gordon and Scolyer, Richard and Alomari, Ahmed and Huang, Kun and Zhang, Jie},
  booktitle = {Proceedings of the 17th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3807503.3819367}
}
```

## License

Please see the `LICENSE` file for details.
