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

Install the required packages:

```bash
pip install -r requirements.txt
```

## Data Format

The code assumes that each WSI has already been converted into patch-level feature embeddings.

A typical input CSV file should contain one row per slide:

```csv
slide_id,label,feature_path
slide_001,1,/path/to/slide_001_features.npy
slide_002,0,/path/to/slide_002_features.npy
```

Each feature file should contain a NumPy array with shape:

```text
number_of_patches × feature_dimension
```

For example, if 1024-dimensional patch features are used, each slide-level feature file should have shape:

```text
number_of_patches × 1024
```

## Training

A simple training command is:

```bash
python train.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --input_dim 1024 \
    --num_evidence_axes 64 \
    --lambda_l1 5.0 \
    --lr 1e-4 \
    --epochs 100 \
    --output_dir outputs/
```

Example arguments:

* `--train_csv`: path to the training CSV file.
* `--val_csv`: path to the validation CSV file.
* `--input_dim`: dimension of the patch-level feature embeddings.
* `--num_evidence_axes`: number of positive and negative evidence axes.
* `--lambda_l1`: strength of L1 regularization on evidence weights.
* `--lr`: learning rate.
* `--epochs`: number of training epochs.
* `--output_dir`: directory for saving checkpoints and logs.

## Output

The training script saves model checkpoints and validation results to the specified output directory.

Example:

```text
outputs/
├── best_model.pt
├── last_model.pt
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

