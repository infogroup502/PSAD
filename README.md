# Lightweight Periodicity-Similarity-Guided Spatio-Temporal Anomaly Detection

This repository provides a PyTorch implementation of PSAD for multivariate time-series anomaly detection.

## Requirements

The recommended environment is Python 3.9 or later with a PyTorch version compatible with your CUDA or CPU environment. The remaining dependencies are:

- numpy==1.26.4
- pandas==2.2.2
- scipy==1.14.1
- scikit-learn==1.5.1
- Pillow==10.4.0
- tqdm==4.66.5
- matplotlib==3.9.2
- statsmodels==0.14.2
- tsfresh==0.20.3
- hurst==0.0.5
- arch==7.0.0

Install PyTorch first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data

PSAD supports multivariate time-series anomaly detection datasets stored as NumPy arrays.

Place each dataset under `dataset/<dataset>/` using the following layout:

```text
dataset/<dataset>/
├── <dataset>_train.npy
├── <dataset>_test.npy
└── <dataset>_test_label.npy
```

- Training and test arrays must have shape `[time, variables]`.
- Test labels must align with the test time axis, where `0` denotes normal and `1` denotes anomalous.
- Dataset files are not included in this repository.

## Usage

1. Install Python, PyTorch, and the dependencies listed above.
2. Place the SKAB files under `dataset/SKAB/`.
3. Run the provided local SKAB configuration from the project root:

```bash
conda activate PSAD
bash script/skab.sh
```

The script uses `python` from the active Conda environment and stops with a clear error if CUDA is unavailable.

The script trains and evaluates PSAD on SKAB. Checkpoints, logs, scores, and metrics are written under `result/SKAB/`.
# PSAD
