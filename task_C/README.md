# Task C - Code Classification with BERT Models

This folder contains the complete replication package for Task C, implementing CodeBERT and ModernBERT training with comprehensive evaluation and calibration methods.

## Overview

The notebook provides an end-to-end pipeline for code classification including:

- **Model Training**: CodeBERT and ModernBERT (1024 tokens)
- **Data Processing**: Tokenization caching for faster iteration
- **Calibration**: Temperature scaling and bias tuning
- **Analysis**: Hard/easy confidence analysis and truncation statistics
- **Evaluation**: Macro-F1, per-class F1, confusion matrices
- **Submission**: Automated prediction generation

All results can be reproduced by running the notebook from top to bottom. No additional scripts are required.

---

##  Quick Start

### Pretrained Checkpoints (Recommended)

To skip training and directly evaluate/generate submissions:

1. Download pretrained models: [**Google Drive Link**](https://drive.google.com/drive/folders/1VcON9IhEWg761JsnjE22vyl1cbzPo299?usp=drive_link)
2. Place checkpoints in the appropriate `outputs/runs/` directory
3. Run evaluation, calibration, and submission cells directly

---

## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Dataset Format

Organize your data folder as follows:

```
Task_C/
├── train.parquet
├── validation.parquet
└── test.parquet (optional)
```

---

## Training Experiments

The notebook implements the following experiments:

### CodeBERT Variants
- **Baseline**: Standard training
- **Label Smoothing**: Regularization technique
- **Balanced Sampling**: Address class imbalance

### ModernBERT
- **1024-token training**: Extended context for longer code snippets

### Training Pipeline

Each experiment automatically:
1. Tokenizes data (cached for reuse)
2. Trains the model
3. Saves the best checkpoint
4. Exports validation logits for analysis

---

## Output Structure

The notebook generates the following directory structure:

```
outputs/
├── tokenized_cache_local/     # Cached tokenized datasets
└── runs/
    ├── run1_baseline/
    ├── run2_label_smoothing/
    ├── run3_balanced_sampling/
    └── run_modernbert_1024/
```

### Contents of Each Run Folder

```
run_name/
├── checkpoint files           # Model weights
├── logs/                      # Training logs
├── val_logits.npy            # Validation logits
├── val_labels.npy            # Validation labels
├── temp_T.npy                # Temperature scaling parameter
├── bias_b.npy                # Bias tuning parameter
├── hard_mask.npy             # Hard samples mask
├── easy_mask.npy             # Easy samples mask
├── conf.npy                  # Confidence scores
```

---

## Logits Export

After training, the notebook exports:
- `val_logits.npy`: Raw model predictions on validation set
- `val_labels.npy`: Ground truth labels

**Benefits:**
- Enables calibration without retraining
- Facilitates evaluation and analysis
- Improves reproducibility
- Allows rapid experimentation

---

## Calibration

Two calibration methods are implemented to improve prediction confidence:

### 1. Temperature Scaling
Learns optimal temperature `T` to scale logits

### 2. Bias Tuning
Learns class-specific bias vector `b`

### Combined Calibration
Final calibrated logits: `logits / T + b`

Parameters are automatically learned and saved as:
- `temp_T.npy`
- `bias_b.npy`

---

## Hard/Easy Confidence Analysis

Samples are stratified by prediction confidence:

- **Hard samples**: Bottom 25% confidence
- **Easy samples**: Top 25% confidence

### Saved Artifacts
- `hard_mask.npy`: Boolean mask for hard samples
- `easy_mask.npy`: Boolean mask for easy samples
- `conf.npy`: Confidence scores for all samples

### Reported Metrics
- Macro-F1 (full dataset)
- Macro-F1 (hard samples)
- Macro-F1 (easy samples)
- Per-class F1 scores

---

## Submission Generation

Predictions are generated using streaming inference to avoid memory issues.

### Output Files
- `submission.csv`: Uncalibrated predictions
- `submission_calibrated.csv`: Calibrated predictions

### To Regenerate
Simply run the **"submission"** cell in the notebook.

---

## Truncation Analysis

The notebook computes sequence length statistics:

- Average length
- Median length
- 90th percentile (p90)
- 95th percentile (p95)
- Truncation rate at 512 tokens
- Truncation rate at 1024 tokens

**Purpose:** Justifies the use of ModernBERT with extended 1024-token context.

---

##  Reproducibility

### Design Principles
- All hyperparameters are fixed within cells
- Tokenization is cached for consistency
- Logits and calibration parameters are saved
- Paths are defined at the beginning of each cell

### Steps to Reproduce

1. **Prepare data**: Place `train.parquet`, `validation.parquet`, and optionally `test.parquet` in `Task_C/`
2. **Configure paths**: At the top of the notebook, edit the path configuration cell and set your local directories:
2. **Open notebook**: Launch the Jupyter notebook
3. **Run all cells**: Execute sequentially from top to bottom

**No additional configuration required.**


