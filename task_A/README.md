# README – Task A

This repository contains a classification pipeline for SemEval-2026 Task A, aimed at distinguishing AI-generated code from human-written code. The solution combines frozen pretrained code models, manual feature engineering and ensemble methods.

## Contents
- Task-A.ipynb

Main notebook that runs the full pipeline:
- data loading
- feature extraction
- model training
- validation and test evaluation
- ensemble comparison

## Implemented Approaches

- Frozen Encoder + Logistic Regression
    - Pretrained code models used as frozen feature extractors (e.g. UniXCoder)
    - Class-weighted Logistic Regression classifier
- Feature-Based Model
    - Manual code features (length, structure, comments, indentation, symbols)
    - Random Forest classifier
- Ensembles
    - Probability-level ensembles across multiple frozen encoders
    - Hybrid ensemble combining neural and feature-based models

## Datasets

The datasets required for Task A are available on Kaggle.
To run the code:
1. Download the Task A datasets from Kaggle
2. Place the parquet files inside the Task A repository folder
3. Update dataset paths in Task-A.ipynb if needed

## Quick Start

Install dependencies:

```pip install torch transformers datasets scikit-learn numpy pandas tqdm ```

Run:
- Open Task-A.ipynb
- Execute all cells from top to bottom

The notebook will automatically train the models and report Macro F1 and accuracy on validation and test sets.

## Notes

Only a subset of the training data is used.