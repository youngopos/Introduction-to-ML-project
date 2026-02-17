# ML Academic Project – End-to-End Pipeline

## Overview

This project implements a complete machine learning workflow, including:

- Exploratory Data Analysis (EDA)
- Feature selection
- Hyperparameter tuning
- Model training
- Final evaluation and comparison

The pipeline is designed to ensure reproducibility and clear separation between data preparation, training, and evaluation stages.

---

## Requirements

- Python 3.12+
- numpy  
- pandas  
- scikit-learn  
- optuna  
- matplotlib  
- joblib  

Standard library modules used:
- `json`
- `pathlib (Path)`

They do not require installation.

---

## Project Structure

```
.
├── data/              # Raw input data
├── artefacts/         # Trained models, best hyperparameters, selected features
├── notebooks/
│   ├── EDA.ipynb      # Data preprocessing + feature selection
│   └── evaluation.ipynb
├── src/
│   ├── Pipelines/     # Model definitions + hyperparameter search spaces
│   └── ...            # Training & tuning logic
└── README.md
```

---

## Execution Workflow

### Data Preparation & Feature Selection

Run:

```
notebooks/EDA.ipynb
```

This step:
- performs preprocessing
- splits data into training and test sets
- selects relevant features
- saves artifacts

This is a critical stage of the project.

---

### Hyperparameter Tuning & Model Training (Optional)

Run training and tuning scripts from:

```
src/
```

⚠ **Warning:**  
This step is computationally expensive and may take several hours depending on your hardware.

The tuning stage can use full CPU parallelization (`n_jobs = -1`).

All results (trained models, best hyperparameters, selected variables) are already available in the `artefacts/` directory.

If you only want to inspect results, you can skip this step.

---

### Final Evaluation

Run:

```
notebooks/evaluation.ipynb
```

This notebook:
- loads trained models
- evaluates performance
- compares results

---

## Artifacts

The `artefacts/` directory contains:

- Trained models  
- Best hyperparameters  
- Selected features  
- Evaluation outputs  

This ensures reproducibility without rerunning the full tuning process.

---

## Reproducibility Notes

- Train/test split is performed during the EDA stage.
- Selected features are persisted.
- Hyperparameter tuning results are saved.
- Evaluation is performed on a held-out test set.

---

# Recommended Minimal Run

If you want to quickly review results:

1. Run `EDA.ipynb`
2. Run `evaluation.ipynb`

No need to execute the full tuning pipeline.