# ML Academic Project – End-to-End Classification Pipeline

## Project Overview

This project was developed as the final assignment for the **Introduction to Machine Learning** course.

The objective was to build the best possible classifier for a fully artificial dataset containing a mixture of informative and noisy features.  
Model performance was evaluated using **Balanced Accuracy** on a hidden test set.

A key challenge of this project was effective feature selection — separating pure noise from truly informative attributes.

Although this was an academic assignment, the repository is structured to resemble a production-style ML pipeline.

---

## Final Results

The final model achieved:

**Balanced Accuracy: 0.912**

This was one of the top results in the course cohort.

---

## Methodology

The project follows a structured machine learning workflow:

1. Exploratory Data Analysis (EDA)
2. Feature selection
3. Hyperparameter tuning
4. Model training
5. Final evaluation and comparison

The pipeline is designed to ensure:
- clear separation of responsibilities,
- reproducibility,
- prevention of data leakage.

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

No additional installation is required for them.

---

## Project Structure

```
.
├── data/              # Raw input data
├── notebooks/
│   ├── EDA.ipynb      # Preprocessing + feature selection
│   └── evaluation.ipynb
├── src/
│   ├── Pipelines/     # Model definitions + hyperparameter search spaces
│   └── ...            # Training & tuning logic
└── README.md
```

---

## Execution Workflow

### 1️⃣ Data Preparation & Feature Selection

Run:

```
notebooks/EDA.ipynb
```

This step:
- performs preprocessing,
- splits the dataset into training and test sets,
- selects relevant features,
- persists intermediate results.

This is the most critical stage of the project.

---

### 2️⃣ Hyperparameter Tuning & Model Training (Optional)

Run training and tuning scripts from:

```
src/
```

⚠ **Warning:**  
This step is computationally intensive and may take several hours depending on your hardware.

Hyperparameter optimization uses parallelization (`n_jobs = -1`).
If you are only interested in reviewing results, this step can be skipped.
Moreover, run training to generate artefacts locally (not tracked in git)

---

### 3️⃣ Final Evaluation

Run:

```
notebooks/evaluation.ipynb
```

This notebook:
- loads trained models,
- evaluates performance on the test set,
- compares final results.

---

## Reproducibility Notes

- Train/test split is performed during the EDA stage.
- Feature selection is saved and reused.
- Hyperparameter tuning results are persisted.
- Evaluation is performed on a strictly held-out test set.

---

## Recommended Minimal Run

To quickly review the project:

1. Run `EDA.ipynb`
2. Run `evaluation.ipynb`

No need to execute the full tuning pipeline.
