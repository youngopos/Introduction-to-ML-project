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

---

## Requirements

- Python 3.10+
- pip

## Installation

```bash
pip install -r requirements.txt
---
```
## Project Structure

```
├── data/              
├── artefacts/         
├── notebooks/
│   ├── EDA.ipynb      # Preprocessing + feature selection
│   └── evaluation.ipynb
├── src/
│   ├── Pipelines/     # Model definitions + hyperparameter search spaces
│   └── ...            # Training & tuning logic
└── README.md
```
## Pretrained Models

Trained models are available in the GitHub Releases section:

https://github.com/youngopos/Introduction-to-ML-project/releases/tag/v1.0
Download the `.joblib` files and place them in the `artefacts/` directory.

---

## Recommended Minimal Run

To quickly review the project:

1. Run `EDA.ipynb`
2. Make sure that the `.joblib` files from the release are placed in the artefacts directory.
3. Run `evaluation.ipynb`

No need to execute the full tuning pipeline.
