# Credit Card Fraud Detection

## Overview

A machine learning project focused on detecting fraudulent credit card
transactions using classification algorithms and techniques for handling
highly imbalanced data.

The project compares multiple classification models, cross-validation
strategies, and class-balancing techniques to identify effective approaches
for fraud detection.

## Problem Statement

Credit card fraud detection is a highly imbalanced binary classification
problem where fraudulent transactions represent only a small fraction of
all transactions.

The objective of this project is to develop and evaluate machine learning
models that can distinguish fraudulent transactions from legitimate ones
while maintaining strong predictive performance.

## Dataset

The dataset contains credit card transactions made by European cardholders
over two days in September 2013.

- Total transactions: 284,807
- Fraudulent transactions: 492
- Fraud rate: 0.172%
- Target variable: `Class`
- Features: PCA-transformed variables along with `Time` and `Amount`

Due to confidentiality, most original feature information has been
transformed using PCA.

## Project Workflow

1. Data loading and understanding
2. Exploratory Data Analysis
3. Class imbalance analysis
4. Feature transformation
5. Train-test split
6. Model development
7. Cross-validation
8. Class imbalance handling
9. Hyperparameter tuning
10. Model evaluation

## Exploratory Data Analysis

The analysis includes:

- Dataset structure and data types
- Missing-value checks
- Statistical summaries
- Fraud vs. non-fraud distribution
- Correlation analysis
- Feature distributions
- Transaction-time analysis

The dataset is highly imbalanced, with fraudulent transactions representing
only 0.172% of the total observations.

## Models Evaluated

The project evaluates multiple classification approaches:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine

## Cross-Validation

Two cross-validation approaches were explored:

- RepeatedKFold
- StratifiedKFold

StratifiedKFold was particularly useful for maintaining the class distribution
across folds given the severe imbalance in the dataset.

## Handling Class Imbalance

Multiple approaches were compared:

- Random Oversampling
- SMOTE
- ADASYN

These techniques were evaluated together with StratifiedKFold cross-validation
to understand their effect on model performance.

## Hyperparameter Tuning

XGBoost was further tuned using randomized hyperparameter search.

The selected model used parameters including:

- `max_depth = 7`
- `learning_rate = 0.125`
- `n_estimators = 60`
- `subsample = 0.8`
- `colsample_bytree = 0.7`

## Results

### Tuned XGBoost

| Metric | Result |
|---|---:|
| Accuracy | 99.93% |
| ROC-AUC | 0.9815 |
| Selected Threshold | 0.0172 |

The tuned XGBoost model achieved a ROC-AUC of **0.9815** on the test data.

Among the class-balancing approaches, XGBoost with Random Oversampling
and StratifiedKFold produced the strongest results in this project.

The baseline comparison also showed that Logistic Regression with L2
regularization using StratifiedKFold without resampling performed strongly
among the non-oversampled approaches.

## Key Takeaways

- Fraud detection requires careful handling of severe class imbalance.
- Stratified cross-validation helps preserve class distribution during model evaluation.
- Different oversampling techniques can produce substantially different results.
- XGBoost performed strongly after class balancing and hyperparameter tuning.
- ROC-AUC provides a more informative view of ranking performance than accuracy alone
  in this highly imbalanced classification problem.

## Technologies

Python | Pandas | NumPy | Scikit-learn | XGBoost |
Imbalanced-learn | Matplotlib | Seaborn

## Project Files

- `Credit_Card_Fraud_Detection.ipynb` – Complete analysis and modeling workflow
- `Credit Card Fraud Detection.pdf` – Project presentation

## Author

### Shaila Magline J

Data Analyst | MSc Data Science

[LinkedIn](YOUR_LINKEDIN_LINK)
