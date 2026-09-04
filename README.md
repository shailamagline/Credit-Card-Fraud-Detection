# 💳 Credit Card Fraud Detection
### Binary Classification · Class Imbalance Handling · Random Forest · SMOTE

---

## 📌 Project Overview

Credit card fraud is a **rare but high-stakes event** — fraudulent transactions represent only **0.17%** of all activity, making this a classic imbalanced binary classification problem. Standard accuracy metrics fail completely in this context; a model that predicts "legitimate" for every transaction would achieve 99.83% accuracy while catching zero fraud.

This project tackles that challenge head-on. Using a structured ML pipeline, I compared multiple classification algorithms, applied three oversampling techniques to address class imbalance, tuned a Random Forest model, and evaluated performance using metrics that actually matter for fraud — **Recall, F1-Score, PR-AUC, and ROC-AUC**.

---

## 🎯 Objective

> *Build a model that maximises fraud detection (Recall) while maintaining a viable Precision — minimising both missed fraud and false alarms.*

---

## 📂 Project Structure

```
credit-card-fraud-detection/
│
├── Credit_card_Fraud_Detection_Final_fixed.ipynb   # Full analysis notebook
├── creditcard.csv                                   # Dataset
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — ULB Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Transactions | 284,807 |
| Fraudulent Transactions | 492 |
| Fraud Rate | **0.172%** |
| Features | 30 (V1–V28 PCA-transformed + `Time` + `Amount`) |
| Target | `Class` (0 = Legitimate, 1 = Fraud) |

> **Note:** Features V1–V28 are PCA-transformed for confidentiality. Only `Time` and `Amount` retain their original scale.

---

## 🔬 Methodology

### 1️⃣ Exploratory Data Analysis
- Dataset shape, data types, missing values, and duplicate detection (1,081 duplicates removed before the train-test split to prevent leakage)
- Class distribution visualisation — confirmed severe imbalance
- Transaction amount and time analysis by class
- Correlation heatmap and class-correlated feature ranking
- Skewness analysis across all numerical features

### 2️⃣ Data Preprocessing
- Target/feature separation
- **Stratified train-test split** (80/20) — preserves fraud ratio in both sets
- **StandardScaler** fitted on training data only, then applied to test — no leakage

### 3️⃣ Baseline Modelling
Five models evaluated without any imbalance correction:

| Model | Focus |
|---|---|
| Logistic Regression | Linear baseline |
| K-Nearest Neighbors | Distance-based |
| Decision Tree | Rule-based, interpretable |
| Random Forest | Ensemble, tree-based |
| XGBoost | Gradient boosting |

Evaluated on: **ROC-AUC, Recall, F1-Score, PR-AUC**

### 4️⃣ Handling Class Imbalance
Three oversampling techniques were applied and compared on the training set:

| Technique | Description |
|---|---|
| **Random Oversampling** | Duplicates existing minority-class samples |
| **SMOTE** | Generates synthetic minority samples via interpolation |
| **ADASYN** | Focuses synthetic generation on harder-to-classify samples |
| **Class Weighting** | Penalises misclassification of minority class — no data augmentation |

All resampling was applied to the **training set only**. The test set remained untouched throughout.

### 5️⃣ Hyperparameter Tuning
Random Forest configurations were compared across:
- `n_estimators`: [100, 150, 200]
- `max_depth`: [None, 10, 20]
- `min_samples_split`: [2, 5]

Best configuration selected by **F1-Score** on the original (unbalanced) test set.

### 6️⃣ Final Model Evaluation
- Confusion matrix
- ROC curve with AUC
- Precision-Recall curve
- Full classification report (Precision, Recall, F1 per class)

### 7️⃣ Feature Importance
Top 15 features ranked by Random Forest importance — V14 emerged as the most influential feature (~21.70% importance), followed by V10, V4, and V12.

---

## 📈 Results

| Metric | Score |
|---|---|
| Recall (Fraud) | **82.65%** |
| F1-Score (Fraud) | **83.08%** |
| ROC-AUC | High |
| PR-AUC | High |

> The final model — **Random Forest + SMOTE** — correctly identifies the majority of fraudulent transactions while keeping false positives at a manageable level.

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Manipulation | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| Modelling | scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE, ADASYN, RandomOverSampler) |
| Environment | Jupyter Notebook |

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook Credit_card_Fraud_Detection_Final_fixed.ipynb
   ```

4. **Ensure `creditcard.csv` is in the same directory** as the notebook before running.

---

## 💡 Key Decisions & Learnings

- **Accuracy is not the right metric** for imbalanced fraud data — Recall and F1 are far more informative
- **Duplicates were removed before the train-test split**, not after — preventing identical transactions from appearing in both sets
- **Scaler was fitted on training data only** — applying it before the split would constitute data leakage
- **SMOTE outperformed Random Oversampling** because synthetic generation exposes the model to a broader feature space rather than just repeating existing samples
- **Class weighting** is a valid and computationally cheaper alternative to resampling — worth testing first before augmenting data
- **V14 dominates feature importance** (~21.70%), suggesting this PCA component captures the most discriminative signal between legitimate and fraudulent transactions

---

