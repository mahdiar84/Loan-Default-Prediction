# Loan Default Prediction

## 📌 Project Overview
This project focuses on predicting loan defaults using machine learning.  
The goal is to build models that can identify whether a loan applicant is likely to default based on historical loan data.

We worked with the [Loan Default Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default) from Kaggle.

---

## ⚙️ Tech Stack
- Python 🐍
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE for handling class imbalance)

---

## 🚀 Features
- Data preprocessing: handling missing values, encoding categorical features, scaling numerical features.
- Applied **SMOTE** to fix class imbalance.
- Models used:
  - Logistic Regression
  - Random Forest (with RandomizedSearchCV)
  - XGBoost (with RandomizedSearchCV)
  - MLP Classifier
- Evaluations:
  - Accuracy
  - ROC-AUC
  - Confusion Matrix & Classification Report
- Visualization of results:
  - Confusion matrix heatmaps
  - ROC curves
  - Accuracy comparison bar chart

---

## 📊 Results
The project compared multiple ML models and tuned them using `RandomizedSearchCV`.  
The best-performing models achieved strong results while handling imbalance effectively.

---

## 📂 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Loan-Default-Prediction.git
   cd Loan-Default-Prediction

🔗 Dataset

The dataset can be found here: Kaggle Loan Default Dataset

📌 Author

Created by Mahdiar
