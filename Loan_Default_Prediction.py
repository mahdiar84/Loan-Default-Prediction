# loan_default_pipeline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support
)


# ------------------------------
# Config / path
# ------------------------------
DATA_PATH = r"C:\Users\saraye tel\OneDrive\Desktop\ARCH_Roadmap\Datasets\Loan_default.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

# ------------------------------
# Load dataset
# ------------------------------
df_raw = pd.read_csv(DATA_PATH)
df = df_raw.copy()
print("Dataset shape:", df.shape)

# ------------------------------
# Identify / fix target column (robust)
# ------------------------------
# Try common target names
possible_targets = ["loan_status", "loan_default", "loan_default_status", "default", "target"]
target_col = None
for t in possible_targets:
    if t in df.columns:
        target_col = t
        break

# fallback: assume last column is target if it's binary-like
if target_col is None:
    last_col = df.columns[-1]
    unique_vals = df[last_col].dropna().unique()
    if len(unique_vals) <= 5:
        target_col = last_col

if target_col is None:
    raise ValueError("Could not detect target column automatically. Please set 'target_col' manually.")

print("Detected target column:", target_col)

# ------------------------------
# Drop obvious ID or leakage columns (adjust names as necessary)
# ------------------------------
drop_cols = []
# common id column names
for c in ["LoanID", "loan_id", "id", "ID", "CustomerID", "customerID"]:
    if c in df.columns:
        drop_cols.append(c)

# heuristic: drop columns that obviously contain "date" after default or explicit probability columns
for c in df.columns:
    lowc = c.lower()
    if ("paid" in lowc and "date" in lowc) or ("default" in lowc and "prob" in lowc) or ("paidoff" in lowc):
        drop_cols.append(c)

drop_cols = list(set(drop_cols))
if drop_cols:
    print("Dropping columns:", drop_cols)
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

# ------------------------------
# Basic missing-value handling
# ------------------------------
# If many missing in a column, you might want to drop or engineer — here we do a safe fill:
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# Fill numeric with median, categorical with mode
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for c in obj_cols:
    df[c] = df[c].fillna(df[c].mode().iloc[0])

# ------------------------------
# Extract X_full, y_full and encode target to 0/1
# ------------------------------
y_full_raw = df[target_col]

# If object labels (like 'Yes'/'No' or 'Default'/'Paid'), label-encode them to 0/1
if y_full_raw.dtype == "object" or y_full_raw.dtype.name == "category":
    le = LabelEncoder()
    y_full = le.fit_transform(y_full_raw)
    print("Target classes after LabelEncoder:", le.classes_)
else:
    # numeric: if values are e.g. {0,2} map to {0,1}
    uniq = set(np.unique(y_full_raw.dropna()))
    if uniq == {0, 2}:
        y_full = y_full_raw.map({0: 0, 2: 1}).astype(int)
    else:
        y_full = y_full_raw.astype(int)

# Drop target column to get features
X_full = df.drop(columns=[target_col])

print("Features shape:", X_full.shape, "Target distribution:", pd.Series(y_full).value_counts().to_dict())

# ------------------------------
# Split BEFORE preprocessing (avoid leakage)
# ------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_full
)
print("Train shape:", X_train_raw.shape, "Test shape:", X_test_raw.shape)

# ------------------------------
# Identify numerical and categorical columns (after split we can re-evaluate)
# ------------------------------
numerical_cols = X_train_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
print("Numerical cols:", numerical_cols)
print("Categorical cols:", categorical_cols)

# ------------------------------
# Preprocessing pipeline (fit on train only)
# ------------------------------
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols),
    ],
    remainder="drop"
)

# Fit preprocessor on training data
preprocessor.fit(X_train_raw)

# Transform train and test
X_train_trans = preprocessor.transform(X_train_raw)
X_test_trans = preprocessor.transform(X_test_raw)

# Build feature names for later interpretability
feature_names_num = numerical_cols
feature_names_cat = []
if len(categorical_cols) > 0:
    try:
        feature_names_cat = list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
    except Exception:
        # older sklearn fallback
        feature_names_cat = []
feature_names = feature_names_num + feature_names_cat
print("Total features after transform:", len(feature_names))

# ------------------------------
# Balance training data with SMOTE (only on training set)
# ------------------------------
print("Before SMOTE counts:", pd.Series(y_train).value_counts().to_dict())
sm = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = sm.fit_resample(X_train_trans, y_train)
print("After SMOTE counts:", pd.Series(y_train_res).value_counts().to_dict())

# ------------------------------
# Helper: cross-validation check on training set (ROC-AUC)
# ------------------------------
def cv_check(model, X, y, cv=CV_FOLDS):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return scores

# ------------------------------
# Define models (regularized / constrained)
# ------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="auc"),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=500, random_state=RANDOM_STATE)
}

# ------------------------------
# Quick CV check (on balanced training set) to detect overfitting early
# ------------------------------
print("\nCross-validation (ROC-AUC) on balanced training set:")
for name, model in models.items():
    try:
        start = time.time()
        scores = cv_check(model, X_train_res, y_train_res)
        print(f"{name}: mean AUC = {scores.mean():.4f} (+/- {scores.std():.4f}), time {time.time()-start:.1f}s")
    except Exception as e:
        print(f"{name}: CV failed with error: {e}")

# ------------------------------
# Train on X_train_res / evaluate on X_test_trans
# ------------------------------
results = {}
for name, model in models.items():
    print("\n" + "=" * 40)
    print("Training and evaluating:", name)
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_trans)

    # Probabilities (if available)
    y_proba = model.predict_proba(X_test_trans)[:, 1] if hasattr(model, "predict_proba") else None

    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    cr = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {"model": model, "auc": auc, "report": cr, "cm": cm}

    print(f"Test ROC-AUC: {auc:.4f}" if auc is not None else "ROC-AUC: N/A")
    print("Classification report:\n", cr)
    print("Confusion matrix:\n", cm)

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# ------------------------------
# ROC curves comparison (if probabilities available)
# ------------------------------
plt.figure(figsize=(8, 6))
for name, r in results.items():
    model = r["model"]
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_trans)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = r["auc"]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

# ------------------------------
# Feature importance for tree models (RandomForest, XGBoost)
# ------------------------------
def plot_feature_importance(model, feature_names, top_n=25):
    try:
        importances = model.feature_importances_
    except Exception:
        print("Model has no feature_importances_ attribute.")
        return
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=fi.values, y=fi.index)
    plt.title("Feature importances")
    plt.tight_layout()
    plt.show()

# Try RandomForest
if "RandomForest" in results:
    print("\nRandomForest top features:")
    plot_feature_importance(results["RandomForest"]["model"], feature_names)

# Try XGBoost
if "XGBoost" in results:
    print("\nXGBoost top features:")
    plot_feature_importance(results["XGBoost"]["model"], feature_names)

# ------------------------------
# Save best model (example: highest AUC)
# ------------------------------
best_name = max(results.keys(), key=lambda n: results[n]["auc"] if results[n]["auc"] is not None else -1)
best_model = results[best_name]["model"]
print(f"\nBest model by AUC: {best_name} (AUC={results[best_name]['auc']:.4f})")
joblib.dump(best_model, "best_loan_default_model.pkl")
print("Saved best model to best_loan_default_model.pkl")