import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.friction_engine import FrictionEngine, BASELINE_FEATURES

# Load raw dataset to fit engine and train full model
df_raw = pd.read_csv("data/raw/student-mat.csv")
df_raw["failure"] = (df_raw["G3"] < 10).astype(int)

# Fit Friction Engine
engine = FrictionEngine()
df = engine.fit_transform(df_raw, target_col="failure")

X_base = df[BASELINE_FEATURES]
X_friction = df[BASELINE_FEATURES + ["friction_index"]]
y = df["failure"]

# 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_base_list = []
auc_friction_list = []

for train_idx, test_idx in skf.split(df, y):
    Xb_tr, Xb_te = X_base.iloc[train_idx], X_base.iloc[test_idx]
    Xf_tr, Xf_te = X_friction.iloc[train_idx], X_friction.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    mb = LogisticRegression(max_iter=1000)
    mb.fit(Xb_tr, y_tr)
    auc_base_list.append(roc_auc_score(y_te, mb.predict_proba(Xb_te)[:, 1]))

    mf = LogisticRegression(max_iter=1000)
    mf.fit(Xf_tr, y_tr)
    auc_friction_list.append(roc_auc_score(y_te, mf.predict_proba(Xf_te)[:, 1]))

print(f"5-Fold CV Baseline AUC: {np.mean(auc_base_list):.4f} ± {np.std(auc_base_list):.4f}")
print(f"5-Fold CV With Friction AUC: {np.mean(auc_friction_list):.4f} ± {np.std(auc_friction_list):.4f}")

# Train final model on full dataset
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_friction, y)

# Save model artifacts into models/
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/friction_model.joblib")
joblib.dump(engine, "models/friction_engine.joblib")

print("Saved trained model to models/friction_model.joblib")
print("Saved friction engine to models/friction_engine.joblib")
