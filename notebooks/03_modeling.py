import datetime
import json
import os
import subprocess

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.friction_engine import BASELINE_FEATURES, FrictionEngine


# Version Tag Helper
def get_version_tag() -> str:
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        if git_hash:
            return f"v1.0.0-{git_hash}"
    except (subprocess.SubprocessError, OSError):
        pass
    return "v1.0.0"

# Load raw dataset to fit engine and train full model
df_raw = pd.read_csv("data/raw/student-mat.csv")
df_raw["failure"] = (df_raw["G3"] < 10).astype(int)

# Fit Friction Engine
engine = FrictionEngine()
df = engine.fit_transform(df_raw, target_col="failure")

X_base = df[BASELINE_FEATURES]
feature_list = BASELINE_FEATURES + ["friction_index"]
X_friction = df[feature_list]
y = df["failure"]

# 5-Fold Stratified Cross-Validation for multiple models
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidate_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}

print("==================================================")
print("     MODEL BENCHMARKING (5-FOLD STRATIFIED CV)    ")
print("==================================================")
print(f"{'Model Name':<28} | {'Base AUC':<15} | {'With Friction AUC':<15}")
print("-" * 65)

best_model_name = None
best_auc = -1.0
best_model_obj = None

for name, model_cls in candidate_models.items():
    auc_base_list = []
    auc_friction_list = []

    for train_idx, test_idx in skf.split(df, y):
        Xb_tr, Xb_te = X_base.iloc[train_idx], X_base.iloc[test_idx]
        Xf_tr, Xf_te = X_friction.iloc[train_idx], X_friction.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # Base model clone/fit
        mb = model_cls.__class__(**model_cls.get_params())
        mb.fit(Xb_tr, y_tr)
        auc_base_list.append(roc_auc_score(y_te, mb.predict_proba(Xb_te)[:, 1]))

        # Friction model clone/fit
        mf = model_cls.__class__(**model_cls.get_params())
        mf.fit(Xf_tr, y_tr)
        auc_friction_list.append(roc_auc_score(y_te, mf.predict_proba(Xf_te)[:, 1]))

    base_mean = float(np.mean(auc_base_list))
    base_std = float(np.std(auc_base_list))
    fric_mean = float(np.mean(auc_friction_list))
    fric_std = float(np.std(auc_friction_list))

    results[name] = {
        "base_auc_mean": round(base_mean, 4),
        "base_auc_std": round(base_std, 4),
        "friction_auc_mean": round(fric_mean, 4),
        "friction_auc_std": round(fric_std, 4),
    }

    print(f"{name:<28} | {base_mean:.4f} ± {base_std:.4f} | {fric_mean:.4f} ± {fric_std:.4f}")

    if fric_mean > best_auc:
        best_auc = fric_mean
        best_model_name = name
        best_model_obj = model_cls

print("-" * 65)
print(f"Best Performing Model: {best_model_name} with Friction AUC: {best_auc:.4f}")

# Train best model on full dataset
final_model = best_model_obj.__class__(**best_model_obj.get_params())
final_model.fit(X_friction, y)

version_tag = get_version_tag()
training_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

metadata = {
    "version": version_tag,
    "training_date": training_date,
    "model_type": best_model_name,
    "best_cv_auc": round(best_auc, 4),
    "feature_list": feature_list,
    "benchmark_results": results,
}

# Save model artifacts into models/
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/friction_model.joblib")
joblib.dump(engine, "models/friction_engine.joblib")

metadata_path = "models/friction_model_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\nSaved trained best model ({best_model_name}) to models/friction_model.joblib")
print("Saved friction engine to models/friction_engine.joblib")
print(f"Saved companion metadata to {metadata_path}")
