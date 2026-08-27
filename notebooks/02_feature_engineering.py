import os
import pandas as pd
from src.friction_engine import FrictionEngine

# Load raw student performance dataset
df = pd.read_csv("data/raw/student-mat.csv", sep=",")
df["failure"] = (df["G3"] < 10).astype(int)

# Fit and transform friction engine
engine = FrictionEngine()
df_transformed = engine.fit_transform(df, target_col="failure")

os.makedirs("data/processed", exist_ok=True)
df_transformed.to_csv("data/processed/friction_data.csv", index=False)

print("Saved data/processed/friction_data.csv")
print(f"Friction Index range: min={df_transformed['friction_index'].min():.3f}, max={df_transformed['friction_index'].max():.3f}")
print("Risk Band Distribution:")
print(df_transformed["risk_band"].value_counts())
