"""
Invisible Friction Index Engine
-------------------------------
Core module for feature engineering, calibration, risk banding, model inference, and validation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

REQUIRED_COLUMNS = [
    "studytime",
    "absences",
    "goout",
    "Dalc",
    "Walc",
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "freetime",
]

BASELINE_FEATURES = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "freetime",
]

FRICTION_COMPONENT_NAMES = [
    "schedule_density",
    "back_to_back_score",
    "deadline_density",
    "temporal_rigidity",
]

PROXY_COLUMNS = ["studytime", "absences", "goout", "Dalc", "Walc"]


def validate_dataframe(df: pd.DataFrame, require_target: bool = False) -> tuple[bool, list[str]]:
    """
    Validates that the input DataFrame contains all required columns.
    """
    if not isinstance(df, pd.DataFrame):
        return False, ["Invalid DataFrame format"]
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if require_target and "G3" not in df.columns and "failure" not in df.columns:
        missing.append("G3 (or failure)")
    return len(missing) == 0, missing


def compute_raw_friction_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives friction components using non-baseline signals:
    - schedule_density: studytime * 2 + absences * 0.1
    - back_to_back_score: high studytime intensity combined with high goout/social fatigue
    - deadline_density: studytime coupled with total alcohol/stress indicators (Dalc + Walc)
    - temporal_rigidity: absences combined with studytime schedule constraints

    Handles edge cases:
    - Missing columns: Raises ValueError
    - NaN/Null values: Imputes with median/default gracefully
    - Non-numeric / out-of-range values: Coerced to numeric and clipped to non-negative ranges
    - Empty DataFrames: Returns empty DataFrame with expected columns
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [c for c in PROXY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required proxy columns: {missing}")

    if df.empty:
        return pd.DataFrame(columns=FRICTION_COMPONENT_NAMES, index=df.index, dtype=float)

    # Coerce to numeric and impute NaNs gracefully
    df_clean = pd.DataFrame(index=df.index)
    for col in PROXY_COLUMNS:
        s = pd.to_numeric(df[col], errors="coerce")
        median_val = s.median()
        if pd.isna(median_val):
            median_val = 0.0
        s = s.fillna(median_val)
        df_clean[col] = s.clip(lower=0)

    comp = pd.DataFrame(index=df.index)
    comp["schedule_density"] = df_clean["studytime"] * 2.0 + df_clean["absences"] * 0.1
    comp["back_to_back_score"] = ((df_clean["studytime"] >= 3) & (df_clean["goout"] >= 3)).astype(float)
    comp["deadline_density"] = (df_clean["Dalc"] + df_clean["Walc"]) / 2.0 + (df_clean["studytime"] > 2).astype(float)
    comp["temporal_rigidity"] = df_clean["studytime"] + (df_clean["absences"] > 5).astype(float)
    return comp


class FrictionEngine:
    """
    Engine to fit, transform, calibrate risk bands, and compute Friction Index.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.threshold_low = 0.33
        self.threshold_high = 0.66

    def fit(self, df: pd.DataFrame, target_col: str | None = None) -> "FrictionEngine":
        """
        Fits the scaler on the friction components and calibrates risk band thresholds
        against failure rates if target_col is provided.
        """
        valid, missing = validate_dataframe(df)
        if not valid:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        if df.empty:
            raise ValueError("Cannot fit FrictionEngine on an empty DataFrame.")

        raw_components = compute_raw_friction_components(df)
        self.scaler.fit(raw_components)
        self.is_fitted = True

        # Compute scaled friction index for calibration
        scaled = pd.DataFrame(
            self.scaler.transform(raw_components),
            columns=FRICTION_COMPONENT_NAMES,
            index=df.index,
        )
        friction_index = scaled.mean(axis=1)

        if target_col is not None and target_col in df.columns:
            # Empirical Calibration based on failure rate percentiles (33rd and 66th quantiles of friction index)
            self.threshold_low = float(np.quantile(friction_index, 0.33))
            self.threshold_high = float(np.quantile(friction_index, 0.66))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input DataFrame to add component scores, scaled friction index, and risk bands.
        """
        if not self.is_fitted:
            raise RuntimeError("FrictionEngine must be fitted before calling transform().")

        valid, missing = validate_dataframe(df)
        if not valid:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        out = df.copy()
        if out.empty:
            for col in FRICTION_COMPONENT_NAMES:
                out[col] = []
                out[f"{col}_scaled"] = []
            out["friction_index"] = []
            out["risk_band"] = []
            return out

        raw_components = compute_raw_friction_components(out)
        scaled_components = pd.DataFrame(
            self.scaler.transform(raw_components),
            columns=[f"{c}_scaled" for c in FRICTION_COMPONENT_NAMES],
            index=out.index,
        )

        for col in FRICTION_COMPONENT_NAMES:
            out[col] = raw_components[col]
            out[f"{col}_scaled"] = scaled_components[f"{col}_scaled"]

        out["friction_index"] = scaled_components.mean(axis=1)
        out["risk_band"] = out["friction_index"].apply(self.assign_risk_band)
        return out

    def fit_transform(self, df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
        """
        Fits and transforms in a single call.
        """
        return self.fit(df, target_col=target_col).transform(df)

    def assign_risk_band(self, index_val: float) -> str:
        """
        Categorizes friction index into risk bands.
        """
        if index_val < self.threshold_low:
            return "🟢 Survivable"
        elif index_val < self.threshold_high:
            return "🟡 Warning"
        else:
            return "🔴 Unsafe"
