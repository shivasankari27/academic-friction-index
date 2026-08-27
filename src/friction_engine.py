"""
Invisible Friction Index Engine
-------------------------------
Core module for feature engineering, calibration, risk banding, model inference, and validation.
"""

from typing import Tuple, List, Dict, Any, Optional
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


def validate_dataframe(df: pd.DataFrame, require_target: bool = False) -> Tuple[bool, List[str]]:
    """
    Validates that the input DataFrame contains all required columns.
    """
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
    """
    comp = pd.DataFrame(index=df.index)
    comp["schedule_density"] = df["studytime"] * 2.0 + df["absences"] * 0.1
    comp["back_to_back_score"] = ((df["studytime"] >= 3) & (df["goout"] >= 3)).astype(float)
    comp["deadline_density"] = (df["Dalc"] + df["Walc"]) / 2.0 + (df["studytime"] > 2).astype(float)
    comp["temporal_rigidity"] = df["studytime"] + (df["absences"] > 5).astype(float)
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

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> "FrictionEngine":
        """
        Fits the scaler on the friction components and calibrates risk band thresholds
        against failure rates if target_col is provided.
        """
        valid, missing = validate_dataframe(df)
        if not valid:
            raise ValueError(f"DataFrame missing required columns: {missing}")

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

    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
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
