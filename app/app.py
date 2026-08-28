import json
import os
import sys
from typing import Any

import joblib
import pandas as pd
import streamlit as st

# Add repo root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.friction_engine import (
    BASELINE_FEATURES,
    FRICTION_COMPONENT_NAMES,
    FrictionEngine,
    validate_dataframe,
)


# --------------------------------------------------
# REFACTORED CORE LOGIC FUNCTIONS
# --------------------------------------------------
@st.cache_resource
def load_artifacts(
    model_path: str = "models/friction_model.joblib",
    engine_path: str = "models/friction_engine.joblib",
    metadata_path: str = "models/friction_model_metadata.json",
) -> tuple[Any | None, FrictionEngine, dict[str, Any] | None]:
    """
    Loads model, friction engine, and companion metadata JSON.
    Handles missing or corrupt artifact paths gracefully.
    """
    model = None
    engine = None
    metadata = None

    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except (OSError, ValueError, TypeError, KeyError):
            model = None

    if os.path.exists(engine_path):
        try:
            engine = joblib.load(engine_path)
        except (OSError, ValueError, TypeError, KeyError):
            engine = FrictionEngine()
    else:
        engine = FrictionEngine()

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            metadata = None

    return model, engine, metadata


def run_inference(
    df_raw: pd.DataFrame,
    engine: FrictionEngine,
    model: Any | None = None,
) -> tuple[pd.DataFrame | None, bool, list]:
    """
    Validates input DataFrame schema, transforms it using FrictionEngine,
    and runs model inference if model is available.
    """
    is_valid, missing_cols = validate_dataframe(df_raw)
    if not is_valid:
        return None, False, missing_cols

    # Transform dataframe using persisted engine scaler & calibrated thresholds
    df = engine.transform(df_raw)

    # Model Prediction if model exists
    if model is not None:
        try:
            X_infer = df[BASELINE_FEATURES + ["friction_index"]]
            df["predicted_failure_prob"] = model.predict_proba(X_infer)[:, 1]
        except (ValueError, KeyError, AttributeError):
            pass

    return df, True, []


def compute_summary_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Computes summary metrics for display on dashboard.
    """
    avg_friction = round(float(df["friction_index"].mean()), 3)
    band_pct = (df["risk_band"].value_counts(normalize=True) * 100).to_dict()

    unsafe_pct = round(float(band_pct.get("🔴 Unsafe", 0.0)), 1)
    warning_pct = round(float(band_pct.get("🟡 Warning", 0.0)), 1)
    survivable_pct = round(float(band_pct.get("🟢 Survivable", 0.0)), 1)

    avg_pred_risk = None
    if "predicted_failure_prob" in df.columns:
        avg_pred_risk = round(float(df["predicted_failure_prob"].mean()) * 100, 1)

    return {
        "avg_friction": avg_friction,
        "unsafe_pct": unsafe_pct,
        "warning_pct": warning_pct,
        "survivable_pct": survivable_pct,
        "avg_pred_risk": avg_pred_risk,
    }


def main():
    # --------------------------------------------------
    # PAGE CONFIG
    # --------------------------------------------------
    st.set_page_config(
        page_title="Invisible Friction Index",
        layout="wide"
    )

    st.title("Invisible Friction Index")
    st.caption("System-level academic design friction — measuring failing academic design, not failing students.")

    # --------------------------------------------------
    # LOAD MODEL & ENGINE ARTIFACTS
    # --------------------------------------------------
    model, engine, metadata = load_artifacts()

    # --------------------------------------------------
    # DATA UPLOAD / DEMO DATASET SELECTION
    # --------------------------------------------------
    st.sidebar.header("Data Selection")
    use_demo = st.sidebar.button("Load Demo Dataset (student-mat.csv)")
    uploaded_file = st.sidebar.file_uploader("Or Upload Custom CSV", type=["csv"])

    df_raw = None

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file, sep=None, engine="python")
        except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
            st.error(f"Error reading uploaded file: {e}")
    elif use_demo:
        if os.path.exists("data/raw/student-mat.csv"):
            df_raw = pd.read_csv("data/raw/student-mat.csv")
        else:
            st.info("Please upload a CSV file with student/schedule data to proceed.")
    else:
        st.info("Upload a CSV file or click 'Load Demo Dataset (student-mat.csv)' to analyze friction.")

    if df_raw is not None:
        df, is_valid, missing_cols = run_inference(df_raw, engine, model)
        if not is_valid:
            st.error(
                f"❌ Uploaded dataset is missing required columns: {', '.join(missing_cols)}.\n"
                "Please upload a dataset that matches the required schema."
            )
        else:
            metrics = compute_summary_metrics(df)

            # --------------------------------------------------
            # SYSTEM-LEVEL DASHBOARD
            # --------------------------------------------------
            st.subheader("System-Level Friction Summary")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Average Friction Index", metrics["avg_friction"])
            c2.metric("Unsafe (%)", f"{metrics['unsafe_pct']}%")
            c3.metric("Warning (%)", f"{metrics['warning_pct']}%")
            if metrics["avg_pred_risk"] is not None:
                c4.metric("Predicted Failure Risk (%)", f"{metrics['avg_pred_risk']}%")
            else:
                c4.metric("Survivable (%)", f"{metrics['survivable_pct']}%")

            st.divider()

            # --------------------------------------------------
            # RISK DISTRIBUTION CHART
            # --------------------------------------------------
            st.subheader("Friction Risk Distribution")

            chart_df = (
                df["risk_band"]
                .value_counts()
                .reindex(["🟢 Survivable", "🟡 Warning", "🔴 Unsafe"])
                .fillna(0)
            )

            st.bar_chart(chart_df)

            st.divider()

            # --------------------------------------------------
            # DESIGN FIX SUGGESTIONS
            # --------------------------------------------------
            st.subheader("Design-Level Fix Suggestions")

            mean_components = df[FRICTION_COMPONENT_NAMES].mean()
            top_issue = mean_components.idxmax()

            if top_issue == "back_to_back_score":
                st.warning(
                    "High back-to-back intensity detected. "
                    "Insert 15–30 minute buffer gaps between sessions."
                )
            elif top_issue == "deadline_density":
                st.warning(
                    "Deadline clustering & daily workload density detected. "
                    "Spread assessments across weeks instead of stacking them."
                )
            elif top_issue == "temporal_rigidity":
                st.warning(
                    "Early + late scheduling rigidity detected. "
                    "Introduce flexible start times or protected recovery windows."
                )
            else:
                st.warning(
                    "High daily schedule density detected. "
                    "Reduce overall daily course compression to preserve cognitive energy."
                )

            st.divider()

            # --------------------------------------------------
            # DATA PREVIEW
            # --------------------------------------------------
            st.subheader("Sample System Friction Analysis")

            cols_to_show = ["studytime", "absences", "friction_index", "risk_band"]
            if "predicted_failure_prob" in df.columns:
                cols_to_show.append("predicted_failure_prob")

            st.dataframe(
                df[cols_to_show].head(10),
                use_container_width=True,
            )

            st.divider()

            # --------------------------------------------------
            # MODEL CARD & LIMITATIONS
            # --------------------------------------------------
            with st.expander("📄 Model Card, Calibrated Thresholds & Scope"):
                meta_info = ""
                if metadata is not None:
                    meta_info = f"""
                    **Model Metadata:**
                    - Version: `{metadata.get('version', 'N/A')}`
                    - Model Type: `{metadata.get('model_type', 'N/A')}`
                    - Training Date: `{metadata.get('training_date', 'N/A')}`
                    - Best CV AUC: `{metadata.get('best_cv_auc', 'N/A')}`
                    """

                st.markdown(
                    f"""
                    **Calibrated Thresholds:**
                    - Low Risk threshold (Survivable / Warning): `{engine.threshold_low:.3f}`
                    - High Risk threshold (Warning / Unsafe): `{engine.threshold_high:.3f}`

                    {meta_info}

                    **What this model measures:**
                    - Academic design friction derived from schedule and workload proxies.
                    - System-imposed stress, not student behavior.

                    **What this model does NOT do:**
                    - No mental health diagnosis.
                    - No behavioral surveillance.
                    - No student labeling or grading decisions.

                    **Intended users:**
                    - Timetable designers
                    - Academic planners
                    - Institutional policy teams
                    """
                )


if __name__ == "__main__":
    main()
