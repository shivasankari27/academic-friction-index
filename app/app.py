import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Add repo root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.friction_engine import (
    FrictionEngine,
    validate_dataframe,
    BASELINE_FEATURES,
    FRICTION_COMPONENT_NAMES,
)

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
@st.cache_resource
def load_artifacts():
    model_path = "models/friction_model.joblib"
    engine_path = "models/friction_engine.joblib"
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    engine = joblib.load(engine_path) if os.path.exists(engine_path) else FrictionEngine()
    return model, engine

model, engine = load_artifacts()

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
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
elif use_demo or True:  # Default fallback
    if os.path.exists("data/raw/student-mat.csv"):
        df_raw = pd.read_csv("data/raw/student-mat.csv")
    else:
        st.info("Please upload a CSV file with student/schedule data to proceed.")

if df_raw is not None:
    # Validate Schema
    is_valid, missing_cols = validate_dataframe(df_raw)
    if not is_valid:
        st.error(
            f"❌ Uploaded dataset is missing required columns: {', '.join(missing_cols)}.\n"
            "Please upload a dataset that matches the required schema."
        )
    else:
        # Transform dataframe using persisted engine scaler & calibrated thresholds
        df = engine.transform(df_raw)

        # Model Prediction if model exists
        if model is not None:
            X_infer = df[BASELINE_FEATURES + ["friction_index"]]
            df["predicted_failure_prob"] = model.predict_proba(X_infer)[:, 1]

        # --------------------------------------------------
        # SYSTEM-LEVEL DASHBOARD
        # --------------------------------------------------
        st.subheader("System-Level Friction Summary")

        avg_friction = round(df["friction_index"].mean(), 3)
        band_pct = df["risk_band"].value_counts(normalize=True) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average Friction Index", avg_friction)
        c2.metric("Unsafe (%)", f"{round(band_pct.get('🔴 Unsafe', 0), 1)}%")
        c3.metric("Warning (%)", f"{round(band_pct.get('🟡 Warning', 0), 1)}%")
        if "predicted_failure_prob" in df.columns:
            avg_pred = round(df["predicted_failure_prob"].mean() * 100, 1)
            c4.metric("Predicted Failure Risk (%)", f"{avg_pred}%")
        else:
            c4.metric("Survivable (%)", f"{round(band_pct.get('🟢 Survivable', 0), 1)}%")

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
            st.markdown(
                f"""
                **Calibrated Thresholds:**
                - Low Risk threshold (Survivable / Warning): `{engine.threshold_low:.3f}`
                - High Risk threshold (Warning / Unsafe): `{engine.threshold_high:.3f}`

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
