# Invisible Friction Index

**We don’t predict failing students — we measure failing academic design.**

---

## Overview

Capable students often struggle or burn out not because of lack of effort or ability, but because academic systems silently overload them.

The **Invisible Friction Index (IFI)** reframes student failure as a **system-level design problem**, not an individual weakness. It quantifies hidden workload pressure created by dense schedules, rigid timing, and clustered demands — long before burnout or failure becomes visible.

Instead of asking *“Which student is at risk?”*, this project asks:  
**“Which academic designs are unsafe?”**

---

## The Problem

Academic systems create **invisible friction** through design choices such as:

- Back-to-back classes with no recovery buffers  
- Compressed daily schedules  
- Clustered deadlines and assessments  
- Rigid early and late class timings  

This friction:
- Is **not graded**
- Is **not tracked**
- Is **not measured**

Yet it steadily drains cognitive and physical energy, reduces learning quality, and increases failure risk. Most interventions occur **after** collapse, while the underlying design remains unchanged.

---

## Data Dictionary & Proxy Mapping

Because public datasets with granular minute-by-minute timetable logs are scarce, we utilize the UCI Student Performance Dataset (`student-mat.csv`) as a proxy environment.

| Dataset Column | Category | Proxy Domain Interpretation | Friction Component Signal |
| :--- | :--- | :--- | :--- |
| `studytime` | Friction Signal | Daily schedule commitment & course load density | `schedule_density`, `back_to_back_score`, `temporal_rigidity` |
| `absences` | Friction Signal | Schedule fatigue, travel friction & class clash proxy | `schedule_density`, `temporal_rigidity` |
| `goout` | Friction Signal | Out-of-class time constraints / social exhaustion | `back_to_back_score` |
| `Dalc` / `Walc` | Friction Signal | Stress load & recovery imbalance | `deadline_density` |
| `age`, `Medu`, `Fedu` | Baseline Demographic | Individual background baseline covariates | Isolated from friction scoring |
| `traveltime`, `freetime` | Baseline Contextual | Environmental baseline covariates | Isolated from friction scoring |
| `G3` | Target Outcome | Final course performance (`failure` = `G3 < 10`) | Used for empirical risk calibration |

---

## What This Project Does

The Invisible Friction Index is a **system-level analytics tool** that:

- Engineers timetable-like design features from non-demographic schedule/stress proxies
- Persists global scaling (`MinMaxScaler`) and calibrated percentile risk thresholds
- Computes a composite **Invisible Friction Index**  
- Categorizes conditions into **Survivable**, **Warning**, and **Unsafe** bands  
- Integrates a serialized machine learning inference model in Streamlit
- Suggests **concrete design fixes**, not student interventions  

---

## How It Works

### 1. Feature Independence & Engine Architecture
Baseline individual covariates (`age`, `Medu`, `Fedu`, `traveltime`, `freetime`) are kept distinct from friction components. All feature scaling and risk band cutoffs are calculated in a central engine (`src/friction_engine.py`) and persisted for dashboard inference.

### 2. Risk Calibration
Thresholds for **Survivable**, **Warning**, and **Unsafe** bands are empirically calibrated based on the 33rd and 66th quantiles of the friction index across training data.

### 3. Validation (5-Fold Stratified CV Ablation)
We evaluate model performance using 5-Fold Stratified Cross-Validation to assess baseline vs. friction-augmented models without data leakage or session-dependent scale distortion.

---

## Ethics, Responsibility & Limitations

**What this model does:**
- Measures system-imposed academic design friction  
- Identifies unsafe scheduling patterns  

**What this model does NOT do:**
- No mental health diagnosis  
- No behavioral surveillance  
- No student labeling or grading decisions  

**Stated Limitations:**
This project currently uses student survey metrics as proxy inputs for academic scheduling friction. For real-world production deployment, input data should be directly ingested from institutional SIS / ERP timetable databases (`course_id`, `room_buffer_min`, `daily_lecture_hours`, `exam_gap_days`).

---

## Tech Stack & Verification

- Python 3.12
- Pandas, NumPy — Data Processing
- Scikit-learn — Modeling & Calibration
- Joblib — Artifact Serialization
- Pytest — Automated Unit Testing
- Streamlit — Interactive Dashboard

---

## Quickstart & Running the App

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run Pipeline & Tests:**
```bash
PYTHONPATH=. python3 notebooks/02_feature_engineering.py
PYTHONPATH=. python3 notebooks/03_modeling.py
PYTHONPATH=. python3 -m pytest tests/
```

3. **Launch Dashboard:**
```bash
streamlit run app/app.py
```
