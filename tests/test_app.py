import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.app import compute_summary_metrics, load_artifacts, run_inference
from src.friction_engine import FrictionEngine


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame({
        "studytime": [1, 2, 3, 4],
        "absences": [0, 4, 10, 20],
        "goout": [1, 2, 4, 5],
        "Dalc": [1, 1, 2, 5],
        "Walc": [1, 2, 3, 5],
        "age": [15, 16, 17, 18],
        "Medu": [4, 3, 2, 1],
        "Fedu": [4, 3, 2, 1],
        "traveltime": [1, 1, 2, 3],
        "freetime": [3, 3, 2, 1],
    })


def test_load_artifacts_success():
    _model, engine, _metadata = load_artifacts()
    # Should successfully return engine instance even if models don't exist yet
    assert engine is not None
    assert isinstance(engine, FrictionEngine)


def test_load_artifacts_missing_or_corrupt_path(tmp_path):
    invalid_model = str(tmp_path / "corrupt_model.joblib")
    invalid_engine = str(tmp_path / "corrupt_engine.joblib")
    invalid_meta = str(tmp_path / "invalid_meta.json")

    with open(invalid_model, "w") as f:
        f.write("not a joblib file")
    with open(invalid_engine, "w") as f:
        f.write("not a joblib file")
    with open(invalid_meta, "w") as f:
        f.write("{invalid json")

    model, engine, metadata = load_artifacts(invalid_model, invalid_engine, invalid_meta)
    assert model is None
    assert isinstance(engine, FrictionEngine)
    assert metadata is None


def test_run_inference_full_run(sample_raw_df):
    engine = FrictionEngine()
    engine.fit(sample_raw_df)

    df_out, is_valid, missing = run_inference(sample_raw_df, engine, model=None)
    assert is_valid is True
    assert len(missing) == 0
    assert "friction_index" in df_out.columns
    assert "risk_band" in df_out.columns


def test_run_inference_missing_cols():
    invalid_df = pd.DataFrame({"studytime": [1, 2]})
    engine = FrictionEngine()

    df_out, is_valid, missing = run_inference(invalid_df, engine)
    assert is_valid is False
    assert len(missing) > 0
    assert df_out is None


def test_compute_summary_metrics(sample_raw_df):
    engine = FrictionEngine()
    df_transformed = engine.fit_transform(sample_raw_df)

    metrics = compute_summary_metrics(df_transformed)
    assert "avg_friction" in metrics
    assert "unsafe_pct" in metrics
    assert "warning_pct" in metrics
    assert "survivable_pct" in metrics
    assert metrics["avg_friction"] >= 0.0


def test_app_test_framework_run():
    at = AppTest.from_file("app/app.py").run()
    assert not at.exception
    # Verify title rendered
    assert len(at.title) > 0
    assert "Invisible Friction Index" in at.title[0].value


def test_app_test_framework_demo_button():
    at = AppTest.from_file("app/app.py").run()
    # Click the demo button
    if len(at.button) > 0:
        at.button[0].click().run()
        assert not at.exception
