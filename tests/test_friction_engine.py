import numpy as np
import pandas as pd
import pytest

from src.friction_engine import (
    FRICTION_COMPONENT_NAMES,
    FrictionEngine,
    compute_raw_friction_components,
    validate_dataframe,
)


@pytest.fixture
def sample_data():
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
        "G3": [15, 12, 8, 4],
    })


def test_validate_dataframe_valid(sample_data):
    valid, missing = validate_dataframe(sample_data, require_target=True)
    assert valid is True
    assert len(missing) == 0


def test_validate_dataframe_invalid():
    invalid_df = pd.DataFrame({"studytime": [1, 2]})
    valid, missing = validate_dataframe(invalid_df)
    assert valid is False
    assert "absences" in missing


def test_friction_engine_fit_transform(sample_data):
    engine = FrictionEngine()
    df_transformed = engine.fit_transform(sample_data, target_col="G3")

    assert engine.is_fitted is True
    assert "friction_index" in df_transformed.columns
    assert "risk_band" in df_transformed.columns
    assert df_transformed["friction_index"].min() >= 0.0
    assert df_transformed["friction_index"].max() <= 1.0


def test_unfitted_engine_raises_error(sample_data):
    engine = FrictionEngine()
    with pytest.raises(RuntimeError):
        engine.transform(sample_data)


def test_risk_band_assignment():
    engine = FrictionEngine()
    engine.threshold_low = 0.33
    engine.threshold_high = 0.66

    assert engine.assign_risk_band(0.20) == "🟢 Survivable"
    assert engine.assign_risk_band(0.50) == "🟡 Warning"
    assert engine.assign_risk_band(0.80) == "🔴 Unsafe"


# Edge Case Tests for compute_raw_friction_components and FrictionEngine
def test_compute_raw_friction_missing_columns():
    df_missing = pd.DataFrame({
        "studytime": [1, 2],
        "absences": [0, 5],
    })
    with pytest.raises(ValueError, match="DataFrame missing required proxy columns"):
        compute_raw_friction_components(df_missing)


def test_compute_raw_friction_nan_values():
    df_nan = pd.DataFrame({
        "studytime": [1, np.nan, 3, 4],
        "absences": [0, 4, np.nan, 20],
        "goout": [1, np.nan, 4, 5],
        "Dalc": [1, 1, np.nan, 5],
        "Walc": [1, 2, 3, np.nan],
    })
    res = compute_raw_friction_components(df_nan)
    assert not res.isnull().values.any()
    assert len(res) == 4
    for col in FRICTION_COMPONENT_NAMES:
        assert col in res.columns


def test_compute_raw_friction_out_of_range_and_strings():
    df_strings = pd.DataFrame({
        "studytime": ["1", "2", "-5", "invalid"],
        "absences": [0, -10, 10, "99"],
        "goout": [1, 2, 4, 5],
        "Dalc": [1, 1, 2, 5],
        "Walc": [1, 2, 3, 5],
    })
    res = compute_raw_friction_components(df_strings)
    assert not res.isnull().values.any()
    assert (res["schedule_density"] >= 0).all()


def test_compute_raw_friction_empty_dataframe():
    df_empty = pd.DataFrame(columns=["studytime", "absences", "goout", "Dalc", "Walc"])
    res = compute_raw_friction_components(df_empty)
    assert res.empty
    assert list(res.columns) == FRICTION_COMPONENT_NAMES


def test_engine_fit_empty_dataframe_raises():
    df_empty = pd.DataFrame(columns=["studytime", "absences", "goout", "Dalc", "Walc", "age", "Medu", "Fedu", "traveltime", "freetime"])
    engine = FrictionEngine()
    with pytest.raises(ValueError, match="Cannot fit FrictionEngine on an empty DataFrame"):
        engine.fit(df_empty)
