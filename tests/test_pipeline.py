"""
Automated pytest suite for the Satellite-Derived Active Fire Triage Dashboard.
Verifies CSV ingestion, schema validation, coordinate range checks, missing-value
handling, Haversine repeat detection, explainable priority scoring, and functional evaluation.
"""

from pathlib import Path
import pandas as pd

from src.evaluator import run_functional_evaluation
from src.sample_data import (
    ensure_sample_csv_files,
    get_clean_synthetic_records,
    get_mixed_synthetic_records,
)
from src.schema import ScoringWeights
from src.scorer import (
    compute_haversine_matrix_km,
    get_observation_breakdown,
    score_observations,
)
from src.validator import load_and_validate_csv, validate_fire_dataframe


def test_mixed_synthetic_dataset_validation():
    """Verify that the mixed synthetic dataset separates 25 valid and 8 invalid records."""
    df_raw = pd.DataFrame(get_mixed_synthetic_records())
    result = validate_fire_dataframe(df_raw)

    assert result.is_schema_valid is True
    assert result.total_rows == 33
    assert result.valid_count == 25
    assert result.rejected_count == 8
    assert result.warning_count == 3


def test_coordinate_and_type_rejection():
    """Verify invalid latitudes, longitudes, and non-numeric strings are caught."""
    bad_df = pd.DataFrame(
        [
            {
                "obs_id": "E1",
                "latitude": "91.0",
                "longitude": "-120.0",
                "brightness": "340",
                "acq_date": "2026-10-16",
                "confidence": "80",
            },
            {
                "obs_id": "E2",
                "latitude": "40.0",
                "longitude": "-185.0",
                "brightness": "340",
                "acq_date": "2026-10-16",
                "confidence": "80",
            },
            {
                "obs_id": "E3",
                "latitude": "abc",
                "longitude": "-120.0",
                "brightness": "340",
                "acq_date": "2026-10-16",
                "confidence": "80",
            },
        ]
    )
    res = validate_fire_dataframe(bad_df)
    assert res.valid_count == 0
    assert res.rejected_count == 3


def test_explainable_scoring_and_ranking():
    """Verify priority scoring, tier assignment, and factor breakdown sum."""
    clean_df = pd.DataFrame(get_clean_synthetic_records())
    res = validate_fire_dataframe(clean_df)
    weights = ScoringWeights(
        recency=0.25,
        confidence=0.25,
        brightness=0.20,
        frp=0.15,
        repeat_detections=0.15,
        cluster_radius_km=5.0,
    )
    scored = score_observations(res.valid_df, weights=weights)

    assert len(scored) == 25
    assert list(scored["priority_rank"]) == list(range(1, 26))
    # Top ranked should be SYN-001 (Plumas NF intense cluster core)
    top = scored.iloc[0]
    assert top["obs_id"] == "SYN-001"
    assert top["priority_tier"] == "Critical"
    assert top["repeat_count"] == 4
    assert "Primary drivers:" in top["score_explanation"]

    breakdown = get_observation_breakdown(top, weights)
    assert len(breakdown) == 5
    points_sum = round(sum(item["Points Added"] for item in breakdown), 2)
    assert abs(points_sum - top["priority_score"]) < 0.05


def test_csv_file_generation_and_loading(tmp_path: Path):
    """Verify sample CSV files are generated and loaded accurately from disk."""
    paths = ensure_sample_csv_files(tmp_path)
    assert paths["mixed"].exists()
    assert paths["clean"].exists()

    res_mixed = load_and_validate_csv(str(paths["mixed"]))
    assert res_mixed.valid_count == 25
    assert res_mixed.rejected_count == 8


def test_functional_evaluation_all_pass():
    """Ensure all 10 functional evaluation test cases pass."""
    eval_df = run_functional_evaluation()
    assert len(eval_df) == 10
    assert (eval_df["Status"] == "PASS").all()


def test_streamlit_dashboard_renders_without_errors():
    """Verify the Streamlit dashboard renders all tabs and metrics with zero runtime exceptions."""
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not at.exception
    assert len(at.metric) >= 15

