"""
Functional evaluation harness for the Satellite-Derived Active Fire Triage Dashboard.

Executes end-to-end verification tests and produces structured reports comparing
Expected Behavior vs. Actual Behavior as required by PRD Section 2.2 & 4.1.
"""

from typing import Any, Dict, List
import pandas as pd

from src.sample_data import get_mixed_synthetic_records
from src.schema import ScoringWeights
from src.scorer import get_observation_breakdown, score_observations
from src.validator import validate_fire_dataframe


def run_functional_evaluation() -> pd.DataFrame:
    """
    Execute the 10 functional evaluation test cases and return a DataFrame
    documenting Test ID, Requirement Area, Input Scenario, Expected Behavior,
    Actual Behavior, and Status (PASS / FAIL).
    """
    results: List[Dict[str, Any]] = []

    # --- TC-01: VIIRS & MODIS Column Alias Normalization ---
    df_viirs = pd.DataFrame(
        [
            {
                "lat": "40.0",
                "lon": "-121.0",
                "bright_ti4": "380.0",
                "acq_date": "2026-10-16",
                "conf": "h",
                "frp": "120.0",
            }
        ]
    )
    res_01 = validate_fire_dataframe(df_viirs)
    pass_01 = (
        res_01.is_schema_valid
        and res_01.valid_count == 1
        and res_01.valid_df.iloc[0]["confidence_num"] == 95.0
    )
    results.append(
        {
            "Test ID": "TC-01",
            "PRD Requirement": "CSV Parsing & Schema Alias Support",
            "Input Scenario": "VIIRS-style headers (`lat`, `lon`, `bright_ti4`, `conf='h'`)",
            "Expected Behavior": "Schema accepted; mapped to canonical columns; 'h' normalized to 95.0%",
            "Actual Behavior": (
                f"Schema valid={res_01.is_schema_valid}, valid_count={res_01.valid_count}, "
                f"confidence_num={res_01.valid_df.iloc[0]['confidence_num'] if res_01.valid_count else 'N/A'}"
            ),
            "Status": "PASS" if pass_01 else "FAIL",
        }
    )

    # --- TC-02: Missing Required Column Detection ---
    df_missing_col = pd.DataFrame(
        [{"latitude": "40.0", "longitude": "-121.0", "acq_date": "2026-10-16"}]
    )
    res_02 = validate_fire_dataframe(df_missing_col)
    pass_02 = (not res_02.is_schema_valid) and ("brightness" in res_02.missing_required_columns)
    results.append(
        {
            "Test ID": "TC-02",
            "PRD Requirement": "Required-Field Schema Validation",
            "Input Scenario": "CSV missing `brightness` and `confidence` columns",
            "Expected Behavior": "Schema marked invalid; missing columns explicitly listed",
            "Actual Behavior": f"Schema valid={res_02.is_schema_valid}, missing={res_02.missing_required_columns}",
            "Status": "PASS" if pass_02 else "FAIL",
        }
    )

    # --- TC-03: Out-of-Bounds Coordinate Validation ---
    df_coords = pd.DataFrame(
        [
            {
                "obs_id": "LAT-OUT",
                "latitude": "96.45",
                "longitude": "-121.0",
                "brightness": "350.0",
                "acq_date": "2026-10-16",
                "confidence": "80",
            },
            {
                "obs_id": "LON-OUT",
                "latitude": "39.50",
                "longitude": "-195.82",
                "brightness": "350.0",
                "acq_date": "2026-10-16",
                "confidence": "80",
            },
        ]
    )
    res_03 = validate_fire_dataframe(df_coords)
    pass_03 = res_03.valid_count == 0 and res_03.rejected_count == 2
    results.append(
        {
            "Test ID": "TC-03",
            "PRD Requirement": "Latitude & Longitude Range Validation",
            "Input Scenario": "Row 1: lat=96.45 (>90); Row 2: lon=-195.82 (<-180)",
            "Expected Behavior": "Both rows rejected (0 valid, 2 rejected) with coordinate bound errors",
            "Actual Behavior": f"valid={res_03.valid_count}, rejected={res_03.rejected_count}, issues={len(res_03.issues)}",
            "Status": "PASS" if pass_03 else "FAIL",
        }
    )

    # --- TC-04: Non-Numeric & Missing Coordinate Rejection ---
    df_malformed_coord = pd.DataFrame(
        [
            {
                "obs_id": "STR-LAT",
                "latitude": "NOT_A_NUMBER",
                "longitude": "-120.5",
                "brightness": "350.0",
                "acq_date": "2026-10-16",
                "confidence": "75",
            },
            {
                "obs_id": "NULL-LON",
                "latitude": "42.1",
                "longitude": "",
                "brightness": "360.0",
                "acq_date": "2026-10-16",
                "confidence": "82",
            },
        ]
    )
    res_04 = validate_fire_dataframe(df_malformed_coord)
    pass_04 = res_04.valid_count == 0 and res_04.rejected_count == 2
    results.append(
        {
            "Test ID": "TC-04",
            "PRD Requirement": "Data-Type & Required Value Validation",
            "Input Scenario": "Row 1: lat='NOT_A_NUMBER'; Row 2: lon='' (empty)",
            "Expected Behavior": "Both rows rejected with type/missing coordinate error messages",
            "Actual Behavior": f"valid={res_04.valid_count}, rejected={res_04.rejected_count}",
            "Status": "PASS" if pass_04 else "FAIL",
        }
    )

    # --- TC-05: Date Validation & Missing Time Handling ---
    df_dates = pd.DataFrame(
        [
            {
                "obs_id": "BAD-DATE",
                "latitude": "41.2",
                "longitude": "-122.1",
                "brightness": "348.0",
                "acq_date": "2026-13-45",
                "acq_time": "0915",
                "confidence": "70",
            },
            {
                "obs_id": "NO-TIME",
                "latitude": "41.8",
                "longitude": "-123.1",
                "brightness": "359.2",
                "acq_date": "2026-10-16",
                "acq_time": "",
                "confidence": "77",
            },
        ]
    )
    res_05 = validate_fire_dataframe(df_dates)
    pass_05 = (
        res_05.valid_count == 1
        and res_05.rejected_count == 1
        and res_05.valid_df.iloc[0]["acq_time"] == "0000"
        and bool(res_05.valid_df.iloc[0]["has_missing_values"])
    )
    results.append(
        {
            "Test ID": "TC-05",
            "PRD Requirement": "Date Parsing & Missing Time Handling",
            "Input Scenario": "Row 1: acq_date='2026-13-45'; Row 2: valid date, empty acq_time",
            "Expected Behavior": "Row 1 rejected; Row 2 accepted with acq_time='0000' and warning flag",
            "Actual Behavior": (
                f"valid={res_05.valid_count}, rejected={res_05.rejected_count}, "
                f"Row 2 acq_time='{res_05.valid_df.iloc[0]['acq_time'] if res_05.valid_count else ''}'"
            ),
            "Status": "PASS" if pass_05 else "FAIL",
        }
    )

    # --- TC-06: Physical Bounds & Confidence Range Checks ---
    df_bounds = pd.DataFrame(
        [
            {
                "obs_id": "NEG-BRIGHT",
                "latitude": "40.1",
                "longitude": "-121.5",
                "brightness": "-45.0",
                "acq_date": "2026-10-16",
                "confidence": "90",
                "frp": "55.0",
            },
            {
                "obs_id": "HIGH-CONF",
                "latitude": "38.5",
                "longitude": "-120.8",
                "brightness": "345.0",
                "acq_date": "2026-10-16",
                "confidence": "145",
                "frp": "35.0",
            },
            {
                "obs_id": "NEG-FRP",
                "latitude": "37.8",
                "longitude": "-119.5",
                "brightness": "352.0",
                "acq_date": "2026-10-16",
                "confidence": "80",
                "frp": "-25.4",
            },
        ]
    )
    res_06 = validate_fire_dataframe(df_bounds)
    pass_06 = res_06.valid_count == 0 and res_06.rejected_count == 3
    results.append(
        {
            "Test ID": "TC-06",
            "PRD Requirement": "Sensor Range & Measurement Validation",
            "Input Scenario": "3 rows with brightness=-45K, confidence=145, and frp=-25.4 MW",
            "Expected Behavior": "All 3 rows rejected for violating physical/sensor measurement bounds",
            "Actual Behavior": f"valid={res_06.valid_count}, rejected={res_06.rejected_count}",
            "Status": "PASS" if pass_06 else "FAIL",
        }
    )

    # --- TC-07: Missing Optional FRP Imputation & Flagging ---
    df_missing_frp = pd.DataFrame(
        [
            {
                "obs_id": "NO-FRP",
                "latitude": "40.74",
                "longitude": "-122.94",
                "brightness": "374.5",
                "acq_date": "2026-10-16",
                "acq_time": "1330",
                "confidence": "88",
                "frp": "",
            }
        ]
    )
    res_07 = validate_fire_dataframe(df_missing_frp)
    scored_07 = score_observations(res_07.valid_df)
    pass_07 = (
        res_07.valid_count == 1
        and bool(scored_07.iloc[0]["frp_missing"])
        and scored_07.iloc[0]["score_frp"] == 0.0
        and scored_07.iloc[0]["quality_status"] == "Partial (Imputed)"
    )
    results.append(
        {
            "Test ID": "TC-07",
            "PRD Requirement": "Missing-Value Handling (Optional Telemetry)",
            "Input Scenario": "Valid coordinates & brightness, but empty `frp` field",
            "Expected Behavior": "Record retained; frp imputed as 0.0 MW; flagged as 'Partial (Imputed)'",
            "Actual Behavior": (
                f"valid={res_07.valid_count}, status='{scored_07.iloc[0]['quality_status']}', "
                f"score_frp={scored_07.iloc[0]['score_frp']}"
            ),
            "Status": "PASS" if pass_07 else "FAIL",
        }
    )

    # --- TC-08: Spatial Repeat-Detection Clustering (Haversine) ---
    mixed_df = pd.DataFrame(get_mixed_synthetic_records())
    res_mixed = validate_fire_dataframe(mixed_df)
    scored_mixed = score_observations(res_mixed.valid_df, ScoringWeights(cluster_radius_km=5.0))
    syn001_row = scored_mixed[scored_mixed["obs_id"] == "SYN-001"].iloc[0]
    syn018_row = scored_mixed[scored_mixed["obs_id"] == "SYN-018"].iloc[0]
    pass_08 = int(syn001_row["repeat_count"]) == 4 and int(syn018_row["repeat_count"]) == 0
    results.append(
        {
            "Test ID": "TC-08",
            "PRD Requirement": "Repeat-Detection Spatial Clustering",
            "Input Scenario": "5 Plumas NF detections within 2 km (`SYN-001`..`005`) vs isolated Mojave pixel (`SYN-018`)",
            "Expected Behavior": "SYN-001 detects 4 nearby observations within 5 km; SYN-018 detects 0",
            "Actual Behavior": f"SYN-001 repeat_count={int(syn001_row['repeat_count'])}, SYN-018 repeat_count={int(syn018_row['repeat_count'])}",
            "Status": "PASS" if pass_08 else "FAIL",
        }
    )

    # --- TC-09: Explainable Priority Scoring & Ranking Order ---
    rank1_obs = scored_mixed.iloc[0]
    last_obs = scored_mixed.iloc[-1]
    breakdown = get_observation_breakdown(rank1_obs, ScoringWeights())
    pass_09 = (
        rank1_obs["obs_id"] == "SYN-001"
        and rank1_obs["priority_tier"] == "Critical"
        and rank1_obs["priority_score"] > last_obs["priority_score"]
        and len(breakdown) == 5
    )
    results.append(
        {
            "Test ID": "TC-09",
            "PRD Requirement": "Explainable Priority Scoring & Ranking",
            "Input Scenario": "Rank all 25 valid synthetic observations using default weights",
            "Expected Behavior": "SYN-001 ranks #1 (Critical tier) with 5-factor breakdown; low-conf old anomaly ranks last",
            "Actual Behavior": (
                f"Rank #1={rank1_obs['obs_id']} ({rank1_obs['priority_score']}/100, {rank1_obs['priority_tier']}); "
                f"Rank #{len(scored_mixed)}={last_obs['obs_id']} ({last_obs['priority_score']}/100)"
            ),
            "Status": "PASS" if pass_09 else "FAIL",
        }
    )

    # --- TC-10: End-to-End Mixed Dataset Audit ---
    pass_10 = (
        res_mixed.total_rows == 33
        and res_mixed.valid_count == 25
        and res_mixed.rejected_count == 8
        and res_mixed.warning_count == 3
    )
    results.append(
        {
            "Test ID": "TC-10",
            "PRD Requirement": "End-to-End Synthetic Dataset Triage Audit",
            "Input Scenario": "Full 33-row Synthetic Mixed Dataset (`synthetic_firms_triage_mixed.csv`)",
            "Expected Behavior": "33 total -> 25 valid (22 complete + 3 partial/imputed) and 8 rejected records",
            "Actual Behavior": (
                f"total={res_mixed.total_rows}, valid={res_mixed.valid_count} "
                f"(warnings={res_mixed.warning_count}), rejected={res_mixed.rejected_count}"
            ),
            "Status": "PASS" if pass_10 else "FAIL",
        }
    )

    return pd.DataFrame(results)
