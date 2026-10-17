"""
Explainable priority scoring engine for satellite active-fire observations.

Computes normalized sub-scores (0-100), weighted point contributions, spatial repeat
detection clusters via Haversine distance, priority tiers, deterministic rankings,
and human-readable explanations for each observation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.schema import PRIORITY_TIERS, ScoringWeights


EARTH_RADIUS_KM = 6371.0088


def compute_haversine_matrix_km(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Haversine distance matrix (in km) for arrays of latitudes and longitudes.
    """
    if len(latitudes) == 0:
        return np.empty((0, 0), dtype=float)

    lat_rad = np.radians(latitudes.astype(float))
    lon_rad = np.radians(longitudes.astype(float))

    dlat = lat_rad[:, np.newaxis] - lat_rad[np.newaxis, :]
    dlon = lon_rad[:, np.newaxis] - lon_rad[np.newaxis, :]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_rad[:, np.newaxis]) * np.cos(lat_rad[np.newaxis, :]) * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def assign_priority_tier(score: float) -> Tuple[str, str]:
    """Return (tier_name, hex_color) for a 0-100 priority score."""
    for tier_name, threshold, color in PRIORITY_TIERS:
        if score >= threshold:
            return tier_name, color
    return "Low", "#3B82F6"


def _build_explanation_text(row: pd.Series, norm_weights: Dict[str, float]) -> Tuple[str, str]:
    """
    Create a concise human-readable explanation and top drivers summary for an observation.
    Returns (top_drivers_short, full_explanation_sentence).
    """
    contributions = [
        ("Recency", row["contrib_recency"], f"{row['hours_since_latest']:.1f}h ago"),
        ("Confidence", row["contrib_confidence"], f"{row['confidence_num']:.0f}%"),
        ("Brightness", row["contrib_brightness"], f"{row['brightness']:.1f}K"),
        ("FRP", row["contrib_frp"], "missing/0 MW" if row.get("frp_missing", False) else f"{row['frp']:.1f} MW"),
        ("Repeat Detections", row["contrib_repeat"], f"{int(row['repeat_count'])} nearby"),
    ]
    contributions_sorted = sorted(contributions, key=lambda x: x[1], reverse=True)
    top_two = contributions_sorted[:2]
    top_drivers = ", ".join(
        f"{name} (+{pts:.1f} pts)" for name, pts, _ in top_two if pts > 0
    )
    if not top_drivers:
        top_drivers = "Baseline minimal signals"

    parts = [
        f"Scored {row['priority_score']:.1f}/100 ({row['priority_tier']} priority).",
        f"Primary drivers: {top_two[0][0]} ({top_two[0][2]}, +{top_two[0][1]:.1f} pts)",
    ]
    if len(top_two) > 1 and top_two[1][1] > 0:
        parts.append(f"and {top_two[1][0]} ({top_two[1][2]}, +{top_two[1][1]:.1f} pts).")
    else:
        parts[-1] += "."

    if int(row["repeat_count"]) > 0:
        parts.append(
            f"Part of a {int(row['repeat_count']) + 1}-detection spatial cluster within {row['cluster_radius_km']:.1f} km."
        )
    else:
        parts.append("Isolated detection (no neighbors within cluster radius).")

    if row.get("has_missing_values", False):
        parts.append(f"Data quality note: {row.get('quality_notes', 'Partial record')}.")

    return top_drivers, " ".join(parts)


def score_observations(
    valid_df: pd.DataFrame,
    weights: Optional[ScoringWeights] = None,
    reference_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Calculate explainable sub-scores, weighted contributions, priority tiers,
    and rankings for all validated observations.
    """
    if valid_df.empty:
        return valid_df.copy()

    if weights is None:
        weights = ScoringWeights()
    norm_w = weights.normalized()

    df = valid_df.copy().reset_index(drop=True)

    # 1. Detection Recency Sub-Score (0 - 100)
    if reference_time is None:
        reference_time = df["acq_datetime"].max()

    hours_diff = (
        (reference_time - df["acq_datetime"]).dt.total_seconds() / 3600.0
    ).clip(lower=0.0)
    df["hours_since_latest"] = hours_diff.round(2)

    half_life = max(1.0, float(weights.recency_half_life_hours))
    df["score_recency"] = (100.0 * np.power(2.0, -df["hours_since_latest"] / half_life)).clip(
        0.0, 100.0
    )

    # 2. Confidence Sub-Score (0 - 100)
    df["score_confidence"] = df["confidence_num"].astype(float).clip(0.0, 100.0)

    # 3. Brightness Sub-Score (0 - 100): 300K baseline -> 0, 420K intense -> 100
    df["score_brightness"] = (
        (df["brightness"].astype(float) - 300.0) / (420.0 - 300.0) * 100.0
    ).clip(0.0, 100.0)

    # 4. Fire Radiative Power (FRP) Sub-Score (0 - 100): Log-scaled up to 500 MW
    frp_vals = df["frp"].astype(float).clip(lower=0.0)
    df["score_frp"] = (np.log1p(frp_vals) / np.log1p(500.0) * 100.0).clip(0.0, 100.0)
    if "frp_missing" in df.columns:
        df.loc[df["frp_missing"], "score_frp"] = 0.0

    # 5. Repeat Detections / Spatial Persistence Sub-Score (0 - 100)
    dist_matrix = compute_haversine_matrix_km(
        df["latitude"].to_numpy(), df["longitude"].to_numpy()
    )
    radius_km = max(0.1, float(weights.cluster_radius_km))
    # Exclude self-distance on diagonal
    within_radius = (dist_matrix <= radius_km) & (~np.eye(len(df), dtype=bool))
    repeat_counts = within_radius.sum(axis=1)

    df["repeat_count"] = repeat_counts.astype(int)
    df["cluster_radius_km"] = radius_km
    # Exponential saturation curve: 0->0, 1->45.1, 2->69.9, 3->83.5, 4->90.9, 5+->95-100
    df["score_repeat"] = np.where(
        df["repeat_count"] > 0,
        np.minimum(100.0, 100.0 * (1.0 - np.exp(-0.60 * df["repeat_count"]))),
        0.0,
    )

    # Store neighbor observation IDs for explainability
    obs_ids = df["obs_id"].tolist()
    neighbor_ids_list: List[str] = []
    for i in range(len(df)):
        nbr_indices = np.where(within_radius[i])[0]
        nbr_ids = [obs_ids[j] for j in nbr_indices]
        neighbor_ids_list.append(", ".join(nbr_ids[:8]) + ("..." if len(nbr_ids) > 8 else ""))
    df["nearby_obs_ids"] = neighbor_ids_list

    # Weighted Point Contributions
    df["contrib_recency"] = (df["score_recency"] * norm_w["recency"]).round(2)
    df["contrib_confidence"] = (df["score_confidence"] * norm_w["confidence"]).round(2)
    df["contrib_brightness"] = (df["score_brightness"] * norm_w["brightness"]).round(2)
    df["contrib_frp"] = (df["score_frp"] * norm_w["frp"]).round(2)
    df["contrib_repeat"] = (df["score_repeat"] * norm_w["repeat_detections"]).round(2)

    df["priority_score"] = (
        df["contrib_recency"]
        + df["contrib_confidence"]
        + df["contrib_brightness"]
        + df["contrib_frp"]
        + df["contrib_repeat"]
    ).clip(0.0, 100.0).round(2)

    # Assign Priority Tier and Color
    tiers_and_colors = [assign_priority_tier(s) for s in df["priority_score"]]
    df["priority_tier"] = [tc[0] for tc in tiers_and_colors]
    df["tier_color"] = [tc[1] for tc in tiers_and_colors]

    # Sort deterministically and assign Priority Rank
    df = df.sort_values(
        by=["priority_score", "confidence_num", "brightness", "frp"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    df["priority_rank"] = np.arange(1, len(df) + 1)

    # Build human-readable explanations
    drivers_list: List[str] = []
    explanations_list: List[str] = []
    for _, row in df.iterrows():
        top_drv, expl = _build_explanation_text(row, norm_w)
        drivers_list.append(top_drv)
        explanations_list.append(expl)

    df["top_drivers"] = drivers_list
    df["score_explanation"] = explanations_list

    return df


def get_observation_breakdown(
    row: pd.Series, weights: ScoringWeights
) -> List[Dict[str, Any]]:
    """
    Return a structured factor-by-factor breakdown for a single scored observation
    for rendering in the Observation Detail & Explainability view.
    """
    norm_w = weights.normalized()
    return [
        {
            "Factor": "Detection Recency",
            "Raw Measurement": f"{row['hours_since_latest']:.1f} hrs before latest ({row['acq_date']} {row['acq_time']} UTC)",
            "Sub-Score (0–100)": round(float(row["score_recency"]), 1),
            "Weight (%)": round(norm_w["recency"] * 100.0, 1),
            "Points Added": round(float(row["contrib_recency"]), 2),
            "Max Possible Points": round(norm_w["recency"] * 100.0, 2),
            "Rationale": f"Exponential decay with {weights.recency_half_life_hours:.0f}h half-life relative to latest observation.",
        },
        {
            "Factor": "Sensor Confidence",
            "Raw Measurement": f"{row['confidence_raw']} ({row['confidence_label']})",
            "Sub-Score (0–100)": round(float(row["score_confidence"]), 1),
            "Weight (%)": round(norm_w["confidence"] * 100.0, 1),
            "Points Added": round(float(row["contrib_confidence"]), 2),
            "Max Possible Points": round(norm_w["confidence"] * 100.0, 2),
            "Rationale": "Normalized confidence (0–100 scale; VIIRS l/n/h mapped to 30/65/95).",
        },
        {
            "Factor": "Thermal Brightness",
            "Raw Measurement": f"{row['brightness']:.1f} K",
            "Sub-Score (0–100)": round(float(row["score_brightness"]), 1),
            "Weight (%)": round(norm_w["brightness"] * 100.0, 1),
            "Points Added": round(float(row["contrib_brightness"]), 2),
            "Max Possible Points": round(norm_w["brightness"] * 100.0, 2),
            "Rationale": "Linearly scaled between 300 K (ambient anomaly threshold) and 420 K (high-intensity anomaly).",
        },
        {
            "Factor": "Fire Radiative Power (FRP)",
            "Raw Measurement": "Missing (Imputed 0.0 MW)"
            if row.get("frp_missing", False)
            else f"{row['frp']:.1f} MW",
            "Sub-Score (0–100)": round(float(row["score_frp"]), 1),
            "Weight (%)": round(norm_w["frp"] * 100.0, 1),
            "Points Added": round(float(row["contrib_frp"]), 2),
            "Max Possible Points": round(norm_w["frp"] * 100.0, 2),
            "Rationale": "Log-scaled radiative heat output (0 to 500+ MW); 0 pts when missing.",
        },
        {
            "Factor": "Repeat Detections (Cluster)",
            "Raw Measurement": f"{int(row['repeat_count'])} nearby within {row['cluster_radius_km']:.1f} km",
            "Sub-Score (0–100)": round(float(row["score_repeat"]), 1),
            "Weight (%)": round(norm_w["repeat_detections"] * 100.0, 1),
            "Points Added": round(float(row["contrib_repeat"]), 2),
            "Max Possible Points": round(norm_w["repeat_detections"] * 100.0, 2),
            "Rationale": "Spatial persistence based on Haversine distance to neighboring observations.",
        },
    ]
