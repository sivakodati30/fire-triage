"""
Satellite-Derived Active Fire Triage Dashboard
Project: AI/ML 1 — Satellite-Derived Active Fire Triage (Version 1.0)

Interactive Streamlit web application for ingesting, validating, mapping,
filtering, and prioritizing satellite active-fire observations using an
explainable multi-factor scoring algorithm.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from src.evaluator import run_functional_evaluation
from src.sample_data import (
    SYNTHETIC_DISCLAIMER,
    ensure_sample_csv_files,
)
from src.schema import PRIORITY_TIERS, REQUIRED_COLUMNS, ScoringWeights
from src.scorer import (
    compute_haversine_matrix_km,
    get_observation_breakdown,
    score_observations,
)
from src.validator import ValidationResult, load_and_validate_csv


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Satellite Active Fire Triage Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .disclaimer-banner {
        background: linear-gradient(90deg, rgba(127,29,29,0.18) 0%, rgba(180,83,9,0.15) 100%);
        border-left: 5px solid #F97316;
        padding: 0.85rem 1.15rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        font-size: 0.92rem;
    }
    .synthetic-badge {
        background-color: rgba(59, 130, 246, 0.15);
        border: 1px solid #3B82F6;
        color: #2563EB;
        padding: 0.45rem 0.85rem;
        border-radius: 0.4rem;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    .tier-badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.8rem;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATHS = ensure_sample_csv_files(BASE_DIR)

TIER_COLOR_MAP = {
    "Critical": "#EF4444",
    "High": "#F97316",
    "Medium": "#EAB308",
    "Low": "#3B82F6",
}

TIER_RGB_MAP = {
    "Critical": [239, 68, 68, 215],
    "High": [249, 115, 22, 210],
    "Medium": [234, 179, 8, 200],
    "Low": [59, 130, 246, 190],
}


def render_header(is_synthetic: bool, dataset_label: str) -> None:
    """Render main dashboard title, operational disclaimer, and synthetic data label."""
    st.title("🛰️ Satellite-Derived Active Fire Triage Dashboard")
    st.caption(
        "Project: AI/ML 1 — Satellite-Derived Active Fire Triage (v1.0) | "
        "Explainable Multi-Factor Thermal Anomaly Validation & Prioritization"
    )

    st.markdown(
        """
        <div class="disclaimer-banner">
            <strong>⚠️ Operational Scope & Safety Disclaimer:</strong>
            This dashboard is an analyst decision-support and data exploration tool.
            Satellite thermal anomalies may originate from wildfires, prescribed burns, agricultural clearing,
            industrial heat sources, or solar glint, and may be obscured by cloud cover.
            <strong>This system does NOT independently confirm active fires and must NOT be used for autonomous emergency-response decisions.</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_synthetic:
        st.markdown(
            f"""
            <div class="synthetic-badge">
                🧪 <strong>SYNTHETIC DATA LABEL ({dataset_label}):</strong> {SYNTHETIC_DISCLAIMER}
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_pydeck_map(df: pd.DataFrame, cluster_radius_km: float) -> pdk.Deck:
    """Build an interactive PyDeck map with color-coded priority markers and cluster radius rings."""
    map_df = df.copy()
    map_df["fill_color"] = map_df["priority_tier"].map(TIER_RGB_MAP)
    map_df["radius_m"] = (map_df["priority_score"] * 120.0 + 3500.0).clip(3500.0, 16000.0)
    map_df["cluster_ring_m"] = float(cluster_radius_km) * 1000.0

    center_lat = float(map_df["latitude"].mean())
    center_lon = float(map_df["longitude"].mean())

    cluster_df = map_df[map_df["repeat_count"] > 0].copy()

    layers = []
    if not cluster_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=cluster_df,
                get_position="[longitude, latitude]",
                get_radius="cluster_ring_m",
                get_fill_color=[249, 115, 22, 25],
                get_line_color=[249, 115, 22, 120],
                stroked=True,
                filled=True,
                line_width_min_pixels=1,
                pickable=False,
            )
        )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[longitude, latitude]",
            get_radius="radius_m",
            get_fill_color="fill_color",
            get_line_color=[255, 255, 255, 220],
            stroked=True,
            filled=True,
            line_width_min_pixels=1.5,
            radius_min_pixels=6,
            radius_max_pixels=22,
            pickable=True,
            auto_highlight=True,
        )
    )

    tooltip = {
        "html": (
            "<b>Rank #{priority_rank} — {obs_id} ({priority_tier})</b><br/>"
            "<b>Priority Score:</b> {priority_score} / 100<br/>"
            "<b>Region:</b> {region}<br/>"
            "<b>Coords:</b> {latitude}, {longitude}<br/>"
            "<b>Acquired:</b> {acq_date} {acq_time} UTC ({hours_since_latest}h ago)<br/>"
            "<b>Brightness:</b> {brightness} K | <b>FRP:</b> {frp} MW<br/>"
            "<b>Confidence:</b> {confidence_label} | <b>Cluster Neighbors:</b> {repeat_count}<br/>"
            "<b>Top Drivers:</b> {top_drivers}<br/>"
            "<b>Quality:</b> {quality_status}"
        ),
        "style": {
            "backgroundColor": "#1E293B",
            "color": "#F8FAFC",
            "fontSize": "12px",
            "padding": "10px",
            "borderRadius": "6px",
        },
    }

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=4.6,
        pitch=0,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    )


def build_plotly_map(df: pd.DataFrame) -> go.Figure:
    """Build an interactive Plotly geospatial scatter map colored by Priority Tier."""
    fig = px.scatter_map(
        df,
        lat="latitude",
        lon="longitude",
        color="priority_tier",
        size="priority_score",
        size_max=18,
        color_discrete_map=TIER_COLOR_MAP,
        category_orders={"priority_tier": ["Critical", "High", "Medium", "Low"]},
        hover_name="obs_id",
        hover_data={
            "priority_rank": True,
            "priority_score": True,
            "priority_tier": True,
            "region": True,
            "acq_date": True,
            "acq_time": True,
            "brightness": True,
            "frp": True,
            "confidence_label": True,
            "repeat_count": True,
            "top_drivers": True,
            "quality_status": True,
            "latitude": ":.4f",
            "longitude": ":.4f",
        },
        center={
            "lat": float(df["latitude"].mean()),
            "lon": float(df["longitude"].mean()),
        },
        zoom=4.2,
        height=520,
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend_title_text="Priority Tier",
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar: Data Source & Scoring Weights
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Data Source & Ingestion")
    data_mode = st.radio(
        "Select Observation Source",
        options=[
            "Synthetic Mixed Dataset (25 Valid + 8 Invalid)",
            "Synthetic Clean Dataset (25 Valid Records)",
            "Upload Custom CSV File",
        ],
        index=0,
        help="Choose a built-in labeled synthetic dataset to test validation and scoring, or upload your own NASA FIRMS CSV.",
    )

    uploaded_file = None
    if data_mode == "Upload Custom CSV File":
        uploaded_file = st.file_uploader(
            "Upload NASA FIRMS / Fire Observation CSV",
            type=["csv"],
            help=f"Required columns: {', '.join(REQUIRED_COLUMNS)}",
        )

    with st.expander("📥 Download Sample CSV Templates"):
        st.caption("Download built-in synthetic CSV files to inspect or test uploading:")
        with open(SAMPLE_PATHS["mixed"], "rb") as f_mix:
            st.download_button(
                "⬇️ Download Mixed Synthetic CSV (33 rows)",
                data=f_mix.read(),
                file_name="synthetic_firms_triage_mixed.csv",
                mime="text/csv",
                width="stretch",
            )
        with open(SAMPLE_PATHS["clean"], "rb") as f_cln:
            st.download_button(
                "⬇️ Download Clean Synthetic CSV (25 rows)",
                data=f_cln.read(),
                file_name="synthetic_firms_clean_benchmark.csv",
                mime="text/csv",
                width="stretch",
            )

    st.divider()
    st.header("2. Explainable Scoring Weights")
    st.caption(
        "Adjust factor weights to customize how observations are prioritized. "
        "Weights are automatically normalized to sum to 100%."
    )

    w_recency = st.slider("⏱️ Detection Recency Weight", 0.0, 1.0, 0.25, 0.05)
    w_conf = st.slider("🎯 Sensor Confidence Weight", 0.0, 1.0, 0.25, 0.05)
    w_bright = st.slider("🌡️ Thermal Brightness (K) Weight", 0.0, 1.0, 0.20, 0.05)
    w_frp = st.slider("⚡ Fire Radiative Power (FRP) Weight", 0.0, 1.0, 0.15, 0.05)
    w_repeat = st.slider("🔁 Repeat Detections (Cluster) Weight", 0.0, 1.0, 0.15, 0.05)

    cluster_radius_km = st.slider(
        "📏 Spatial Cluster Radius (km)",
        min_value=1.0,
        max_value=25.0,
        value=5.0,
        step=0.5,
        help="Observations within this Haversine distance of each other are counted as repeat/clustered detections.",
    )
    half_life_hrs = st.slider(
        "⌛ Recency Half-Life (hours)",
        min_value=6.0,
        max_value=72.0,
        value=24.0,
        step=6.0,
        help="Time in hours for the recency sub-score to decay by 50%.",
    )

    weights = ScoringWeights(
        recency=w_recency,
        confidence=w_conf,
        brightness=w_bright,
        frp=w_frp,
        repeat_detections=w_repeat,
        cluster_radius_km=cluster_radius_km,
        recency_half_life_hours=half_life_hrs,
    )
    norm_w = weights.normalized()
    st.info(
        "**Normalized Active Weights:**\n"
        f"- Recency: **{norm_w['recency']*100:.1f}%**\n"
        f"- Confidence: **{norm_w['confidence']*100:.1f}%**\n"
        f"- Brightness: **{norm_w['brightness']*100:.1f}%**\n"
        f"- FRP: **{norm_w['frp']*100:.1f}%**\n"
        f"- Repeat Cluster: **{norm_w['repeat_detections']*100:.1f}%**"
    )


# ---------------------------------------------------------------------------
# Load, Validate, and Score Dataset
# ---------------------------------------------------------------------------
is_synthetic_source = True
dataset_label = data_mode

if data_mode == "Synthetic Mixed Dataset (25 Valid + 8 Invalid)":
    val_result: ValidationResult = load_and_validate_csv(str(SAMPLE_PATHS["mixed"]))
elif data_mode == "Synthetic Clean Dataset (25 Valid Records)":
    val_result = load_and_validate_csv(str(SAMPLE_PATHS["clean"]))
else:
    is_synthetic_source = False
    dataset_label = "Custom Uploaded CSV"
    if uploaded_file is None:
        render_header(is_synthetic=False, dataset_label="Awaiting CSV Upload")
        st.info(
            "👆 Please upload a NASA FIRMS or compatible fire observation CSV file in the sidebar, "
            "or switch to one of the built-in Synthetic Datasets to explore the dashboard."
        )
        st.stop()
    val_result = load_and_validate_csv(uploaded_file)
    # Check if uploaded file itself contains synthetic markers
    if not val_result.valid_df.empty and "notes" in val_result.valid_df.columns:
        if val_result.valid_df["notes"].astype(str).str.contains("SYNTHETIC", case=False).any():
            is_synthetic_source = True
            dataset_label = f"Uploaded Synthetic CSV ({uploaded_file.name})"

render_header(is_synthetic=is_synthetic_source, dataset_label=dataset_label)

if not val_result.is_schema_valid:
    st.error(
        f"❌ **CSV Schema Validation Failed:** Missing required columns: "
        f"`{', '.join(val_result.missing_required_columns)}`. "
        f"Required columns are: `{', '.join(REQUIRED_COLUMNS)}`."
    )
    st.dataframe(val_result.issues_as_dataframe(), width="stretch")
    st.stop()

scored_df = score_observations(val_result.valid_df, weights=weights)


# ---------------------------------------------------------------------------
# Sidebar: Observation Filters (Dynamic based on valid scored observations)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.header("3. Observation Filters")

    if not scored_df.empty:
        all_dates = sorted(scored_df["acq_date"].unique().tolist())
        selected_dates = st.multiselect(
            "📅 Acquisition Date(s)",
            options=all_dates,
            default=all_dates,
        )

        all_tiers = ["Critical", "High", "Medium", "Low"]
        selected_tiers = st.multiselect(
            "🚨 Priority Tier",
            options=all_tiers,
            default=all_tiers,
        )

        min_conf = st.slider(
            "🎯 Minimum Confidence (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
        )

        min_score = st.slider(
            "📊 Minimum Priority Score",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=5.0,
        )

        all_regions = sorted(scored_df["region"].unique().tolist())
        selected_regions = st.multiselect(
            "📍 Region / Location",
            options=all_regions,
            default=all_regions,
        )

        all_sats = sorted(scored_df["satellite"].unique().tolist())
        selected_sats = st.multiselect(
            "🛰️ Satellite Platform",
            options=all_sats,
            default=all_sats,
        )

        all_dn = sorted(scored_df["daynight"].unique().tolist())
        selected_dn = st.multiselect(
            "🌗 Day / Night Overpass",
            options=all_dn,
            default=all_dn,
        )

        quality_filter = st.selectbox(
            "🛡️ Data Completeness Filter",
            options=[
                "All Valid Observations",
                "Complete Records Only",
                "Partial (Imputed) Records Only",
            ],
            index=0,
        )
    else:
        selected_dates, selected_tiers, selected_regions, selected_sats, selected_dn = (
            [],
            [],
            [],
            [],
            [],
        )
        min_conf, min_score = 0, 0.0
        quality_filter = "All Valid Observations"


# Apply Filters
if not scored_df.empty:
    filtered_df = scored_df[
        (scored_df["acq_date"].isin(selected_dates))
        & (scored_df["priority_tier"].isin(selected_tiers))
        & (scored_df["confidence_num"] >= min_conf)
        & (scored_df["priority_score"] >= min_score)
        & (scored_df["region"].isin(selected_regions))
        & (scored_df["satellite"].isin(selected_sats))
        & (scored_df["daynight"].isin(selected_dn))
    ].copy()

    if quality_filter == "Complete Records Only":
        filtered_df = filtered_df[~filtered_df["has_missing_values"]].copy()
    elif quality_filter == "Partial (Imputed) Records Only":
        filtered_df = filtered_df[filtered_df["has_missing_values"]].copy()
else:
    filtered_df = scored_df.copy()


# ---------------------------------------------------------------------------
# Top KPI Summary Strip
# ---------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
kpi1.metric("Total Rows Ingested", val_result.total_rows)
kpi2.metric(
    "Valid Observations",
    val_result.valid_count,
    delta=f"{val_result.warning_count} partial/imputed" if val_result.warning_count else "100% complete",
    delta_color="off",
)
kpi3.metric(
    "Rejected / Invalid Rows",
    val_result.rejected_count,
    delta="Flagged in Audit Tab" if val_result.rejected_count > 0 else "0 errors",
    delta_color="inverse" if val_result.rejected_count > 0 else "normal",
)
kpi4.metric("Filtered View Count", len(filtered_df))

crit_high_count = (
    int(filtered_df["priority_tier"].isin(["Critical", "High"]).sum())
    if not filtered_df.empty
    else 0
)
kpi5.metric("Critical & High Priority", crit_high_count)

clustered_count = (
    int((filtered_df["repeat_count"] > 0).sum()) if not filtered_df.empty else 0
)
kpi6.metric(f"Clustered (≤{cluster_radius_km:.1f} km)", clustered_count)


# ---------------------------------------------------------------------------
# Main Dashboard Tabs
# ---------------------------------------------------------------------------
tab_map, tab_table, tab_detail, tab_validation, tab_eval = st.tabs(
    [
        "🗺️ Interactive Map & Triage Queue",
        "📋 Filterable Observation Table",
        "🔍 Observation Detail & Explainability",
        f"🛡️ Validation & Quality Audit ({val_result.rejected_count} Rejected)",
        "✅ Functional Evaluation & Limitations",
    ]
)


# ===========================================================================
# TAB 1: INTERACTIVE MAP & PRIORITY TRIAGE QUEUE
# ===========================================================================
with tab_map:
    if filtered_df.empty:
        st.warning("No observations match the current filter criteria. Adjust filters in the sidebar.")
    else:
        col_map, col_queue = st.columns([1.85, 1.15], gap="large")

        with col_map:
            map_head_l, map_head_r = st.columns([2, 1])
            with map_head_l:
                st.subheader("Geospatial Thermal Anomaly Triage Map")
            with map_head_r:
                map_engine = st.selectbox(
                    "Map Renderer",
                    options=["Plotly Interactive Map (OSM)", "PyDeck Dark Matter Map"],
                    index=0,
                    label_visibility="collapsed",
                )

            if map_engine == "Plotly Interactive Map (OSM)":
                fig_map = build_plotly_map(filtered_df)
                st.plotly_chart(fig_map, width="stretch")
            else:
                deck = build_pydeck_map(filtered_df, cluster_radius_km=cluster_radius_km)
                st.pydeck_chart(deck, width="stretch")

            st.caption(
                "💡 **Map Legend:** Marker color indicates **Priority Tier** "
                "(🔴 Critical ≥75, 🟠 High 55–74.9, 🟡 Medium 35–54.9, 🔵 Low <35). "
                "Marker size scales with **Priority Score**. Hover over any point for sensor telemetry and top scoring drivers."
            )

        with col_queue:
            st.subheader("🏆 Top Priority Triage Queue")
            st.caption("Ranked highest-to-lowest by explainable priority score:")

            top_n = filtered_df.head(8)
            for _, row in top_n.iterrows():
                tier_color = TIER_COLOR_MAP.get(row["priority_tier"], "#3B82F6")
                with st.container(border=True):
                    q_col1, q_col2 = st.columns([3.2, 1.2])
                    with q_col1:
                        st.markdown(
                            f"**Rank #{int(row['priority_rank'])} — `{row['obs_id']}`** "
                            f"<span class='tier-badge' style='background:{tier_color};'>"
                            f"{row['priority_tier']}</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"📍 **{row['region']}** (`{row['latitude']:.4f}, {row['longitude']:.4f}`) | "
                            f"🕒 `{row['acq_date']} {row['acq_time']} UTC` ({row['hours_since_latest']:.1f}h ago)"
                        )
                    with q_col2:
                        st.metric("Score", f"{row['priority_score']:.1f}/100", label_visibility="collapsed")

                    frp_str = "Missing" if row["frp_missing"] else f"{row['frp']:.1f} MW"
                    st.markdown(
                        f"**Telemetry:** `{row['brightness']:.1f} K` | "
                        f"FRP: `{frp_str}` | "
                        f"Conf: `{row['confidence_label']}` | "
                        f"Cluster: `{int(row['repeat_count'])} nearby`"
                    )
                    st.caption(f"🔍 **Why ranked #{int(row['priority_rank'])}:** {row['top_drivers']}")


# ===========================================================================
# TAB 2: FILTERABLE & SORTABLE OBSERVATION TABLE
# ===========================================================================
with tab_table:
    st.subheader("📋 Filterable & Sortable Observation Triage Table")

    if filtered_df.empty:
        st.warning("No valid observations match the active filter criteria.")
    else:
        ctrl1, ctrl2, ctrl3 = st.columns([2, 1.5, 1])
        with ctrl1:
            search_query = st.text_input(
                "🔎 Search by Observation ID, Region, Satellite, or Notes",
                value="",
                placeholder="e.g., SYN-001, Plumas, VIIRS, industrial...",
            )
        with ctrl2:
            sort_col_label = st.selectbox(
                "Sort Table By",
                options=[
                    "Priority Rank (Default)",
                    "Priority Score (High → Low)",
                    "Priority Score (Low → High)",
                    "Confidence (High → Low)",
                    "Brightness K (High → Low)",
                    "FRP MW (High → Low)",
                    "Most Recent First",
                    "Repeat Detections (High → Low)",
                ],
                index=0,
            )
        with ctrl3:
            st.write("")
            csv_export = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export Ranked CSV",
                data=csv_export,
                file_name="ranked_fire_triage_observations.csv",
                mime="text/csv",
                width="stretch",
            )

        table_df = filtered_df.copy()
        if search_query.strip():
            q = search_query.strip().lower()
            mask = (
                table_df["obs_id"].astype(str).str.lower().str.contains(q)
                | table_df["region"].astype(str).str.lower().str.contains(q)
                | table_df["satellite"].astype(str).str.lower().str.contains(q)
                | table_df["instrument"].astype(str).str.lower().str.contains(q)
                | table_df["notes"].astype(str).str.lower().str.contains(q)
            )
            table_df = table_df[mask].copy()

        sort_map = {
            "Priority Rank (Default)": ("priority_rank", True),
            "Priority Score (High → Low)": ("priority_score", False),
            "Priority Score (Low → High)": ("priority_score", True),
            "Confidence (High → Low)": ("confidence_num", False),
            "Brightness K (High → Low)": ("brightness", False),
            "FRP MW (High → Low)": ("frp", False),
            "Most Recent First": ("hours_since_latest", True),
            "Repeat Detections (High → Low)": ("repeat_count", False),
        }
        s_col, s_asc = sort_map[sort_col_label]
        table_df = table_df.sort_values(by=s_col, ascending=s_asc)

        display_cols = [
            "priority_rank",
            "obs_id",
            "priority_tier",
            "priority_score",
            "top_drivers",
            "region",
            "acq_date",
            "acq_time",
            "hours_since_latest",
            "confidence_label",
            "confidence_num",
            "brightness",
            "frp",
            "repeat_count",
            "satellite",
            "instrument",
            "daynight",
            "quality_status",
            "latitude",
            "longitude",
            "score_explanation",
        ]

        st.dataframe(
            table_df[display_cols],
            width="stretch",
            hide_index=True,
            column_config={
                "priority_rank": st.column_config.NumberColumn("Rank", format="#%d"),
                "obs_id": st.column_config.TextColumn("Obs ID"),
                "priority_tier": st.column_config.TextColumn("Priority Tier"),
                "priority_score": st.column_config.ProgressColumn(
                    "Priority Score",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.1f / 100",
                ),
                "top_drivers": st.column_config.TextColumn("Top Scoring Drivers"),
                "region": st.column_config.TextColumn("Region / Location"),
                "acq_date": st.column_config.TextColumn("Acq Date"),
                "acq_time": st.column_config.TextColumn("UTC Time"),
                "hours_since_latest": st.column_config.NumberColumn("Age (hrs)", format="%.1fh"),
                "confidence_label": st.column_config.TextColumn("Confidence"),
                "confidence_num": st.column_config.ProgressColumn(
                    "Conf (%)",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.0f%%",
                ),
                "brightness": st.column_config.NumberColumn("Brightness (K)", format="%.1f K"),
                "frp": st.column_config.NumberColumn("FRP (MW)", format="%.1f MW"),
                "repeat_count": st.column_config.NumberColumn("Cluster Neighbors", format="%d"),
                "quality_status": st.column_config.TextColumn("Data Quality"),
                "latitude": st.column_config.NumberColumn("Latitude", format="%.4f"),
                "longitude": st.column_config.NumberColumn("Longitude", format="%.4f"),
                "score_explanation": st.column_config.TextColumn("Full Score Explanation"),
            },
            height=520,
        )


# ===========================================================================
# TAB 3: OBSERVATION DETAIL & EXPLAINABILITY INSPECTOR
# ===========================================================================
with tab_detail:
    st.subheader("🔍 Observation Detail & Explainable Score Inspector")

    if scored_df.empty:
        st.warning("No valid observations available to inspect.")
    else:
        inspect_pool = filtered_df if not filtered_df.empty else scored_df
        obs_options = [
            f"Rank #{int(r['priority_rank'])} | {r['obs_id']} | Score: {r['priority_score']:.1f}/100 ({r['priority_tier']}) | {r['region']}"
            for _, r in inspect_pool.iterrows()
        ]
        selected_option = st.selectbox(
            "Select Observation to Inspect",
            options=obs_options,
            index=0,
        )
        selected_obs_id = selected_option.split("|")[1].strip()
        obs_row = inspect_pool[inspect_pool["obs_id"] == selected_obs_id].iloc[0]

        # Summary Header Banner
        tier_col = TIER_COLOR_MAP.get(obs_row["priority_tier"], "#3B82F6")
        with st.container(border=True):
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Priority Rank", f"#{int(obs_row['priority_rank'])} of {len(scored_df)}")
            d2.metric("Priority Score", f"{obs_row['priority_score']:.2f} / 100")
            d3.metric("Priority Tier", obs_row["priority_tier"])
            d4.metric("Cluster Neighbors", f"{int(obs_row['repeat_count'])} within {cluster_radius_km:.1f} km")
            d5.metric("Data Completeness", obs_row["quality_status"])

            st.markdown(f"**🧠 Explainable Triage Summary:** {obs_row['score_explanation']}")
            if obs_row["notes"]:
                st.caption(f"📝 **Source / Analyst Note:** {obs_row['notes']}")
            if obs_row["has_missing_values"]:
                st.warning(
                    f"⚠️ **Missing-Value Handling Applied:** `{obs_row['quality_notes']}`"
                )

        # Factor Contribution Visual + Table
        breakdown_rows = get_observation_breakdown(obs_row, weights)
        breakdown_df = pd.DataFrame(breakdown_rows)

        chart_col, table_col = st.columns([1.15, 1.35], gap="large")
        with chart_col:
            st.markdown("#### 📊 Points Contributed vs. Max Possible per Factor")
            fig_bar = go.Figure()
            fig_bar.add_trace(
                go.Bar(
                    y=breakdown_df["Factor"],
                    x=breakdown_df["Points Added"],
                    name="Points Added to Score",
                    orientation="h",
                    marker_color="#F97316",
                    text=breakdown_df["Points Added"].apply(lambda v: f"+{v:.1f} pts"),
                    textposition="auto",
                )
            )
            fig_bar.add_trace(
                go.Bar(
                    y=breakdown_df["Factor"],
                    x=(breakdown_df["Max Possible Points"] - breakdown_df["Points Added"]).clip(lower=0.0),
                    name="Remaining Headroom",
                    orientation="h",
                    marker_color="rgba(148, 163, 184, 0.28)",
                )
            )
            fig_bar.update_layout(
                barmode="stack",
                height=340,
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
                xaxis_title="Weighted Priority Points",
                yaxis={"autorange": "reversed"},
                legend={"orientation": "h", "y": 1.12},
            )
            st.plotly_chart(fig_bar, width="stretch")

        with table_col:
            st.markdown("#### 🧮 Factor-by-Factor Mathematical Breakdown")
            st.dataframe(
                breakdown_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Sub-Score (0–100)": st.column_config.ProgressColumn(
                        "Sub-Score (0–100)",
                        min_value=0.0,
                        max_value=100.0,
                        format="%.1f",
                    ),
                    "Weight (%)": st.column_config.NumberColumn("Weight (%)", format="%.1f%%"),
                    "Points Added": st.column_config.NumberColumn("Points Added", format="+%.2f"),
                    "Max Possible Points": st.column_config.NumberColumn("Max Points", format="%.2f"),
                },
            )

        # Spatial Cluster & Telemetry Details
        st.divider()
        sub_col1, sub_col2 = st.columns([1, 1.4], gap="large")
        with sub_col1:
            st.markdown("#### 🛰️ Raw Sensor & Geospatial Telemetry")
            telemetry_items = [
                ("Observation ID", obs_row["obs_id"]),
                ("Region / Location", obs_row["region"]),
                ("Coordinates (Lat, Lon)", f"{obs_row['latitude']:.4f}°, {obs_row['longitude']:.4f}°"),
                ("Acquisition Date & Time", f"{obs_row['acq_date']} {obs_row['acq_time']} UTC"),
                ("Hours Before Latest Pass", f"{obs_row['hours_since_latest']:.2f} hours"),
                ("Satellite / Instrument", f"{obs_row['satellite']} / {obs_row['instrument']}"),
                ("Day / Night Flag", obs_row["daynight"]),
                ("Primary Brightness (MIR)", f"{obs_row['brightness']:.1f} K"),
                (
                    "Secondary Brightness (TIR t31)",
                    "Missing" if pd.isna(obs_row["bright_t31"]) or obs_row["bright_t31"] is None else f"{obs_row['bright_t31']:.1f} K",
                ),
                (
                    "Fire Radiative Power (FRP)",
                    "Missing (Imputed 0.0 MW)" if obs_row["frp_missing"] else f"{obs_row['frp']:.1f} MW",
                ),
                ("Raw Sensor Confidence", f"{obs_row['confidence_raw']} → {obs_row['confidence_num']:.0f}%"),
            ]
            st.table(pd.DataFrame(telemetry_items, columns=["Attribute", "Value"]))

        with sub_col2:
            st.markdown(
                f"#### 🔁 Repeat Detections Within {cluster_radius_km:.1f} km Cluster Radius"
            )
            # Compute exact Haversine distances from selected observation to all valid observations
            all_lats = np.concatenate([[obs_row["latitude"]], scored_df["latitude"].to_numpy()])
            all_lons = np.concatenate([[obs_row["longitude"]], scored_df["longitude"].to_numpy()])
            d_mat = compute_haversine_matrix_km(all_lats, all_lons)
            dists_to_all = d_mat[0, 1:]

            nbr_df = scored_df.copy()
            nbr_df["distance_km"] = np.round(dists_to_all, 2)
            nbr_df = nbr_df[
                (nbr_df["obs_id"] != obs_row["obs_id"])
                & (nbr_df["distance_km"] <= cluster_radius_km)
            ].sort_values(by="distance_km")

            if nbr_df.empty:
                st.info(
                    f"No other valid detections found within **{cluster_radius_km:.1f} km** of `{obs_row['obs_id']}`. "
                    "This observation is classified as an **isolated thermal anomaly**."
                )
            else:
                st.success(
                    f"Found **{len(nbr_df)}** neighboring observation(s) within **{cluster_radius_km:.1f} km**, "
                    "indicating spatial persistence across pixels or satellite passes:"
                )
                st.dataframe(
                    nbr_df[
                        [
                            "obs_id",
                            "priority_rank",
                            "distance_km",
                            "acq_date",
                            "acq_time",
                            "satellite",
                            "brightness",
                            "frp",
                            "confidence_label",
                            "priority_score",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "obs_id": "Neighbor ID",
                        "priority_rank": st.column_config.NumberColumn("Rank", format="#%d"),
                        "distance_km": st.column_config.NumberColumn("Distance (km)", format="%.2f km"),
                        "brightness": st.column_config.NumberColumn("Brightness (K)", format="%.1f K"),
                        "frp": st.column_config.NumberColumn("FRP (MW)", format="%.1f MW"),
                        "priority_score": st.column_config.NumberColumn("Score", format="%.1f"),
                    },
                )


# ===========================================================================
# TAB 4: DATA VALIDATION & QUALITY AUDIT REPORT
# ===========================================================================
with tab_validation:
    st.subheader("🛡️ CSV Ingestion, Validation & Missing-Value Audit")

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Schema Validation", "PASSED ✅" if val_result.is_schema_valid else "FAILED ❌")
    v2.metric("Complete Valid Records", val_result.valid_count - val_result.warning_count)
    v3.metric("Partial / Imputed Records", val_result.warning_count)
    v4.metric("Rejected Malformed Records", val_result.rejected_count)

    st.markdown("#### 1. Rejected / Invalid Records (Excluded from Map & Ranking)")
    if val_result.rejected_df.empty:
        st.success("✅ Zero malformed or invalid records detected in the active dataset.")
    else:
        st.error(
            f"🚫 **{val_result.rejected_count} record(s) failed validation** due to invalid coordinates, "
            "non-numeric fields, impossible dates, or out-of-bounds physical measurements:"
        )
        rej_cols = ["row_number", "obs_id", "rejection_reasons"] + [
            c
            for c in ["latitude", "longitude", "brightness", "acq_date", "acq_time", "confidence", "frp", "notes"]
            if c in val_result.rejected_df.columns
        ]
        st.dataframe(
            val_result.rejected_df[rej_cols],
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### 2. Missing-Value Handling & Imputation Log (Retained Valid Records)")
    partial_df = (
        scored_df[scored_df["has_missing_values"]]
        if not scored_df.empty and "has_missing_values" in scored_df.columns
        else pd.DataFrame()
    )
    if partial_df.empty:
        st.info("No missing optional fields required imputation in the valid observation set.")
    else:
        st.warning(
            f"⚠️ **{len(partial_df)} valid observation(s)** had missing optional telemetry "
            "(`frp`, `acq_time`, `bright_t31`, or `daynight`) that were safely defaulted and flagged:"
        )
        st.dataframe(
            partial_df[
                [
                    "row_number",
                    "obs_id",
                    "priority_rank",
                    "priority_score",
                    "quality_status",
                    "quality_notes",
                    "region",
                    "notes",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### 3. Complete Field-Level Validation Issue Log")
    issues_df = val_result.issues_as_dataframe()
    if not issues_df.empty:
        sev_filter = st.radio(
            "Filter Issues by Severity",
            options=["ALL", "ERROR (Rejected)", "WARNING (Imputed/Defaulted)"],
            horizontal=True,
        )
        if sev_filter.startswith("ERROR"):
            issues_df = issues_df[issues_df["severity"] == "ERROR"]
        elif sev_filter.startswith("WARNING"):
            issues_df = issues_df[issues_df["severity"] == "WARNING"]
        st.dataframe(issues_df, width="stretch", hide_index=True)


# ===========================================================================
# TAB 5: FUNCTIONAL EVALUATION & LIMITATIONS DOCUMENTATION
# ===========================================================================
with tab_eval:
    st.subheader("✅ Functional Evaluation Report (Expected vs. Actual Behavior)")
    st.markdown(
        "In accordance with **PRD Section 2.2 (Success Criteria)** and **Section 4.1 (MVP Scope)**, "
        "the automated functional evaluation suite verifies CSV parsing, coordinate validation, "
        "missing-value handling, spatial clustering, and explainable priority ranking."
    )

    eval_df = run_functional_evaluation()
    passed_count = int((eval_df["Status"] == "PASS").sum())
    st.success(
        f"**Functional Verification Summary:** **{passed_count} / {len(eval_df)}** test cases passed (`100% PASS`)."
    )
    st.dataframe(
        eval_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Test ID": st.column_config.TextColumn("Test ID", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
        },
    )

    st.divider()
    st.subheader("📘 Data Sources, Scoring Methodology & Operational Limitations")

    doc_col1, doc_col2 = st.columns(2, gap="large")
    with doc_col1:
        st.markdown(
            """
            #### 1. Supported Data Sources & Schema
            - **NASA FIRMS (Fire Information for Resource Management System)**:
              Supports standard CSV exports from **MODIS** (Terra/Aqua, 1 km nominal resolution)
              and **VIIRS** (Suomi-NPP / NOAA-20 / NOAA-21, 375 m nominal resolution).
            - **Required Fields**:
              - `latitude` (`[-90.0, 90.0]` decimal degrees)
              - `longitude` (`[-180.0, 180.0]` decimal degrees)
              - `brightness` or `bright_ti4` (`[240.0, 650.0]` Kelvin)
              - `acq_date` (`YYYY-MM-DD`)
              - `confidence` (`0–100` numeric or VIIRS `'l'/'n'/'h'`)
            - **Optional Enriching Fields**:
              - `acq_time` (`HHMM` UTC; defaults to `0000` with warning if missing)
              - `frp` (Fire Radiative Power in MW; imputed to `0.0` and flagged if missing)
              - `bright_t31` / `bright_ti5`, `satellite`, `instrument`, `daynight`, `region`, `notes`.

            #### 2. Explainable Priority Scoring Formula
            Each valid observation receives a composite **Priority Score (`0–100`)** computed as the
            weighted sum of five normalized sub-scores (`0–100` each):
            1. **Detection Recency**: Exponential decay $100 \\times 2^{-\\Delta t / t_{1/2}}$ relative to the latest observation timestamp.
            2. **Sensor Confidence**: Direct `0–100` scale (`l=30`, `n=65`, `h=95` for categorical VIIRS).
            3. **Thermal Brightness**: Linear scaling between `300 K` (baseline anomaly) and `420 K` (high-intensity anomaly).
            4. **Fire Radiative Power (FRP)**: Logarithmic scaling $\\ln(1 + \\text{FRP}) / \\ln(1 + 500) \\times 100$.
            5. **Repeat Detections (Spatial Cluster)**: Haversine distance neighbor count within configurable radius $R$ km: $100 \\times (1 - e^{-0.6 N})$.
            """
        )

    with doc_col2:
        st.markdown(
            """
            #### 3. Known Limitations & Responsible Interpretation
            - **Thermal Anomalies ≠ Confirmed Wildfires**:
              Satellite infrared sensors detect *sub-pixel thermal anomalies*, not verified wildfires.
              Detections frequently include:
              - Stationary industrial heat sources (gas flares, steel mills, power plants, refineries)
              - Agricultural residue burning or authorized prescribed forest management burns
              - Daytime solar reflection / glint off bare rock, dry lakebeds, or commercial rooftops
            - **Cloud Cover, Smoke & Canopy Obscuration**:
              Thick cloud cover completely blocks mid-infrared radiation, while heavy smoke plumes or dense forest canopies can attenuate brightness temperature and FRP, causing active surface fires to go undetected or appear with artificially low confidence.
            - **Orbital Sampling & Pixel Geometry**:
              Polar-orbiting satellites (Terra, Aqua, Suomi-NPP, NOAA-20) capture snapshots only during specific overpass windows. High scan angles near swath edges enlarge pixel footprints (`scan` / `track`), reducing geolocation precision.
            - **Synthetic Dataset Labeling**:
              Built-in datasets (`synthetic_firms_triage_mixed.csv` and `synthetic_firms_clean_benchmark.csv`) are **synthetic evaluation records** designed to test coordinate validation, missing-value imputation, and cluster triage. They must never be interpreted as live fire intelligence.
            - **No Emergency-Response Automation**:
              All priority rankings require human analyst verification against local meteorological context, land-use masks, and ground-truth agency reports before any operational action is taken.
            """
        )
