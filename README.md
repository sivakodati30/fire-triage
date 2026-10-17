# Satellite-Derived Active Fire Triage Dashboard

**Project:** AI/ML 1 — Satellite-Derived Active Fire Triage  
**Version:** 1.0  
**Product Type:** Web-based data analysis dashboard (Python + Streamlit + Plotly/PyDeck + Pandas)

---

## 1. Overview & Purpose

The **Satellite-Derived Active Fire Triage Dashboard** helps environmental analysts, researchers, students, and data science practitioners ingest, validate, map, filter, and rank satellite thermal-anomaly observations exported from NASA's **Fire Information for Resource Management System (FIRMS)** (MODIS and VIIRS sensors) or compatible CSV sources.

> **⚠️ Operational Scope & Safety Disclaimer**  
> This application supports data exploration and analyst review. Satellite detections represent thermal anomalies (which may include wildfires, prescribed burns, agricultural clearing, industrial heat sources, or solar glint) and are subject to cloud cover and orbital sampling limitations. **The dashboard does not independently confirm active fires and must not be used for emergency-response decisions.**

---

## 2. Project Structure

```text
fire-triage-dashboard/
├── app.py                        # Main Streamlit web application (5 interactive tabs)
├── requirements.txt              # Python package dependencies
├── README.md                     # Documentation, scoring methodology & evaluation report
├── data/
│   ├── synthetic_firms_triage_mixed.csv     # Labeled synthetic dataset (25 valid + 8 invalid records)
│   └── synthetic_firms_clean_benchmark.csv  # Labeled synthetic dataset (25 valid records)
├── src/
│   ├── __init__.py
│   ├── schema.py                 # FIRMS column aliases, physical bounds, and ScoringWeights
│   ├── validator.py              # CSV ingestion, coordinate/type checks, missing-value handling
│   ├── scorer.py                 # Explainable 5-factor scoring & Haversine spatial clustering
│   ├── sample_data.py            # Synthetic dataset generator with explicit synthetic labeling
│   └── evaluator.py              # Functional test & evaluation harness (Expected vs. Actual)
└── tests/
    └── test_pipeline.py          # Automated pytest suite
```

---

## 3. Quick Start

### 3.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.2 Launch the Dashboard

```bash
streamlit run app.py
```

### 3.3 Run the Automated Test Suite

```bash
python -m pytest -v
```

---

## 4. Explainable Priority Scoring Algorithm

Each valid observation receives a composite **Priority Score (`0–100`)** calculated from five transparent, configurable factors:

| Factor | Default Weight | Normalization (`0–100` Sub-Score) | Rationale |
| :--- | :---: | :--- | :--- |
| **1. Detection Recency** | `25%` | $100 \times 2^{-\Delta t / t_{1/2}}$ (default $t_{1/2}=24\text{h}$) | Prioritizes newly detected thermal anomalies over stale historical passes. |
| **2. Sensor Confidence** | `25%` | Direct `0–100` scale (`l`=30, `n`=65, `h`=95 for VIIRS) | Reflects sensor algorithm certainty and reduces cloud-edge/glint false positives. |
| **3. Thermal Brightness** | `20%` | Linearly scaled between `300 K` and `420 K` | Captures mid-infrared brightness temperature intensity (`brightness` / `bright_ti4`). |
| **4. Fire Radiative Power (FRP)** | `15%` | Log-scaled: $\frac{\ln(1 + \text{FRP})}{\ln(1 + 500)} \times 100$ | Measures radiative heat output in Megawatts (`MW`). Imputed as `0` and flagged if missing. |
| **5. Repeat Detections (Cluster)** | `15%` | $100 \times (1 - e^{-0.6 N})$ within radius $R\text{ km}$ | Uses Haversine distance (default $R=5\text{ km}$) to identify persistent multi-pixel or multi-pass clusters. |

### Priority Tiers
- **Critical (`75.0 – 100.0`)**: Recent, high-confidence, intense, or multi-detection clustered thermal anomalies.
- **High (`55.0 – 74.9`)**: Strong thermal signature or developing spatial cluster requiring analyst review.
- **Medium (`35.0 – 54.9`)**: Moderate or isolated detections, older passes, or partial telemetry records.
- **Low (`0.0 – 34.9`)**: Low-confidence, decayed, or likely non-wildfire anomalies (e.g., solar glint, old isolated pixels).

---

## 5. Functional Evaluation Summary (Expected vs. Actual Behavior)

| Test ID | PRD Requirement | Input Scenario | Expected Behavior | Actual Behavior | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **TC-01** | CSV Parsing & Schema Alias Support | VIIRS-style headers (`lat`, `lon`, `bright_ti4`, `conf='h'`) | Schema accepted; mapped to canonical columns; `'h'` normalized to `95.0%` | Schema valid=True, valid_count=1, confidence_num=95.0 | **PASS** |
| **TC-02** | Required-Field Schema Validation | CSV missing `brightness` and `confidence` columns | Schema marked invalid; missing columns explicitly listed | Schema valid=False, missing=`['brightness', 'confidence']` | **PASS** |
| **TC-03** | Latitude & Longitude Range Validation | Row 1: `lat=96.45` (`>90`); Row 2: `lon=-195.82` (`<-180`) | Both rows rejected (`0` valid, `2` rejected) with coordinate bound errors | valid=0, rejected=2, issues=2 | **PASS** |
| **TC-04** | Data-Type & Required Value Validation | Row 1: `lat='NOT_A_NUMBER'`; Row 2: `lon=''` (empty) | Both rows rejected with type/missing coordinate error messages | valid=0, rejected=2 | **PASS** |
| **TC-05** | Date Parsing & Missing Time Handling | Row 1: `acq_date='2026-13-45'`; Row 2: valid date, empty `acq_time` | Row 1 rejected; Row 2 accepted with `acq_time='0000'` and warning flag | valid=1, rejected=1, Row 2 `acq_time='0000'` | **PASS** |
| **TC-06** | Sensor Range & Measurement Validation | 3 rows with `brightness=-45K`, `confidence=145`, and `frp=-25.4 MW` | All 3 rows rejected for violating physical/sensor measurement bounds | valid=0, rejected=3 | **PASS** |
| **TC-07** | Missing-Value Handling (Optional Telemetry) | Valid coordinates & brightness, but empty `frp` field | Record retained; `frp` imputed as `0.0 MW`; flagged as `'Partial (Imputed)'` | valid=1, status=`'Partial (Imputed)'`, score_frp=0.0 | **PASS** |
| **TC-08** | Repeat-Detection Spatial Clustering | 5 Plumas NF detections within 2 km (`SYN-001`..`005`) vs isolated Mojave pixel (`SYN-018`) | `SYN-001` detects 4 nearby observations within 5 km; `SYN-018` detects 0 | `SYN-001` repeat_count=4, `SYN-018` repeat_count=0 | **PASS** |
| **TC-09** | Explainable Priority Scoring & Ranking | Rank all 25 valid synthetic observations using default weights | `SYN-001` ranks `#1` (`Critical` tier) with 5-factor breakdown; low-conf old anomaly ranks last | Rank `#1=SYN-001` (`95.07/100`, Critical); Rank `#25=SYN-022` (`9.92/100`) | **PASS** |
| **TC-10** | End-to-End Synthetic Dataset Triage Audit | Full 33-row Synthetic Mixed Dataset (`synthetic_firms_triage_mixed.csv`) | 33 total → 25 valid (22 complete + 3 partial/imputed) and 8 rejected records | total=33, valid=25 (warnings=3), rejected=8 | **PASS** |

---

## 6. Synthetic Data & Limitations

- **Synthetic Labeling**: All records in `data/synthetic_firms_triage_mixed.csv` and `data/synthetic_firms_clean_benchmark.csv` are explicitly marked with `[SYNTHETIC]` and surfaced with a prominent banner in the UI.
- **Physical & Sensor Limitations**: Satellite active-fire detections can be triggered by industrial facilities, gas flares, agricultural burns, or daytime solar glint, and can be obscured by cloud cover or heavy forest canopy. Always cross-reference triage rankings with meteorological data and ground-truth reporting.
