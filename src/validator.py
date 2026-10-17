"""
Data validation, schema normalization, coordinate verification, and missing-value
handling for satellite-derived active fire observations.
"""

from dataclasses import dataclass, field
from datetime import datetime
import io
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.schema import (
    BRIGHTNESS_BOUNDS,
    CATEGORICAL_CONFIDENCE_MAP,
    COLUMN_ALIASES,
    FRP_BOUNDS,
    LAT_BOUNDS,
    LON_BOUNDS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
)


@dataclass
class ValidationIssue:
    """Represents a single validation error or warning on a specific row."""

    row_number: int  # 1-indexed CSV data row (excluding header)
    obs_id: str
    severity: str  # "ERROR" (rejected) or "WARNING" (accepted with missing-value handling)
    field_name: str
    raw_value: str
    message: str


@dataclass
class ValidationResult:
    """Complete result of CSV parsing and record validation."""

    is_schema_valid: bool
    missing_required_columns: List[str]
    column_mapping: Dict[str, str]
    total_rows: int
    valid_df: pd.DataFrame
    rejected_df: pd.DataFrame
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def valid_count(self) -> int:
        return len(self.valid_df)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_df)

    @property
    def warning_count(self) -> int:
        if self.valid_df.empty or "has_missing_values" not in self.valid_df.columns:
            return 0
        return int(self.valid_df["has_missing_values"].sum())

    def issues_as_dataframe(self) -> pd.DataFrame:
        """Convert validation issues list to a DataFrame for display."""
        if not self.issues:
            return pd.DataFrame(
                columns=["row_number", "obs_id", "severity", "field_name", "raw_value", "message"]
            )
        return pd.DataFrame([vars(issue) for issue in self.issues])


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Map input CSV column names to canonical FIRMS schema names.
    Returns (normalized_df, mapping_dict, missing_required_columns).
    """
    lower_to_orig = {str(col).strip().lower(): col for col in df.columns}
    mapping: Dict[str, str] = {}
    rename_map: Dict[str, str] = {}

    all_canonical = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    for canonical in all_canonical:
        aliases = COLUMN_ALIASES.get(canonical, [canonical])
        for alias in aliases:
            if alias.lower() in lower_to_orig:
                orig_col = lower_to_orig[alias.lower()]
                mapping[canonical] = orig_col
                if orig_col != canonical:
                    rename_map[orig_col] = canonical
                break

    normalized_df = df.rename(columns=rename_map).copy()
    missing_req = [col for col in REQUIRED_COLUMNS if col not in normalized_df.columns]
    return normalized_df, mapping, missing_req


def _is_missing(val: Any) -> bool:
    """Check whether a value is null, NaN, or an empty/whitespace string."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    s = str(val).strip()
    return s == "" or s.lower() in {"nan", "null", "none", "n/a", "na", "-"}


def _parse_confidence(val: Any) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Parse confidence value (numeric 0-100 or categorical l/n/h).
    Returns (numeric_confidence, confidence_label, error_message).
    """
    if _is_missing(val):
        return None, None, "Missing required confidence value"

    s = str(val).strip()
    s_lower = s.lower()

    if s_lower in CATEGORICAL_CONFIDENCE_MAP:
        num = CATEGORICAL_CONFIDENCE_MAP[s_lower]
        label_map = {
            30.0: "Low (l)",
            65.0: "Nominal (n)",
            95.0: "High (h)",
        }
        return num, label_map.get(num, s), None

    try:
        num = float(s)
    except ValueError:
        return (
            None,
            None,
            f"Invalid confidence '{s}': must be 0–100 or categorical (l/n/h, low/nominal/high)",
        )

    if not np.isfinite(num) or num < 0.0 or num > 100.0:
        return None, None, f"Confidence {num} out of valid range [0, 100]"

    if num >= 80:
        label = f"High ({int(round(num))}%)"
    elif num >= 50:
        label = f"Nominal ({int(round(num))}%)"
    else:
        label = f"Low ({int(round(num))}%)"
    return float(num), label, None


def _parse_acq_datetime(
    date_val: Any, time_val: Any
) -> Tuple[Optional[datetime], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Parse acquisition date and optional acquisition time.
    Returns (acq_datetime, formatted_date, formatted_time, error_msg, warning_msg).
    """
    if _is_missing(date_val):
        return None, None, None, "Missing required acquisition date (acq_date)", None

    date_str = str(date_val).strip()
    parsed_date = None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue

    if parsed_date is None:
        return (
            None,
            None,
            None,
            f"Malformed acquisition date '{date_str}' (expected YYYY-MM-DD)",
            None,
        )

    warning_msg = None
    if _is_missing(time_val):
        hour, minute = 0, 0
        formatted_time = "0000"
        warning_msg = "Missing acq_time; defaulted to 0000 UTC"
    else:
        raw_t = str(time_val).strip()
        # Handle float representations like "1430.0" from pandas
        if raw_t.endswith(".0"):
            raw_t = raw_t[:-2]
        raw_t_clean = raw_t.replace(":", "")
        if not raw_t_clean.isdigit():
            return (
                None,
                None,
                None,
                f"Malformed acquisition time '{raw_t}' (expected HHMM or HH:MM)",
                None,
            )
        raw_t_padded = raw_t_clean.zfill(4)
        if len(raw_t_padded) != 4:
            return (
                None,
                None,
                None,
                f"Acquisition time '{raw_t}' has invalid length (expected HHMM)",
                None,
            )
        hour = int(raw_t_padded[:2])
        minute = int(raw_t_padded[2:])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return (
                None,
                None,
                None,
                f"Acquisition time '{raw_t}' out of valid 24-hour range (0000–2359)",
                None,
            )
        formatted_time = raw_t_padded

    acq_dt = parsed_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return acq_dt, parsed_date.strftime("%Y-%m-%d"), formatted_time, None, warning_msg


def validate_fire_dataframe(df_raw: pd.DataFrame) -> ValidationResult:
    """
    Validate a DataFrame of satellite active fire observations.
    Identifies schema errors, coordinate errors, type errors, and missing values.
    """
    df_norm, col_mapping, missing_req = normalize_columns(df_raw)

    if missing_req:
        issues = [
            ValidationIssue(
                row_number=0,
                obs_id="SCHEMA",
                severity="ERROR",
                field_name=col,
                raw_value="MISSING_COLUMN",
                message=f"Required column '{col}' not found in uploaded CSV header.",
            )
            for col in missing_req
        ]
        return ValidationResult(
            is_schema_valid=False,
            missing_required_columns=missing_req,
            column_mapping=col_mapping,
            total_rows=len(df_raw),
            valid_df=pd.DataFrame(),
            rejected_df=df_raw.copy(),
            issues=issues,
        )

    valid_records: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []
    issues: List[ValidationIssue] = []

    for idx, row in df_norm.iterrows():
        row_num = int(idx) + 1 if isinstance(idx, (int, np.integer)) else len(valid_records) + len(rejected_records) + 1
        obs_id = str(row.get("obs_id", f"OBS-{row_num:04d}")).strip()
        if _is_missing(obs_id):
            obs_id = f"OBS-{row_num:04d}"

        row_errors: List[Tuple[str, str, str]] = []
        row_warnings: List[Tuple[str, str, str]] = []

        # 1. Validate Latitude
        raw_lat = row.get("latitude")
        lat_val: Optional[float] = None
        if _is_missing(raw_lat):
            row_errors.append(("latitude", str(raw_lat), "Missing required latitude value"))
        else:
            try:
                lat_val = float(str(raw_lat).strip())
                if not np.isfinite(lat_val) or not (LAT_BOUNDS[0] <= lat_val <= LAT_BOUNDS[1]):
                    row_errors.append(
                        (
                            "latitude",
                            str(raw_lat),
                            f"Latitude {lat_val} out of valid bounds [{LAT_BOUNDS[0]}, {LAT_BOUNDS[1]}]",
                        )
                    )
            except ValueError:
                row_errors.append(
                    ("latitude", str(raw_lat), f"Non-numeric latitude '{raw_lat}'")
                )

        # 2. Validate Longitude
        raw_lon = row.get("longitude")
        lon_val: Optional[float] = None
        if _is_missing(raw_lon):
            row_errors.append(("longitude", str(raw_lon), "Missing required longitude value"))
        else:
            try:
                lon_val = float(str(raw_lon).strip())
                if not np.isfinite(lon_val) or not (LON_BOUNDS[0] <= lon_val <= LON_BOUNDS[1]):
                    row_errors.append(
                        (
                            "longitude",
                            str(raw_lon),
                            f"Longitude {lon_val} out of valid bounds [{LON_BOUNDS[0]}, {LON_BOUNDS[1]}]",
                        )
                    )
            except ValueError:
                row_errors.append(
                    ("longitude", str(raw_lon), f"Non-numeric longitude '{raw_lon}'")
                )

        # 3. Validate Brightness (Kelvin)
        raw_bright = row.get("brightness")
        bright_val: Optional[float] = None
        if _is_missing(raw_bright):
            row_errors.append(("brightness", str(raw_bright), "Missing required brightness value"))
        else:
            try:
                bright_val = float(str(raw_bright).strip())
                if not np.isfinite(bright_val) or not (
                    BRIGHTNESS_BOUNDS[0] <= bright_val <= BRIGHTNESS_BOUNDS[1]
                ):
                    row_errors.append(
                        (
                            "brightness",
                            str(raw_bright),
                            f"Brightness {bright_val}K outside physical sensor range [{BRIGHTNESS_BOUNDS[0]}K, {BRIGHTNESS_BOUNDS[1]}K]",
                        )
                    )
            except ValueError:
                row_errors.append(
                    ("brightness", str(raw_bright), f"Non-numeric brightness '{raw_bright}'")
                )

        # 4. Validate Acquisition Date & Time
        raw_date = row.get("acq_date")
        raw_time = row.get("acq_time") if "acq_time" in df_norm.columns else None
        acq_dt, fmt_date, fmt_time, dt_err, dt_warn = _parse_acq_datetime(raw_date, raw_time)
        if dt_err:
            err_field = "acq_date" if "date" in dt_err.lower() else "acq_time"
            err_raw = str(raw_date) if err_field == "acq_date" else str(raw_time)
            row_errors.append((err_field, err_raw, dt_err))
        if dt_warn:
            row_warnings.append(("acq_time", str(raw_time), dt_warn))

        # 5. Validate Confidence
        raw_conf = row.get("confidence")
        conf_num, conf_label, conf_err = _parse_confidence(raw_conf)
        if conf_err:
            row_errors.append(("confidence", str(raw_conf), conf_err))

        # 6. Validate Optional FRP (Fire Radiative Power in MW)
        raw_frp = row.get("frp") if "frp" in df_norm.columns else None
        frp_val: float = 0.0
        frp_missing: bool = False
        if _is_missing(raw_frp):
            frp_missing = True
            frp_val = 0.0
            row_warnings.append(
                ("frp", str(raw_frp), "Missing FRP; set to 0.0 MW (flagged as imputed)")
            )
        else:
            try:
                parsed_frp = float(str(raw_frp).strip())
                if not np.isfinite(parsed_frp) or not (FRP_BOUNDS[0] <= parsed_frp <= FRP_BOUNDS[1]):
                    row_errors.append(
                        (
                            "frp",
                            str(raw_frp),
                            f"FRP {parsed_frp} MW outside valid bounds [{FRP_BOUNDS[0]}, {FRP_BOUNDS[1]}]",
                        )
                    )
                else:
                    frp_val = parsed_frp
            except ValueError:
                row_errors.append(("frp", str(raw_frp), f"Non-numeric FRP '{raw_frp}'"))

        # 7. Validate Optional Secondary Brightness (bright_t31 / bright_ti5)
        raw_t31 = row.get("bright_t31") if "bright_t31" in df_norm.columns else None
        t31_val: Optional[float] = None
        if _is_missing(raw_t31):
            row_warnings.append(
                ("bright_t31", str(raw_t31), "Missing secondary channel brightness (bright_t31)")
            )
        else:
            try:
                t31_val = float(str(raw_t31).strip())
            except ValueError:
                row_warnings.append(
                    ("bright_t31", str(raw_t31), f"Non-numeric bright_t31 '{raw_t31}' ignored")
                )

        # 8. Optional metadata fields (satellite, instrument, daynight, region, notes)
        raw_sat = row.get("satellite") if "satellite" in df_norm.columns else None
        sat_val = "Unknown" if _is_missing(raw_sat) else str(raw_sat).strip()

        raw_inst = row.get("instrument") if "instrument" in df_norm.columns else None
        inst_val = "VIIRS/MODIS" if _is_missing(raw_inst) else str(raw_inst).strip()

        raw_dn = row.get("daynight") if "daynight" in df_norm.columns else None
        if _is_missing(raw_dn):
            dn_val = "U"
            row_warnings.append(("daynight", str(raw_dn), "Missing daynight flag; set to 'U'"))
        else:
            dn_clean = str(raw_dn).strip().upper()
            if dn_clean in {"D", "DAY"}:
                dn_val = "D"
            elif dn_clean in {"N", "NIGHT"}:
                dn_val = "N"
            else:
                dn_val = "U"
                row_warnings.append(
                    ("daynight", str(raw_dn), f"Unrecognized daynight '{raw_dn}'; set to 'U'")
                )

        raw_region = row.get("region") if "region" in df_norm.columns else None
        region_val = "Unspecified Region" if _is_missing(raw_region) else str(raw_region).strip()

        raw_notes = row.get("notes") if "notes" in df_norm.columns else None
        notes_val = "" if _is_missing(raw_notes) else str(raw_notes).strip()

        # Record any errors or warnings
        if row_errors:
            for f_name, r_val, msg in row_errors:
                issues.append(
                    ValidationIssue(
                        row_number=row_num,
                        obs_id=obs_id,
                        severity="ERROR",
                        field_name=f_name,
                        raw_value=r_val,
                        message=msg,
                    )
                )
            rej_dict = row.to_dict()
            rej_dict["row_number"] = row_num
            rej_dict["obs_id"] = obs_id
            rej_dict["rejection_reasons"] = "; ".join(msg for _, _, msg in row_errors)
            rejected_records.append(rej_dict)
        else:
            for f_name, r_val, msg in row_warnings:
                issues.append(
                    ValidationIssue(
                        row_number=row_num,
                        obs_id=obs_id,
                        severity="WARNING",
                        field_name=f_name,
                        raw_value=r_val,
                        message=msg,
                    )
                )
            has_missing = len(row_warnings) > 0
            valid_records.append(
                {
                    "row_number": row_num,
                    "obs_id": obs_id,
                    "latitude": float(lat_val),
                    "longitude": float(lon_val),
                    "brightness": float(bright_val),
                    "bright_t31": t31_val,
                    "acq_date": fmt_date,
                    "acq_time": fmt_time,
                    "acq_datetime": acq_dt,
                    "confidence_raw": str(raw_conf).strip(),
                    "confidence_num": float(conf_num),
                    "confidence_label": conf_label,
                    "frp": float(frp_val),
                    "frp_missing": frp_missing,
                    "satellite": sat_val,
                    "instrument": inst_val,
                    "daynight": dn_val,
                    "region": region_val,
                    "notes": notes_val,
                    "has_missing_values": has_missing,
                    "quality_status": "Partial (Imputed)" if has_missing else "Complete",
                    "quality_notes": "; ".join(msg for _, _, msg in row_warnings)
                    if has_missing
                    else "All required & optional fields present",
                }
            )

    valid_df = pd.DataFrame(valid_records)
    rejected_df = pd.DataFrame(rejected_records)

    return ValidationResult(
        is_schema_valid=True,
        missing_required_columns=[],
        column_mapping=col_mapping,
        total_rows=len(df_raw),
        valid_df=valid_df,
        rejected_df=rejected_df,
        issues=issues,
    )


def load_and_validate_csv(
    source: Union[str, io.BytesIO, io.StringIO]
) -> ValidationResult:
    """Read a CSV file or buffer and run full validation."""
    try:
        df_raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    except Exception as exc:
        issue = ValidationIssue(
            row_number=0,
            obs_id="FILE",
            severity="ERROR",
            field_name="csv_parser",
            raw_value="",
            message=f"Failed to parse CSV file: {exc}",
        )
        return ValidationResult(
            is_schema_valid=False,
            missing_required_columns=REQUIRED_COLUMNS,
            column_mapping={},
            total_rows=0,
            valid_df=pd.DataFrame(),
            rejected_df=pd.DataFrame(),
            issues=[issue],
        )

    return validate_fire_dataframe(df_raw)
