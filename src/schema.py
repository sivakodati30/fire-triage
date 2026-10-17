"""
Schema definitions, constants, and configuration defaults for NASA FIRMS
and compatible satellite active-fire observation CSV files.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Standard column aliases to normalize both MODIS and VIIRS FIRMS CSV exports
COLUMN_ALIASES: Dict[str, List[str]] = {
    "latitude": ["latitude", "lat", "y"],
    "longitude": ["longitude", "lon", "lng", "long", "x"],
    "brightness": ["brightness", "bright_ti4", "temp_k", "bt_mir"],
    "bright_t31": ["bright_t31", "bright_ti5", "bt_tir"],
    "acq_date": ["acq_date", "date", "observation_date"],
    "acq_time": ["acq_time", "time", "observation_time"],
    "confidence": ["confidence", "conf"],
    "frp": ["frp", "fire_radiative_power"],
    "satellite": ["satellite", "sat", "platform"],
    "instrument": ["instrument", "sensor"],
    "daynight": ["daynight", "day_night", "dn"],
    "scan": ["scan"],
    "track": ["track"],
    "version": ["version", "collection"],
    "region": ["region", "location", "area", "zone"],
    "notes": ["notes", "source_note", "synthetic_label"],
}

# Required canonical columns for a record to be valid
REQUIRED_COLUMNS: List[str] = [
    "latitude",
    "longitude",
    "brightness",
    "acq_date",
    "confidence",
]

# Optional canonical columns that enrich scoring and filtering
OPTIONAL_COLUMNS: List[str] = [
    "acq_time",
    "frp",
    "bright_t31",
    "satellite",
    "instrument",
    "daynight",
    "scan",
    "track",
    "version",
    "region",
    "notes",
]

# Coordinate & physical bounds for validation
LAT_BOUNDS: Tuple[float, float] = (-90.0, 90.0)
LON_BOUNDS: Tuple[float, float] = (-180.0, 180.0)
# Brightness temperature in Kelvin: terrestrial thermal anomalies typically 250K - 600K
BRIGHTNESS_BOUNDS: Tuple[float, float] = (240.0, 650.0)
# Fire Radiative Power in MW: must be non-negative (>= 0)
FRP_BOUNDS: Tuple[float, float] = (0.0, 15000.0)

# VIIRS / categorical confidence mappings to 0-100 numeric scale
CATEGORICAL_CONFIDENCE_MAP: Dict[str, float] = {
    "l": 30.0,
    "low": 30.0,
    "n": 65.0,
    "nominal": 65.0,
    "m": 65.0,
    "medium": 65.0,
    "h": 95.0,
    "high": 95.0,
}


@dataclass
class ScoringWeights:
    """Configurable weights for the explainable priority scoring algorithm."""

    recency: float = 0.25
    confidence: float = 0.25
    brightness: float = 0.20
    frp: float = 0.15
    repeat_detections: float = 0.15
    cluster_radius_km: float = 5.0
    recency_half_life_hours: float = 24.0

    def normalized(self) -> Dict[str, float]:
        """Return weights normalized so they sum to 1.0 (or equal weights if all zero)."""
        raw = {
            "recency": max(0.0, float(self.recency)),
            "confidence": max(0.0, float(self.confidence)),
            "brightness": max(0.0, float(self.brightness)),
            "frp": max(0.0, float(self.frp)),
            "repeat_detections": max(0.0, float(self.repeat_detections)),
        }
        total = sum(raw.values())
        if total <= 0.0:
            return {k: 0.20 for k in raw}
        return {k: v / total for k, v in raw.items()}


# Priority tier thresholds on 0-100 scale
PRIORITY_TIERS: List[Tuple[str, float, str]] = [
    ("Critical", 75.0, "#EF4444"),  # Red
    ("High", 55.0, "#F97316"),      # Orange
    ("Medium", 35.0, "#EAB308"),    # Amber/Yellow
    ("Low", 0.0, "#3B82F6"),        # Blue
]
