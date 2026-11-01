"""Spectral indices computed from a reflectance Stack."""
from __future__ import annotations

import numpy as np


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


def compute(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    NDVI  (B08, B04)  canopy greenness / vigour.            Healthy crop ≈ 0.6–0.9.
    NDMI  (B08, B11)  water held in leaves and canopy.      Higher = wetter.
    NDRE  (B8A, B05)  red-edge chlorophyll, tracks leaf nitrogen and keeps
                      responding after NDVI saturates in dense canopies.
    NDWI  (B03, B08)  open water (McFeeters). > 0 means standing water.
    """
    return {
        "NDVI": _nd(bands["B08"], bands["B04"]),
        "NDMI": _nd(bands["B08"], bands["B11"]),
        "NDRE": _nd(bands["B8A"], bands["B05"]),
        "NDWI": _nd(bands["B03"], bands["B08"]),
    }
