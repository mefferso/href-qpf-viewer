#!/usr/bin/env python3
"""
Build lightweight, sampleable HREF QPF grids for the web viewer.

This script is intentionally boring and readable. It:
  1. Finds the latest available HREF cycle on NOMADS.
  2. Downloads selected ensemble product GRIB2 files.
  3. Finds total precipitation messages inside those GRIB2 files.
  4. Crops them to the configured map domain.
  5. Converts values to inches.
  6. Writes gzipped JSON grid files for the browser.
  7. Writes docs/data/catalog.json so the website knows what exists.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pygrib
import requests

# =============================================================================
# EASY SETTINGS TO CHANGE
# =============================================================================

# Broad Gulf Coast / southern CONUS domain. Tighten this later if you want.
DOMAIN = {
    "west": -95.0,
    "east": -83.0,
    "south": 24.0,
    "north": 34.0,
}

# Forecast hours to process. Keep this short at first so the repo doesn't bloat.
FORECAST_HOURS = [6, 12, 18, 24, 30, 36, 42, 48]

# Products to try. The file code is what NOMADS uses in the filename.
# avrg = ensemble mean/average. pmmn = probability matched mean.
PRODUCTS = [
    {"file_code": "avrg", "label": "Mean"},
    {"file_code": "max", "label": "Max"},
    {"file_code": "pmmn", "label": "PMM"},
    {"file_code": "lpmm", "label": "LPMM"},
]

# Use every Nth grid point after cropping. 1 = full density. 2 = lighter/faster.
# Start at 2. If the map looks too blocky, change to 1 later.
SPATIAL_STRIDE = 2

# Keep local downloads out of docs so only final web data gets published.
CACHE_DIR = Path(".cache/href_grib")
DATA_DIR = Path("docs/data")

# NOMADS base path.
NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod"

# User-agent: don't look like random malware doing grabby internet stuff.
HEADERS = {"User-Agent": "href-qpf-viewer/1.0 (GitHub Actions)"}

# =============================================================================


@dataclass
class Cycle:
    yyyymmdd: str
    hour: int

    @property
    def cycle_string(self) -> str:
        return f"{self.yyyymmdd}{self.hour:02d}"

    @property
    def cycle_label(self) -> str:
        return f"{self.yyyymmdd} {self.hour:02d}Z"


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}] {msg}", flush=True)


def candidate_cycles(now: Optional[datetime] = None) -> List[Cycle]:
    """Return likely HREF cycles, newest first."""
    now = now or datetime.now(timezone.utc)
    candidates: List[Cycle] = []

    # HREF CONUS ensemble products are generally 00Z and 12Z.
    # Check today and yesterday so early UTC hours don't fail.
    for day_offset in range(0, 3):
        day = now - timedelta(days=day_offset)
        ymd = day.strftime("%Y%m%d")
        for hr in (12, 0):
            cycle_dt = datetime(day.year, day.month, day.day, hr, tzinfo=timezone.utc)
            if cycle_dt <= now:
                candidates.append(Cycle(yyyymmdd=ymd, hour=hr))

    # Newest first.
    candidates.sort(key=lambda c: c.cycle_string, reverse=True)
    return candidates


def grib_url(cycle: Cycle, product_code: str, fhour: int) -> str:
    return (
        f"{NOMADS_BASE}/href.{cycle.yyyymmdd}/ensprod/"
        f"href.t{cycle.hour:02d}z.conus.{product_code}.f{fhour:02d}.grib2"
    )


def url_exists(session: requests.Session, url: str) -> bool:
    try:
        r = session.head(url, headers=HEADERS, timeout=20, allow_redirects=True)
        if r.status_code == 200:
            return True
        # Some servers are grumpy about HEAD. Try a tiny ranged GET.
        r = session.get(url, headers={**HEADERS, "Range": "bytes=0-99"}, timeout=20, stream=True)
        return r.status_code in (200, 206)
    except Exception:
        return False


def find_latest_cycle(session: requests.Session) -> Cycle:
    """Find newest cycle that has at least one usable PMM f06 file."""
    for c in candidate_cycles():
        test_url = grib_url(c, "pmmn", 6)
        log(f"Checking {c.cycle_label}: {test_url}")
        if url_exists(session, test_url):
            log(f"Using HREF cycle {c.cycle_label}")
            return c
    raise RuntimeError("Could not find a recent HREF cycle on NOMADS.")


def download_file(session: requests.Session, url: str, out_path: Path) -> bool:
    """Download a URL if needed. Returns True when file exists and looks usable."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 10_000:
        return True

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    log(f"Downloading {url}")
    try:
        with session.get(url, headers=HEADERS, timeout=90, stream=True) as r:
            if r.status_code != 200:
                log(f"  Skip: HTTP {r.status_code}")
                return False
            with tmp_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        if tmp_path.stat().st_size < 10_000:
            log("  Skip: downloaded file is suspiciously small")
            tmp_path.unlink(missing_ok=True)
            return False
        tmp_path.replace(out_path)
        return True
    except Exception as e:
        log(f"  Download failed: {e}")
        tmp_path.unlink(missing_ok=True)
        return False


def safe_get(grb, attr: str, default=None):
    try:
        return getattr(grb, attr)
    except Exception:
        return default


def message_is_precip(grb) -> bool:
    """Be flexible because GRIB metadata labels vary a bit."""
    short_name = str(safe_get(grb, "shortName", "")).lower()
    name = str(safe_get(grb, "name", "")).lower()
    parameter_name = str(safe_get(grb, "parameterName", "")).lower()

    haystack = " ".join([short_name, name, parameter_name])
    return (
        short_name in {"tp", "apcp"}
        or "total precipitation" in haystack
        or "total precip" in haystack
        or "precipitation" in haystack and "probability" not in haystack
    )


def accum_hours_from_message(grb) -> Optional[int]:
    """Try to extract accumulation period in hours from GRIB metadata."""
    for key in ("lengthOfTimeRange", "stepRange"):
        val = safe_get(grb, key, None)
        if val is None:
            continue
        if key == "lengthOfTimeRange":
            try:
                n = int(val)
                if n > 0:
                    return n
            except Exception:
                pass
        else:
            text = str(val)
            # Examples: "0-6", "18-24", "24"
            m = re.match(r"^(\d+)\s*-\s*(\d+)$", text)
            if m:
                return max(1, int(m.group(2)) - int(m.group(1)))
    return None


def value_units_to_inches(values: np.ndarray, units: str) -> np.ndarray:
    """
    Convert precip values to inches.

    HREF APCP is typically kg/m^2, which is numerically equal to mm of water.
    If the units already say inches, keep them.
    """
    u = (units or "").lower().replace("**", "^")
    if "in" in u and "kg" not in u and "mm" not in u:
        return values.astype(float)
    # kg/m^2 and mm both become inches.
    return values.astype(float) / 25.4


def crop_and_stride(lats: np.ndarray, lons: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop to DOMAIN and thin with SPATIAL_STRIDE."""
    lons = np.where(lons > 180, lons - 360, lons)

    mask = (
        (lats >= DOMAIN["south"])
        & (lats <= DOMAIN["north"])
        & (lons >= DOMAIN["west"])
        & (lons <= DOMAIN["east"])
        & np.isfinite(vals)
    )

    rows, cols = np.where(mask)
    if rows.size == 0:
        return np.array([]), np.array([]), np.array([])

    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()

    sl_r = slice(rmin, rmax + 1, SPATIAL_STRIDE)
    sl_c = slice(cmin, cmax + 1, SPATIAL_STRIDE)

    crop_lats = lats[sl_r, sl_c]
    crop_lons = lons[sl_r, sl_c]
    crop_vals = vals[sl_r, sl_c]
    crop_mask = mask[sl_r, sl_c]

    return crop_lats[crop_mask], crop_lons[crop_mask], crop_vals[crop_mask]


def round_list(arr: np.ndarray, ndigits: int) -> List[float]:
    return [round(float(x), ndigits) for x in arr]


def write_json_gz(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


def process_grib_file(
    grib_path: Path,
    cycle: Cycle,
    product_code: str,
    product_label: str,
    fhour: int,
) -> List[dict]:
    """Process all precip messages in one GRIB2 file."""
    layers: List[dict] = []

    try:
        grbs = pygrib.open(str(grib_path))
    except Exception as e:
        log(f"Could not open {grib_path}: {e}")
        return layers

    try:
        messages = [g for g in grbs if message_is_precip(g)]
    except Exception as e:
        log(f"Could not scan {grib_path}: {e}")
        grbs.close()
        return layers

    if not messages:
        log(f"No precipitation messages found in {grib_path.name}")
        grbs.close()
        return layers

    log(f"Found {len(messages)} precip message(s) in {grib_path.name}")

    for idx, grb in enumerate(messages, start=1):
        try:
            accum = accum_hours_from_message(grb) or fhour
            units = str(safe_get(grb, "units", ""))
            name = str(safe_get(grb, "name", "Total precipitation"))
            step_range = str(safe_get(grb, "stepRange", ""))
            valid_date = safe_get(grb, "validDate", None)
            anal_date = safe_get(grb, "analDate", None)

            vals_raw = grb.values
            if np.ma.isMaskedArray(vals_raw):
                vals_raw = vals_raw.filled(np.nan)
            vals_in = value_units_to_inches(np.array(vals_raw, dtype=float), units)
            vals_in[vals_in < 0] = np.nan

            lats, lons = grb.latlons()
            pts_lat, pts_lon, pts_val = crop_and_stride(lats, lons, vals_in)
            if pts_val.size == 0:
                continue

            # Hide microscopic precip noise, but keep trace-ish values sampleable.
            pts_val = np.where(pts_val < 0.001, 0.0, pts_val)

            layer_id = f"{cycle.cycle_string}_{product_code}_f{fhour:02d}_a{accum:02d}_{idx}"
            rel_path = f"data/grids/{layer_id}.json.gz"
            out_path = Path("docs") / rel_path

            payload = {
                "metadata": {
                    "id": layer_id,
                    "run": cycle.cycle_string,
                    "runLabel": cycle.cycle_label,
                    "product": product_code,
                    "productLabel": product_label,
                    "forecastHour": fhour,
                    "accumHours": accum,
                    "units": "inches",
                    "sourceUnits": units,
                    "name": name,
                    "stepRange": step_range,
                    "validTimeUTC": valid_date.isoformat() if hasattr(valid_date, "isoformat") else None,
                    "analysisTimeUTC": anal_date.isoformat() if hasattr(anal_date, "isoformat") else None,
                    "domain": DOMAIN,
                    "spatialStride": SPATIAL_STRIDE,
                },
                "lat": round_list(pts_lat, 4),
                "lon": round_list(pts_lon, 4),
                "value": round_list(pts_val, 3),
            }

            write_json_gz(payload, out_path)

            max_val = float(np.nanmax(pts_val)) if pts_val.size else 0.0
            layer_record = {
                "id": layer_id,
                "url": rel_path,
                "run": cycle.cycle_string,
                "runLabel": cycle.cycle_label,
                "product": product_code,
                "productLabel": product_label,
                "forecastHour": fhour,
                "accumHours": accum,
                "units": "inches",
                "validTimeUTC": payload["metadata"]["validTimeUTC"],
                "stepRange": step_range,
                "name": name,
                "maxValue": round(max_val, 2),
                "pointCount": int(pts_val.size),
            }
            layers.append(layer_record)
            log(f"  Wrote {layer_id}: {pts_val.size:,} points; max {max_val:.2f} in")
        except Exception as e:
            log(f"  Message {idx} failed: {e}")

    grbs.close()
    return layers


def clean_old_data() -> None:
    grids = DATA_DIR / "grids"
    if grids.exists():
        shutil.rmtree(grids)
    grids.mkdir(parents=True, exist_ok=True)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clean_old_data()

    session = requests.Session()
    cycle = find_latest_cycle(session)

    all_layers: List[dict] = []

    for fhour in FORECAST_HOURS:
        for product in PRODUCTS:
            code = product["file_code"]
            label = product["label"]
            url = grib_url(cycle, code, fhour)
            local_path = CACHE_DIR / cycle.cycle_string / f"href.t{cycle.hour:02d}z.conus.{code}.f{fhour:02d}.grib2"

            if not download_file(session, url, local_path):
                continue

            layers = process_grib_file(local_path, cycle, code, label, fhour)
            all_layers.extend(layers)
            time.sleep(1.0)

    if not all_layers:
        raise RuntimeError("No HREF QPF layers were generated. Check NOMADS availability or GRIB parsing.")

    all_layers.sort(key=lambda x: (x["run"], x["forecastHour"], x["accumHours"], x["productLabel"]))

    catalog = {
        "generatedUTC": datetime.now(timezone.utc).isoformat(),
        "domain": DOMAIN,
        "source": "NCEP NOMADS HREF CONUS ensprod GRIB2",
        "defaultLayerId": all_layers[0]["id"],
        "colorScale": [
            {"min": 0.001, "max": 0.10, "color": "#d7f9d0", "label": "Trace-0.10"},
            {"min": 0.10, "max": 0.25, "color": "#9beb7f", "label": "0.10-0.25"},
            {"min": 0.25, "max": 0.50, "color": "#52d24c", "label": "0.25-0.50"},
            {"min": 0.50, "max": 1.00, "color": "#1faa3e", "label": "0.50-1.00"},
            {"min": 1.00, "max": 2.00, "color": "#1787d4", "label": "1.00-2.00"},
            {"min": 2.00, "max": 3.00, "color": "#1f49d8", "label": "2.00-3.00"},
            {"min": 3.00, "max": 5.00, "color": "#7b2cbf", "label": "3.00-5.00"},
            {"min": 5.00, "max": 8.00, "color": "#d000ff", "label": "5.00-8.00"},
            {"min": 8.00, "max": 12.00, "color": "#ff2d2d", "label": "8.00-12.00"},
            {"min": 12.00, "max": 15.00, "color": "#ff9f1c", "label": "12.00-15.00"},
            {"min": 15.00, "max": 999.00, "color": "#ffffff", "label": ">15.00"},
        ],
        "layers": all_layers,
    }

    catalog_path = DATA_DIR / "catalog.json"
    with catalog_path.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    log(f"Wrote {catalog_path} with {len(all_layers)} layer(s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise
