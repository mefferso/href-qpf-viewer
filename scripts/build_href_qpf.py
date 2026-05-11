#!/usr/bin/env python3
"""Build HREF QPF data for the web viewer.

Outputs per layer:
  - gzipped JSON points for click sampling
  - transparent PNG raster for Leaflet imageOverlay display
  - catalog.json with layer metadata
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pygrib
import requests
from PIL import Image

DOMAIN = {"west": -95.0, "east": -83.0, "south": 24.0, "north": 34.0}
FORECAST_HOURS = [6, 12, 18, 24, 30, 36, 42, 48]
PRODUCTS = [
    {"file_code": "avrg", "label": "Mean"},
    {"file_code": "max", "label": "Max"},
    {"file_code": "pmmn", "label": "PMM"},
    {"file_code": "lpmm", "label": "LPMM"},
]

# JSON sampling can stay lighter; the displayed PNG uses the full cropped grid.
SAMPLE_STRIDE = 2
RASTER_UPSCALE = 4

CACHE_DIR = Path(".cache/href_grib")
DATA_DIR = Path("docs/data")
GRID_DIR = DATA_DIR / "grids"
RASTER_DIR = DATA_DIR / "rasters"
NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod"
HEADERS = {"User-Agent": "href-qpf-viewer/1.0"}

COLOR_SCALE = [
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
]


def hex_to_rgba(hex_color: str, alpha: int = 210) -> Tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha


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
    now = now or datetime.now(timezone.utc)
    out: List[Cycle] = []
    for day_offset in range(0, 3):
        day = now - timedelta(days=day_offset)
        ymd = day.strftime("%Y%m%d")
        for hr in (12, 0):
            if datetime(day.year, day.month, day.day, hr, tzinfo=timezone.utc) <= now:
                out.append(Cycle(ymd, hr))
    return sorted(out, key=lambda c: c.cycle_string, reverse=True)


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
        r = session.get(url, headers={**HEADERS, "Range": "bytes=0-99"}, timeout=20, stream=True)
        return r.status_code in (200, 206)
    except Exception:
        return False


def find_latest_cycle(session: requests.Session) -> Cycle:
    for cycle in candidate_cycles():
        test_url = grib_url(cycle, "pmmn", 6)
        log(f"Checking {cycle.cycle_label}: {test_url}")
        if url_exists(session, test_url):
            log(f"Using HREF cycle {cycle.cycle_label}")
            return cycle
    raise RuntimeError("Could not find a recent HREF cycle on NOMADS.")


def download_file(session: requests.Session, url: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return True

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)
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
            log("  Skip: downloaded file is too small")
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
    short_name = str(safe_get(grb, "shortName", "")).lower()
    name = str(safe_get(grb, "name", "")).lower()
    parameter_name = str(safe_get(grb, "parameterName", "")).lower()
    haystack = " ".join([short_name, name, parameter_name])
    return (
        short_name in {"tp", "apcp"}
        or "total precipitation" in haystack
        or "total precip" in haystack
        or ("precipitation" in haystack and "probability" not in haystack)
    )


def accum_hours_from_message(grb) -> Optional[int]:
    val = safe_get(grb, "lengthOfTimeRange", None)
    try:
        if val is not None and int(val) > 0:
            return int(val)
    except Exception:
        pass

    text = str(safe_get(grb, "stepRange", ""))
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", text)
    if m:
        return max(1, int(m.group(2)) - int(m.group(1)))
    return None


def value_units_to_inches(values: np.ndarray, units: str) -> np.ndarray:
    u = (units or "").lower().replace("**", "^")
    if "in" in u and "kg" not in u and "mm" not in u:
        return values.astype(float)
    return values.astype(float) / 25.4


def crop_2d(lats: np.ndarray, lons: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        empty = np.array([])
        return empty, empty, empty, empty

    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    return (
        lats[rmin : rmax + 1, cmin : cmax + 1],
        lons[rmin : rmax + 1, cmin : cmax + 1],
        vals[rmin : rmax + 1, cmin : cmax + 1],
        mask[rmin : rmax + 1, cmin : cmax + 1],
    )


def sample_points(lats2d: np.ndarray, lons2d: np.ndarray, vals2d: np.ndarray, mask2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lats_s = lats2d[::SAMPLE_STRIDE, ::SAMPLE_STRIDE]
    lons_s = lons2d[::SAMPLE_STRIDE, ::SAMPLE_STRIDE]
    vals_s = vals2d[::SAMPLE_STRIDE, ::SAMPLE_STRIDE]
    mask_s = mask2d[::SAMPLE_STRIDE, ::SAMPLE_STRIDE] & np.isfinite(vals_s)
    return lats_s[mask_s], lons_s[mask_s], vals_s[mask_s]


def round_list(arr: np.ndarray, ndigits: int) -> List[float]:
    return [round(float(x), ndigits) for x in arr]


def write_json_gz(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


def resize_float_grid(values: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    arr = np.nan_to_num(values.astype(np.float32), nan=0.0)
    img = Image.fromarray(arr, mode="F")
    img = img.resize(out_size, resample=Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.float32)


def resize_mask(mask: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize(out_size, resample=Image.Resampling.NEAREST)
    return np.array(img) > 0


def make_raster_png(vals2d: np.ndarray, mask2d: np.ndarray, out_path: Path) -> None:
    h, w = vals2d.shape
    out_size = (max(1, w * RASTER_UPSCALE), max(1, h * RASTER_UPSCALE))
    vals_hi = resize_float_grid(vals2d, out_size)
    mask_hi = resize_mask(mask2d, out_size)

    rgba = np.zeros((out_size[1], out_size[0], 4), dtype=np.uint8)
    visible = mask_hi & np.isfinite(vals_hi) & (vals_hi >= 0.001)

    for b in COLOR_SCALE:
        m = visible & (vals_hi >= b["min"]) & (vals_hi < b["max"])
        rgba[m] = hex_to_rgba(b["color"], 215)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_path, optimize=True)


def bounds_from_crop(lats2d: np.ndarray, lons2d: np.ndarray, mask2d: np.ndarray) -> dict:
    return {
        "south": round(float(np.nanmin(lats2d[mask2d])), 5),
        "north": round(float(np.nanmax(lats2d[mask2d])), 5),
        "west": round(float(np.nanmin(lons2d[mask2d])), 5),
        "east": round(float(np.nanmax(lons2d[mask2d])), 5),
    }


def process_grib_file(grib_path: Path, cycle: Cycle, product_code: str, product_label: str, fhour: int) -> List[dict]:
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
            lats2d, lons2d, vals2d, mask2d = crop_2d(lats, lons, vals_in)
            if vals2d.size == 0:
                continue

            vals2d = np.where(vals2d < 0.001, 0.0, vals2d)
            pts_lat, pts_lon, pts_val = sample_points(lats2d, lons2d, vals2d, mask2d)
            if pts_val.size == 0:
                continue

            layer_id = f"{cycle.cycle_string}_{product_code}_f{fhour:02d}_a{accum:02d}_{idx}"
            grid_rel = f"data/grids/{layer_id}.json.gz"
            raster_rel = f"data/rasters/{layer_id}.png"
            raster_bounds = bounds_from_crop(lats2d, lons2d, mask2d)

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
                    "sampleStride": SAMPLE_STRIDE,
                    "rasterUpscale": RASTER_UPSCALE,
                },
                "lat": round_list(pts_lat, 4),
                "lon": round_list(pts_lon, 4),
                "value": round_list(pts_val, 3),
            }

            write_json_gz(payload, Path("docs") / grid_rel)
            make_raster_png(vals2d, mask2d, Path("docs") / raster_rel)

            max_val = float(np.nanmax(vals2d[mask2d])) if np.any(mask2d) else 0.0
            layer_record = {
                "id": layer_id,
                "url": grid_rel,
                "rasterUrl": raster_rel,
                "rasterBounds": raster_bounds,
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
            log(f"  Wrote {layer_id}: {pts_val.size:,} sample points; raster max {max_val:.2f} in")
        except Exception as e:
            log(f"  Message {idx} failed: {e}")

    grbs.close()
    return layers


def clean_old_data() -> None:
    for p in (GRID_DIR, RASTER_DIR):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)


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
            all_layers.extend(process_grib_file(local_path, cycle, code, label, fhour))
            time.sleep(1.0)

    if not all_layers:
        raise RuntimeError("No HREF QPF layers were generated.")

    all_layers.sort(key=lambda x: (x["run"], x["forecastHour"], x["accumHours"], x["productLabel"]))
    catalog = {
        "generatedUTC": datetime.now(timezone.utc).isoformat(),
        "domain": DOMAIN,
        "source": "NCEP NOMADS HREF CONUS ensprod GRIB2",
        "defaultLayerId": all_layers[0]["id"],
        "colorScale": COLOR_SCALE,
        "layers": all_layers,
    }

    with (DATA_DIR / "catalog.json").open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    log(f"Wrote {DATA_DIR / 'catalog.json'} with {len(all_layers)} layer(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
