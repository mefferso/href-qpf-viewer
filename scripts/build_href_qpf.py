#!/usr/bin/env python3
"""Build HREF 6-hour Ensemble Max QPF data for the web viewer.

This script ingests HREF ensprod ``avrg`` GRIB2 files and publishes only
APCP/total-precipitation messages that are explicitly marked as statistical
maximum fields for 6-hour accumulation windows.
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
from scipy.interpolate import griddata

DOMAIN = {"west": -95.0, "east": -83.0, "south": 24.0, "north": 34.0}
FORECAST_HOURS = [6, 12, 18, 24, 30, 36, 42, 48]
PRODUCT_CODE = "avrg"
PRODUCT_LABEL = "Ensemble Max"
SAMPLE_STRIDE = 2
RASTER_UPSCALE = 4
MISSING_VALUE = -9999
CACHE_DIR = Path(".cache/href_grib")
DATA_DIR = Path("docs/data")
GRID_DIR = DATA_DIR / "grids"
VALUE_GRID_DIR = DATA_DIR / "value_grids"
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


@dataclass(frozen=True)
class Cycle:
    yyyymmdd: str
    hour: int

    @property
    def cycle_string(self) -> str:
        return f"{self.yyyymmdd}{self.hour:02d}"

    @property
    def cycle_label(self) -> str:
        return f"{self.yyyymmdd} {self.hour:02d}Z"

    @property
    def dt(self) -> datetime:
        return datetime.strptime(self.cycle_string, "%Y%m%d%H").replace(tzinfo=timezone.utc)


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}] {msg}", flush=True)


def candidate_cycles(now: Optional[datetime] = None) -> List[Cycle]:
    now = now or datetime.now(timezone.utc)
    out: List[Cycle] = []
    for day_offset in range(3):
        day = now - timedelta(days=day_offset)
        for hour in (12, 0):
            if datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc) <= now:
                out.append(Cycle(day.strftime("%Y%m%d"), hour))
    return sorted(out, key=lambda c: c.cycle_string, reverse=True)


def product_filename(cycle: Cycle, code: str, fhour: int) -> str:
    return f"href.t{cycle.hour:02d}z.conus.{code}.f{fhour:02d}.grib2"


def product_url(cycle: Cycle, code: str, fhour: int) -> str:
    return f"{NOMADS_BASE}/href.{cycle.yyyymmdd}/ensprod/{product_filename(cycle, code, fhour)}"


def cycle_base_url(cycle: Cycle) -> str:
    return f"{NOMADS_BASE}/href.{cycle.yyyymmdd}"


def url_exists(session: requests.Session, url: str) -> bool:
    try:
        r = session.get(url, headers={**HEADERS, "Range": "bytes=0-99"}, timeout=20, stream=True)
        return r.status_code in (200, 206)
    except Exception:
        return False


def list_grib_files(session: requests.Session, dir_url: str) -> List[str]:
    try:
        r = session.get(f"{dir_url}/", headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return []
    except Exception:
        return []
    return sorted(set(re.findall(r'href\.t\d{2}z\.conus\.[a-z0-9_]+\.f\d{2}\.grib2', r.text, flags=re.IGNORECASE)))


def find_latest_cycle(session: requests.Session) -> Cycle:
    for cycle in candidate_cycles():
        sentinel = product_url(cycle, PRODUCT_CODE, 6)
        log(f"Checking {cycle.cycle_label}: {sentinel}")
        if url_exists(session, sentinel):
            log(f"Using HREF cycle {cycle.cycle_label}")
            return cycle
    raise RuntimeError("Could not find a recent HREF cycle on NOMADS.")


def find_member_directories(session: requests.Session, cycle: Cycle) -> List[str]:
    base = cycle_base_url(cycle)
    candidates = ["mem", "members", "member", "hiresw", ""]
    found: List[str] = []
    for d in candidates:
        url = f"{base}/{d}" if d else base
        files = list_grib_files(session, url)
        if files:
            found.append(url)
    return found


def member_file_candidates(session: requests.Session, cycle: Cycle, fhour: int) -> List[str]:
    member_urls = find_member_directories(session, cycle)
    include = []
    forecast_tag = f"f{fhour:02d}.grib2"
    stat_tokens = {"avrg", "mean", "eas", "lpmm", "pmmn", "sprd", "prob"}
    for base_url in member_urls:
        for name in list_grib_files(session, base_url):
            if not name.endswith(forecast_tag):
                continue
            parts = name.split(".")
            code = parts[4].lower() if len(parts) > 4 else ""
            if code in stat_tokens:
                continue
            include.append(f"{base_url}/{name}")
    return sorted(set(include))


def download_file(session: requests.Session, url: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return True
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    log(f"Downloading {url}")
    try:
        with session.get(url, headers=HEADERS, timeout=90, stream=True) as r:
            if r.status_code != 200:
                log(f"  Skip: HTTP {r.status_code}")
                return False
            ctype = r.headers.get("content-type", "").lower()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
        if tmp.stat().st_size < 10_000 or "text/html" in ctype:
            log("  Skip: response was not a usable GRIB")
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(out_path)
        return True
    except Exception as e:
        log(f"  Download failed: {e}")
        tmp.unlink(missing_ok=True)
        return False


def safe_get(grb, attr: str, default=None):
    try:
        return getattr(grb, attr)
    except Exception:
        return default


def grib_inventory(grib_path: Path, limit: int = 80) -> str:
    rows: List[str] = []
    try:
        grbs = pygrib.open(str(grib_path))
        for i, g in enumerate(grbs, start=1):
            if i > limit:
                rows.append(f"...and more messages after {limit}")
                break
            rows.append(
                f"#{i}: shortName={safe_get(g, 'shortName', '')} | name={safe_get(g, 'name', '')} | "
                f"param={safe_get(g, 'parameterName', '')} | units={safe_get(g, 'units', '')} | "
                f"stepRange={safe_get(g, 'stepRange', '')} | stepType={safe_get(g, 'stepType', '')} | "
                f"derivedForecast={safe_get(g, 'derivedForecast', '')} | "
                f"typeOfStatisticalProcessing={safe_get(g, 'typeOfStatisticalProcessing', '')} | "
                f"typeOfStatisticalProcessingOfOverallTimeInterval={safe_get(g, 'typeOfStatisticalProcessingOfOverallTimeInterval', '')} | "
                f"length={safe_get(g, 'lengthOfTimeRange', '')}"
            )
        grbs.close()
    except Exception as e:
        rows.append(f"Could not inventory GRIB: {e}")
    return "\n".join(rows)


def message_is_precip(grb) -> bool:
    parts = [str(safe_get(grb, x, "")).lower() for x in ("shortName", "name", "parameterName")]
    text = " ".join(parts)
    return "apcp" in text or "total precipitation" in text or "total precip" in text or ("precipitation" in text and "probability" not in text)


def message_is_maximum(grb) -> bool:
    stat_raw = safe_get(grb, "typeOfStatisticalProcessing", None)
    stat_overall_raw = safe_get(grb, "typeOfStatisticalProcessingOfOverallTimeInterval", None)
    try:
        stat = int(stat_raw) if stat_raw is not None else None
    except Exception:
        stat = None
    try:
        stat_overall = int(stat_overall_raw) if stat_overall_raw is not None else None
    except Exception:
        stat_overall = None
    step_type = str(safe_get(grb, "stepType", "")).lower()
    derived = str(safe_get(grb, "derivedForecast", "")).lower()
    name = str(safe_get(grb, "name", "")).lower()
    parameter_name = str(safe_get(grb, "parameterName", "")).lower()
    if stat == 2 or stat_overall == 2:
        return True
    if step_type == "max":
        return True
    if "maximum" in derived:
        return True
    if "maximum" in name or "maximum" in parameter_name:
        return True
    return False


def parse_step_range(step_range: str, fhour: int, accum: int) -> Tuple[int, int]:
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", str(step_range or ""))
    if m:
        return int(m.group(1)), int(m.group(2))
    return max(0, fhour - accum), fhour


def accum_hours_from_message(grb) -> Optional[int]:
    val = safe_get(grb, "lengthOfTimeRange", None)
    try:
        if val is not None and int(val) > 0:
            return int(val)
    except Exception:
        pass
    start, end = parse_step_range(str(safe_get(grb, "stepRange", "")), 0, 0)
    return end - start if end > start else None


def value_units_to_inches(values: np.ndarray, units: str) -> np.ndarray:
    u = (units or "").lower().replace("**", "^")
    if "in" in u and "mm" not in u and "kg" not in u:
        return values.astype(float)
    return values.astype(float) / 25.4


def crop_2d(lats: np.ndarray, lons: np.ndarray, vals: np.ndarray):
    lons = np.where(lons > 180, lons - 360, lons)
    mask = (lats >= DOMAIN["south"]) & (lats <= DOMAIN["north"]) & (lons >= DOMAIN["west"]) & (lons <= DOMAIN["east"]) & np.isfinite(vals)
    rows, cols = np.where(mask)
    if rows.size == 0:
        empty = np.array([])
        return empty, empty, empty, empty
    rmin, rmax, cmin, cmax = rows.min(), rows.max(), cols.min(), cols.max()
    return lats[rmin:rmax + 1, cmin:cmax + 1], lons[rmin:rmax + 1, cmin:cmax + 1], vals[rmin:rmax + 1, cmin:cmax + 1], mask[rmin:rmax + 1, cmin:cmax + 1]


def orient_for_leaflet(lats2d, lons2d, vals2d, mask2d):
    if float(np.nanmean(lats2d[0, :])) < float(np.nanmean(lats2d[-1, :])):
        lats2d, lons2d, vals2d, mask2d = np.flipud(lats2d), np.flipud(lons2d), np.flipud(vals2d), np.flipud(mask2d)
    if float(np.nanmean(lons2d[:, 0])) > float(np.nanmean(lons2d[:, -1])):
        lats2d, lons2d, vals2d, mask2d = np.fliplr(lats2d), np.fliplr(lons2d), np.fliplr(vals2d), np.fliplr(mask2d)
    return lats2d, lons2d, vals2d, mask2d


def bounds_from_crop(lats2d, lons2d, mask2d) -> dict:
    return {
        "south": round(float(np.nanmin(lats2d[mask2d])), 5),
        "north": round(float(np.nanmax(lats2d[mask2d])), 5),
        "west": round(float(np.nanmin(lons2d[mask2d])), 5),
        "east": round(float(np.nanmax(lons2d[mask2d])), 5),
    }


def max_point_from_grid(lats2d, lons2d, vals2d, mask2d) -> dict:
    valid = mask2d & np.isfinite(vals2d)
    if not np.any(valid):
        return {"lat": None, "lon": None, "value": 0.0}
    work = np.where(valid, vals2d, np.nan)
    row, col = np.unravel_index(np.nanargmax(work), work.shape)
    return {"lat": round(float(lats2d[row, col]), 4), "lon": round(float(lons2d[row, col]), 4), "value": round(float(vals2d[row, col]), 3)}


def sample_points(lats2d, lons2d, vals2d, mask2d):
    m = mask2d[::SAMPLE_STRIDE, ::SAMPLE_STRIDE] & np.isfinite(vals2d[::SAMPLE_STRIDE, ::SAMPLE_STRIDE])
    return lats2d[::SAMPLE_STRIDE, ::SAMPLE_STRIDE][m], lons2d[::SAMPLE_STRIDE, ::SAMPLE_STRIDE][m], vals2d[::SAMPLE_STRIDE, ::SAMPLE_STRIDE][m]


def round_list(arr: np.ndarray, ndigits: int) -> List[float]:
    return [round(float(x), ndigits) for x in arr]


def write_json_gz(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


def resize_float_grid(values: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    arr = np.nan_to_num(values.astype(np.float32), nan=0.0)
    return np.array(Image.fromarray(arr, mode="F").resize(out_size, resample=Image.Resampling.BILINEAR), dtype=np.float32)


def resize_mask(mask: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(out_size, resample=Image.Resampling.NEAREST)) > 0


def hex_to_rgba(hex_color: str, alpha: int = 215):
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha


def write_value_grid_and_raster(vals2d, mask2d, bounds: dict, value_path: Path, raster_path: Path) -> None:
    h, w = vals2d.shape
    out_size = (max(1, w * RASTER_UPSCALE), max(1, h * RASTER_UPSCALE))
    vals_hi = resize_float_grid(vals2d, out_size)
    mask_hi = resize_mask(mask2d, out_size)
    grid = np.where(mask_hi & np.isfinite(vals_hi), vals_hi, MISSING_VALUE)
    write_json_gz({
        "bounds": bounds,
        "width": int(grid.shape[1]),
        "height": int(grid.shape[0]),
        "missing": MISSING_VALUE,
        "units": "inches",
        "values": [round(float(v), 3) for v in grid.ravel()],
    }, value_path)
    rgba = np.zeros((out_size[1], out_size[0], 4), dtype=np.uint8)
    visible = mask_hi & np.isfinite(vals_hi) & (vals_hi >= 0.001)
    for b in COLOR_SCALE:
        m = visible & (vals_hi >= b["min"]) & (vals_hi < b["max"])
        rgba[m] = hex_to_rgba(b["color"], 215)
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(raster_path, optimize=True)


def reproject_to_regular_grid(lats2d, lons2d, vals2d, mask2d):
    valid = mask2d & np.isfinite(vals2d)
    if not np.any(valid):
        raise RuntimeError("No valid points available for reprojection.")
    src_points = np.column_stack((lons2d[valid], lats2d[valid]))
    src_values = vals2d[valid]
    ny, nx = vals2d.shape
    lon_axis = np.linspace(float(np.nanmin(lons2d[valid])), float(np.nanmax(lons2d[valid])), nx)
    lat_axis = np.linspace(float(np.nanmax(lats2d[valid])), float(np.nanmin(lats2d[valid])), ny)
    lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)
    interp_vals = griddata(src_points, src_values, (lon_grid, lat_grid), method="linear")
    if np.isnan(interp_vals).any():
        nearest_vals = griddata(src_points, src_values, (lon_grid, lat_grid), method="nearest")
        interp_vals = np.where(np.isnan(interp_vals), nearest_vals, interp_vals)
    out_mask = np.isfinite(interp_vals)
    return lat_grid, lon_grid, interp_vals, out_mask


def iso_for_step(cycle: Cycle, hour: int) -> str:
    return (cycle.dt + timedelta(hours=hour)).isoformat()


def process_file(grib_path: Path, cycle: Cycle, fhour: int) -> Optional[dict]:
    try:
        grbs = pygrib.open(str(grib_path))
        messages = [g for g in grbs if message_is_precip(g)]
    except Exception as e:
        log(f"Could not open {grib_path.name}: {e}")
        return None
    max_messages = [g for g in messages if message_is_maximum(g)]
    if not max_messages:
        grbs.close()
        inventory = grib_inventory(grib_path)
        raise RuntimeError(
            f"No max APCP message found in {grib_path.name}.\n"
            f"Detailed GRIB inventory:\n{inventory}"
        )
    best = None
    for grb in max_messages:
        accum = accum_hours_from_message(grb) or 6
        step_range = str(safe_get(grb, "stepRange", ""))
        step_start, step_end = parse_step_range(step_range, fhour, accum)
        score = (1 if accum == 6 else 0, 1 if step_end == fhour else 0, -abs(accum - 6))
        if best is None or score > best[0]:
            best = (score, grb, accum, step_start, step_end, step_range)
    _, grb, accum, step_start, step_end, step_range = best
    vals_raw = grb.values
    if np.ma.isMaskedArray(vals_raw):
        vals_raw = vals_raw.filled(np.nan)
    vals = value_units_to_inches(np.array(vals_raw, dtype=float), str(safe_get(grb, "units", "")))
    vals[vals < 0] = np.nan
    lats, lons = grb.latlons()
    grbs.close()
    lats2d, lons2d, vals2d, mask2d = crop_2d(lats, lons, vals)
    if vals2d.size == 0:
        return None
    vals2d = np.where(vals2d < 0.001, 0.0, vals2d)
    lats2d, lons2d, vals2d, mask2d = orient_for_leaflet(lats2d, lons2d, vals2d, mask2d)
    lats2d, lons2d, vals2d, mask2d = reproject_to_regular_grid(lats2d, lons2d, vals2d, mask2d)
    bounds = bounds_from_crop(lats2d, lons2d, mask2d)
    max_point = max_point_from_grid(lats2d, lons2d, vals2d, mask2d)
    pts_lat, pts_lon, pts_val = sample_points(lats2d, lons2d, vals2d, mask2d)
    layer_id = f"{cycle.cycle_string}_max_f{fhour:02d}_a06"
    grid_rel = f"data/grids/{layer_id}.json.gz"
    value_rel = f"data/value_grids/{layer_id}.json.gz"
    raster_rel = f"data/rasters/{layer_id}.png"
    filename = product_filename(cycle, PRODUCT_CODE, fhour)
    period_label = f"F{step_start:02d}-F{step_end:02d} ({accum}-hr Ensemble Max QPF)"
    start_iso, end_iso = iso_for_step(cycle, step_start), iso_for_step(cycle, step_end)
    write_json_gz({
        "metadata": {
            "id": layer_id,
            "run": cycle.cycle_string,
            "runLabel": cycle.cycle_label,
            "product": "max",
            "productLabel": PRODUCT_LABEL,
            "forecastHour": fhour,
            "accumHours": accum,
            "periodLabel": period_label,
            "units": "inches",
            "name": "HREF ensprod avrg Ensemble Max QPF",
            "stepRange": step_range,
            "startTimeUTC": start_iso,
            "validTimeUTC": end_iso,
            "sourceFile": filename,
        },
        "lat": round_list(pts_lat, 4),
        "lon": round_list(pts_lon, 4),
        "value": round_list(pts_val, 3),
    }, Path("docs") / grid_rel)
    write_value_grid_and_raster(vals2d, mask2d, bounds, Path("docs") / value_rel, Path("docs") / raster_rel)
    max_val = float(max_point["value"] or 0.0)
    log(f"  Wrote {filename}: {period_label}; max {max_val:.2f} in")
    return {
        "id": layer_id,
        "url": grid_rel,
        "valueGridUrl": value_rel,
        "rasterUrl": raster_rel,
        "rasterBounds": bounds,
        "maxPoint": max_point,
        "periodKey": f"f{fhour:02d}_a{accum:02d}_{step_start:02d}_{step_end:02d}",
        "periodLabel": period_label,
        "stepStart": step_start,
        "stepEnd": step_end,
        "run": cycle.cycle_string,
        "runLabel": cycle.cycle_label,
        "product": "max",
        "productLabel": PRODUCT_LABEL,
        "forecastHour": fhour,
        "accumHours": accum,
        "units": "inches",
        "startTimeUTC": start_iso,
        "validTimeUTC": end_iso,
        "stepRange": step_range,
        "name": "HREF ensprod avrg Ensemble Max QPF",
        "sourceFile": filename,
        "maxValue": round(max_val, 2),
        "pointCount": int(pts_val.size),
        "native": False,
    }


def extract_member_apcp(grib_path: Path, fhour: int):
    grbs = pygrib.open(str(grib_path))
    messages = [g for g in grbs if message_is_precip(g)]
    if not messages:
        grbs.close()
        return None
    best = None
    for grb in messages:
        accum = accum_hours_from_message(grb) or 0
        step_range = str(safe_get(grb, "stepRange", ""))
        start, end = parse_step_range(step_range, fhour, accum if accum > 0 else 6)
        score = (1 if end == fhour else 0, accum)
        if best is None or score > best[0]:
            best = (score, grb, start, end)
    _, grb, start, end = best
    vals_raw = grb.values
    if np.ma.isMaskedArray(vals_raw):
        vals_raw = vals_raw.filled(np.nan)
    vals = value_units_to_inches(np.array(vals_raw, dtype=float), str(safe_get(grb, "units", "")))
    lats, lons = grb.latlons()
    grbs.close()
    return {"lats": lats, "lons": lons, "vals": vals, "start": start, "end": end}


def compute_member_6hr(session: requests.Session, cycle: Cycle, member_url: str, fhour: int):
    name = member_url.rsplit("/", 1)[-1]
    local = CACHE_DIR / cycle.cycle_string / "members" / name
    if not download_file(session, member_url, local):
        return None
    curr = extract_member_apcp(local, fhour)
    if curr is None:
        return None
    vals = curr["vals"]
    if curr["end"] == fhour and curr["start"] == max(0, fhour - 6):
        return curr["lats"], curr["lons"], vals, name
    if fhour < 6:
        return None
    prev_name = re.sub(rf"\.f{fhour:02d}\.grib2$", f".f{fhour - 6:02d}.grib2", name)
    prev_url = member_url.rsplit("/", 1)[0] + "/" + prev_name
    prev_local = CACHE_DIR / cycle.cycle_string / "members" / prev_name
    if not download_file(session, prev_url, prev_local):
        return None
    prev = extract_member_apcp(prev_local, fhour - 6)
    if prev is None:
        return None
    if prev["vals"].shape != vals.shape:
        return None
    inc = vals - prev["vals"]
    return curr["lats"], curr["lons"], inc, name


def build_layer_from_members(session: requests.Session, cycle: Cycle, fhour: int) -> Optional[dict]:
    files = member_file_candidates(session, cycle, fhour)
    if not files:
        log(f"F{fhour:02d}: no member files discovered; skipping.")
        return None
    member_fields = []
    used = []
    base_lats = base_lons = None
    for url in files:
        out = compute_member_6hr(session, cycle, url, fhour)
        if out is None:
            continue
        lats, lons, vals, fname = out
        if base_lats is None:
            base_lats, base_lons = lats, lons
            member_fields.append(vals)
            used.append(fname)
        else:
            if lats.shape != base_lats.shape:
                points = np.column_stack((lons[np.isfinite(vals)], lats[np.isfinite(vals)]))
                values = vals[np.isfinite(vals)]
                target = (base_lons, base_lats)
                vals = griddata(points, values, target, method="linear")
                if np.isnan(vals).any():
                    near = griddata(points, values, target, method="nearest")
                    vals = np.where(np.isnan(vals), near, vals)
            member_fields.append(vals)
            used.append(fname)
    if not member_fields:
        log(f"F{fhour:02d}: no usable member APCP data found; skipping.")
        return None
    log(f"F{fhour:02d}: using {len(used)} member files: {', '.join(used)}")
    ens_max = np.nanmax(np.stack(member_fields, axis=0), axis=0)
    tmp_name = CACHE_DIR / cycle.cycle_string / f"computed_member_max_f{fhour:02d}.npz"
    tmp_name.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tmp_name, lats=base_lats, lons=base_lons, vals=ens_max)
    # Reuse downstream metadata/output pipeline via in-memory flow:
    lats2d, lons2d, vals2d, mask2d = crop_2d(base_lats, base_lons, ens_max)
    if vals2d.size == 0:
        return None
    vals2d = np.where(vals2d < 0.001, 0.0, vals2d)
    lats2d, lons2d, vals2d, mask2d = orient_for_leaflet(lats2d, lons2d, vals2d, mask2d)
    lats2d, lons2d, vals2d, mask2d = reproject_to_regular_grid(lats2d, lons2d, vals2d, mask2d)
    bounds = bounds_from_crop(lats2d, lons2d, mask2d)
    max_point = max_point_from_grid(lats2d, lons2d, vals2d, mask2d)
    pts_lat, pts_lon, pts_val = sample_points(lats2d, lons2d, vals2d, mask2d)
    layer_id = f"{cycle.cycle_string}_max_f{fhour:02d}_a06"
    grid_rel = f"data/grids/{layer_id}.json.gz"
    value_rel = f"data/value_grids/{layer_id}.json.gz"
    raster_rel = f"data/rasters/{layer_id}.png"
    period_label = f"F{max(0, fhour-6):02d}-F{fhour:02d} (6-hr Ensemble Max QPF)"
    start_iso, end_iso = iso_for_step(cycle, max(0, fhour - 6)), iso_for_step(cycle, fhour)
    write_json_gz({
        "metadata": {
            "id": layer_id, "run": cycle.cycle_string, "runLabel": cycle.cycle_label, "product": "max",
            "productLabel": PRODUCT_LABEL, "forecastHour": fhour, "accumHours": 6, "periodLabel": period_label,
            "units": "inches", "name": "HREF member-computed Ensemble Max QPF", "stepRange": f"{max(0, fhour-6)}-{fhour}",
            "startTimeUTC": start_iso, "validTimeUTC": end_iso, "sourceFile": f"{len(used)} member files",
        },
        "lat": round_list(pts_lat, 4), "lon": round_list(pts_lon, 4), "value": round_list(pts_val, 3),
    }, Path("docs") / grid_rel)
    write_value_grid_and_raster(vals2d, mask2d, bounds, Path("docs") / value_rel, Path("docs") / raster_rel)
    max_val = float(max_point["value"] or 0.0)
    return {
        "id": layer_id, "url": grid_rel, "valueGridUrl": value_rel, "rasterUrl": raster_rel, "rasterBounds": bounds,
        "maxPoint": max_point, "periodKey": f"f{fhour:02d}_a06_{max(0, fhour-6):02d}_{fhour:02d}", "periodLabel": period_label,
        "stepStart": max(0, fhour - 6), "stepEnd": fhour, "run": cycle.cycle_string, "runLabel": cycle.cycle_label,
        "product": "max", "productLabel": PRODUCT_LABEL, "forecastHour": fhour, "accumHours": 6, "units": "inches",
        "startTimeUTC": start_iso, "validTimeUTC": end_iso, "stepRange": f"{max(0, fhour-6)}-{fhour}",
        "name": "HREF member-computed Ensemble Max QPF", "sourceFile": f"{len(used)} member files",
        "maxValue": round(max_val, 2), "pointCount": int(pts_val.size), "native": False,
    }


def clean_old_data() -> None:
    for p in (GRID_DIR, VALUE_GRID_DIR, RASTER_DIR):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clean_old_data()
    session = requests.Session()
    cycle = find_latest_cycle(session)
    layers: List[dict] = []
    for fhour in FORECAST_HOURS:
        layer = build_layer_from_members(session, cycle, fhour)
        if layer:
            layers.append(layer)
        time.sleep(0.5)
    if not layers:
        raise RuntimeError("No forecast hours were successfully built from individual HREF member APCP fields.")
    layers.sort(key=lambda x: (x["run"], x["forecastHour"]))
    catalog = {
        "generatedUTC": datetime.now(timezone.utc).isoformat(),
        "domain": DOMAIN,
        "source": "NCEP NOMADS HREF member GRIB2 APCP fields; Ensemble Max computed as gridpoint max across members",
        "defaultLayerId": layers[0]["id"],
        "colorScale": COLOR_SCALE,
        "layers": layers,
    }
    with (DATA_DIR / "catalog.json").open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    log(f"Wrote {DATA_DIR / 'catalog.json'} with {len(layers)} Ensemble Max layer(s) computed from members")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
