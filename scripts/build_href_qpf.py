#!/usr/bin/env python3
"""Build HREF Max QPF data for the web viewer.

This keeps the known-working NOMADS ensprod data path, derives Max layers from
available HREF ensemble summary products when a native Max file is unavailable,
and publishes only one Max layer at each 6-hour forecast interval.
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
SOURCE_PRODUCTS = [
    {"file_code": "avrg", "label": "Mean"},
    {"file_code": "pmmn", "label": "PMM"},
    {"file_code": "lpmm", "label": "LPMM"},
]
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


def hex_to_rgba(hex_color: str, alpha: int = 215) -> Tuple[int, int, int, int]:
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

    @property
    def dt(self) -> datetime:
        return datetime.strptime(f"{self.yyyymmdd}{self.hour:02d}", "%Y%m%d%H").replace(tzinfo=timezone.utc)


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
    return f"{NOMADS_BASE}/href.{cycle.yyyymmdd}/ensprod/href.t{cycle.hour:02d}z.conus.{product_code}.f{fhour:02d}.grib2"


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
    return short_name in {"tp", "apcp"} or "total precipitation" in haystack or "total precip" in haystack or ("precipitation" in haystack and "probability" not in haystack)


def parse_step_range(step_range: str, fhour: int, accum: int) -> Tuple[int, int]:
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", str(step_range or ""))
    if m:
        return int(m.group(1)), int(m.group(2))
    end = fhour
    return max(0, end - accum), end


def accum_hours_from_message(grb) -> Optional[int]:
    val = safe_get(grb, "lengthOfTimeRange", None)
    try:
        if val is not None and int(val) > 0:
            return int(val)
    except Exception:
        pass
    start, end = parse_step_range(str(safe_get(grb, "stepRange", "")), 0, 0)
    if end > start:
        return end - start
    return None


def value_units_to_inches(values: np.ndarray, units: str) -> np.ndarray:
    u = (units or "").lower().replace("**", "^")
    if "in" in u and "kg" not in u and "mm" not in u:
        return values.astype(float)
    return values.astype(float) / 25.4


def crop_2d(lats: np.ndarray, lons: np.ndarray, vals: np.ndarray):
    lons = np.where(lons > 180, lons - 360, lons)
    mask = (lats >= DOMAIN["south"]) & (lats <= DOMAIN["north"]) & (lons >= DOMAIN["west"]) & (lons <= DOMAIN["east"]) & np.isfinite(vals)
    rows, cols = np.where(mask)
    if rows.size == 0:
        empty = np.array([])
        return empty, empty, empty, empty
    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    return lats[rmin:rmax+1, cmin:cmax+1], lons[rmin:rmax+1, cmin:cmax+1], vals[rmin:rmax+1, cmin:cmax+1], mask[rmin:rmax+1, cmin:cmax+1]


def orient_for_leaflet(lats2d, lons2d, vals2d, mask2d):
    out_lats, out_lons, out_vals, out_mask = lats2d, lons2d, vals2d, mask2d
    if float(np.nanmean(out_lats[0, :])) < float(np.nanmean(out_lats[-1, :])):
        out_lats, out_lons, out_vals, out_mask = np.flipud(out_lats), np.flipud(out_lons), np.flipud(out_vals), np.flipud(out_mask)
    if float(np.nanmean(out_lons[:, 0])) > float(np.nanmean(out_lons[:, -1])):
        out_lats, out_lons, out_vals, out_mask = np.fliplr(out_lats), np.fliplr(out_lons), np.fliplr(out_vals), np.fliplr(out_mask)
    return out_lats, out_lons, out_vals, out_mask


def sample_points(lats2d, lons2d, vals2d, mask2d):
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


def read_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def resize_float_grid(values: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    arr = np.nan_to_num(values.astype(np.float32), nan=0.0)
    img = Image.fromarray(arr, mode="F")
    img = img.resize(out_size, resample=Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.float32)


def resize_mask(mask: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize(out_size, resample=Image.Resampling.NEAREST)
    return np.array(img) > 0


def make_raster_png(vals2d, mask2d, out_path: Path) -> None:
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


def write_value_grid(vals2d, mask2d, bounds: dict, out_path: Path) -> None:
    grid = np.where(mask2d & np.isfinite(vals2d), vals2d, MISSING_VALUE)
    payload = {"bounds": bounds, "width": int(grid.shape[1]), "height": int(grid.shape[0]), "missing": MISSING_VALUE, "units": "inches", "values": [round(float(v), 3) for v in grid.ravel()]}
    write_json_gz(payload, out_path)


def iso_for_step(cycle: Cycle, step_hour: int) -> str:
    return (cycle.dt + timedelta(hours=step_hour)).isoformat()


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
            step_start, step_end = parse_step_range(step_range, fhour, accum)
            period_label = f"F{step_start:02d}-F{step_end:02d} ({accum}-hr QPF)"
            period_key = f"f{fhour:02d}_a{accum:02d}_{step_start:02d}_{step_end:02d}_{idx}"
            valid_time_utc = iso_for_step(cycle, step_end)
            start_time_utc = iso_for_step(cycle, step_start)
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
            lats2d, lons2d, vals2d, mask2d = orient_for_leaflet(lats2d, lons2d, vals2d, mask2d)
            pts_lat, pts_lon, pts_val = sample_points(lats2d, lons2d, vals2d, mask2d)
            if pts_val.size == 0:
                continue
            layer_id = f"{cycle.cycle_string}_{product_code}_f{fhour:02d}_a{accum:02d}_{idx}"
            grid_rel = f"data/grids/{layer_id}.json.gz"
            value_grid_rel = f"data/value_grids/{layer_id}.json.gz"
            raster_rel = f"data/rasters/{layer_id}.png"
            raster_bounds = bounds_from_crop(lats2d, lons2d, mask2d)
            max_point = max_point_from_grid(lats2d, lons2d, vals2d, mask2d)
            metadata = {"id": layer_id, "run": cycle.cycle_string, "runLabel": cycle.cycle_label, "product": product_code, "productLabel": product_label, "forecastHour": fhour, "accumHours": accum, "periodLabel": period_label, "periodKey": period_key, "units": "inches", "sourceUnits": units, "name": name, "stepRange": step_range, "startTimeUTC": start_time_utc, "validTimeUTC": valid_time_utc, "analysisTimeUTC": anal_date.isoformat() if hasattr(anal_date, "isoformat") else None, "domain": DOMAIN, "sampleStride": SAMPLE_STRIDE, "rasterUpscale": RASTER_UPSCALE}
            payload = {"metadata": metadata, "lat": round_list(pts_lat, 4), "lon": round_list(pts_lon, 4), "value": round_list(pts_val, 3)}
            write_json_gz(payload, Path("docs") / grid_rel)
            write_value_grid(vals2d, mask2d, raster_bounds, Path("docs") / value_grid_rel)
            make_raster_png(vals2d, mask2d, Path("docs") / raster_rel)
            max_val = float(max_point["value"] or 0.0)
            layers.append({"id": layer_id, "url": grid_rel, "valueGridUrl": value_grid_rel, "rasterUrl": raster_rel, "rasterBounds": raster_bounds, "maxPoint": max_point, "periodKey": period_key, "periodLabel": period_label, "stepStart": step_start, "stepEnd": step_end, "run": cycle.cycle_string, "runLabel": cycle.cycle_label, "product": product_code, "productLabel": product_label, "forecastHour": fhour, "accumHours": accum, "units": "inches", "startTimeUTC": start_time_utc, "validTimeUTC": valid_time_utc, "stepRange": step_range, "name": name, "maxValue": round(max_val, 2), "pointCount": int(pts_val.size)})
            log(f"  Wrote {layer_id}: {period_label}; ending {valid_time_utc}; max {max_val:.2f} in")
        except Exception as e:
            log(f"  Message {idx} failed: {e}")
    grbs.close()
    return layers


def sample_regular_value_grid(grid: dict, lat: float, lon: float) -> float:
    b = grid["bounds"]
    if lat < b["south"] or lat > b["north"] or lon < b["west"] or lon > b["east"]:
        return np.nan
    width = int(grid["width"])
    height = int(grid["height"])
    x = round((lon - b["west"]) / (b["east"] - b["west"]) * (width - 1))
    y = round((b["north"] - lat) / (b["north"] - b["south"]) * (height - 1))
    if x < 0 or y < 0 or x >= width or y >= height:
        return np.nan
    v = float(grid["values"][int(y) * width + int(x)])
    if v == grid.get("missing", MISSING_VALUE) or v < -1000:
        return np.nan
    return v


def derive_missing_max_layers(all_layers: List[dict]) -> List[dict]:
    derived: List[dict] = []
    groups = {}
    for layer in all_layers:
        key = (layer["run"], layer["forecastHour"], layer["accumHours"], layer["stepStart"], layer["stepEnd"])
        groups.setdefault(key, []).append(layer)

    for key, layers in groups.items():
        if any(l["product"] == "max" for l in layers):
            continue
        source_layers = [l for l in layers if l["product"] in {"avrg", "pmmn", "lpmm"}]
        if len(source_layers) < 2:
            continue

        grids = []
        for layer in source_layers:
            path = Path("docs") / layer["valueGridUrl"]
            if path.exists():
                grids.append((layer, read_json_gz(path)))
        if len(grids) < 2:
            continue

        base_layer, base_grid = max(grids, key=lambda item: int(item[1]["width"]) * int(item[1]["height"]))
        width = int(base_grid["width"])
        height = int(base_grid["height"])
        b = base_grid["bounds"]
        lons_1d = np.linspace(b["west"], b["east"], width)
        lats_1d = np.linspace(b["north"], b["south"], height)
        lons2d, lats2d = np.meshgrid(lons_1d, lats_1d)
        vals2d = np.full((height, width), np.nan, dtype=float)

        for r in range(height):
            for c in range(width):
                lat = float(lats2d[r, c])
                lon = float(lons2d[r, c])
                values = [sample_regular_value_grid(g, lat, lon) for _, g in grids]
                finite = [v for v in values if np.isfinite(v)]
                if finite:
                    vals2d[r, c] = max(finite)

        mask2d = np.isfinite(vals2d)
        if not np.any(mask2d):
            continue
        vals2d = np.where(vals2d < 0.001, 0.0, vals2d)
        pts_lat, pts_lon, pts_val = sample_points(lats2d, lons2d, vals2d, mask2d)
        run, fhour, accum, step_start, step_end = key
        layer_id = f"{run}_max_f{fhour:02d}_a{accum:02d}_{step_start:02d}_{step_end:02d}"
        grid_rel = f"data/grids/{layer_id}.json.gz"
        value_grid_rel = f"data/value_grids/{layer_id}.json.gz"
        raster_rel = f"data/rasters/{layer_id}.png"
        bounds = base_grid["bounds"]
        max_point = max_point_from_grid(lats2d, lons2d, vals2d, mask2d)
        period_label = f"F{step_start:02d}-F{step_end:02d} ({accum}-hr Max QPF)"
        period_key = f"f{fhour:02d}_a{accum:02d}_{step_start:02d}_{step_end:02d}"
        source_labels = [l["productLabel"] for l, _ in grids]
        payload = {
            "metadata": {"id": layer_id, "run": run, "runLabel": base_layer["runLabel"], "product": "max", "productLabel": "Max", "forecastHour": fhour, "accumHours": accum, "periodLabel": period_label, "periodKey": period_key, "units": "inches", "name": "Derived maximum precipitation", "stepRange": base_layer.get("stepRange", ""), "startTimeUTC": base_layer.get("startTimeUTC"), "validTimeUTC": base_layer.get("validTimeUTC"), "domain": DOMAIN, "sampleStride": SAMPLE_STRIDE, "rasterUpscale": RASTER_UPSCALE, "derivedFromProducts": source_labels},
            "lat": round_list(pts_lat, 4), "lon": round_list(pts_lon, 4), "value": round_list(pts_val, 3),
        }
        write_json_gz(payload, Path("docs") / grid_rel)
        write_value_grid(vals2d, mask2d, bounds, Path("docs") / value_grid_rel)
        make_raster_png(vals2d, mask2d, Path("docs") / raster_rel)
        max_val = float(max_point["value"] or 0.0)
        layer_record = {"id": layer_id, "url": grid_rel, "valueGridUrl": value_grid_rel, "rasterUrl": raster_rel, "rasterBounds": bounds, "maxPoint": max_point, "periodKey": period_key, "periodLabel": period_label, "stepStart": step_start, "stepEnd": step_end, "run": run, "runLabel": base_layer["runLabel"], "product": "max", "productLabel": "Max", "forecastHour": fhour, "accumHours": accum, "units": "inches", "startTimeUTC": base_layer.get("startTimeUTC"), "validTimeUTC": base_layer.get("validTimeUTC"), "stepRange": base_layer.get("stepRange", ""), "name": "Derived maximum precipitation", "maxValue": round(max_val, 2), "pointCount": int(pts_val.size), "derived": True, "derivedFromProducts": source_labels}
        derived.append(layer_record)
        log(f"  Derived Max {layer_id}: {period_label}; max {max_val:.2f} in")

    return derived


def select_published_max_layers(all_layers: List[dict]) -> List[dict]:
    """Publish one Max layer per forecast hour.

    Prefer a true 6-hour accumulation when present. If the GRIB set does not
    include one for a forecast hour, use the largest accumulation up to 6 hours.
    This keeps the site simple and prevents a total failure when HREF metadata
    only provides 1-hour or 3-hour fields for a given forecast hour.
    """
    max_layers = [l for l in all_layers if l.get("product") == "max" and int(l.get("forecastHour", -1)) in FORECAST_HOURS]
    groups = {}
    for layer in max_layers:
        groups.setdefault((layer["run"], int(layer["forecastHour"])), []).append(layer)

    selected: List[dict] = []
    for _, layers in sorted(groups.items(), key=lambda item: item[0][1]):
        def rank(layer: dict):
            accum = int(layer.get("accumHours") or 0)
            exact_six = 1 if accum == 6 else 0
            usable_short = 1 if accum <= 6 else 0
            return (exact_six, usable_short, min(accum, 6), float(layer.get("maxValue") or 0.0))
        chosen = max(layers, key=rank)
        selected.append(chosen)
        log(f"  Publish {chosen['id']}: {chosen['periodLabel']}")
    return selected


def prune_unpublished_layer_files(all_layers: List[dict], published_layers: List[dict]) -> None:
    keep = {l["id"] for l in published_layers}
    for layer in all_layers:
        if layer.get("id") in keep:
            continue
        for key in ("url", "valueGridUrl", "rasterUrl"):
            rel = layer.get(key)
            if rel:
                (Path("docs") / rel).unlink(missing_ok=True)


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
    all_layers: List[dict] = []
    for fhour in FORECAST_HOURS:
        for product in SOURCE_PRODUCTS:
            code = product["file_code"]
            label = product["label"]
            url = grib_url(cycle, code, fhour)
            local_path = CACHE_DIR / cycle.cycle_string / f"href.t{cycle.hour:02d}z.conus.{code}.f{fhour:02d}.grib2"
            if not download_file(session, url, local_path):
                continue
            all_layers.extend(process_grib_file(local_path, cycle, code, label, fhour))
            time.sleep(1.0)
    if not all_layers:
        raise RuntimeError("No HREF QPF source layers were generated.")
    all_layers.extend(derive_missing_max_layers(all_layers))
    published_layers = select_published_max_layers(all_layers)
    if not published_layers:
        raise RuntimeError("No publishable HREF Max QPF layers were generated from ensprod files.")
    prune_unpublished_layer_files(all_layers, published_layers)
    published_layers.sort(key=lambda x: (x["run"], x["forecastHour"]))
    catalog = {"generatedUTC": datetime.now(timezone.utc).isoformat(), "domain": DOMAIN, "source": "NCEP NOMADS HREF CONUS ensprod GRIB2; Max-only catalog derived from available ensemble summary products", "defaultLayerId": published_layers[0]["id"], "colorScale": COLOR_SCALE, "layers": published_layers}
    with (DATA_DIR / "catalog.json").open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    log(f"Wrote {DATA_DIR / 'catalog.json'} with {len(published_layers)} Max layer(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
