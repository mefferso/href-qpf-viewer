#!/usr/bin/env python3
"""Build HREF 6-hour Max QPF data for the web viewer.

Important: this script will only publish a native HREF Max QPF product. It does
not derive Max from Mean/PMM/LPMM. If NOMADS does not expose native Max GRIB2
files in ensprod, it fails loudly and prints the available ensprod filenames.
"""

from __future__ import annotations

import gzip
import html
import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlencode, urlparse

import numpy as np
import pygrib
import requests
from PIL import Image

DOMAIN = {"west": -95.0, "east": -83.0, "south": 24.0, "north": 34.0}
FORECAST_HOURS = [6, 12, 18, 24, 30, 36, 42, 48]
SAMPLE_STRIDE = 2
RASTER_UPSCALE = 4
MISSING_VALUE = -9999
CACHE_DIR = Path(".cache/href_grib")
DATA_DIR = Path("docs/data")
GRID_DIR = DATA_DIR / "grids"
VALUE_GRID_DIR = DATA_DIR / "value_grids"
RASTER_DIR = DATA_DIR / "rasters"
NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod"
NOMADS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_href.pl"
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


@dataclass(frozen=True)
class ProductFile:
    filename: str
    fhour: int


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}] {msg}", flush=True)


def candidate_cycles(now: Optional[datetime] = None) -> List[Cycle]:
    now = now or datetime.now(timezone.utc)
    cycles: List[Cycle] = []
    for day_offset in range(3):
        day = now - timedelta(days=day_offset)
        for hour in (12, 0):
            if datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc) <= now:
                cycles.append(Cycle(day.strftime("%Y%m%d"), hour))
    return sorted(cycles, key=lambda c: c.cycle_string, reverse=True)


def ensprod_url(cycle: Cycle, filename: str) -> str:
    return f"{NOMADS_BASE}/href.{cycle.yyyymmdd}/ensprod/{filename}"


def known_product_url(cycle: Cycle, product_code: str, fhour: int) -> str:
    return f"{NOMADS_BASE}/href.{cycle.yyyymmdd}/ensprod/href.t{cycle.hour:02d}z.conus.{product_code}.f{fhour:02d}.grib2"


def filter_url(cycle: Cycle, filename: str) -> str:
    return f"{NOMADS_FILTER}?{urlencode({'file': filename, 'dir': f'/href.{cycle.yyyymmdd}/ensprod'})}"


def url_exists(session: requests.Session, url: str) -> bool:
    try:
        r = session.get(url, headers={**HEADERS, "Range": "bytes=0-99"}, timeout=20, stream=True)
        return r.status_code in (200, 206)
    except Exception:
        return False


def find_latest_cycle(session: requests.Session) -> Cycle:
    for cycle in candidate_cycles():
        sentinel = known_product_url(cycle, "pmmn", 6)
        log(f"Checking {cycle.cycle_label}: {sentinel}")
        if url_exists(session, sentinel):
            log(f"Using HREF cycle {cycle.cycle_label}")
            return cycle
    raise RuntimeError("Could not find a recent HREF cycle on NOMADS.")


def parse_listing(text: str) -> List[str]:
    files: List[str] = []
    for raw in re.findall(r'href=["\']([^"\']+)["\']', text, flags=re.I):
        href = html.unescape(raw)
        if href.startswith("?"):
            qs = parse_qs(urlparse(href).query)
            if "file" in qs:
                name = unquote(qs["file"][0]).split("/")[-1]
                if name.endswith(".grib2"):
                    files.append(name)
            continue
        name = unquote(urlparse(href).path or href).strip("/").split("/")[-1]
        if name.endswith(".grib2"):
            files.append(name)
    for raw in re.findall(r'name=["\']file["\'][^>]*value=["\']([^"\']+)["\']', text, flags=re.I):
        name = html.unescape(raw).split("/")[-1]
        if name.endswith(".grib2"):
            files.append(name)
    return sorted(set(files))


def list_ensprod_files(session: requests.Session, cycle: Cycle) -> List[str]:
    urls = [
        f"{NOMADS_BASE}/href.{cycle.yyyymmdd}/ensprod/",
        f"{NOMADS_FILTER}?{urlencode({'dir': f'/href.{cycle.yyyymmdd}/ensprod'})}",
    ]
    for url in urls:
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                log(f"Ensprod listing failed {r.status_code}: {url}")
                continue
            files = parse_listing(r.text)
            log(f"Listed ensprod: {len(files)} grib2 file(s)")
            if files:
                for f in files[:60]:
                    log(f"  ensprod file: {f}")
                if len(files) > 60:
                    log(f"  ...and {len(files) - 60} more")
                return files
        except Exception as e:
            log(f"Ensprod listing error for {url}: {e}")
    return []


def fhour_from_filename(filename: str) -> Optional[int]:
    m = re.search(r"\.f(\d{2,3})(?:\.|_).*\.grib2$", filename, flags=re.I)
    if not m:
        m = re.search(r"f(\d{2,3}).*\.grib2$", filename, flags=re.I)
    return int(m.group(1)) if m else None


def looks_like_native_max_qpf(filename: str, cycle: Cycle) -> bool:
    low = filename.lower()
    if f"t{cycle.hour:02d}z" not in low or not low.endswith(".grib2"):
        return False
    if any(token in low for token in ["avrg", "mean", "pmm", "pmmn", "lpmm", "prob", "stamp", "paintball"]):
        return False
    if "max" not in low:
        return False
    return any(token in low for token in ["qpf", "apcp", ".max.", "_max_", "max."])


def discover_native_max_files(session: requests.Session, cycle: Cycle) -> Tuple[List[ProductFile], List[str]]:
    files = list_ensprod_files(session, cycle)
    candidates: List[ProductFile] = []
    for filename in files:
        fhour = fhour_from_filename(filename)
        if fhour not in FORECAST_HOURS:
            continue
        if looks_like_native_max_qpf(filename, cycle):
            candidates.append(ProductFile(filename, fhour))
    candidates.sort(key=lambda x: x.fhour)
    for c in candidates:
        log(f"Native Max candidate F{c.fhour:02d}: {c.filename}")
    return candidates, files


def download_file(session: requests.Session, cycle: Cycle, filename: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return True
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    for url in (ensprod_url(cycle, filename), filter_url(cycle, filename)):
        log(f"Downloading {url}")
        try:
            with session.get(url, headers=HEADERS, timeout=90, stream=True) as r:
                if r.status_code != 200:
                    log(f"  Skip: HTTP {r.status_code}")
                    continue
                ctype = r.headers.get("content-type", "").lower()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            if tmp.stat().st_size < 10_000 or "text/html" in ctype:
                log("  Skip: response was not a usable GRIB")
                tmp.unlink(missing_ok=True)
                continue
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


def message_is_precip(grb) -> bool:
    parts = [str(safe_get(grb, x, "")).lower() for x in ("shortName", "name", "parameterName")]
    text = " ".join(parts)
    return "apcp" in text or "total precipitation" in text or "total precip" in text or ("precipitation" in text and "probability" not in text)


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
    return lats[rmin:rmax+1, cmin:cmax+1], lons[rmin:rmax+1, cmin:cmax+1], vals[rmin:rmax+1, cmin:cmax+1], mask[rmin:rmax+1, cmin:cmax+1]


def orient_for_leaflet(lats2d, lons2d, vals2d, mask2d):
    if float(np.nanmean(lats2d[0, :])) < float(np.nanmean(lats2d[-1, :])):
        lats2d, lons2d, vals2d, mask2d = np.flipud(lats2d), np.flipud(lons2d), np.flipud(vals2d), np.flipud(mask2d)
    if float(np.nanmean(lons2d[:, 0])) > float(np.nanmean(lons2d[:, -1])):
        lats2d, lons2d, vals2d, mask2d = np.fliplr(lats2d), np.fliplr(lons2d), np.fliplr(vals2d), np.fliplr(mask2d)
    return lats2d, lons2d, vals2d, mask2d


def bounds_from_crop(lats2d, lons2d, mask2d) -> dict:
    return {"south": round(float(np.nanmin(lats2d[mask2d])), 5), "north": round(float(np.nanmax(lats2d[mask2d])), 5), "west": round(float(np.nanmin(lons2d[mask2d])), 5), "east": round(float(np.nanmax(lons2d[mask2d])), 5)}


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
    write_json_gz({"bounds": bounds, "width": int(grid.shape[1]), "height": int(grid.shape[0]), "missing": MISSING_VALUE, "units": "inches", "values": [round(float(v), 3) for v in grid.ravel()]}, value_path)
    rgba = np.zeros((out_size[1], out_size[0], 4), dtype=np.uint8)
    visible = mask_hi & np.isfinite(vals_hi) & (vals_hi >= 0.001)
    for b in COLOR_SCALE:
        m = visible & (vals_hi >= b["min"]) & (vals_hi < b["max"])
        rgba[m] = hex_to_rgba(b["color"], 215)
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(raster_path, optimize=True)


def iso_for_step(cycle: Cycle, hour: int) -> str:
    return (cycle.dt + timedelta(hours=hour)).isoformat()


def process_file(grib_path: Path, cycle: Cycle, product_file: ProductFile) -> Optional[dict]:
    try:
        grbs = pygrib.open(str(grib_path))
        messages = [g for g in grbs if message_is_precip(g)]
    except Exception as e:
        log(f"Could not open {grib_path.name}: {e}")
        return None
    best = None
    for grb in messages:
        accum = accum_hours_from_message(grb) or 6
        step_range = str(safe_get(grb, "stepRange", ""))
        step_start, step_end = parse_step_range(step_range, product_file.fhour, accum)
        score = (1 if accum == 6 else 0, 1 if step_end == product_file.fhour else 0, -abs(accum - 6))
        if best is None or score > best[0]:
            best = (score, grb, accum, step_start, step_end, step_range)
    if best is None:
        grbs.close()
        return None
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
    bounds = bounds_from_crop(lats2d, lons2d, mask2d)
    max_point = max_point_from_grid(lats2d, lons2d, vals2d, mask2d)
    pts_lat, pts_lon, pts_val = sample_points(lats2d, lons2d, vals2d, mask2d)
    layer_id = f"{cycle.cycle_string}_max_f{product_file.fhour:02d}_a06"
    grid_rel = f"data/grids/{layer_id}.json.gz"
    value_rel = f"data/value_grids/{layer_id}.json.gz"
    raster_rel = f"data/rasters/{layer_id}.png"
    period_label = f"F{step_start:02d}-F{step_end:02d} ({accum}-hr Max QPF)"
    start_iso, end_iso = iso_for_step(cycle, step_start), iso_for_step(cycle, step_end)
    write_json_gz({"metadata": {"id": layer_id, "run": cycle.cycle_string, "runLabel": cycle.cycle_label, "product": "max", "productLabel": "Max", "forecastHour": product_file.fhour, "accumHours": accum, "periodLabel": period_label, "units": "inches", "name": "Native HREF ensemble maximum precipitation", "stepRange": step_range, "startTimeUTC": start_iso, "validTimeUTC": end_iso, "sourceFile": product_file.filename}, "lat": round_list(pts_lat, 4), "lon": round_list(pts_lon, 4), "value": round_list(pts_val, 3)}, Path("docs") / grid_rel)
    write_value_grid_and_raster(vals2d, mask2d, bounds, Path("docs") / value_rel, Path("docs") / raster_rel)
    max_val = float(max_point["value"] or 0.0)
    log(f"  Wrote native Max F{product_file.fhour:02d}: {period_label}; max {max_val:.2f} in")
    return {"id": layer_id, "url": grid_rel, "valueGridUrl": value_rel, "rasterUrl": raster_rel, "rasterBounds": bounds, "maxPoint": max_point, "periodKey": f"f{product_file.fhour:02d}_a{accum:02d}_{step_start:02d}_{step_end:02d}", "periodLabel": period_label, "stepStart": step_start, "stepEnd": step_end, "run": cycle.cycle_string, "runLabel": cycle.cycle_label, "product": "max", "productLabel": "Max", "forecastHour": product_file.fhour, "accumHours": accum, "units": "inches", "startTimeUTC": start_iso, "validTimeUTC": end_iso, "stepRange": step_range, "name": "Native HREF ensemble maximum precipitation", "sourceFile": product_file.filename, "maxValue": round(max_val, 2), "pointCount": int(pts_val.size), "native": True}


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
    native_files, all_files = discover_native_max_files(session, cycle)
    if not native_files:
        preview = "\n".join(all_files[:120]) if all_files else "<no grib2 files listed>"
        raise RuntimeError("No native Max QPF files were found in NOMADS ensprod. Refusing to publish fake Max from Mean/PMM/LPMM. First listed ensprod files:\n" + preview)
    layers: List[dict] = []
    for pf in native_files:
        local = CACHE_DIR / cycle.cycle_string / pf.filename
        if not download_file(session, cycle, pf.filename, local):
            continue
        layer = process_file(local, cycle, pf)
        if layer:
            layers.append(layer)
        time.sleep(0.5)
    if not layers:
        raise RuntimeError("Native Max QPF files were found but no publishable precip layers could be built.")
    layers.sort(key=lambda x: (x["run"], x["forecastHour"]))
    catalog = {"generatedUTC": datetime.now(timezone.utc).isoformat(), "domain": DOMAIN, "source": "NCEP NOMADS HREF ensprod native Max QPF GRIB2", "defaultLayerId": layers[0]["id"], "colorScale": COLOR_SCALE, "layers": layers}
    with (DATA_DIR / "catalog.json").open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    log(f"Wrote {DATA_DIR / 'catalog.json'} with {len(layers)} native Max layer(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
