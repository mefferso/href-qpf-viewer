#!/usr/bin/env python3
"""Build true 6-hour HREF Max QPF for the viewer.

Max is computed from HREF member APCP fields, not Mean/PMM/LPMM summary fields.
The script discovers available member GRIB2 files on NOMADS, pulls surface APCP
through filter_href.pl, builds 6-hour totals for each member, then computes the
pointwise member maximum.
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
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlencode, urljoin, urlparse

import numpy as np
import pygrib
import requests
from PIL import Image

DOMAIN = {"west": -95.0, "east": -83.0, "south": 24.0, "north": 34.0}
PUBLISH_FORECAST_HOURS = [6, 12, 18, 24, 30, 36, 42, 48]
SOURCE_FORECAST_HOURS = list(range(0, 49))
RASTER_UPSCALE = 4
SAMPLE_STRIDE = 2
MISSING_VALUE = -9999
CACHE_DIR = Path(".cache/href_grib")
DATA_DIR = Path("docs/data")
GRID_DIR = DATA_DIR / "grids"
VALUE_GRID_DIR = DATA_DIR / "value_grids"
RASTER_DIR = DATA_DIR / "rasters"
NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod"
NOMADS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_href.pl"
HEADERS = {"User-Agent": "href-qpf-viewer/1.0"}
SUMMARY_TOKENS = ("ensprod", "avrg", "mean", "pmm", "pmmn", "lpmm", "prob", "paintball", "stamp")
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
        return datetime.strptime(f"{self.cycle_string}", "%Y%m%d%H").replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class CandidateFile:
    dir_path: str
    filename: str
    member_id: str
    fhour: int


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}] {msg}", flush=True)


def candidate_cycles(now: Optional[datetime] = None) -> List[Cycle]:
    now = now or datetime.now(timezone.utc)
    out: List[Cycle] = []
    for day_offset in range(3):
        day = now - timedelta(days=day_offset)
        for hr in (12, 0):
            if datetime(day.year, day.month, day.day, hr, tzinfo=timezone.utc) <= now:
                out.append(Cycle(day.strftime("%Y%m%d"), hr))
    return sorted(out, key=lambda c: c.cycle_string, reverse=True)


def ensprod_url(cycle: Cycle, product_code: str, fhour: int) -> str:
    return f"{NOMADS_BASE}/href.{cycle.yyyymmdd}/ensprod/href.t{cycle.hour:02d}z.conus.{product_code}.f{fhour:02d}.grib2"


def url_exists(session: requests.Session, url: str) -> bool:
    try:
        r = session.get(url, headers={**HEADERS, "Range": "bytes=0-99"}, timeout=20, stream=True)
        return r.status_code in (200, 206)
    except Exception:
        return False


def find_latest_cycle(session: requests.Session) -> Cycle:
    for cycle in candidate_cycles():
        test_url = ensprod_url(cycle, "pmmn", 6)
        log(f"Checking {cycle.cycle_label}: {test_url}")
        if url_exists(session, test_url):
            log(f"Using HREF cycle {cycle.cycle_label}")
            return cycle
    raise RuntimeError("Could not find a recent HREF cycle on NOMADS.")


def listing_urls(dir_path: str) -> List[str]:
    clean = dir_path.strip("/")
    return [
        f"{NOMADS_BASE}/{clean}/",
        f"{NOMADS_FILTER}?{urlencode({'dir': '/' + clean})}",
    ]


def parse_listing(text: str) -> Tuple[List[str], List[str]]:
    subdirs: List[str] = []
    files: List[str] = []
    for raw in re.findall(r'href=["\']([^"\']+)["\']', text, flags=re.I):
        href = html.unescape(raw)
        if href.startswith("?"):
            qs = parse_qs(urlparse(href).query)
            if "file" in qs:
                files.append(unquote(qs["file"][0]))
            continue
        name = unquote(href.split("?")[0]).strip()
        if name in ("", "/", "../"):
            continue
        if name.endswith("/"):
            subdirs.append(name.strip("/"))
        elif name.endswith(".grib2"):
            files.append(name.split("/")[-1])
    for raw in re.findall(r'name=["\']file["\'][^>]*value=["\']([^"\']+)["\']', text, flags=re.I):
        files.append(html.unescape(raw))
    return sorted(set(subdirs)), sorted(set(files))


def list_dir(session: requests.Session, dir_path: str) -> Tuple[List[str], List[str]]:
    for url in listing_urls(dir_path):
        try:
            r = session.get(url, headers=HEADERS, timeout=25)
            if r.status_code != 200:
                log(f"  Directory listing failed {r.status_code}: {url}")
                continue
            subdirs, files = parse_listing(r.text)
            if subdirs or files:
                return subdirs, files
        except Exception as e:
            log(f"  Directory listing error for {url}: {e}")
    return [], []


def member_id_from_filename(filename: str) -> str:
    name = filename.replace(".grib2", "")
    name = re.sub(r"^href\.t\d{2}z\.", "", name)
    name = re.sub(r"\.f\d{2,3}.*$", "", name)
    name = name.replace("conus.", "").replace(".conus", "")
    return name or filename


def fhour_from_filename(filename: str) -> Optional[int]:
    m = re.search(r"\.f(\d{2,3})(?:\.|_).*\.grib2$", filename, flags=re.I)
    if not m:
        m = re.search(r"f(\d{2,3}).*\.grib2$", filename, flags=re.I)
    return int(m.group(1)) if m else None


def discover_member_files(session: requests.Session, cycle: Cycle) -> List[CandidateFile]:
    root = f"href.{cycle.yyyymmdd}"
    queue: List[Tuple[str, int]] = [(root, 0)]
    seen = {root}
    found: Dict[Tuple[str, int], CandidateFile] = {}
    log("Discovering HREF member GRIB2 files")
    while queue:
        dir_path, depth = queue.pop(0)
        subdirs, files = list_dir(session, dir_path)
        low_dir = dir_path.lower()
        for filename in files:
            low = filename.lower()
            if not filename.endswith(".grib2"):
                continue
            if f"t{cycle.hour:02d}z" not in low:
                continue
            if any(tok in low or tok in low_dir for tok in SUMMARY_TOKENS):
                continue
            fhour = fhour_from_filename(filename)
            if fhour is None or fhour not in SOURCE_FORECAST_HOURS:
                continue
            member_id = member_id_from_filename(filename)
            found.setdefault((member_id, fhour), CandidateFile(dir_path, filename, member_id, fhour))
        if depth >= 4:
            continue
        for sub in subdirs:
            next_path = f"{dir_path.rstrip('/')}/{sub}"
            if next_path not in seen:
                seen.add(next_path)
                queue.append((next_path, depth + 1))
    out = sorted(found.values(), key=lambda c: (c.fhour, c.member_id, c.dir_path, c.filename))
    log(f"Discovered {len(out)} member file candidates")
    for c in out[:25]:
        log(f"  {c.member_id} F{c.fhour:02d}: /{c.dir_path}/{c.filename}")
    if len(out) > 25:
        log(f"  ...and {len(out) - 25} more")
    return out


def filter_url(c: CandidateFile) -> str:
    return f"{NOMADS_FILTER}?{urlencode({'file': c.filename, 'lev_surface': 'on', 'var_APCP': 'on', 'dir': '/' + c.dir_path.strip('/')})}"


def raw_url(c: CandidateFile) -> str:
    return f"{NOMADS_BASE}/{c.dir_path.strip('/')}/{c.filename}"


def download_candidate(session: requests.Session, c: CandidateFile, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 500:
        return True
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    for url in (filter_url(c), raw_url(c)):
        log(f"Downloading APCP {c.member_id} F{c.fhour:02d}: {url}")
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
            if tmp.stat().st_size < 500 or "text/html" in ctype:
                log("  Skip: response was not a usable GRIB subset")
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


def extract_member_layers(grib_path: Path, c: CandidateFile) -> List[dict]:
    layers: List[dict] = []
    try:
        grbs = pygrib.open(str(grib_path))
        messages = [g for g in grbs if message_is_precip(g)]
    except Exception as e:
        log(f"Could not read {grib_path.name}: {e}")
        return layers
    for idx, grb in enumerate(messages, start=1):
        try:
            accum = accum_hours_from_message(grb) or c.fhour
            step_range = str(safe_get(grb, "stepRange", ""))
            step_start, step_end = parse_step_range(step_range, c.fhour, accum)
            if step_end <= step_start:
                continue
            vals_raw = grb.values
            if np.ma.isMaskedArray(vals_raw):
                vals_raw = vals_raw.filled(np.nan)
            vals = value_units_to_inches(np.array(vals_raw, dtype=float), str(safe_get(grb, "units", "")))
            vals[vals < 0] = np.nan
            lats, lons = grb.latlons()
            lats2d, lons2d, vals2d, mask2d = crop_2d(lats, lons, vals)
            if vals2d.size == 0:
                continue
            vals2d = np.where(vals2d < 0.001, 0.0, vals2d)
            lats2d, lons2d, vals2d, mask2d = orient_for_leaflet(lats2d, lons2d, vals2d, mask2d)
            layers.append({"memberId": c.member_id, "stepStart": int(step_start), "stepEnd": int(step_end), "accumHours": int(accum), "lats2d": lats2d, "lons2d": lons2d, "vals2d": vals2d, "mask2d": mask2d})
            log(f"  Loaded {c.member_id} F{step_start:02d}-F{step_end:02d} ({accum}-hr)")
        except Exception as e:
            log(f"  Message {idx} failed in {grib_path.name}: {e}")
    grbs.close()
    return layers


def choose_exact_cover(layers: List[dict], start: int, end: int) -> Optional[List[dict]]:
    by_start: Dict[int, List[dict]] = {}
    for layer in layers:
        s, e = int(layer["stepStart"]), int(layer["stepEnd"])
        if s >= start and e <= end and e > s:
            by_start.setdefault(s, []).append(layer)
    for s in by_start:
        by_start[s].sort(key=lambda l: (int(l["stepEnd"]) - int(l["stepStart"])), reverse=True)

    @lru_cache(None)
    def solve(pos: int):
        if pos == end:
            return tuple()
        best = None
        for layer in by_start.get(pos, []):
            tail = solve(int(layer["stepEnd"]))
            if tail is not None:
                cover = (layer,) + tail
                if best is None or len(cover) < len(best):
                    best = cover
        return best
    ans = solve(start)
    return list(ans) if ans else None


def sum_layers(cover: List[dict]) -> Optional[dict]:
    base = cover[0]
    vals_sum = np.zeros_like(base["vals2d"], dtype=float)
    mask_any = np.zeros_like(base["mask2d"], dtype=bool)
    for layer in cover:
        if layer["vals2d"].shape != base["vals2d"].shape:
            return None
        valid = layer["mask2d"] & np.isfinite(layer["vals2d"])
        vals_sum += np.where(valid, layer["vals2d"], 0.0)
        mask_any |= valid
    vals_sum = np.where(mask_any, vals_sum, np.nan)
    return {"lats2d": base["lats2d"], "lons2d": base["lons2d"], "vals2d": vals_sum, "mask2d": mask_any & np.isfinite(vals_sum), "pieces": [f"F{int(x['stepStart']):02d}-F{int(x['stepEnd']):02d}" for x in cover]}


def member_total_for_period(layers: List[dict], start: int, end: int) -> Optional[dict]:
    cover = choose_exact_cover(layers, start, end)
    if cover:
        return sum_layers(cover)
    end_layers = [x for x in layers if int(x["stepStart"]) == 0 and int(x["stepEnd"]) == end]
    if start == 0 and end_layers:
        total = sum_layers([end_layers[0]])
        if total:
            total["pieces"] = [f"F00-F{end:02d}"]
        return total
    start_layers = [x for x in layers if int(x["stepStart"]) == 0 and int(x["stepEnd"]) == start]
    if end_layers and start_layers and end_layers[0]["vals2d"].shape == start_layers[0]["vals2d"].shape:
        e, s = end_layers[0], start_layers[0]
        valid = e["mask2d"] & s["mask2d"] & np.isfinite(e["vals2d"]) & np.isfinite(s["vals2d"])
        vals = np.where(valid, e["vals2d"] - s["vals2d"], np.nan)
        vals = np.where(vals < 0, 0.0, vals)
        return {"lats2d": e["lats2d"], "lons2d": e["lons2d"], "vals2d": vals, "mask2d": valid & np.isfinite(vals), "pieces": [f"F00-F{end:02d} minus F00-F{start:02d}"]}
    return None


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
    return np.array(Image.fromarray(np.nan_to_num(values.astype(np.float32), nan=0.0), mode="F").resize(out_size, resample=Image.Resampling.BILINEAR), dtype=np.float32)


def resize_mask(mask: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(out_size, resample=Image.Resampling.NEAREST)) > 0


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


def hex_to_rgba(hex_color: str, alpha: int = 215):
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha


def iso_for_step(cycle: Cycle, hour: int) -> str:
    return (cycle.dt + timedelta(hours=hour)).isoformat()


def build_layers(cycle: Cycle, source_layers: List[dict]) -> List[dict]:
    by_member: Dict[str, List[dict]] = {}
    for layer in source_layers:
        by_member.setdefault(layer["memberId"], []).append(layer)
    out: List[dict] = []
    for end in PUBLISH_FORECAST_HOURS:
        start = end - 6
        totals, member_notes = [], []
        for member, layers in sorted(by_member.items()):
            total = member_total_for_period(layers, start, end)
            if total is None:
                continue
            total["memberId"] = member
            totals.append(total)
            member_notes.append(f"{member}: {' + '.join(total['pieces'])}")
        log(f"F{start:02d}-F{end:02d}: {len(totals)} members with 6-hr totals")
        if not totals:
            continue
        base = totals[0]
        stacks, masks = [], []
        for total in totals:
            if total["vals2d"].shape != base["vals2d"].shape:
                log(f"  Skip {total['memberId']}: grid shape mismatch")
                continue
            valid = total["mask2d"] & np.isfinite(total["vals2d"])
            stacks.append(np.where(valid, total["vals2d"], np.nan))
            masks.append(valid)
        if not stacks:
            continue
        vals2d = np.nanmax(np.stack(stacks), axis=0)
        mask2d = np.any(np.stack(masks), axis=0) & np.isfinite(vals2d)
        lats2d, lons2d = base["lats2d"], base["lons2d"]
        bounds = bounds_from_crop(lats2d, lons2d, mask2d)
        max_point = max_point_from_grid(lats2d, lons2d, vals2d, mask2d)
        pts_lat, pts_lon, pts_val = sample_points(lats2d, lons2d, vals2d, mask2d)
        layer_id = f"{cycle.cycle_string}_max_f{end:02d}_a06"
        grid_rel = f"data/grids/{layer_id}.json.gz"
        value_rel = f"data/value_grids/{layer_id}.json.gz"
        raster_rel = f"data/rasters/{layer_id}.png"
        period_label = f"F{start:02d}-F{end:02d} (6-hr Max QPF)"
        start_iso, end_iso = iso_for_step(cycle, start), iso_for_step(cycle, end)
        write_json_gz({"metadata": {"id": layer_id, "run": cycle.cycle_string, "runLabel": cycle.cycle_label, "product": "max", "productLabel": "Max", "forecastHour": end, "accumHours": 6, "periodLabel": period_label, "units": "inches", "name": "True 6-hour maximum precipitation across HREF members", "stepRange": f"{start}-{end}", "startTimeUTC": start_iso, "validTimeUTC": end_iso, "memberCount": len(totals), "derivedFromMembers": member_notes}, "lat": round_list(pts_lat, 4), "lon": round_list(pts_lon, 4), "value": round_list(pts_val, 3)}, Path("docs") / grid_rel)
        write_value_grid_and_raster(vals2d, mask2d, bounds, Path("docs") / value_rel, Path("docs") / raster_rel)
        max_val = float(max_point["value"] or 0.0)
        out.append({"id": layer_id, "url": grid_rel, "valueGridUrl": value_rel, "rasterUrl": raster_rel, "rasterBounds": bounds, "maxPoint": max_point, "periodKey": f"f{end:02d}_a06_{start:02d}_{end:02d}", "periodLabel": period_label, "stepStart": start, "stepEnd": end, "run": cycle.cycle_string, "runLabel": cycle.cycle_label, "product": "max", "productLabel": "Max", "forecastHour": end, "accumHours": 6, "units": "inches", "startTimeUTC": start_iso, "validTimeUTC": end_iso, "stepRange": f"{start}-{end}", "name": "True 6-hour maximum precipitation across HREF members", "maxValue": round(max_val, 2), "pointCount": int(pts_val.size), "derived": True, "memberCount": len(totals), "derivedFromMembers": member_notes})
        log(f"  Wrote {layer_id}; max {max_val:.2f} in from {len(totals)} members")
    return out


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
    candidates = discover_member_files(session, cycle)
    if not candidates:
        raise RuntimeError("No HREF member files were discovered; refusing to publish Max from Mean/PMM/LPMM summary products.")
    source_layers: List[dict] = []
    for c in candidates:
        local = CACHE_DIR / cycle.cycle_string / re.sub(r"[^A-Za-z0-9_.-]+", "_", c.member_id) / f"f{c.fhour:03d}.apcp.grib2"
        if not download_candidate(session, c, local):
            continue
        source_layers.extend(extract_member_layers(local, c))
        time.sleep(0.15)
    if not source_layers:
        raise RuntimeError("No member APCP layers were generated; refusing to publish Max from Mean/PMM/LPMM summary products.")
    layers = build_layers(cycle, source_layers)
    if not layers:
        raise RuntimeError("No true 6-hour member-max QPF layers could be derived.")
    layers.sort(key=lambda x: (x["run"], x["forecastHour"]))
    catalog = {"generatedUTC": datetime.now(timezone.utc).isoformat(), "domain": DOMAIN, "source": "NCEP NOMADS HREF member GRIB2 via filter_href.pl; true pointwise 6-hour max across members", "defaultLayerId": layers[0]["id"], "colorScale": COLOR_SCALE, "layers": layers}
    with (DATA_DIR / "catalog.json").open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    log(f"Wrote {DATA_DIR / 'catalog.json'} with {len(layers)} true member-max layers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
