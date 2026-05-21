#!/usr/bin/env python3
"""Fast HREF 6-hour Ensemble Max QPF builder.

This wrapper reuses the existing output/publishing code from build_href_qpf.py,
but replaces the slow member-file download path with NOMADS .idx byte-range
fetches. Instead of downloading full CONUS GRIB2 files for every member/hour, it
pulls only the APCP message needed for each forecast hour.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from scipy.interpolate import griddata
from urllib3.util.retry import Retry

import build_href_qpf as base

CONNECT_TIMEOUT = int(os.getenv("HREF_CONNECT_TIMEOUT_SECONDS", "10"))
READ_TIMEOUT = int(os.getenv("HREF_READ_TIMEOUT_SECONDS", "60"))
MAX_RUNTIME_SECONDS = int(os.getenv("HREF_MAX_RUNTIME_SECONDS", "2100"))
START_MONOTONIC = time.monotonic()
MIN_GRIB_MESSAGE_BYTES = 500

_DIR_CACHE: dict[str, List[str]] = {}
_MEMBER_DIR_CACHE: dict[str, List[str]] = {}
_ORIG_LIST_GRIB_FILES = base.list_grib_files
_ORIG_FIND_MEMBER_DIRECTORIES = base.find_member_directories
_ORIG_MEMBER_FILE_CANDIDATES = base.member_file_candidates


@dataclass(frozen=True)
class IndexRecord:
    message_no: int
    offset: int
    next_offset: Optional[int]
    text: str


def log(msg: str) -> None:
    base.log(msg)


def check_deadline() -> None:
    elapsed = time.monotonic() - START_MONOTONIC
    if elapsed > MAX_RUNTIME_SECONDS:
        raise TimeoutError(
            f"HREF build exceeded {MAX_RUNTIME_SECONDS}s runtime budget. "
            "Stopping before GitHub Actions burns half the damn day."
        )


def make_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=None,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def cached_list_grib_files(session: requests.Session, dir_url: str) -> List[str]:
    if dir_url not in _DIR_CACHE:
        check_deadline()
        _DIR_CACHE[dir_url] = _ORIG_LIST_GRIB_FILES(session, dir_url)
    return _DIR_CACHE[dir_url]


def cached_find_member_directories(session: requests.Session, cycle: base.Cycle) -> List[str]:
    key = cycle.cycle_string
    if key not in _MEMBER_DIR_CACHE:
        check_deadline()
        _MEMBER_DIR_CACHE[key] = _ORIG_FIND_MEMBER_DIRECTORIES(session, cycle)
    return _MEMBER_DIR_CACHE[key]


def member_file_candidates_fast(session: requests.Session, cycle: base.Cycle, fhour: int) -> List[str]:
    check_deadline()
    files = _ORIG_MEMBER_FILE_CANDIDATES(session, cycle, fhour)
    if files:
        log(f"F{fhour:02d}: {len(files)} member candidate(s) after cached directory scan")
    return files


def parse_idx(text: str) -> List[IndexRecord]:
    raw = []
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        try:
            raw.append((int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue

    records: List[IndexRecord] = []
    for idx, (message_no, offset, msg_text) in enumerate(raw):
        next_offset = raw[idx + 1][1] if idx + 1 < len(raw) else None
        records.append(IndexRecord(message_no, offset, next_offset, msg_text))
    return records


def period_matches(text: str) -> list[tuple[int, int]]:
    # Handles common NOMADS idx text like "6-12 hour acc fcst".
    return [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*-\s*(\d+)\s*hour", text)]


def select_apcp_record(records: List[IndexRecord], fhour: int) -> Optional[IndexRecord]:
    target_start = max(0, fhour - 6)
    best: Optional[tuple[int, IndexRecord]] = None

    for rec in records:
        text = rec.text.lower()
        if "apcp" not in text and "total precip" not in text and "total precipitation" not in text:
            continue
        if "prob" in text or "ptype" in text:
            continue

        score = 0
        if "apcp" in text:
            score += 20
        if "surface" in text:
            score += 10
        if "acc" in text:
            score += 10

        periods = period_matches(text)
        for start, end in periods:
            if start == target_start and end == fhour:
                score += 100
            elif start == 0 and end == fhour:
                score += 70
            elif end == fhour:
                score += 40
            else:
                score -= min(abs(end - fhour), 24)

        if f"f{fhour:02d}" in text:
            score += 5

        if best is None or score > best[0]:
            best = (score, rec)

    return best[1] if best else None


def download_idx(session: requests.Session, grib_url: str) -> Optional[List[IndexRecord]]:
    idx_url = f"{grib_url}.idx"
    try:
        check_deadline()
        r = session.get(idx_url, headers=base.HEADERS, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except Exception as e:
        log(f"  IDX fetch failed for {idx_url}: {e}")
        return None

    if r.status_code != 200 or not r.text.strip():
        log(f"  IDX unavailable HTTP {r.status_code}: {idx_url}")
        return None

    records = parse_idx(r.text)
    if not records:
        log(f"  IDX had no parseable records: {idx_url}")
        return None
    return records


def download_apcp_message(session: requests.Session, grib_url: str, fhour: int, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > MIN_GRIB_MESSAGE_BYTES:
        return True

    records = download_idx(session, grib_url)
    if not records:
        return False

    rec = select_apcp_record(records, fhour)
    if rec is None:
        log(f"  No APCP record in IDX for F{fhour:02d}: {grib_url}")
        return False

    if rec.next_offset is not None:
        byte_range = f"bytes={rec.offset}-{rec.next_offset - 1}"
    else:
        # Last message in file. This still avoids most of the file in normal GRIBs.
        byte_range = f"bytes={rec.offset}-"

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    headers = {**base.HEADERS, "Range": byte_range}
    log(f"  Byte-range APCP F{fhour:02d}: {grib_url.rsplit('/', 1)[-1]} [{byte_range}]")

    try:
        check_deadline()
        with session.get(grib_url, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), stream=True) as r:
            if r.status_code != 206:
                log(f"  Skip byte-range download: HTTP {r.status_code} for {grib_url}")
                return False
            ctype = r.headers.get("content-type", "").lower()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(256 * 1024):
                    check_deadline()
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        log(f"  Byte-range APCP download failed: {e}")
        tmp.unlink(missing_ok=True)
        return False

    if tmp.stat().st_size <= MIN_GRIB_MESSAGE_BYTES or "text/html" in ctype:
        log("  Skip: byte-range response was not a usable GRIB message")
        tmp.unlink(missing_ok=True)
        return False

    tmp.replace(out_path)
    return True


def crop_member_field(lats: np.ndarray, lons: np.ndarray, vals: np.ndarray):
    vals = np.array(vals, dtype=float)
    vals[vals < 0] = np.nan
    lats2d, lons2d, vals2d, mask2d = base.crop_2d(lats, lons, vals)
    if vals2d.size == 0:
        return None
    vals2d = np.where(mask2d & np.isfinite(vals2d), vals2d, np.nan)
    return lats2d, lons2d, vals2d


def compute_member_6hr_fast(session: requests.Session, cycle: base.Cycle, member_url: str, fhour: int):
    check_deadline()
    name = member_url.rsplit("/", 1)[-1]
    safe_name = name.replace(".grib2", f".f{fhour:02d}.apcp.grib2")
    local = base.CACHE_DIR / cycle.cycle_string / "members_apcp" / safe_name

    if not download_apcp_message(session, member_url, fhour, local):
        return None

    curr = base.extract_member_apcp(local, fhour)
    if curr is None:
        return None

    vals = curr["vals"]
    if curr["end"] == fhour and curr["start"] == max(0, fhour - 6):
        cropped = crop_member_field(curr["lats"], curr["lons"], vals)
        if cropped is None:
            return None
        return (*cropped, name)

    if fhour < 6:
        return None

    prev_name = base.replace_forecast_hour(name, fhour - 6)
    prev_url = member_url.rsplit("/", 1)[0] + "/" + prev_name
    prev_safe_name = prev_name.replace(".grib2", f".f{fhour - 6:02d}.apcp.grib2")
    prev_local = base.CACHE_DIR / cycle.cycle_string / "members_apcp" / prev_safe_name

    if not download_apcp_message(session, prev_url, fhour - 6, prev_local):
        return None

    prev = base.extract_member_apcp(prev_local, fhour - 6)
    if prev is None or prev["vals"].shape != vals.shape:
        return None

    inc = vals - prev["vals"]
    inc[inc < -0.001] = np.nan
    inc = np.where(inc < 0.001, 0.0, inc)
    cropped = crop_member_field(curr["lats"], curr["lons"], inc)
    if cropped is None:
        return None
    return (*cropped, name)


def build_layer_from_members_fast(session: requests.Session, cycle: base.Cycle, fhour: int):
    check_deadline()
    files = base.member_file_candidates(session, cycle, fhour)
    if not files:
        log(f"F{fhour:02d}: no member files discovered; skipping.")
        return None

    member_fields = []
    used = []
    base_lats = base_lons = None

    for url in files:
        check_deadline()
        out = base.compute_member_6hr(session, cycle, url, fhour)
        if out is None:
            continue
        lats, lons, vals, fname = out

        if base_lats is None:
            base_lats, base_lons = lats, lons
            member_fields.append(vals)
            used.append(fname)
            continue

        if lats.shape != base_lats.shape:
            valid = np.isfinite(vals)
            if not np.any(valid):
                continue
            points = np.column_stack((lons[valid], lats[valid]))
            values = vals[valid]
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

    log(f"F{fhour:02d}: using {len(used)} member APCP slice(s): {', '.join(used)}")
    ens_max = np.nanmax(np.stack(member_fields, axis=0), axis=0)

    lats2d, lons2d, vals2d, mask2d = base.crop_2d(base_lats, base_lons, ens_max)
    if vals2d.size == 0:
        return None

    vals2d = np.where(vals2d < 0.001, 0.0, vals2d)
    lats2d, lons2d, vals2d, mask2d = base.orient_for_leaflet(lats2d, lons2d, vals2d, mask2d)
    lats2d, lons2d, vals2d, mask2d = base.reproject_to_regular_grid(lats2d, lons2d, vals2d, mask2d)
    bounds = base.bounds_from_crop(lats2d, lons2d, mask2d)
    max_point = base.max_point_from_grid(lats2d, lons2d, vals2d, mask2d)
    pts_lat, pts_lon, pts_val = base.sample_points(lats2d, lons2d, vals2d, mask2d)

    layer_id = f"{cycle.cycle_string}_max_f{fhour:02d}_a06"
    grid_rel = f"data/grids/{layer_id}.json.gz"
    value_rel = f"data/value_grids/{layer_id}.json.gz"
    raster_rel = f"data/rasters/{layer_id}.png"
    period_label = f"F{max(0, fhour - 6):02d}-F{fhour:02d} (6-hr Ensemble Max QPF)"
    start_iso = base.iso_for_step(cycle, max(0, fhour - 6))
    end_iso = base.iso_for_step(cycle, fhour)

    base.write_json_gz({
        "metadata": {
            "id": layer_id,
            "run": cycle.cycle_string,
            "runLabel": cycle.cycle_label,
            "product": "max",
            "productLabel": base.PRODUCT_LABEL,
            "forecastHour": fhour,
            "accumHours": 6,
            "periodLabel": period_label,
            "units": "inches",
            "name": "HREF member-computed Ensemble Max QPF",
            "stepRange": f"{max(0, fhour - 6)}-{fhour}",
            "startTimeUTC": start_iso,
            "validTimeUTC": end_iso,
            "sourceFile": f"{len(used)} member APCP slices",
        },
        "lat": base.round_list(pts_lat, 4),
        "lon": base.round_list(pts_lon, 4),
        "value": base.round_list(pts_val, 3),
    }, Path("docs") / grid_rel)

    base.write_value_grid_and_raster(vals2d, mask2d, bounds, Path("docs") / value_rel, Path("docs") / raster_rel)
    max_val = float(max_point["value"] or 0.0)

    return {
        "id": layer_id,
        "url": grid_rel,
        "valueGridUrl": value_rel,
        "rasterUrl": raster_rel,
        "rasterBounds": bounds,
        "maxPoint": max_point,
        "periodKey": f"f{fhour:02d}_a06_{max(0, fhour - 6):02d}_{fhour:02d}",
        "periodLabel": period_label,
        "stepStart": max(0, fhour - 6),
        "stepEnd": fhour,
        "run": cycle.cycle_string,
        "runLabel": cycle.cycle_label,
        "product": "max",
        "productLabel": base.PRODUCT_LABEL,
        "forecastHour": fhour,
        "accumHours": 6,
        "units": "inches",
        "startTimeUTC": start_iso,
        "validTimeUTC": end_iso,
        "stepRange": f"{max(0, fhour - 6)}-{fhour}",
        "name": "HREF member-computed Ensemble Max QPF",
        "sourceFile": f"{len(used)} member APCP slices",
        "maxValue": round(max_val, 2),
        "pointCount": int(pts_val.size),
        "native": False,
    }


def install_fast_overrides() -> None:
    base.list_grib_files = cached_list_grib_files
    base.find_member_directories = cached_find_member_directories
    base.member_file_candidates = member_file_candidates_fast
    base.compute_member_6hr = compute_member_6hr_fast
    base.build_layer_from_members = build_layer_from_members_fast


def main() -> int:
    install_fast_overrides()
    base.DATA_DIR.mkdir(parents=True, exist_ok=True)
    base.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base.clean_old_data()

    session = make_session()
    cycle = base.find_latest_cycle(session)
    layers = []

    for fhour in base.FORECAST_HOURS:
        check_deadline()
        started = time.monotonic()
        layer = base.build_layer_from_members(session, cycle, fhour)
        if layer:
            layers.append(layer)
        log(f"F{fhour:02d}: completed in {time.monotonic() - started:.1f}s")
        time.sleep(0.2)

    if not layers:
        raise RuntimeError("No forecast hours were successfully built from individual HREF member APCP fields.")

    layers.sort(key=lambda x: (x["run"], x["forecastHour"]))
    catalog = {
        "generatedUTC": datetime.now(timezone.utc).isoformat(),
        "domain": base.DOMAIN,
        "source": "NCEP NOMADS HREF/HIRESW member GRIB2 APCP byte ranges; Ensemble Max computed as gridpoint max across member APCP slices",
        "defaultLayerId": layers[0]["id"],
        "colorScale": base.COLOR_SCALE,
        "layers": layers,
    }
    with (base.DATA_DIR / "catalog.json").open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    elapsed = time.monotonic() - START_MONOTONIC
    log(f"Wrote {base.DATA_DIR / 'catalog.json'} with {len(layers)} Ensemble Max layer(s) in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
