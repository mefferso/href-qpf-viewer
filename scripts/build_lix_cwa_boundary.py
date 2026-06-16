#!/usr/bin/env python3
"""Build a web-ready LIX CWA boundary GeoJSON from the NWS CWA shapefile.

This mirrors the boundary source/style used by the lix_precip repo:
- download the national NWS CWA shapefile if missing
- extract it under data/shapes/cwa/
- filter CWA == "LIX"
- reproject to EPSG:4326 for Leaflet/web map rendering
- write docs/data/lix_cwa_boundary.geojson
"""

from __future__ import annotations

import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import geopandas as gpd
import requests

CWA_SHAPE_URL = "https://www.weather.gov/source/gis/Shapefiles/WSOM/w_16ap26.zip"
CWA_SHAPE_DIR = Path("data/shapes/cwa")
CWA_SHAPE_ZIP = CWA_SHAPE_DIR / "w_16ap26.zip"
CWA_SHAPE_PATH = CWA_SHAPE_DIR / "w_16ap26.shp"
LIX_CWA_GEOJSON = Path("docs/data/lix_cwa_boundary.geojson")
HEADERS = {"User-Agent": "href-qpf-viewer/1.0"}


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}] {msg}", flush=True)


def download_cwa_zip(session: requests.Session) -> None:
    CWA_SHAPE_DIR.mkdir(parents=True, exist_ok=True)
    if CWA_SHAPE_ZIP.exists() and CWA_SHAPE_ZIP.stat().st_size > 10_000:
        return

    tmp = CWA_SHAPE_ZIP.with_suffix(".zip.tmp")
    tmp.unlink(missing_ok=True)
    log(f"Downloading NWS CWA shapefile: {CWA_SHAPE_URL}")
    with session.get(CWA_SHAPE_URL, headers=HEADERS, timeout=90, stream=True) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    if tmp.stat().st_size <= 10_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded CWA shapefile zip is too small to be valid.")
    tmp.replace(CWA_SHAPE_ZIP)


def copy_shapefile_sidecars(source_shp: Path, target_dir: Path) -> Path:
    """Move/copy extracted shapefile sidecars to data/shapes/cwa/w_16ap26.* if needed."""
    target_dir.mkdir(parents=True, exist_ok=True)
    if source_shp.parent == target_dir and source_shp.name == CWA_SHAPE_PATH.name:
        return source_shp

    for sidecar in source_shp.parent.glob(f"{source_shp.stem}.*"):
        shutil.copy2(sidecar, target_dir / f"w_16ap26{sidecar.suffix}")
    return CWA_SHAPE_PATH


def ensure_cwa_shapefile(session: Optional[requests.Session] = None) -> Path:
    if CWA_SHAPE_PATH.exists():
        return CWA_SHAPE_PATH

    close_session = False
    if session is None:
        session = requests.Session()
        close_session = True
    try:
        download_cwa_zip(session)
    finally:
        if close_session:
            session.close()

    log(f"Extracting {CWA_SHAPE_ZIP} to {CWA_SHAPE_DIR}")
    with zipfile.ZipFile(CWA_SHAPE_ZIP) as zf:
        zf.extractall(CWA_SHAPE_DIR)

    if CWA_SHAPE_PATH.exists():
        return CWA_SHAPE_PATH

    matches = list(CWA_SHAPE_DIR.rglob("w_16ap26.shp"))
    if not matches:
        matches = list(CWA_SHAPE_DIR.rglob("*.shp"))
    if not matches:
        raise RuntimeError(f"No shapefile found after extracting {CWA_SHAPE_ZIP}")

    return copy_shapefile_sidecars(matches[0], CWA_SHAPE_DIR)


def build_lix_cwa_boundary(session: Optional[requests.Session] = None) -> Path:
    shp_path = ensure_cwa_shapefile(session)
    log(f"Reading CWA shapefile: {shp_path}")
    cwa = gpd.read_file(shp_path)
    if "CWA" not in cwa.columns:
        raise RuntimeError(f"Expected CWA field not found. Available fields: {', '.join(cwa.columns)}")

    lix = cwa[cwa["CWA"].astype(str).str.upper() == "LIX"].copy()
    if lix.empty:
        raise RuntimeError("CWA shapefile loaded, but CWA == 'LIX' returned no features.")

    if lix.crs is None:
        # NWS WSOM shapefiles are geographic NAD83. EPSG:4269 is explicit here;
        # converting to EPSG:4326 below keeps the web map coordinate order clean.
        lix = lix.set_crs("EPSG:4269", allow_override=True)

    # The Leaflet viewer and QPF rasters use geographic lon/lat bounds, so export WGS84.
    lix = lix.to_crs("EPSG:4326")

    LIX_CWA_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    LIX_CWA_GEOJSON.write_text(lix.to_json(drop_id=True), encoding="utf-8")
    log(f"Wrote LIX CWA boundary: {LIX_CWA_GEOJSON}")
    return LIX_CWA_GEOJSON


def refresh_lix_cwa_boundary_safely(session: Optional[requests.Session] = None) -> Optional[Path]:
    try:
        return build_lix_cwa_boundary(session)
    except Exception as exc:
        log(f"WARNING: Could not build LIX CWA boundary overlay: {exc}")
        return None


def main() -> int:
    build_lix_cwa_boundary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
