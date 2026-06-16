#!/usr/bin/env python3
"""Build a web-ready land-only LIX CWA boundary GeoJSON.

This keeps the same CWA source used by the lix_precip repo, but removes the
marine/offshore CWA segments by intersecting the LIX CWA polygon with the LIX
county/parish zone polygons before the web map draws the outline.
"""

from __future__ import annotations

import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import geopandas as gpd
import requests
from shapely.ops import unary_union

CWA_SHAPE_URL = "https://www.weather.gov/source/gis/Shapefiles/WSOM/w_16ap26.zip"
CWA_SHAPE_DIR = Path("data/shapes/cwa")
CWA_SHAPE_ZIP = CWA_SHAPE_DIR / "w_16ap26.zip"
CWA_SHAPE_PATH = CWA_SHAPE_DIR / "w_16ap26.shp"
LIX_CWA_GEOJSON = Path("docs/data/lix_cwa_boundary.geojson")
HEADERS = {"User-Agent": "href-qpf-viewer/1.0"}

LIX_COUNTY_ZONE_AREAS = ("LA", "MS")
LIX_COUNTY_ZONE_IDS = {
    "LAC005", "LAC007", "LAC033", "LAC037", "LAC047", "LAC051", "LAC057", "LAC063",
    "LAC071", "LAC075", "LAC077", "LAC087", "LAC089", "LAC091", "LAC093", "LAC095",
    "LAC103", "LAC105", "LAC109", "LAC117", "LAC121", "LAC125",
    "MSC005", "MSC045", "MSC047", "MSC059", "MSC109", "MSC113", "MSC147", "MSC157",
}


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


def zone_id_from_feature(feature: dict) -> str:
    props = feature.get("properties") or {}
    raw = str(props.get("id") or props.get("zoneId") or props.get("ugc") or "").upper()
    return raw.rsplit("/", 1)[-1]


def fetch_lix_land_zones(session: requests.Session) -> gpd.GeoDataFrame:
    features = []
    for area in LIX_COUNTY_ZONE_AREAS:
        url = f"https://api.weather.gov/zones?type=county&area={area}"
        log(f"Downloading NWS county zones for land mask: {area}")
        response = session.get(url, headers={**HEADERS, "Accept": "application/geo+json, application/json"}, timeout=60)
        response.raise_for_status()
        collection = response.json()
        for feature in collection.get("features", []):
            if feature.get("geometry") and zone_id_from_feature(feature) in LIX_COUNTY_ZONE_IDS:
                features.append(feature)

    if not features:
        raise RuntimeError("No LIX county/parish zones were found for land-only CWA clipping.")

    land = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    land = land[land.geometry.notna()].copy()
    land["geometry"] = land.geometry.buffer(0)
    if land.empty:
        raise RuntimeError("LIX county/parish zone land mask is empty after geometry cleanup.")
    return land


def build_lix_cwa_boundary(session: Optional[requests.Session] = None) -> Path:
    close_session = False
    if session is None:
        session = requests.Session()
        close_session = True

    try:
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

        land = fetch_lix_land_zones(session).to_crs(lix.crs)
        land_geom = unary_union(land.geometry)
        lix_geom = unary_union(lix.geometry)
        land_only_geom = land_geom.intersection(lix_geom)
        if land_only_geom.is_empty:
            raise RuntimeError("Land-only LIX CWA intersection is empty.")

        # The Leaflet viewer and QPF rasters use geographic lon/lat bounds, so export WGS84.
        out = gpd.GeoDataFrame(
            {"CWA": ["LIX"], "boundary": ["land_only"], "marine_removed": [True]},
            geometry=[land_only_geom],
            crs=lix.crs,
        ).to_crs("EPSG:4326")

        LIX_CWA_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
        LIX_CWA_GEOJSON.write_text(out.to_json(drop_id=True), encoding="utf-8")
        log(f"Wrote land-only LIX CWA boundary: {LIX_CWA_GEOJSON}")
        return LIX_CWA_GEOJSON
    finally:
        if close_session:
            session.close()


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
