#!/usr/bin/env python3
"""Build a catalog of SPC official HREF finished-product image URLs.

This does not replace the computed interactive grid. It simply writes
`docs/data/spc_official_catalog.json` with predictable SPC PNG URLs for the
official HREF 6-hr QPF ensemble max product.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

OUT = Path("docs/data/spc_official_catalog.json")
GRAPHICS_ROOT = "https://www.spc.noaa.gov/exper/href/graphics"
MODEL = "href"
PRODUCT = "qpf_006h_max"
SECTORS = ["se", "sp"]
FORECAST_HOURS = [6, 12, 18, 24, 30, 36, 42, 48]
HEADERS = {"User-Agent": "href-qpf-viewer-spc-official/1.0"}


def candidate_cycles(now: Optional[datetime] = None) -> list[datetime]:
    now = now or datetime.now(timezone.utc)
    out: list[datetime] = []
    for d in range(3):
        day = now - timedelta(days=d)
        for hour in (12, 0):
            dt = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
            if dt <= now:
                out.append(dt)
    return sorted(out, reverse=True)


def spc_image_url(cycle: datetime, sector: str, fhour: int) -> str:
    yyyy = cycle.strftime("%Y")
    mm = cycle.strftime("%m")
    dd = cycle.strftime("%d")
    hhmm = cycle.strftime("%H%M")
    ftoken = f"f{fhour:03d}00"
    return f"{GRAPHICS_ROOT}/models/{MODEL}/{yyyy}/{mm}/{dd}/{hhmm}/{ftoken}/{PRODUCT}.{sector}.{ftoken}.png"


def url_exists(session: requests.Session, url: str) -> bool:
    try:
        r = session.get(url, headers={**HEADERS, "Range": "bytes=0-127"}, timeout=20, stream=True)
        return r.status_code in (200, 206) and "image" in r.headers.get("content-type", "").lower()
    except Exception:
        return False


def find_latest_cycle(session: requests.Session) -> datetime:
    for cycle in candidate_cycles():
        # F012 is often a better sentinel than F006 because some products can lag early.
        test_url = spc_image_url(cycle, "sp", 12)
        print(f"Checking SPC official image: {test_url}", flush=True)
        if url_exists(session, test_url):
            print(f"Using SPC HREF cycle {cycle:%Y%m%d %HZ}", flush=True)
            return cycle
    raise RuntimeError("Could not find recent SPC official HREF qpf_006h_max image cycle.")


def iso(cycle: datetime, fhour: int) -> str:
    return (cycle + timedelta(hours=fhour)).isoformat()


def main() -> int:
    session = requests.Session()
    cycle = find_latest_cycle(session)
    layers = []
    for sector in SECTORS:
        for fhour in FORECAST_HOURS:
            url = spc_image_url(cycle, sector, fhour)
            available = url_exists(session, url)
            layers.append({
                "id": f"spc_{cycle:%Y%m%d%H}_{PRODUCT}_{sector}_f{fhour:03d}00",
                "source": "SPC official HREF finished PNG image",
                "model": MODEL,
                "product": PRODUCT,
                "productLabel": "SPC Official 6-hr QPF Ensemble Max",
                "sector": sector,
                "forecastHour": fhour,
                "accumHours": 6,
                "stepStart": max(0, fhour - 6),
                "stepEnd": fhour,
                "periodLabel": f"F{max(0, fhour - 6):02d}-F{fhour:02d} (SPC Official 6-hr Ensemble Max QPF)",
                "run": cycle.strftime("%Y%m%d%H"),
                "runLabel": cycle.strftime("%Y%m%d %HZ"),
                "startTimeUTC": iso(cycle, max(0, fhour - 6)),
                "validTimeUTC": iso(cycle, fhour),
                "imageUrl": url,
                "available": available,
            })
    catalog = {
        "generatedUTC": datetime.now(timezone.utc).isoformat(),
        "graphicsRoot": GRAPHICS_ROOT,
        "source": "SPC official HREF finished PNG image URL pattern from SPC viewer JavaScript image_url_model()",
        "defaultLayerId": next((l["id"] for l in layers if l["sector"] == "se" and l["forecastHour"] == 12 and l["available"]), layers[0]["id"]),
        "layers": layers,
        "notes": [
            "Official SPC image layers are exact SPC-rendered PNGs but are not value-sampleable.",
            "Use the computed interactive grid for hover values and max marker.",
            "This catalog is intentionally separate from docs/data/catalog.json.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} with {sum(1 for l in layers if l['available'])}/{len(layers)} available SPC image layer(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
