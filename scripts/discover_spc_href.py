#!/usr/bin/env python3
"""Probe SPC's public HREF viewer for official product URLs.

This is intentionally separate from the existing HREF data workflow. It writes a
small debug/catalog JSON file that helps us determine whether SPC exposes a
stable finished image URL or only a dynamic web viewer.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests

OUT = Path("docs/data/spc_official_discovery.json")
BASE_URL = "https://www.spc.noaa.gov/exper/href/"
VIEWER_URL = f"{BASE_URL}?model=href&product=qpf_006h_max&sector=se"
HEADERS = {"User-Agent": "href-qpf-viewer-spc-discovery/1.0"}

URL_RE = re.compile(r"(?:src|href)=[\"']([^\"']+)[\"']", re.IGNORECASE)
IMG_RE = re.compile(r"[\"']([^\"']+\.(?:png|gif|jpg|jpeg)(?:\?[^\"']*)?)[\"']", re.IGNORECASE)
TOKEN_RE = re.compile(r"qpf_006h_max|href|sector|product", re.IGNORECASE)


def fetch(url: str) -> tuple[int, str, str]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    content_type = r.headers.get("content-type", "")
    text = r.text if "text" in content_type or "javascript" in content_type or "html" in content_type else ""
    return r.status_code, content_type, text


def absolute(base: str, href: str) -> str:
    return urljoin(base, href)


def main() -> int:
    checked = []
    assets = []
    image_candidates = []
    text_hits = []

    for url in (BASE_URL, VIEWER_URL):
        status, ctype, text = fetch(url)
        checked.append({"url": url, "status": status, "contentType": ctype, "length": len(text)})
        if text:
            for m in URL_RE.finditer(text):
                href = m.group(1)
                full = absolute(url, href)
                if full not in assets:
                    assets.append(full)
            for m in IMG_RE.finditer(text):
                image_candidates.append(absolute(url, m.group(1)))
            for line in text.splitlines():
                if TOKEN_RE.search(line):
                    text_hits.append(line.strip()[:500])

    # Probe likely page assets, but keep it small and safe.
    for asset in list(assets)[:80]:
        if not re.search(r"\.(?:js|json|css)(?:\?|$)", asset, re.IGNORECASE):
            continue
        try:
            status, ctype, text = fetch(asset)
        except Exception as exc:
            checked.append({"url": asset, "error": str(exc)})
            continue
        checked.append({"url": asset, "status": status, "contentType": ctype, "length": len(text)})
        if text:
            for m in IMG_RE.finditer(text):
                image_candidates.append(absolute(asset, m.group(1)))
            for line in text.splitlines():
                if TOKEN_RE.search(line):
                    text_hits.append(line.strip()[:500])

    out = {
        "generatedUTC": datetime.now(timezone.utc).isoformat(),
        "baseUrl": BASE_URL,
        "viewerUrl": VIEWER_URL,
        "checked": checked,
        "assets": sorted(set(assets)),
        "imageCandidates": sorted(set(image_candidates)),
        "textHits": text_hits[:300],
        "notes": [
            "This file is diagnostic only.",
            "The live computed HREF workflow does not depend on this discovery output.",
            "If a stable SPC finished-image URL is found, a future PR can turn it into a proper official-image layer.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} with {len(out['imageCandidates'])} image candidate(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
