#!/usr/bin/env python3
"""Run the fast HREF builder using an SPC-like QPF color scale."""

from __future__ import annotations

import build_href_qpf as base
import build_href_qpf_fast as fast

# Color bins sampled/approximated from the SPC-style QPF legend provided by the
# user. These colors are used by generated rasters, the web legend, and the
# legend-matched contour overlay because they are written into catalog.json.
SPC_LIKE_QPF_COLOR_SCALE = [
    {"min": 0.01, "max": 0.10, "color": "#7fff00", "label": "0.01-0.10"},
    {"min": 0.10, "max": 0.25, "color": "#3fcc00", "label": "0.10-0.25"},
    {"min": 0.25, "max": 0.50, "color": "#009900", "label": "0.25-0.50"},
    {"min": 0.50, "max": 0.75, "color": "#0f4c8c", "label": "0.50-0.75"},
    {"min": 0.75, "max": 1.00, "color": "#0a7faa", "label": "0.75-1.00"},
    {"min": 1.00, "max": 1.25, "color": "#05b2c7", "label": "1.00-1.25"},
    {"min": 1.25, "max": 1.50, "color": "#00e5e5", "label": "1.25-1.50"},
    {"min": 1.50, "max": 1.75, "color": "#8c66cc", "label": "1.50-1.75"},
    {"min": 1.75, "max": 2.00, "color": "#8c33ac", "label": "1.75-2.00"},
    {"min": 2.00, "max": 2.50, "color": "#8c008c", "label": "2.00-2.50"},
    {"min": 2.50, "max": 3.00, "color": "#8c0000", "label": "2.50-3.00"},
    {"min": 3.00, "max": 4.00, "color": "#cc0000", "label": "3.00-4.00"},
    {"min": 4.00, "max": 5.00, "color": "#e53f00", "label": "4.00-5.00"},
    {"min": 5.00, "max": 7.00, "color": "#ff7f00", "label": "5.00-7.00"},
    {"min": 7.00, "max": 10.00, "color": "#cc8c00", "label": "7.00-10.00"},
    {"min": 10.00, "max": 15.00, "color": "#e5c500", "label": "10.00-15.00"},
    {"min": 15.00, "max": 20.00, "color": "#ffff00", "label": "15.00-20.00"},
    {"min": 20.00, "max": 999.00, "color": "#ffa5bf", "label": ">20.00"},
]


def refresh_lix_cwa_boundary() -> None:
    try:
        import build_lix_cwa_boundary

        build_lix_cwa_boundary.refresh_lix_cwa_boundary_safely()
    except Exception as exc:
        base.log(f"WARNING: LIX CWA boundary setup failed before HREF build: {exc}")


def main() -> int:
    base.COLOR_SCALE = SPC_LIKE_QPF_COLOR_SCALE
    refresh_lix_cwa_boundary()
    return fast.main()


if __name__ == "__main__":
    raise SystemExit(main())
