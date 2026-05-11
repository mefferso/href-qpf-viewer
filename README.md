[README.md](https://github.com/user-attachments/files/27608940/README.md)
# HREF QPF Viewer

A simple GitHub Pages viewer for HREF QPF data.

This repo does three things:

1. Downloads recent HREF ensemble QPF GRIB2 files from NOMADS.
2. Crops and converts the gridded precipitation into lightweight JSON files.
3. Serves a zoomable, clickable Leaflet map from `docs/index.html`.

## What you get in version 1

- Zoomable map
- HREF QPF layers
- Click-to-sample precip values
- Run/product/forecast-hour selector
- Opacity slider
- Data auto-updates through GitHub Actions

## Setup

1. Create a new public GitHub repo.
2. Upload everything in this folder to the repo.
3. Go to **Settings → Pages**.
4. Under **Build and deployment**, choose:
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/docs`
5. Go to **Actions**.
6. Click **Update HREF QPF Viewer**.
7. Click **Run workflow**.
8. Wait for the run to finish.
9. Open the GitHub Pages URL shown under **Settings → Pages**.

## Changing the map domain

Open `scripts/build_href_qpf.py` and edit this block near the top:

```python
DOMAIN = {
    "west": -95.0,
    "east": -83.0,
    "south": 24.0,
    "north": 34.0,
}
```

For a tighter LIX/Gulf Coast view, try:

```python
DOMAIN = {
    "west": -92.5,
    "east": -87.0,
    "south": 27.5,
    "north": 32.5,
}
```

## Changing forecast hours

Open `scripts/build_href_qpf.py` and edit:

```python
FORECAST_HOURS = [6, 12, 18, 24, 30, 36, 42, 48]
```

Keeping this list shorter makes the website update faster and keeps the repo smaller.

## Notes

This starter version intentionally keeps the backend simple. It creates sampleable grids rather than fancy raster tiles. Once this is working, the next upgrade would be proper tiled rasters or Cloud Optimized GeoTIFFs.
