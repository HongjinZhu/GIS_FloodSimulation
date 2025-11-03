"""
Build_Grouped_Rasters.py
-----------------------------------
Reads config.yml and Groups.json, merges DEM tiles for each group into
one grouped raster (groupRaster_<center>.tif) in the grouped folder.

Requirements:
    - Python 3
    - rasterio  (if you don't have GDAL)
or  - GDAL (gdal_merge.py must be callable)

Usage:
    python Build_Grouped_Rasters.py
"""

import os, json, yaml, subprocess
from pathlib import Path

try:
    import rasterio
    from rasterio.merge import merge as rio_merge
    RASTERIO_OK = True
except Exception:
    RASTERIO_OK = False

# ---------------- CONFIG ----------------
CFG_PATH = "../config.yml" if os.path.exists("../config.yml") else "config.yml"
cfg = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))

PROJECT = cfg["project_name"]
DEM_DIR = Path(cfg["dem_folder"])
OUT_DIR = Path(cfg["dem_grouped_folder"])
GROUPS_PATH = Path(cfg["adjacency"]["groups_file"])

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Project: {PROJECT}")
print(f"Ungrouped DEM folder: {DEM_DIR}")
print(f"Output grouped folder: {OUT_DIR}")
print(f"Groups JSON: {GROUPS_PATH}")

groups = json.load(open(GROUPS_PATH, "r", encoding="utf-8"))["groups"]

# ---------------- HELPERS ----------------
def find_dem(idx:int) -> Path:
    """Locate DEM file by division index (search recursively)"""
    pats = [f"*{idx}.tif", f"*{idx}.TIF"]
    for p in pats:
        hits = list(DEM_DIR.rglob(p))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No DEM found for index {idx} in {DEM_DIR}")

def merge_rasters_rasterio(out_path:Path, src_paths:list):
    """Fallback merge using rasterio (works even without gdal_merge)"""
    import numpy as np
    srcs = [rasterio.open(p) for p in src_paths]
    mosaic, transform = rio_merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2],
                 "transform": transform, "compress": "LZW"})
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mosaic)
    for s in srcs: s.close()

def merge_rasters_gdal(out_path:Path, src_paths:list):
    """Merge via gdal_merge.py (if available in PATH)"""
    cmd = ["gdal_merge.py", "-o", str(out_path), "-of", "GTiff", "-n", "nan", "-a_nodata", "nan"] + [str(p) for p in src_paths]
    subprocess.run(cmd, check=True)

# ---------------- MAIN LOOP ----------------
for g in groups:
    name = g.get("name", "Group")
    order = g.get("order", [])
    if not order:
        print(f"[Skip] Empty group {name}")
        continue

    center_idx = order[len(order)//2]   # middle division = center
    out_path = OUT_DIR / f"groupRaster_{center_idx:02d}.tif"
    if out_path.exists():
        print(f"[Skip] {out_path.name} already exists.")
        continue

    # Collect input DEMs
    try:
        dem_paths = [find_dem(i) for i in order]
    except Exception as e:
        print(f"[WARN] Missing DEM in {name}: {e}")
        continue

    print(f"[{name}] merging divisions {order} -> {out_path.name}")

    try:
        if RASTERIO_OK:
            merge_rasters_rasterio(out_path, dem_paths)
        else:
            merge_rasters_gdal(out_path, dem_paths)
    except Exception as e:
        print(f"  !! Error merging {name}: {e}")
        continue

print("All grouped rasters generated")
