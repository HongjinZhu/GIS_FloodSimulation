# surface_volume_csv_processor.py
# Surface-Volume -> CSV generator for UNGROUPED and GROUPED DEMs (side by side)
# - Reads config.yml (paths, num_divs, options)
# - Optional Groups.json to subset/order coastal divisions
# - Writes per-division CSVs to cfg["surface_volume_out"] with suffixes:
#     *_ungrouped.csv  and  *_grouped.csv
#
# Required config keys:
#   project_name, num_divs, dem_folder, dem_grouped_folder, surface_volume_out
#
# Optional config keys:
#   sv_heights_m: [ ... exact levels ... ]  (meters)
#   sv_min_height_m: 0.0
#   sv_max_height_m: 10.0
#   sv_step_height_m: 0.25
#   sv_csv_name_fmt_ungrouped: "{PROJECT}_div{idx:02d}_ungrouped.csv"
#   sv_csv_name_fmt_grouped:   "{PROJECT}_div{idx:02d}_grouped.csv"
#   sv_strict_indexing: true
#   sv_use_groups: true/false
#   sv_only_groups: true/false
#   sv_order_by_groups: true/false
#   adjacency:
#     type: "groups"
#     groups_file: "path/to/Groups.json"

import os, re, sys, json, yaml, glob, math
from pathlib import Path
from typing import List, Tuple, Optional

# ----------------------------
# 0) Load config
# ----------------------------
CFG_PATH = os.path.join("..", "config.yml") if os.path.exists(os.path.join("..", "config.yml")) else "config.yml"
cfg = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))

PROJECT    = cfg["project_name"]
NUM_DIVS   = int(cfg["num_divs"])
DEM_DIR_U  = cfg["dem_folder"]
DEM_DIR_G  = cfg.get("dem_grouped_folder")  # may be None if you don't use grouped
SV_OUT     = cfg["surface_volume_out"]

# Height grid
SV_LEVELS  = cfg.get("sv_heights_m")
SV_MIN     = cfg.get("sv_min_height_m", 0.0)
SV_MAX     = cfg.get("sv_max_height_m", 10.0)
SV_STEP    = cfg.get("sv_step_height_m", 0.25)

# Output name format
CSV_FMT_U  = cfg.get("sv_csv_name_fmt_ungrouped", "{PROJECT}_div{idx:02d}_ungrouped.csv")
CSV_FMT_G  = cfg.get("sv_csv_name_fmt_grouped",   "{PROJECT}_div{idx:02d}_grouped.csv")
STRICT_INDEXING = bool(cfg.get("sv_strict_indexing", True))

# Group-aware switches
SV_USE_GROUPS       = bool(cfg.get("sv_use_groups", False))
SV_ONLY_GROUPS      = bool(cfg.get("sv_only_groups", False))
SV_ORDER_BY_GROUPS  = bool(cfg.get("sv_order_by_groups", SV_ONLY_GROUPS))

# ----------------------------
# 1) Utilities
# ----------------------------
def log(msg: str): print(msg, flush=True)
def ensure_dir(p: str): Path(p).mkdir(parents=True, exist_ok=True)

def build_levels() -> List[float]:
    if isinstance(SV_LEVELS, list) and SV_LEVELS:
        return [float(x) for x in SV_LEVELS]
    n = max(1, int(round((SV_MAX - SV_MIN) / SV_STEP)) + 1)
    return [SV_MIN + i*SV_STEP for i in range(n)]

LEVELS = build_levels()

def extract_first_int(path_str: str) -> Optional[int]:
    m = re.search(r'(\d+)', Path(path_str).stem)
    return int(m.group(1)) if m else None

def list_rasters(folder: str) -> List[str]:
    if not folder: return []
    root = Path(folder).expanduser()
    if not root.exists(): return []
    pats = [
        "*.tif","*.tiff","*.img","*.vrt",
        "**/*.tif","**/*.tiff","**/*.img","**/*.vrt",
        "*.TIF","*.TIFF","*.IMG","*.VRT",
        "**/*.TIF","**/*.TIFF","**/*.IMG","**/*.VRT",
    ]
    paths = []
    for pat in pats:
        paths.extend(glob.glob(str(root / pat), recursive=True))
    # drop sidecars
    paths = [p for p in paths if not p.lower().endswith(".aux.xml")]
    return sorted(set(paths))

def map_idx_to_dem(paths: List[str], label: str) -> dict:
    """Return {index: filepath}. Prefer numeric index embedded in filename."""
    by_num = {}
    for p in paths:
        k = extract_first_int(p)
        if k is not None:
            by_num[k] = p
    if len(by_num) >= min(NUM_DIVS, len(paths)):
        return by_num
    # fallback sequential
    if STRICT_INDEXING and len(paths) != NUM_DIVS:
        raise RuntimeError(f"[{label}] Found {len(paths)} DEM(s), but num_divs={NUM_DIVS}.")
    out = {}
    for i, p in enumerate(sorted(paths)):
        out[i] = p
    return out

# ----------------------------
# 2) Groups.json (or Adjacency.json if graph)
# ----------------------------
def _normalize_groups(obj):
    """
    Accept both:
      {"groups":[{"name":"A","order":[0,1,2]}, ...], "landlocked":[...]}
      [[0,1,2],[10,11]]
    Return (orders_list, landlocked_list)
    """
    if isinstance(obj, dict):
        groups = obj.get("groups", [])
        if groups and isinstance(groups[0], dict):
            orders = [g["order"] for g in groups]
        else:
            orders = groups
        landlocked = obj.get("landlocked", [])
    else:
        orders, landlocked = obj, []
    # dedupe check
    flat = [i for g in orders for i in g]
    if len(set(flat)) != len(flat):
        dupes = sorted([i for i in flat if flat.count(i) > 1])
        raise ValueError(f"Groups.json has duplicate IDs across groups: {dupes}")
    return orders, landlocked

def load_groups_from_cfg(cfg, N):
    adj = cfg.get("adjacency", {})
    if adj.get("type") != "groups":
        return None, [], []
    path = adj.get("groups_file")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"adjacency.type='groups' but groups_file not found: {path}")
    obj = json.load(open(path, "r", encoding="utf-8"))
    orders, landlocked = _normalize_groups(obj)
    bad = [i for i in [x for g in orders for x in g] if (not isinstance(i,int)) or i<0 or i>=N]
    if bad:
        raise ValueError(f"Groups.json out-of-range IDs (0..{N-1}): {sorted(set(bad))}")
    coastal_flat = [i for g in orders for i in g]
    return orders, landlocked, coastal_flat

GROUP_ORDERS, GROUP_LANDLOCKED, COASTAL_IDS = (None, [], [])
if SV_USE_GROUPS:
    GROUP_ORDERS, GROUP_LANDLOCKED, COASTAL_IDS = load_groups_from_cfg(cfg, NUM_DIVS)

def decide_process_indices(idx_to_dem: dict) -> List[int]:
    """Return the list of indices to process and in what order."""
    if SV_USE_GROUPS and SV_ONLY_GROUPS:
        base = list(COASTAL_IDS)
    else:
        base = sorted(idx_to_dem.keys())

    if SV_USE_GROUPS and SV_ORDER_BY_GROUPS and GROUP_ORDERS:
        ordered = []
        for order in GROUP_ORDERS:
            ordered.extend(order)
        return [i for i in ordered if i in set(base)]
    return base  # default numeric index order

# ----------------------------
# 3) Engines: arcpy or numpy
# ----------------------------
ARCPY_OK = False
try:
    import arcpy
    if hasattr(arcpy, "CheckExtension") and arcpy.CheckExtension("3D") == "Available":
        ARCPY_OK = True
except Exception:
    ARCPY_OK = False

if not ARCPY_OK:
    import rasterio
    import numpy as np

def surface_volume_arcgis(dem_path: str, heights_m: List[float]) -> List[Tuple[float,float,float,int]]:
    out = []
    arcpy.CheckOutExtension("3D")
    for h in heights_m:
        tbl = arcpy.CreateUniqueName("svtbl", "in_memory")
        try:
            arcpy.ddd.SurfaceVolume(dem_path, tbl, "BELOW", h, "HORIZONTAL_PLANE", 1.0)
            area = 0.0; vol = 0.0
            with arcpy.da.SearchCursor(tbl, ["AREA_2D", "VOLUME"]) as cur:
                for a2d, v in cur:
                    area += float(a2d or 0.0)
                    vol  += float(v or 0.0)
            out.append((float(h), float(area), float(vol), -1))
        finally:
            try: arcpy.management.Delete(tbl)
            except Exception: pass
    return out

def pixel_area_from_transform(transform) -> float:
    # area = |a*e - b*d| for Affine(a,b,c,d,e,f)
    return abs(transform.a * transform.e - transform.b * transform.d)

def surface_volume_numpy(dem_path: str, heights_m: List[float]) -> List[Tuple[float,float,float,int]]:
    with rasterio.open(dem_path) as ds:
        if ds.crs is None:
            raise RuntimeError(f"{dem_path} has no CRS.")
        # require projected CRS (meters)
        lu = (getattr(ds.crs, "linear_units", None) or "").lower()
        if "degree" in lu:
            raise RuntimeError(f"{dem_path} is in degrees; reproject to a meter-based CRS.")
        nodata = ds.nodata
        A = pixel_area_from_transform(ds.transform)
        elev = ds.read(1, masked=False).astype("float64")
    if nodata is not None:
        mask = elev == nodata
        elev[mask] = float("nan")
    out = []
    finite = np.isfinite(elev)
    for h in heights_m:
        wet = finite & (elev < h)
        cells = int(np.count_nonzero(wet))
        if cells == 0:
            out.append((float(h), 0.0, 0.0, 0))
            continue
        area_m2 = cells * A
        vol_m3  = float(np.nansum((h - elev[wet])) * A)
        out.append((float(h), float(area_m2), float(vol_m3), cells))
    return out

def write_csv(rows: List[Tuple[float,float,float,int]], out_csv: str):
    import csv
    ensure_dir(Path(out_csv).parent.as_posix())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["height_m","area_m2","volume_m3","cell_count"])
        for h,a,v,c in rows:
            w.writerow([f"{h:.6f}", f"{a:.6f}", f"{v:.6f}", c])

# ----------------------------
# 4) Per-source runner
# ----------------------------
def run_source(label: str, dem_dir: str, csv_fmt: str, manifest: dict):
    if not dem_dir:
        log(f"[{label}] Skipped (no folder set).")
        return

    dem_paths = list_rasters(dem_dir)
    if not dem_paths:
        raise RuntimeError(f"[{label}] No rasters found in '{dem_dir}'.")

    idx_to_dem = map_idx_to_dem(dem_paths, label)
    process_idx = decide_process_indices(idx_to_dem)

    log(f"[{label}] DEMs: {len(dem_paths)}  mapped: {len(idx_to_dem)}  to process: {len(process_idx)}")
    items = []

    for k, idx in enumerate(process_idx, 1):
        dem_path = idx_to_dem.get(idx)
        if not dem_path:
            log(f"[{label}] [{k}/{len(process_idx)}] div {idx:02d}: MISSING DEM — skipping")
            items.append({"index": int(idx), "dem": None, "csv": None, "error": "missing DEM"})
            continue

        # Expand {PROJECT} if used in format
        out_name = csv_fmt.format(PROJECT=PROJECT, idx=idx)
        out_csv  = Path(SV_OUT) / out_name

        log(f"[{label}] [{k}/{len(process_idx)}] div {idx:02d}: {Path(dem_path).name} -> {out_name}")
        try:
            rows = surface_volume_arcgis(dem_path, LEVELS) if ARCPY_OK else surface_volume_numpy(dem_path, LEVELS)
            write_csv(rows, out_csv.as_posix())
            items.append({
                "index": int(idx),
                "dem": dem_path,
                "csv": out_csv.as_posix(),
                "min_height_m": float(LEVELS[0]),
                "max_height_m": float(LEVELS[-1]),
                "n_levels": len(LEVELS),
                "source": label
            })
        except Exception as e:
            log(f"[{label}]   !! ERROR on div {idx}: {e}")
            items.append({"index": int(idx), "dem": dem_path, "csv": None, "error": str(e), "source": label})
    manifest[label] = items

# ----------------------------
# 5) Main
# ----------------------------
def main():
    log("=== Surface Volume -> CSV (Ungrouped + Grouped) ===")
    log(f"Project: {PROJECT}   num_divs={NUM_DIVS}")
    log(f"Ungrouped DEM dir: {DEM_DIR_U}")
    log(f"Grouped   DEM dir: {DEM_DIR_G}")
    log(f"Output dir: {SV_OUT}")
    log(f"Levels: {len(LEVELS)} from {LEVELS[0]:.3f} to {LEVELS[-1]:.3f} m")
    log(f"Engine: {'ArcGIS 3D Analyst' if ARCPY_OK else 'NumPy+rasterio'}")
    if SV_USE_GROUPS:
        log(f"Groups: use={SV_USE_GROUPS} only={SV_ONLY_GROUPS} order_by={SV_ORDER_BY_GROUPS}")
        if GROUP_ORDERS:
            log(f"  #groups={len(GROUP_ORDERS)}  coastal_ids={len(COASTAL_IDS)}")

    ensure_dir(SV_OUT)
    manifest = {
        "project": PROJECT,
        "num_divs": NUM_DIVS,
        "levels_m": LEVELS,
        "engine": "arcpy.SurfaceVolume" if ARCPY_OK else "numpy_rasterio",
        "dem_dir_ungrouped": DEM_DIR_U,
        "dem_dir_grouped": DEM_DIR_G,
        "csv_fmt_ungrouped": CSV_FMT_U,
        "csv_fmt_grouped": CSV_FMT_G,
        "use_groups": SV_USE_GROUPS,
        "only_groups": SV_ONLY_GROUPS,
        "order_by_groups": SV_ORDER_BY_GROUPS,
        "groups": GROUP_ORDERS if GROUP_ORDERS is not None else [],
        "landlocked": GROUP_LANDLOCKED,
    }

    # Run both sources
    run_source("ungrouped", DEM_DIR_U, CSV_FMT_U, manifest)
    run_source("grouped",   DEM_DIR_G, CSV_FMT_G, manifest)

    man_path = Path(SV_OUT) / f"{PROJECT}_sv_manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    u_ok = sum(1 for it in manifest.get("ungrouped", []) if it.get("csv"))
    g_ok = sum(1 for it in manifest.get("grouped", []) if it.get("csv"))
    log(f"Done. Wrote {u_ok} ungrouped CSV(s) and {g_ok} grouped CSV(s).")
    log(f"Manifest: {man_path}")

if __name__ == "__main__":
    main()