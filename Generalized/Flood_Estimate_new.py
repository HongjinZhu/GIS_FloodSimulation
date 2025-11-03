#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flood_estimate.py
Config-driven flood estimation runner for GISSR pipeline.

Solvers included:
  - static_bathtub : depth = max(peak_surge - elevation, 0)
  - volume_inverse : enforces consistency with Surface-Volume curves
                     by using the SV curve to translate between
                     height and volume, then invert back to height.

Outputs:
  <outdir>/flood_heights.csv  (columns: div, height_m)
  <outdir>/diagnostics.json   (basic run metadata)

Author: GISSR pipeline (config-driven)
"""

from __future__ import annotations
import argparse
import json
import yaml
import sys
from pathlib import Path
import re
import glob
import math
import datetime as dt
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


# ---------------------------
# Utilities & config loading
# ---------------------------

FT_TO_M = 0.3048

def p(x): 
    return Path(x).expanduser() if isinstance(x, str) else x

def exists(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path

def load_yaml(cfg_path: Path) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def detect_div_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if re.search(r"(div|sector|id)", c, re.I)]
    if not candidates:
        raise ValueError("Could not detect division column in elevation CSV. "
                         "Add a 'div' or 'id' column.")
    return candidates[0]

def assert_permutation(sections: List[int], ndiv: int):
    if sorted(sections) != list(range(ndiv)):
        raise ValueError("sections is not a permutation of 0..ndiv-1. "
                         f"Got len={len(sections)} unique={len(set(sections))}, ndiv={ndiv}")

def load_sections_from_groups(groups_json: Path, ndiv: int) -> List[int]:
    data = json.loads(groups_json.read_text())
    order = []
    for g in data.get("groups", []):
        order += list(map(int, g.get("order", [])))
    if len(order) != ndiv:
        raise ValueError(f"Groups.json order length {len(order)} != ndiv {ndiv}")
    assert_permutation(order, ndiv)
    return order

def load_sections_from_graph(graph_json: Path, ndiv: int, start: Optional[int]=None) -> List[int]:
    G = json.loads(graph_json.read_text())
    nodes = sorted(map(int, G.keys()))
    start = nodes[0] if start is None else int(start)
    seen, order = set(), []
    stack = [start]
    while stack and len(order) < ndiv:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u); order.append(u)
        nbrs = sorted(map(int, G.get(str(u), [])), reverse=True)
        for v in nbrs:
            if v not in seen:
                stack.append(v)
    # add isolated if any
    for u in nodes:
        if u not in seen:
            order.append(u)
    if len(order) != ndiv:
        raise ValueError(f"Graph traversal produced {len(order)} nodes, expected {ndiv}")
    assert_permutation(order, ndiv)
    return order


# ---------------------------
# Surface-Volume helpers
# ---------------------------

class SurfaceVolumeCurve:
    """
    Wraps one division's Surface-Volume CSV (columns: height, volume).
    Provides monotone interpolators:
      - V(h)  via PCHIP
      - h(V)  via PCHIP on swapped arrays (after dedup/monotone checks)
    """
    def __init__(self, heights: np.ndarray, volumes: np.ndarray):
        h = np.asarray(heights, float)
        v = np.asarray(volumes, float)
        # sort and de-dup by height
        idx = np.argsort(h)
        h = h[idx]; v = v[idx]

        # enforce nondecreasing volumes
        if np.any(np.diff(v) < -1e-8):
            # small smoothing: cumulative max
            v = np.maximum.accumulate(v)

        # drop duplicate heights if any
        uniq_mask = np.r_[True, np.diff(h) > 0]
        h = h[uniq_mask]; v = v[uniq_mask]

        # Construct interpolators
        self.h_min = float(h[0]); self.h_max = float(h[-1])
        self.v_min = float(v[0]); self.v_max = float(v[-1])

        self.V_of_h = PchipInterpolator(h, v, extrapolate=True)
        # For inversion, ensure strictly increasing V; enforce a tiny epsilon if needed
        v_inc = np.maximum.accumulate(v + np.linspace(0, 1e-10, len(v)))
        self.h_of_V = PchipInterpolator(v_inc, h, extrapolate=True)

    def volume_from_height(self, h: float|np.ndarray) -> float|np.ndarray:
        return self.V_of_h(h)

    def height_from_volume(self, V: float|np.ndarray) -> float|np.ndarray:
        return self.h_of_V(V)


def load_sv_curves(sv_glob: str, ndiv: int) -> List[Optional[SurfaceVolumeCurve]]:
    """
    Expects filenames containing '_div<id>' and CSV with columns: height, volume
    Returns a list of length ndiv with SurfaceVolumeCurve or None if missing.
    """
    files = sorted(glob.glob(sv_glob))
    out: List[Optional[SurfaceVolumeCurve]] = [None]*ndiv

    def parse_div(path: str) -> Optional[int]:
        m = re.search(r"_div(\d+)", Path(path).stem)
        return int(m.group(1)) if m else None

    for f in files:
        d = parse_div(f)
        if d is None or d >= ndiv: 
            continue
        df = pd.read_csv(f)
        if not {"height","volume"}.issubset(set(df.columns)):
            raise ValueError(f"SV CSV missing required columns height/volume: {f}")
        out[d] = SurfaceVolumeCurve(df["height"].to_numpy(), df["volume"].to_numpy())

    missing = [i for i, s in enumerate(out) if s is None]
    if missing:
        raise ValueError(f"Missing SV CSVs for divisions: {missing[:10]}{'...' if len(missing)>10 else ''}")
    return out


# ---------------------------
# Solvers
# ---------------------------

def solve_static_bathtub(
    ndiv: int,
    elevation_m: np.ndarray,
    surge_series_m: np.ndarray
) -> np.ndarray:
    """
    Simple baseline: flood height = max(peak_surge - ground_elev, 0)
    """
    peak = float(np.nanmax(surge_series_m))
    depths = np.maximum(peak - elevation_m, 0.0)
    return depths


def solve_volume_inverse(
    ndiv: int,
    elevation_m: np.ndarray,
    surge_series_m: np.ndarray,
    sv: List[SurfaceVolumeCurve]
) -> np.ndarray:
    """
    Uses static bathtub depth to define a target height, converts that to
    volume V(h), then maps back to a cleaned height via h(V). This prevents
    oddities from raw DEM edges and respects your SV monotonicity.

    Note: This is NOT dynamic redistribution; it’s a consistent static estimate.
    """
    static_h = solve_static_bathtub(ndiv, elevation_m, surge_series_m)
    out = np.zeros(ndiv, float)
    for i in range(ndiv):
        # volume implied by static depth above local datum (0)=ground?
        # We assume SV height h is water depth (same datum used building SV tables).
        V = float(sv[i].volume_from_height(static_h[i]))
        out[i] = float(sv[i].height_from_volume(V))
        # clamp to [0, h_max] to avoid tiny extrapolation artifacts
        out[i] = max(0.0, min(out[i], sv[i].h_max))
    return out


def solve_kinematic_travel(
    ndiv: int,
    elevation_m: np.ndarray,
    roughness: np.ndarray,
    slope: np.ndarray,
    surge_series_m: np.ndarray,
    sections: List[int],
    sv: List[SurfaceVolumeCurve],
    coast_segment_length_m: float,
    dynamic_cfg: Optional[dict] = None
) -> np.ndarray:
    """
    Division-agnostic, config-driven 1D cascade using Manning + SV curves.

    Routing logic:
      - Treat 'sections' as the coastal-to-inland travel order.
      - At each time step, each division i receives inflow:
          * a coastal inflow based on the current surge depth above its ground,
          * plus any overflow passed from the previous division in 'sections'.
      - Convert inflow discharge to volume over dt, add to current water volume.
      - If volume exceeds a threshold of this division's SV capacity, spill the
        excess to the next division.
      - At the end, convert final volume to water depth via sv[i].height_from_volume(V).

    Notes:
      - This is a practical, stable baseline; refine geometry/width/R as needed.
      - All units are SI.
    """
    DC = dynamic_cfg or {}
    dt = float(DC.get("dt_seconds", 3600.0))
    passes = max(1, int(DC.get("passes", 1)))
    spill_thr = float(DC.get("spill_threshold", 0.99))
    spill_frac = float(DC.get("spill_fraction", 1.0))
    min_slope = float(DC.get("min_slope", 1.0e-5))
    clamp_h = DC.get("max_depth_m", None)

    # Time discretization (assume surge_series is evenly spaced)
    T = int(surge_series_m.size)
    if T < 1:
        raise ValueError("Surge series is empty.")
    # Per-division state: store water volume (m^3) currently held
    V = np.zeros(ndiv, dtype=float)

    # Helper: compute discharge Q (m^3/s) given local depth h (m)
    # Q = v * A, with v from Manning, A ~ h * width.
    # Manning: v = R^(2/3) * sqrt(S) / n ; hydraulic radius R ~ h (simplified).
    width = float(coast_segment_length_m)

    def discharge(i: int, h: float) -> float:
        if h <= 0.0:
            return 0.0
        n = max(1.0e-6, float(roughness[i]))
        S = max(min_slope, float(slope[i]))
        R = max(1.0e-6, h)             # simple proxy; refine if you have geometry
        v = (R ** (2.0 / 3.0)) * math.sqrt(S) / n   # m/s
        A = h * width                                # m^2
        return v * A                                 # m^3/s

    # Precompute per-division SV max capacities for spill thresholding
    Vmax = np.array([s.v_max for s in sv], dtype=float)
    Vspill = spill_thr * Vmax

    # Optional clamp on depths when computing Q from surge
    def clamp_depth(h):
        if clamp_h is None:
            return h
        return max(0.0, min(h, float(clamp_h)))

    # Multiple passes can improve inland penetration when sections are long
    for _ in range(passes):
        # iterate over time
        for t in range(T):
            surge_t = float(surge_series_m[t])
            # overflow buffer for this step (what each div will pass to its successor)
            overflow = np.zeros(ndiv, dtype=float)

            # walk along sections in order (coast -> inland)
            for idx, i in enumerate(sections):
                # 1) Coastal inflow: surge depth above local ground
                coastal_h = clamp_depth(max(surge_t - elevation_m[i], 0.0))
                Qin = discharge(i, coastal_h)  # m^3/s from the coast
                Vin = Qin * dt                  # m^3 this step

                # 2) Add upstream overflow from the previous section (if any)
                if idx > 0:
                    Vin += overflow[sections[idx - 1]]

                # 3) Update local storage
                V[i] += Vin

                # 4) If above spill threshold, spill some fraction to the next section
                if V[i] > Vspill[i] and idx < len(sections) - 1:
                    excess = V[i] - Vspill[i]
                    spill = spill_frac * excess
                    V[i] -= spill
                    overflow[i] += spill

            # Tail division might still exceed absolute max; just cap it
            # (Alternatively, you can let it accumulate and report as-is.)
            tail = sections[-1]
            if V[tail] > Vmax[tail]:
                V[tail] = Vmax[tail]

    # Convert final volumes to depths through SV inversion
    H = np.zeros(ndiv, dtype=float)
    for i in range(ndiv):
        H[i] = float(sv[i].height_from_volume(V[i]))
        if clamp_h is not None:
            H[i] = min(H[i], float(clamp_h))
        # small numeric clean
        if H[i] < 1e-6: 
            H[i] = 0.0
    return H

# ---------------------------
# Runner
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Flood estimate runner (config-driven).")
    ap.add_argument("--config", required=True, help="Path to config.yml")
    ap.add_argument("--solver", default="static_bathtub",
                    choices=["static_bathtub", "volume_inverse", "kinematic_travel"],
                    help="Which solver to run.")
    ap.add_argument("--outdir", default="outputs", help="Directory to write outputs.")
    args = ap.parse_args()

    cfg = load_yaml(p(args.config))
    outdir = p(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Paths & inputs ---
    proj_name  = cfg.get("project_name", "Project")
    num_divs   = cfg.get("num_divs", None)

    dem_folder = exists(p(cfg["dem_folder"]), "DEM folder")
    sv_out     = p(cfg.get("surface_volume_out", "Data/SurfaceVolume"))

    roughness_csv  = exists(p(cfg["roughness_csv"]),  "roughness CSV")
    slope_csv      = exists(p(cfg["slope_csv"]),      "slope CSV")
    elevation_csv  = exists(p(cfg["elevation_csv"]),  "elevation CSV")

    walls_cfg      = cfg.get("walls", {})
    walls_ids_csv  = p(walls_cfg.get("ids_csv")) if walls_cfg.get("ids_csv") else None
    wall_height_ft = float(walls_cfg.get("wall_height_ft", 0.0))
    wall_height_m  = wall_height_ft * FT_TO_M

    storm_cfg      = cfg.get("storm", {})
    storm_mode     = storm_cfg.get("mode", "time_history")
    surge_csv      = p(storm_cfg.get("surge_csv")) if storm_cfg.get("surge_csv") else None

    coast_L        = float(cfg.get("coast_segment_length_m", 100.0))

    adj_cfg        = cfg.get("adjacency", {"type":"groups"})
    adj_type       = adj_cfg.get("type", "groups")
    groups_file    = p(adj_cfg.get("groups_file")) if adj_cfg.get("groups_file") else None
    graph_file     = p(adj_cfg.get("graph_file")) if adj_cfg.get("graph_file") else None

    # --- Elevation table & division ids ---
    elev_df = pd.read_csv(elevation_csv)
    div_col = detect_div_column(elev_df)
    div_ids = elev_df[div_col].astype(int).to_numpy()
    ndiv_detected = int(pd.Series(div_ids).nunique())
    ndiv = int(num_divs) if num_divs is not None else ndiv_detected
    if num_divs is not None and ndiv != ndiv_detected:
        print(f"[warn] config.num_divs={num_divs} but elevation has {ndiv_detected} unique IDs. Using config value {ndiv}.")
    # Ensure 0..ndiv-1 ids by sort + reindex
    # Expectation: your CSV already lines up; if not, we force build a mapping:
    uniq_sorted = sorted(pd.unique(div_ids))
    if uniq_sorted != list(range(ndiv)):
        # Reindex mapping
        remap = {old:i for i, old in enumerate(uniq_sorted[:ndiv])}
        elev_df["_div0"] = elev_df[div_col].map(remap).astype(int)
        div_col = "_div0"
        print("[info] Division IDs were remapped to 0..ndiv-1 for internal use.")

    elev_series = elev_df.sort_values(div_col)["MEAN2"].to_numpy()
    if len(elev_series) != ndiv:
        raise ValueError(f"Elevation MEAN2 length {len(elev_series)} != ndiv {ndiv}. "
                         "Make sure elevation CSV has one row per division.")

    rough_series = pd.read_csv(roughness_csv).sort_values(div_col)["Roughness"].to_numpy()
    slope_series = pd.read_csv(slope_csv).sort_values(div_col)["Slope"].to_numpy()

    # Apply walls (optional)
    if walls_ids_csv and wall_height_m > 0:
        wids = pd.read_csv(walls_ids_csv)["ID"].astype(int).tolist()
        for wid in wids:
            if 0 <= wid < ndiv:
                elev_series[wid] = elev_series[wid] + wall_height_m

    # --- Sections / order ---
    if adj_type == "groups":
        if not groups_file:
            raise ValueError("adjacency.type='groups' but no groups_file set in config.")
        sections = load_sections_from_groups(groups_file, ndiv)
    else:
        if not graph_file:
            raise ValueError("adjacency.type='graph' but no graph_file set in config.")
        # Use the smallest division id as default start
        sections = load_sections_from_graph(graph_file, ndiv)

    # --- Surface-Volume curves ---
    # Accept either a directory or a glob pattern
    sv_glob = str((sv_out / "*_div*.csv") if sv_out.is_dir() else sv_out)
    sv_curves = load_sv_curves(sv_glob, ndiv)

    # --- Surge ---
    if storm_mode not in ("time_history","peak_only"):
        raise ValueError(f"Unknown storm.mode: {storm_mode}")
    if not surge_csv:
        raise ValueError("storm.surge_csv must be provided.")

    surge_series = pd.read_csv(surge_csv)['Verified (m)'].to_numpy()
    if np.all(~np.isfinite(surge_series)) or surge_series.size == 0:
        raise ValueError("Surge series is empty or invalid.")

    if storm_mode == "peak_only":
        # Keep as a 1D series still; we use its max
        pass

    # --- Choose solver ---
    solver = args.solver
    if solver == "static_bathtub":
        heights = solve_static_bathtub(ndiv, elev_series, surge_series)
    elif solver == "volume_inverse":
        heights = solve_volume_inverse(ndiv, elev_series, surge_series, sv_curves)
    elif solver == "kinematic_travel":
        heights = solve_kinematic_travel(
            ndiv=ndiv,
            elevation_m=elev_series,
            roughness=rough_series,
            slope=slope_series,
            surge_series_m=surge_series,
            sections=sections,
            sv=sv_curves,
            coast_segment_length_m=coast_L,
            dynamic_cfg=cfg.get("dynamic", {})
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # --- Write outputs ---
    out_csv = outdir / "flood_heights.csv"
    pd.DataFrame({"div": np.arange(ndiv, dtype=int), "height_m": heights}).to_csv(out_csv, index=False)

    # Minimal diagnostics
    diagnostics = {
        "project": cfg.get("project_name", "Project"),
        "ndiv": ndiv,
        "solver": solver,
        "storm_mode": storm_mode,
        "surge_peak_m": float(np.nanmax(surge_series)),
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
    }
    (outdir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    print(f"[ok] Wrote {out_csv}")
    print(f"[ok] Wrote {(outdir / 'diagnostics.json')}")
    print("[done]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
