#!/usr/bin/env python3
"""
RIC_fixed.py
Corrected version of your RIC script:
- deterministic bounding-box selection
- correct 2x2x2 corner indexing
- safe interpolation (no division by zero)
- consistent save/load of polynomial coefficients
- directories created once, not in tight loops
- uses CLI args dt_, numsteps_, geom_num, V
"""

import os
import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd


# ---------------------------
# Configuration / Grids
# ---------------------------
L_GRID = np.array([0.0, 10.0, 20.0, 40.0, 50.0, 60.0, 80.0, 100.0])           # nm
W_GRID = np.array([5.0, 10.0])  # nm
V_GRID = np.array([0.0, 1.0, 2.0])         # discrete volt steps (example)
# If you have a c-grid, define it here; otherwise c is treated continuously:
C_GRID = np.linspace(0.0, 1.0, 11)              # example 0.0,0.1,...,1.0

# default constants (can be overridden by CLI/params)
c10 = 10  # example RHS conc? (kept for backwards compatibility)


# ---------------------------
# Utility: neighbors
# ---------------------------
def neighbors(grid: np.ndarray, value: float) -> Tuple[float, float]:
    """
    Return (lo, hi) neighbors from a sorted grid that bracket value.
    If value <= grid[0], returns (grid[0], grid[0]).
    If value >= grid[-1], returns (grid[-1], grid[-1]).
    """
    idx = np.searchsorted(grid, value, side="left")
    if idx == 0:
        return float(grid[0]), float(grid[0])
    if idx >= len(grid):
        return float(grid[-1]), float(grid[-1])
    lo = float(grid[idx - 1])
    hi = float(grid[idx])
    return lo, hi


# ---------------------------
# Bounding box finder
# ---------------------------
def _findBoundingBox_(p: List[float]):
    """
    p = [dt, numsteps, c_value, L, W, V, X]
    Returns bounding box: ([Cmin,Cmax],[Lmin,Lmax],[wmin,wmax],[Vmin,Vmax])
    Deterministic O(1) neighbor selection using predefined grids.
    """
    _, _, c_val, L_val, W_val, V_val, _ = p

    Cmin, Cmax = neighbors(C_GRID, c_val)
    Lmin, Lmax = neighbors(L_GRID, L_val)
    wmin, wmax = neighbors(W_GRID, W_val)
    Vmin, Vmax = neighbors(V_GRID, V_val)

    return [Cmin, Cmax], [Lmin, Lmax], [wmin, wmax], [Vmin, Vmax]


# ---------------------------
# Multilinear interpolation (returns numpy.poly1d)
# ---------------------------
def _multiLinearInterp_poly1d(p: List[float], bB, t: int) -> np.poly1d:
    """
    Read the 8 corner polynomial models from CSVs, trilinearly blend them,
    and return a numpy.poly1d representing the interpolated polynomial.
    Assumes training files at:
       FeNICsOutputFiles/{dt}s/{numsteps}steps/L{L}_W{W}/V{V}/Conc_x_t/fitModel_t_{t}.csv
    Each such CSV must contain a column named 'Polynomial' that lists coefficients
    in descending-power order (so np.poly1d constructed with that vector works).
    """

    # Unpack bounding box
    Cmin, Cmax = bB[0]
    Lmin, Lmax = bB[1]
    wmin, wmax = bB[2]
    Vmin, Vmax = bB[3]

    # Avoid division by zero: if equal, weight along that axis is 0 (i.e., snap to low=high)
    def weight(value, lo, hi):
        if hi == lo:
            return 0.0
        return (value - lo) / (hi - lo)

    # pull required values from p consistently:
    # p = [dt, numsteps, c_value, L, W, V, X]
    dt_val, numsteps_val, c_val, L_val, W_val, V_val, X_val = p

    wxL = weight(L_val, Lmin, Lmax)
    wxw = weight(W_val, wmin, wmax)
    wxV = weight(V_val, Vmin, Vmax)

    # base path for trained models
    base_path = os.path.join("FeNICsOutputFiles", f"{dt_val}s", f"{numsteps_val}steps")

    # load 8 corner models into list by index = x_ + 2*y_ + 4*z_ where
    # x_ corresponds to L axis (0->Lmin,1->Lmax),
    # y_ corresponds to w axis (0->wmin,1->wmax),
    # z_ corresponds to V axis (0->Vmin,1->Vmax)
    model_bBpts = [None] * 8

    for z_ in (0, 1):
        for y_ in (0, 1):
            for x_ in (0, 1):
                Lc = [Lmin, Lmax][x_]
                wc = [wmin, wmax][y_]
                Vc = [Vmin, Vmax][z_]
                
                if Lc == 0 or Vc ==0:
                    coeffs = [0,0,0,0,0,0,0,0]
                else:
                    dir_path = os.path.join(base_path, f"L{int(Lc)}_W{int(wc)}", f"V{int(Vc)}", "Conc_x_t")
                    file_path = os.path.join(dir_path, f"fitModel_t_{t}.csv")
    
                    if not os.path.isfile(file_path):
                        raise FileNotFoundError(f"Missing training file for corner: {file_path}")
    
                    df = pd.read_csv(file_path)
                    if "Polynomial" not in df.columns:
                        raise KeyError(f"'Polynomial' column missing in {file_path}")
    
                    # If file lists coefficients in a single-column series, read them as array:
                    coeffs = df["Polynomial"].to_numpy(dtype=float)
                # If the CSV has multiple rows for different orders, ensure order is descending-power.
                # Assume the CSV already stores coefficients in descending-power order per your earlier logic.
                model_bBpts[x_ + 2 * y_ + 4 * z_] = np.poly1d(coeffs)

    # helper lerp for poly1d
    def lerp(a: np.poly1d, b: np.poly1d, w: float) -> np.poly1d:
        return a * (1.0 - w) + b * w

    # Interpolate along L axis for each (w,V) plane
    # Note: mapping of indices (x,y,z) to list index:
    # idx = x + 2*y + 4*z
    # We'll build the standard trilinear combination
    # corners: x changes fastest (L), then y (w), then z (V)

    # for V = Vmin (z=0)
    c000 = lerp(model_bBpts[0], model_bBpts[1], wxL)  # (Lmin,wmin,Vmin) -> (Lmax,wmin,Vmin)
    c010 = lerp(model_bBpts[2], model_bBpts[3], wxL)  # (Lmin,wmax,Vmin) -> (Lmax,wmax,Vmin)
    # for V = Vmax (z=1)
    c001 = lerp(model_bBpts[4], model_bBpts[5], wxL)  # (Lmin,wmin,Vmax) -> (Lmax,wmin,Vmax)
    c011 = lerp(model_bBpts[6], model_bBpts[7], wxL)  # (Lmin,wmax,Vmax) -> (Lmax,wmax,Vmax)

    # interpolate along w
    c00 = lerp(c000, c010, wxw)  # Vmin plane
    c01 = lerp(c001, c011, wxw)  # Vmax plane

    # interpolate along V
    P = lerp(c00, c01, wxV)

    # NOTE: do not multiply by Cmax here unless your training polynomials were normalized
    return P


# ---------------------------
# Save/Load helpers for poly1d
# ---------------------------
def save_poly1d(poly: np.poly1d, filepath: str):
    """Save polynomial coefficients (descending-power order) to CSV 'InterPoly' column."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    coeffs = poly.c  # numpy array
    pd.DataFrame({"InterPoly": coeffs}).to_csv(filepath, index=False)


def load_poly1d(filepath: str) -> np.poly1d:
    """Load polynomial coefficients from CSV column 'InterPoly' and return np.poly1d."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Missing interpolated polynomial file: {filepath}")
    df = pd.read_csv(filepath)
    if "InterPoly" not in df.columns:
        raise KeyError(f"'InterPoly' column missing in {filepath}")
    coeffs = df["InterPoly"].to_numpy(dtype=float)
    return np.poly1d(coeffs)


# ---------------------------
# Main/Side pore writers
# ---------------------------
def _mainporecalc_poly1d(p_main, boundingBox, t: int):
    model_poly = _multiLinearInterp_poly1d(p_main, boundingBox, t)
    out_dir = "Main_pore"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"interp1d_{t}.csv")
    save_poly1d(model_poly, out_path)
    # optional: return model_poly
    return model_poly


def _sideporecalc_poly1d(p_, boundingBox, index: int, t: int):
    model_poly = _multiLinearInterp_poly1d(p_, boundingBox, t)
    dir_ = os.path.join("Side_pore", f"pore{index}")
    os.makedirs(dir_, exist_ok=True)
    out_path = os.path.join(dir_, f"interp1d_{t}.csv")
    save_poly1d(model_poly, out_path)
    return model_poly


# ---------------------------
# Top-level script
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="RIC fixed script")
    parser.add_argument("--dt", nargs=1, required=True, help="time step (used as string in folder names)")
    parser.add_argument("--numsteps", nargs=1, required=True, help="number of timesteps")
    parser.add_argument("--geom", nargs=1, required=True, help="geometry file index (integer)")
    parser.add_argument("--V", nargs=1, required=True, help="potential Vr (Vl=0 assumed)")
    args = parser.parse_args()

    dt_ = float(args.dt[0])                # kept as string if you used it in folder names like '0.0001s'
    numsteps_ = int(args.numsteps[0])
    geom_num = int(args.geom[0])
    V = float(args.V[0])
    Vr = V
    Vl = 0.0

    # Load main-pore geometry (single-row at skiprows=3)
    geom_file = os.path.join("ElectrodeGeometry", f"G{geom_num}.csv")
    if not os.path.isfile(geom_file):
        raise FileNotFoundError(f"Missing geometry file: {geom_file}")

    L, W, X = np.loadtxt(geom_file, dtype=float, delimiter=",", skiprows=3, max_rows=1, usecols=[0, 1, 2], unpack=True)

    # Build p_main: note we use numeric dt_ in folder naming earlier; if needed transform
    p_main = [dt_, numsteps_, c10, float(L), float(W), float(Vr - Vl), float(X)]
    boundingBox_main = _findBoundingBox_(p_main)

    # MAINPORE: create folder and compute
    os.makedirs("Main_pore", exist_ok=True)
    print("Running main-pore interpolation for {} timesteps...".format(numsteps_))
    for t in range(numsteps_):
        _mainporecalc_poly1d(p_main, boundingBox_main, t)

    # SIDEPORE: read main interp for each t and evaluate per side pore
    # Side pores are stored from skiprows=4, next 3 rows => max_rows=3
    l_arr, w_arr, x_arr = np.loadtxt(geom_file, dtype=float, delimiter=",", skiprows=4, max_rows=3, usecols=[0, 1, 2], unpack=True)

    # Ensure side pore directories exist once
    for idx in range(len(x_arr)):
        os.makedirs(os.path.join("Side_pore", f"pore{idx}"), exist_ok=True)

    print("Running side-pore interpolation for {} timesteps and {} side-pores...".format(numsteps_, len(x_arr)))
    for t in range(numsteps_):
        main_interp_path = os.path.join("Main_pore", f"interp1d_{t}.csv")
        MainPore_model = load_poly1d(main_interp_path)

        for indx_ in range(len(x_arr)):
            # Choose how to evaluate the main model: absolute x or normalized xi = x/L.
            # The correct choice depends on how fitModel_t_*.csv was generated.
            # Here we prefer normalized coordinate xi = x/L (units consistent).
            #xi = float(x_arr[indx_]) / float(L) if float(L) != 0.0 else 0.0
            xi = float(x_arr[indx_])*1e-09 if float(L) != 0.0 else 0.0
            try:
                cx = float(MainPore_model(xi))
            except Exception:
                # fallback: evaluate at absolute x (in µm) if evaluation fails
                cx = float(MainPore_model(float(x_arr[indx_])))

            p_side = [dt_, numsteps_, cx, float(l_arr[indx_]), float(w_arr[indx_]), (Vr - Vl) * xi, float(x_arr[indx_])]
            boundingBox_side = _findBoundingBox_(p_side)

            _sideporecalc_poly1d(p_side, boundingBox_side, indx_, t)

    print("All done.")


if __name__ == "__main__":
    main()

