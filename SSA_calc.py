import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse

# -------------------------
# Argument Parsing
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--geom', nargs=1, required=True, help='Electrode geometry file name')
args = parser.parse_args()

geom_num = int(args.geom[0])

# -------------------------
# Load electrode geometry
# -------------------------
L, W, X = np.loadtxt(
    'ElectrodeGeometry/G' + str(geom_num) + '.csv',
    dtype='double',
    delimiter=',',
    skiprows=3,
    max_rows=1,
    unpack=True,
    usecols=[0, 1, 2],
)
l, w, x = np.loadtxt(
    'ElectrodeGeometry/G' + str(geom_num) + '.csv',
    dtype='double',
    delimiter=',',
    skiprows=4,
    unpack=True,
    usecols=[0, 1, 2],
)

thickness = 1e-6  # m (electrode thickness)

# -------------------------
# Calculate Total Internal Surface Area
# -------------------------
L_m = L * 1e-6
W_m = W * 1e-9
A_main = 2 * (L_m * thickness) + 2 * (W_m * thickness)

A_side_total = 0
for i in range(len(l)):
    l_m = l[i] * 1e-6
    w_m = w[i] * 1e-9
    A_side_total += 2 * (l_m * thickness) + 2 * (w_m * thickness)

A_total_surface = A_main + A_side_total

print(f"[INFO] Total internal surface area = {A_total_surface:.4e} m²")
