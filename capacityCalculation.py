import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse

# -------------------------
# Argument Parsing
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--geom', nargs=1, help='Electrode geometry file name')
args = parser.parse_args()
geom_file = str(args.geom[0])

# -------------------------
# Load electrode geometry
# -------------------------
L, W, X = np.loadtxt(
    'ElectrodeGeometry/G' + geom_file + '.csv',
    dtype='double',
    delimiter=',',
    skiprows=3,
    max_rows=1,
    unpack=True,
    usecols=[0, 1, 2],
)
l, w, x = np.loadtxt(
    'ElectrodeGeometry/G' + geom_file + '.csv',
    dtype='double',
    delimiter=',',
    skiprows=4,
    unpack=True,
    usecols=[0, 1, 2],
)

thickness = 1e-12  # m (electrode thickness)

# -------------------------
# Calculate Total Internal Surface Area
# -------------------------
# Main pore surface area (4 walls) [L in um, W in nm → convert both to meters]
L_m = L * 1e-9
W_m = W * 1e-9
A_main = 2 * (L_m * thickness) + 2 * (W_m * thickness)  # Perimeter * thickness
A_main *= 1  # for one main pore

# Side pore surface area: sum over all side pores
A_side_total = 0
for i in range(len(l)):
    l_m = l[i] * 1e-9
    w_m = w[i] * 1e-9
    A_side = 2 * (l_m * thickness) + 2 * (w_m * thickness)
    A_side_total += A_side

A_total_surface = A_main + A_side_total
print(A_total_surface)

print(f"[INFO] Total internal surface area = {A_total_surface:.4e} m²")

# -------------------------
# Capacity calculation
# -------------------------
Cap_total = []
Time = []

# Loop over time steps (detected from Main_pore interp1d files)
interp_dir = "Main_pore"
all_files = sorted(
    [f for f in os.listdir(interp_dir) if f.startswith("interp1d_") and f.endswith(".csv")]
)
numsteps = len(all_files)

for t in range(0, numsteps, 10):  # step every 10 for speed
    # ---- MAIN PORE ----
    main_file = os.path.join("Main_pore", f"interp1d_{t}.csv")
    df_main = pd.read_csv(main_file)
    coeffs_main = df_main["InterPoly"].values
    poly_main = np.poly1d(coeffs_main)

    x_points = np.linspace(0, L, 200)  # micrometers
    c_profile_main = poly_main(x_points * 1e-9)  # convert x to meters
    Cap_main = np.trapz(c_profile_main, x_points) * W_m * thickness

    # ---- SIDE PORES ----
    Cap_side = 0
    for sn in range(len(l)):
        side_file = os.path.join(f"Side_pore/pore{sn}", f"interp1d_{t}.csv")
        if not os.path.exists(side_file):
            continue
        df_side = pd.read_csv(side_file)
        coeffs_side = df_side["InterPoly"].values
        poly_side = np.poly1d(coeffs_side)

        x_points_side = np.linspace(0, l[sn], 200)  # micrometers
        c_profile_side = poly_side(x_points_side * 1e-9)
        Cap_side += np.trapz(c_profile_side, x_points_side) * (w[sn] * 1e-9) * thickness

    Cap_total_raw = Cap_main + Cap_side
    Cap_total_specific = Cap_total_raw / A_total_surface  # normalize by surface area
    Cap_total.append(Cap_total_specific)
    Time.append(t)

# -------------------------
# Plot and Save
# -------------------------
fig, ax = plt.subplots()
ax.plot(Time, Cap_total, label="Specific Capacity (per m²)", linestyle="-", lw=1.5, color="seagreen")
ax.set_xlabel("Time step")
ax.set_ylabel("Specific Capacity (arb. units / m²)")
ax.legend()

out_file = f"capacityResults/capacity_{geom_file}.csv"
with open(out_file, "w") as f:
    f.write("{}, {}\n".format("Time", "SpecificCapacity"))
    for xx, yy in zip(Time, Cap_total):
        f.write("{}, {}\n".format(xx, yy))

print(f"[INFO] Capacity results saved to {out_file}")

#plt.show()

