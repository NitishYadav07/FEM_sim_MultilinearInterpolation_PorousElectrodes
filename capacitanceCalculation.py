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
parser.add_argument('--voltage', nargs=1, required=True, type=float, help='Applied voltage (V)')
args = parser.parse_args()

geom_num = int(args.geom[0])
voltage = float(args.voltage[0])

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

thickness = 1e-12  # m (electrode thickness)

# -------------------------
# Calculate Total Internal Surface Area
# -------------------------
L_m = L * 1e-9
W_m = W * 1e-9
A_main = 2 * (L_m * thickness) + 2 * (W_m * thickness)

A_side_total = 0
for i in range(len(l)):
    l_m = l[i] * 1e-9
    w_m = w[i] * 1e-9
    A_side_total += 2 * (l_m * thickness) + 2 * (w_m * thickness)

A_total_surface = A_main + A_side_total

print(f"[INFO] Total internal surface area = {A_total_surface:.4e} m²")
print(f"[INFO] Applied Voltage = {voltage:.3f} V")

# -------------------------
# Capacity calculation
# -------------------------
Q_total = []
Time = []

interp_dir = "Main_pore"
all_files = sorted(
    [f for f in os.listdir(interp_dir) if f.startswith("interp1d_") and f.endswith(".csv")]
)
numsteps = len(all_files)

for t in range(0, numsteps, 10):
    # ---- MAIN PORE ----
    main_file = os.path.join("Main_pore", f"interp1d_{t}.csv")
    df_main = pd.read_csv(main_file)
    coeffs_main = df_main["InterPoly"].values
    poly_main = np.poly1d(coeffs_main)

    x_points = np.linspace(0, L, 200)
    c_profile_main = poly_main(x_points * 1e-9)
    Q_main = np.trapz(c_profile_main, x_points) * W_m * thickness

    # ---- SIDE PORES ----
    Q_side = 0
    for sn in range(len(l)):
        side_file = os.path.join(f"Side_pore/pore{sn}", f"interp1d_{t}.csv")
        if not os.path.exists(side_file):
            continue
        df_side = pd.read_csv(side_file)
        coeffs_side = df_side["InterPoly"].values
        poly_side = np.poly1d(coeffs_side)

        x_points_side = np.linspace(0, l[sn], 200)
        c_profile_side = poly_side(x_points_side * 1e-9)
        Q_side += np.trapz(c_profile_side, x_points_side) * (w[sn] * 1e-9) * thickness

    Q_total_raw = Q_main + Q_side
    C_specific = Q_total_raw / (voltage * A_total_surface)  # F/m²
    Q_total.append(C_specific)
    Time.append(t)

# -------------------------
# Plot and Save
# -------------------------
fig, ax = plt.subplots()
ax.plot(Time, Q_total, label="Specific Capacitance (F/m²)", linestyle="-", lw=1.5, color="blue")
ax.set_xlabel("Time step")
ax.set_ylabel("C_spec (F/m²)")
ax.legend()

os.makedirs("specificCapacitanceResults", exist_ok=True)
out_file = f"specificCapacitanceResults/specificCapacitance_G{geom_num}_V{voltage}.csv"
with open(out_file, "w") as f:
    f.write("{}, {}\n".format("Time", "C_spec (F/m²)"))
    for xx, yy in zip(Time, Q_total):
        f.write("{}, {}\n".format(xx, yy))

print(f"[INFO] Specific capacitance results saved to {out_file}")

#plt.show()

