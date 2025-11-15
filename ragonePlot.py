import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# ======================================
# Parse arguments
# ======================================
parser = argparse.ArgumentParser()
parser.add_argument('--geometries', nargs='+', required=True, help='List of geometry names (e.g., ElectrodeGeom1 ElectrodeGeom2)')
parser.add_argument('--voltages', nargs='+', required=True, type=float, help='List of voltages (e.g., 0.5 1.0 1.5)')
args = parser.parse_args()

geometries = args.geometries
voltages = args.voltages

# ======================================
# Collect Data
# ======================================
ragone_data = []

for geom in geometries:
    for V in voltages:
        file = f"energyPowerResults/EnergyPower_{geom}_V{V}.csv"
        if not os.path.exists(file):
            print(f"[WARNING] Skipping missing file {file}")
            continue
        
        df = pd.read_csv(file, header=0)  # Expect columns: Time, E_spec (J/m²), P_spec (W/m²)
        E_spec = df.iloc[:, 1].values
        P_spec = df.iloc[:, 2].values
        
        # Take last energy point (steady-state)
        E_final = E_spec[-1]
        
        # Compute mean power over charging duration
        P_mean = np.mean(P_spec[np.isfinite(P_spec)])  # avoid NaNs from first diff
        P_mean = P_spec[-1]
        
        ragone_data.append([geom, V, E_final, P_mean])

# Convert to DataFrame for plotting
ragone_df = pd.DataFrame(ragone_data, columns=["Geometry", "Voltage (V)", "Energy (J/m²)", "Power (W/m²)"])

# Save as CSV for reproducibility
os.makedirs("ragoneResults", exist_ok=True)
ragone_df.to_csv("ragoneResults/ragone_data.csv", index=False)
print(f"[INFO] Ragone data saved to ragoneResults/ragone_data.csv")

# ======================================
# Plot Ragone with Geometry Labels
# ======================================
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    ragone_df["Power (W/m²)"], 
    ragone_df["Energy (J/m²)"],
    c=np.arange(len(ragone_df)),  # color points uniquely
    cmap="tab20",
    s=70,
    edgecolor="black"
)

# Annotate each point with geometry name (and optionally voltage)
for i, row in ragone_df.iterrows():
    label = f"(G{row['Geometry']}, V={row['Voltage (V)']}V)"
    plt.annotate(
        label,
        (row["Power (W/m²)"], row["Energy (J/m²)"]),
        textcoords="offset points",
        xytext=(-20, -15),  # slight offset
        fontsize=12,
        alpha=0.8
    )

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Power Density (W/m²)", fontsize=12)
plt.ylabel("Energy Density (J/m²)", fontsize=12)
plt.title("Ragone Plot (Energy vs Power)")
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("ragoneResults/ragone_plot_annotated.png", dpi=300)
plt.show()

