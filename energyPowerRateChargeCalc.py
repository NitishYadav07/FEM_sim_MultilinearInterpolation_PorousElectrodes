import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# -------------------------
# Argument Parsing
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--geom', nargs=1, required=True, help='Electrode geometry file name')
parser.add_argument('--voltage', nargs=1, required=True, type=float, help='Applied voltage (V)')
args = parser.parse_args()

geom_file = int(args.geom[0])
voltage = float(args.voltage[0])

# -------------------------
# Load specific capacitance results
# -------------------------
input_file = f"specificCapacitanceResults/specificCapacitance_G{geom_file}_V{voltage}.csv"
if not os.path.exists(input_file):
    raise FileNotFoundError(f"[ERROR] Cannot find {input_file}. Run specificCapacitance.py first.")

# Read CSV without assuming headers
df = pd.read_csv(input_file, header=None, skiprows=1)

# Get columns by index
time = df.iloc[:, 0].values   # first column → Time
C_spec = df.iloc[:, 1].values # second column → Specific Capacitance (F/m²)

# -------------------------
# Calculate Energy, Power, Charging Rate
# -------------------------
E_spec = 0.5 * C_spec * (voltage ** 2)  # J/m²
Q_spec = C_spec * voltage               # C/m² (charge per area)

# Finite difference for power and charging rate
P_spec = np.zeros_like(E_spec)
ChargingRate = np.zeros_like(Q_spec)

P_spec[1:] = np.diff(E_spec) / np.diff(time)
ChargingRate[1:] = np.diff(Q_spec) / np.diff(time)

# -------------------------
# Compute integrated total charge
# -------------------------
# Use trapezoidal rule to integrate charging rate over time
Q_total_integrated = np.trapz(ChargingRate, time)  # C/m²

# -------------------------
# Save results
# -------------------------
os.makedirs("energyPowerResults", exist_ok=True)
output_file = f"energyPowerResults/EnergyPower_{geom_file}_V{voltage}.csv"
output_file2 = f"energyPowerResults/IntTotalCharge_{geom_file}_V{voltage}.csv"

with open(output_file, "w") as f:
    f.write("Time, E_spec (J/m²), P_spec (W/m²), Q_spec (C/m²), ChargingRate (A/m²)\n")
    for t, e, p, q, cr in zip(time, E_spec, P_spec, Q_spec, ChargingRate):
        f.write(f"{t}, {e}, {p}, {q}, {cr}\n")

with open(output_file2, "w") as f:
    f.write(f"# Integrated Total Charge (C/m²): {Q_total_integrated}\n")
    
print(f"[INFO] Energy, Power & Charging Rate results saved to {output_file}")
print(f"[INFO] Integrated Total Charge = {Q_total_integrated:.6e} C/m²")

# -------------------------
# Plotting
# -------------------------
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.set_xlabel("Time step")
ax1.set_ylabel("E_spec (J/m²)", color="tab:blue")
ax1.plot(time, E_spec, color="tab:blue", lw=2, label="Energy Density")
ax1.tick_params(axis='y', labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.set_ylabel("P_spec (W/m²)", color="tab:red")
ax2.plot(time, P_spec, color="tab:red", lw=2, linestyle="--", label="Power Density")
ax2.tick_params(axis='y', labelcolor="tab:red")

plt.title(f"Energy & Power Density for {geom_file} at {voltage}V")
plt.tight_layout()
#plt.show()

# Separate plot for Charging Rate
plt.figure(figsize=(8, 4))
plt.plot(time, ChargingRate, color="purple", lw=2)
plt.xlabel("Time step")
plt.ylabel("Charging Rate (A/m²)")
plt.title(f"Charging Rate vs Time for {geom_file} at {voltage}V")
plt.grid(True)
#plt.show()

