#!/bin/bash

# ===============================
# Capacity Calculation Driver
# ===============================

NGeom=5
# Directory where capacity results will be stored
capacityDIR_="capacityResults"
if [ ! -d "$capacityDIR_" ]; then
    echo "[INFO] $capacityDIR_ does not exist. Creating it now..."
    mkdir -p "$capacityDIR_"
fi 

# Number of geometry files (adjust as needed)
N_=$NGeom

# Loop over geometries and run capacity calculation
for (( ind=1; ind<=N_; ind++ ))
do
    geom_name="${ind}"
    echo "[INFO] Processing geometry: $geom_name"
    python3 Interpolate_FittedPoly.py --dt $1 --numsteps $2 --geom $geom_name --V $3
    # Run the Python conc plotting script
    python3 plot_Conc4Geom.py --dt $1 --numsteps $2 --geom $geom_name
    # Run the Python capacity calculation script
    python3 capacityCalculation.py --geom $geom_name
    # Run the Python capacitance calculation script
    python3 capacitanceCalculation.py --geom $geom_name --voltage $3
    # Run the Python script to calculate Energy, Power, Charging Rate
    python3 energyPowerRateChargeCalc.py --geom $geom_name --voltage $3
done

echo "[INFO] All capacity calculations completed. Results are in $capacityDIR_/"


N_GEOM=$NGeom
VOLTAGES=($3)

# Build geometry list dynamically
GEOMS=()
for (( i=1; i<=N_GEOM; i++ ))
do
    GEOMS+=("${i}")
done

# Run the Python script to plot ragone plot
# Call Python script with all geometries and voltages
python3 ragonePlot.py --geometries "${GEOMS[@]}" --voltages "${VOLTAGES[@]}"
