#!/bin/bash

# ===============================
# Capacity Calculation Driver
# ===============================

# Directory where capacity results will be stored
capacityDIR_="capacityResults"
if [ ! -d "$capacityDIR_" ]; then
    echo "[INFO] $capacityDIR_ does not exist. Creating it now..."
    mkdir -p "$capacityDIR_"
fi 

# Number of geometry files (adjust as needed)
N_=5

# Loop over geometries and run capacity calculation
for (( ind=1; ind<=N_; ind++ ))
do
    geom_name="${ind}"
    echo "[INFO] Processing geometry: $geom_name"
    python3  Interpolate_FittedPoly.py --dt $dt_ --numsteps $numsteps_ --geom $geom --V $Vl
    # Run the Python capacity calculation script
    python3 capacityCalculation.py --geom "$geom_name"
done

echo "[INFO] All capacity calculations completed. Results are in $capacityDIR_/"

