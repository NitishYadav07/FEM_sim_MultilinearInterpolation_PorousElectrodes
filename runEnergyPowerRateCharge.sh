#!/bin/bash

# ================================
# Bash Script to Compute Energy, Power, Charging Rate
# for Multiple Geometries and Voltages
# ================================

# --- User settings ---
N_GEOM=18                   # number of geometry files (ElectrodeGeom1 ... ElectrodeGeom18)

# --- Output directory ---
RESULT_DIR="energyPowerResults"
if [ ! -d "$RESULT_DIR" ]; then
    echo "[INFO] Creating output directory: $RESULT_DIR"
    mkdir $RESULT_DIR
fi

# --- Loop over geometries and voltages ---
for (( g=1; g<=N_GEOM; g++ ))
do
    geom_name="${GEOM_PREFIX}${g}"
    echo "[INFO] Processing geometry: $geom_name"
    
    for (( VOLT=1; VOLT<=2; VOLT++ ))
    do
        echo "   --> Voltage: ${VOLT} V"
        python3 energyPowerRateChargeCalc.py --geom "$geom_name" --voltage "$VOLT"
    done
done

echo "[INFO] All energy, power, and charging rate calculations completed."

