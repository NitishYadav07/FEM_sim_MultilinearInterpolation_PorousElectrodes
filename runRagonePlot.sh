#!/bin/bash

# ======================================
# Bash script to generate Ragone Plot
# ======================================

N_GEOM=5
#VOLTAGES=(1.0 2.0)
VOLTAGES=(1.5)

# Build geometry list dynamically
GEOMS=()
for (( i=1; i<=N_GEOM; i++ ))
do
    GEOMS+=("${i}")
done

# Call Python script with all geometries and voltages
python3 ragonePlot.py --geometries "${GEOMS[@]}" --voltages "${VOLTAGES[@]}"
