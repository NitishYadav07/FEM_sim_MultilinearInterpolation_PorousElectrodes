#!/bin/bash

# ===============================
# Capacitance Calculation Driver
# ===============================

# Directory where capacitance results will be stored
capacitanceDIR_="capacitanceResults"
if [ ! -d "$capacitanceDIR_" ]; then
    echo "[INFO] $capacitanceDIR_ does not exist. Creating it now..."
    mkdir -p "$capacitanceDIR_"
fi 

# Number of geometry files (adjust as needed)
N_=5
# Loop over geometries and run capacitance calculation
for (( VOLT=1; VOLT<=2; VOLT++))
do
	for (( ind=1; ind<=N_; ind++ ))
	do
    	geom_name="${ind}"
    	echo "[INFO] Processing geometry: $geom_name"
	    
    	# Run the Python capacitance calculation script
    	python3 capacitanceCalculation.py --geom "G$geom_name" --voltage $VOLT
	done
done
echo "[INFO] All capacitance calculations completed. Results are in $capacitanceDIR_/"

