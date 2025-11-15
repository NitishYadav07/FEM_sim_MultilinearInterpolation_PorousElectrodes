#!/bin/bash

#USER SPECIFIED PARAMETERS
numsteps_=$2
dt_=$1
Vl=$4
geom=$3
python3  Interpolate_FittedPoly.py --dt $dt_ --numsteps $numsteps_ --geom $geom --V $Vl

