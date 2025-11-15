#!/bin/bash

#USER SPECIFIED PARAMETERS
numsteps_=$2
dt_=$1
Vl_low_=1		#lower range of LHS voltage (used in for loop below)
Vl_high_=2		#upper range of LHS voltage (used in for loop below)

for wid in 5 10;
do
	for len in 10 20 40 50 60 80 100;
	do
		for Vl in $Vl_low_ $Vl_high_;
		do
			python3  FitPolynomial_x.py --dt $dt_ --numsteps $numsteps_ --L $len --W $wid --V $Vl
		done
	done
done

