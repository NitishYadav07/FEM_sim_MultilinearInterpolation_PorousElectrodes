#!/bin/bash

#folder_=/media/ubuntu/cff57f71-6a52-4d43-9289-45bc9c34e310/home/sigma/Documents/COMP_PHYS/FEM_ALL/FeNICs/NernstPlanck_NY_2/NEW_Code_08Feb25

#USER SPECIFIED PARAMETERS

numsteps_=$2
dt_=$1

Vl_low_=$3		#lower range of LHS voltage (used in for loop below)
Vl_high_=$4		#upper range of LHS voltage (used in for loop below)

for wid in 5 10;
do
	for len in 10 20 40 50 60 80 100;
	do
		for Vl in $Vl_low_ $Vl_high_;
		do
			opDIR__="FeNICsOutputFiles/${dt_}s/${numsteps_}steps"
			#opDIR_="${folder}/${opDIR__}"
			opDIR_="${opDIR__}"
			if [ ! -d "$opDIR_" ]; then
				echo "$opDIR_ does not exist. Creating $opDIR_ now"
				mkdir $opDIR_
			fi 
			dir_="${opDIR_}/L${len}_W${wid}/V${Vl}"
			echo $dir_
				if [ -z "$( ls -A  $dir_)" ]; then
					echo "Creating directory"
					python3 PERFECTLYWORKINGwBCs_Mixed_Poisson_2_NPN_v3.py --length $len --width $wid --Vl $Vl --dt $dt_ --numsteps $numsteps_ --outputFolder $dir_
				else
					echo "Not Empty - Calculations done for this configuration"
				fi
		done
	done
done



#if [ -f $FILE ]; then
#	echo "File $FILE already exists."
#else
#	echo "File $FILE does not exist."
#fi
