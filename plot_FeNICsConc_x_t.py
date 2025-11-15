from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import os  #to create directory
import pandas as pd
from os import path		# for checking if data file already exists or not

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dt', nargs = 1) #  help='length of pore in micrometers'
parser.add_argument('--numsteps', nargs = 1) #  help='length of pore in micrometers'
parser.add_argument('--L', nargs = 1) #  help='length of pore in micrometers'
parser.add_argument('--W', nargs = 1) #  help='length of pore in micrometers'
parser.add_argument('--V', nargs = 1) #  help='potential at which to calculate CAPACITANCE'
args = parser.parse_args()
dt_ = float(args.dt[0])
numsteps_ = int(args.numsteps[0])
L = int(args.L[0])
W = int(args.W[0])
V = int(args.V[0])
#Vr=V
#Vl=0
color_=['blue','red','green']
c0 = 1.0
#os.makedirs(os.path.join("Side_pore", f"pore{idx}"), exist_ok=True)
out_DIR_="FeNICs_conc_plots/"+"L"+str(L)+"_W"+str(W)+"/V_"+str(V)
in_DIR_=   'FeNICsOutputFiles/'+str(dt_)+'s/'+str(numsteps_)+'steps/'+'L'+str(L)+'_W'+str(W)+'/V'+str(V)
os.makedirs(os.path.join(".", out_DIR_), exist_ok=True)
fig = plt.figure()
for t in range(0, numsteps_):
    ax1 = fig.add_subplot()
    ax2 =   ax1.twinx()
    df_  =   pd.read_csv(in_DIR_+'/Conc_x_t/'+str(t)+'.csv')  # gather the conc profile info, at current time step, for the current geometry's main pore
    X_ = df_["X"].to_numpy(dtype=float)
    conc_ = df_["Conc"].to_numpy(dtype=float)
    ax1.plot(X_, conc_, '-', linewidth='3', color='black', label='Mainpore')
    plt.legend()
    plt.savefig(out_DIR_+'/t_'+str(t)+'.png')
    plt.clf()       # ! makes animation figures
    #plt.close()
    #plt.show()
