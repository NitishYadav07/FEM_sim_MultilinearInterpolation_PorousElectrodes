from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import os  #to create directory
import pandas as pd
from os import path		# for checking if data file already exists or not
from scipy import optimize
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
from ParametersPorousElectrode import *
import FunctionsPorousElectrode as FPE

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dt', nargs = 1) #  help='length of pore in micrometers'
parser.add_argument('--numsteps', nargs = 1) #  help='length of pore in micrometers'
parser.add_argument('--geom', nargs = 1) #  help='length of pore in micrometers'
#parser.add_argument('--V', nargs = 1) #  help='potential at which to calculate CAPACITANCE'
args = parser.parse_args()
dt_ = float(args.dt[0])
numsteps_ = int(args.numsteps[0])
geom_num = int(args.geom[0])
#V = int(args.V[0])
#Vr=V
#Vl=0
color_=['blue','red','green']
#L, W, X for Main pore; l[i], w[i], x[i] for Side pores
L, W, X = np.loadtxt('ElectrodeGeometry/'+'G'+str(geom_num)+'.csv', dtype='double', delimiter=',', skiprows=3, max_rows = 1, unpack=True, usecols=[0, 1, 2])
l, w, x = np.loadtxt('ElectrodeGeometry/'+'G'+str(geom_num)+'.csv', dtype='double', delimiter=',', skiprows=4, max_rows = 3, unpack=True, usecols=[0, 1, 2])
c0 = 1.0
out_DIR_= 'conc_animation/'+str(dt_)+'s/'+str(numsteps_)+'steps/'+'G'+str(geom_num)  #for use when only running one geometry calculation at a time, making figures for articles, presentations etc purpose

#out_DIR_= 'conc_animation/'     # for use when running entire program in one go

#os.makedirs(os.path.join("Side_pore", f"pore{idx}"), exist_ok=True)
os.makedirs(os.path.join(".", out_DIR_), exist_ok=True)
fig = plt.figure()
for t in range(0, numsteps_):
    ax1 = fig.add_subplot()
    ax2 =   ax1.twinx()
    MainPoreDIR_ = 'Main_pore'
    df_mp  =   pd.read_csv(MainPoreDIR_+'/'+'interp1d_'+str(t)+'.csv')  # gather the conc profile info, at current time step, for the current geometry's main pore
    coeff_mp = df_mp["InterPoly"]
    MainPore_model=np.poly1d(coeff_mp)
    X_=   np.linspace(0, L, 50)*1e-09      # multiplcation by 1e012 and then division is necessary to get the correct my_x array in nanometers
    #if t==400:
    ax1.plot(X_, MainPore_model(X_), '-', linewidth='3', color='black', label='Mainpore')
    for indx_ in range(len(x)):
        SidePoreDIR_ = 'Side_pore/'+'pore'+str(indx_)
        df_sp   =   pd.read_csv(SidePoreDIR_+'/'+'interp1d_'+str(t)+'.csv')  # gather the conc profile info, at current time step, for the current geometry's main pore
        coeff_sp = df_sp["InterPoly"]
        SidePore_model=np.poly1d(coeff_sp)
        x_=   np.linspace(0, l[indx_], 50)*1e-09      # multiplcation by 1e-09is necessary to get the correct x_ array in nanometers
        #if t==400:
        ax2.plot(   x_, SidePore_model(x_), '--', color=color_[indx_]   , label=str(indx_))
    plt.legend()
    plt.savefig(out_DIR_+'/'+'t_'+str(t)+'.png')
    plt.clf()       # ! makes animation figures
    #plt.close()
    #plt.show()
