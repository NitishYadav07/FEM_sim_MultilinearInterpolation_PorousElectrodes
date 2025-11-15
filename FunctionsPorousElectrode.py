#!/usr/bin/env python
#RIC: Read, Interpolate, Calculate
# This is the function file for RIC_Mainpore.py and RIC_SidePore.py
import numpy as np
import csv # for handling csv files, data appending
from scipy.interpolate import UnivariateSpline # for interpolating training data
from scipy.signal import savgol_filter
from ParametersPorousElectrode import *
import os  #to create directory
import pandas as pd

# Function to find potential at position x in a pore (Main or Side pore)
#depth = pore depth
def _V_(x, depth):
    return x*(Vr-Vl)/depth

# Find bounding box to carry out multilinear interpolation
def _findBoundingBox_(p):
    l_array = [0, 50, 100] # depth of pore, in micrometers
    w_array = [2500, 5000, 6250, 7500]  # width of pore, in nanometers
    c_min = 0
    L_min = 0
    w_min = 0
    V_min = 0
    c_max=[]
    V_max=[]
    L_max=[]
    w_max=[]
    #finding LOWER bounds:
    for a1 in range(0,int(p[2])+1, 1):  #c0 scan (between 0 and 1)
        for a2 in range(0, int(p[3]), 1):  #L scan  (0, 50 or 100 micrometer)
            for a3 in range(2500, int(p[4]), 1): # w scan (2500, 3750, 5000, 6250, 7500 nm)
                for a4 in range(0, int(p[5])+1, 1):  #V scan (0, 1, 2)
                    c_min = a1/10
                    if a2 in l_array:
                        L_min = a2
                    V_min = a4
                    if a3 in w_array:
                        w_min = a3
    #finding UPPER bounds:
    for a1 in range(int(p[2])+1, 11, 1):  #c0 scan (0 or  1)
        for a2 in range(int(p[3])+1, 150, 1):  #L scan  (0, 50 or 100 micrometer)
            for a3 in range(int(p[4])+1, 8000, 1): # w scan (2500, 3750, 5000, 6250, 7500 nm)
                for a4 in range(int(p[5])+1, 4, 1):  #V scan (0, 1, 2)
                    c_max.append(a1/10)
                    if a2 in l_array:
                        L_max.append(a2)
                    V_max.append(a4)
                    if a3 in w_array:
                        w_max.append(a3)
    return [c_min, min(c_max)], [L_min, min(L_max)], [w_min, min(w_max)], [V_min, min(V_max)]




# Carry out multilinear interpolation
def _multiLinearInterp_(p, bB, t):  #  bB = boundingBox
    # read the 8 (in 3-dimensional space: L, V, w) relevant data files containing training data C(x). c-axis interpolation (between c = 0 and c = 1) is ignored right now. The final 3D interpolation data will be interpolated as a separate calculation along the 4th, c-axis.
    Cmin = bB[0][0]
    Cmax = bB[0][1]
    delC = Cmax - Cmin
    Lmin = bB[1][0]
    Lmax = bB[1][1]
    delL = Lmax - Lmin
    Vmin = bB[2][0]
    Vmax = bB[2][1]
    delV = Vmax - Vmin
    wmin = bB[3][0]
    wmax = bB[3][1]
    delw = wmax - wmin
    
    # Reading the 8 files for EACH TIME STEP
    if wmin == 0 or Vmin == 0:
        P1 = 1e-14*np.ones(1000)
        P2 = 1e-14*np.ones(1000)
        P3 = 1e-14*np.ones(1000)
        P4 = 1e-14*np.ones(1000)
        P5 = 1e-14*np.ones(1000)
        P6 = 1e-14*np.ones(1000)
        P7 = 1e-14*np.ones(1000)
        P8 = 1e-14*np.ones(1000)
    else:
        P1 = np.loadtxt('FeNICsOutputFiles/'+str(dt)+'s'+'/'+str(numsteps_)+'steps'+'/'+"L"+str(Lmax)+"_W"+str(wmin)+'/V'+str(Vmin)+'/'+'Conc_x_t/'+str(t)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0])
        P2 = np.loadtxt('FeNICsOutputFiles/'+str(dt)+'s'+'/'+str(numsteps_)+'steps'+'/'+"L"+str(Lmax)+"_W"+str(wmax)+'/V'+str(Vmin)+'/'+'Conc_x_t/'+str(t)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0])
        P3 = np.loadtxt('FeNICsOutputFiles/'+str(dt)+'s'+'/'+str(numsteps_)+'steps'+'/'+"L"+str(Lmin)+"_W"+str(wmax)+'/V'+str(Vmin)+'/'+'Conc_x_t/'+str(t)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0])
        P4 = np.loadtxt('FeNICsOutputFiles/'+str(dt)+'s'+'/'+str(numsteps_)+'steps'+'/'+"L"+str(Lmin)+"_W"+str(wmin)+'/V'+str(Vmin)+'/'+'Conc_x_t/'+str(t)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0])
        P5 = np.loadtxt('FeNICsOutputFiles/'+str(dt)+'s'+'/'+str(numsteps_)+'steps'+'/'+"L"+str(Lmax)+"_W"+str(wmin)+'/V'+str(Vmax)+'/'+'Conc_x_t/'+str(t)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0])
        P6 = np.loadtxt('FeNICsOutputFiles/'+str(dt)+'s'+'/'+str(numsteps_)+'steps'+'/'+"L"+str(Lmax)+"_W"+str(wmax)+'/V'+str(Vmin)+'/'+'Conc_x_t/'+str(t)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0])
        P7 = np.loadtxt('FeNICsOutputFiles/'+str(dt)+'s'+'/'+str(numsteps_)+'steps'+'/'+"L"+str(Lmin)+"_W"+str(wmax)+'/V'+str(Vmax)+'/'+'Conc_x_t/'+str(t)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0])
        P8 = np.loadtxt('FeNICsOutputFiles/'+str(dt)+'s'+'/'+str(numsteps_)+'steps'+'/'+"L"+str(Lmin)+"_W"+str(wmin)+'/V'+str(Vmax)+'/'+'Conc_x_t/'+str(t)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0])
        # Enter zeroes (1e-14) in shorter arrays at the begining to make them same length as longest array (so that there is no data loss)
    max_size = max(len(P1), len(P2), len(P3), len(P4), len(P5), len(P6), len(P7), len(P8))
    P1 = np.append(1e-14*np.ones(abs(len(P1) - max_size)), P1)
    P2 = np.append(1e-14*np.ones(abs(len(P2) - max_size)), P2)
    P3 = np.append(1e-14*np.ones(abs(len(P3) - max_size)), P3)
    P4 = np.append(1e-14*np.ones(abs(len(P4) - max_size)), P4)
    P5 = np.append(1e-14*np.ones(abs(len(P5) - max_size)), P5)
    P6 = np.append(1e-14*np.ones(abs(len(P6) - max_size)), P6)
    P7 = np.append(1e-14*np.ones(abs(len(P7) - max_size)), P7)
    P8 = np.append(1e-14*np.ones(abs(len(P8) - max_size)), P8)
    
    A = P4*(Lmax-p[1])/delL + P1*(p[1]-Lmin)/delL
    #A = np.sum( np.multiply(P4, (Lmax-p[1])/delL), np.multiply(P1, (p[1]-Lmin)/delL) )
    del P1; del P4
    B = P3*(Lmax-p[1])/delL + P2*(p[1]-Lmin)/delL
    del P2; del P3
    C = P8*(Lmax-p[1])/delL + P5*(p[1]-Lmin)/delL
    del P5; del P8
    D = P7*(Lmax-p[1])/delL + P6*(p[1]-Lmin)/delL
    del P6; del P7
    
    E = B*(wmax-p[3])/delw + A*(p[3]-wmin)/delw
    del A; del B
    F = D*(wmax-p[3])/delw + C*(p[3]-wmin)/delw
    del C; del D
    
    P = F*(Vmax-p[2])/delV + E*(p[2]-Vmin)/delV
    del E; del F
    t = t + 1  # not adding dt as files are not named with 'dt', but with index 'n'
    
    print('timestep = ', t)
    return np.multiply(P, p[0]/10)

'''
https://stackoverflow.com/questions/29156532/python-baseline-correction-library
I found an answer to my question, just sharing for everyone who stumbles upon this.
There is an algorithm called "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005. The paper is free and you can find it on google.

Works perfectly for me. just quoting from that paper what those parameters are: <<There are two parameters: p for asymmetry and λ for smoothness. Both have to be tuned to the data at hand. We found that generally 0.001 ≤ p ≤ 0.1 is a good choice (for a signal with positive peaks) and 10^2 ≤ λ ≤ 10^9 , but exceptions may occur. In any case one should vary λ on a grid that is approximately linear for log λ>>
'''
from scipy import sparse
from scipy.sparse.linalg import spsolve
def _baseline_als_(y, lam, p, niter=10):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z



def _mainporecalc_(p_main, boundingBox, t):
    P = _multiLinearInterp_(p_main, boundingBox, t)
    x = np.linspace(0, 500, len(P))
    lam = 1e4
    p = 0.001
    z_ = _baseline_als_(P, lam, p, niter=10)
    print(z_)
    f = open('Main_pore/'+str(t)+".csv", "w")
    f.write("{}\n".format("conc"))
    for x in zip(z_):
	    f.write("{}\n".format(x[0]))
    f.close()
    #return z_

def _sideporecalc_(p_side, boundingBox, t,  geom_, index, dir_name):
    P = _multiLinearInterp_(p_side, boundingBox, t)
    x = np.linspace(0, 500, len(P))
    lam = 1e4
    p = 0.001
    z_ = _baseline_als_(P, lam, p, niter=10)
    
    f = open(dir_name+'/'+str(t)+'.csv', 'w')
    f.write("{}\n".format("conc"))
    for x in zip(z_):
        f.write("{}\n".format(x[0]))
    f.close()
    

# function to determine conc, C(x), at the inlet of the side pore, L = depth of main pore
def _find_C_x_(c_arr, x, L):
    return c_arr[int(1-(x/L)*len(c_arr))]


# function to determine conc, PROFILE C(x), at the inlet of the side pore, L = depth of main pore
def _find_C_x_poly1d(model, x):
    return model(x)



#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################






def _multiLinearInterp_poly1d(p, bB, t):  #  bB = boundingBox
    # read the 8 (in 3-dimensional space: L, V, w) relevant data files containing training data C(x). c-axis interpolation (between c = 0 and c = 1) is ignored right now. The final 3D interpolation data will be interpolated as a separate calculation along the 4th, c-axis.
    
    Cmin   =    bB[0][0];    Cmax   =   bB[0][1];   delC    =   Cmax - Cmin
    Lmin    =    bB[1][0];    Lmax   =   bB[1][1];    delL    =    Lmax - Lmin
    wmin   =    bB[2][0];    wmax   =   bB[2][1];   delw    =   wmax - wmin
    Vmin   =    bB[3][0];    Vmax   =   bB[3][1];    delV   =    Vmax - Vmin
    
    model_bBpts=[0,0,0,0,0,0,0,0]           #numpy      " list "     of size 8 (8 corners of bBox)
    
    # Reading the 8 models for EACH TIME STEP
    directory_name = "Conc_x_t"
    for ind_ in range(8):
        for x_ in range(2):
            for y_ in range(2):
                for z_ in range(2):
                    path = "FeNICsOutputFiles"+"/"+str(p[0])+"s/"+str(p[1])+"steps"+"/"+"L"+str(int(bB[1][x_]))+"_W"+str(int(bB[2][y_]))+"/"+"V"+str(int(bB[3][z_]))+"/"
                    full_path = path+directory_name
                    df  =   pd.read_csv(full_path + '/' +'fitModel_t_'+str(t)+'.csv')
                    model_bBpts[ind_] = np.poly1d(df['Polynomial'])
    
    A = model_bBpts[    0   ]*(Lmax-p[3])/delL       +   model_bBpts[   4    ]*(p[3]-Lmin)/delL
    B = model_bBpts[    2   ]*(Lmax-p[3])/delL      +    model_bBpts[   6    ]*(p[3]-Lmin)/delL
    C = model_bBpts[   1    ]*(Lmax-p[3])/delL      +    model_bBpts[   5    ]*(p[3]-Lmin)/delL
    D = model_bBpts[    3   ]*(Lmax-p[3])/delL      +   model_bBpts[    7   ]*(p[3]-Lmin)/delL
    E = B*(wmax-p[4])/delw + A*(p[4]-wmin)/delw
    F = D*(wmax-p[4])/delw + C*(p[4]-wmin)/delw
    P = F*(Vmax-p[5])/delV + E*(p[5]-Vmin)/delV
    
    return P*Cmax



def _mainporecalc_poly1d(p_main, boundingBox, t):
    model_ = _multiLinearInterp_poly1d(p_main, boundingBox, t)
    f = open('Main_pore/'+'interp1d_'+str(t)+'.csv', 'w')
    f.write("{}\n".format("InterPoly"))
    for x in zip(model_):
	    f.write(    "{}\n".format(  x[0]  )         )
    f.close()







def _sideporecalc_poly1d(p_, boundingBox, index, t):
    model_ = _multiLinearInterp_poly1d(p_, boundingBox, t)
    
    dir_ = 'Side_pore/'+'pore'+str(index)
    #print(model_)
    f = open(dir_+'/'+'interp1d_'+str(t)+".csv", "w")
    f.write("{}\n".format("InterPoly"))
    for x in zip(model_):
	    f.write(    "{}\n".format(  x[0]  )         )
    f.close()







'''
def _outConc_polymerReg( dt, numsteps_, l, w, x, Vr, Vl  c0 ):
    for t in range(numsteps_):
        c_x = np.loadtxt('FeNICsOutputFiles/'+str(dt)+'s'+'/'+str(numsteps_)+'steps'+'/'+"L"+str(Lmax)+"_W"+str(wmin)+'/V'+str(Vmin)+'/'+'Conc_x_t/'+str(t)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0])
        x = np.arange(0, numsteps_)
'''
