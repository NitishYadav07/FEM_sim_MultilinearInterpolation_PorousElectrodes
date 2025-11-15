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

def fitfunc (x, A0, A1, A2, A3):
    return A0 + A1*x + A2*np.exp(A3*x)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dt', nargs = 1) #  help='length of pore in micrometers'
parser.add_argument('--numsteps', nargs = 1) #  help='length of pore in micrometers'
parser.add_argument('--L', nargs = 1) #  help='length of pore in micrometers'
parser.add_argument('--W', nargs = 1) #  help='length of pore in nanometers'
parser.add_argument('--V', nargs = 1) #  help='potential at which to calculate CAPACITANCE'
args = parser.parse_args()
dt_ = float(args.dt[0])
numsteps_ = int(args.numsteps[0])
L = int(args.L[0])
W = int(args.W[0])
V = int(args.V[0])

# Specify the directory name
path = "FeNICsOutputFiles"+"/"+str(dt_)+"s/"+str(numsteps_)+"steps"+"/"+"L"+str(L)+"_W"+str(W)+"/"+"V"+str(V)+"/"
directory_name = "Conc_x_t"

fig = plt.figure()
ax = fig.add_subplot()

for t in range(0, numsteps_):
    df      =   pd.read_csv(path+directory_name + '/' +str(t)+'.csv')
    df = df.sort_values(by="X", ascending=True)
    x       =   df["X"]
    conc =   df["Conc"]
    x_temp = max(x)*1e09
    my_x=   np.arange(0, x_temp)*1e-09      # multiplcation by 1e09 and then by 1e-09 is necessary to get the correct my_x array in NANOmeters
    mymodel = np.poly1d(np.polyfit(x, conc, 8))
    f = open(path+directory_name + '/' +'fitModel_t_'+str(t)+'.csv', 'w')
    f.write("{}\n".format("Polynomial"))
    for h_ in zip(mymodel):
        f.write("{}\n".format(h_[0]))
    f.close()
    #print(r2_score(conc, mymodel(x))) 
    #popt, pcov = optimize.curve_fit(fitfunc, x, conc)
    #if t == 100:
    #   plt.plot(x, conc, 'o', color='black')
    #    plt.plot(my_x, mymodel(my_x), '-', color='r')
#plt.show()

'''
    #Smoothed data (Zhat):
#Zhat = savgol_filter(Z_, 500, 5) # window size 51, polynomial order 3

col_len = 0
for ind in range(len(X_)):
    if arr[0][ind] < arr[0][ind+1]:
        col_len = col_len + 1
    else:
        break

Z=[]
X=[]
cnt = 0
for k in range(len(arr[0])):
    temp = 0
    for i in range(col_len):
        temp = temp + arr[2][i]
    Z.append(temp/(col_len+1))  #conc at time t, for all x
    X.append(arr[0][col_len*cnt])
    cnt = cnt+1
    
    
    
    Z_fit=[]
for j in range(len(X)):
    Z_fit.append(popt[0] + np.multiply(popt[1], X) + np.multiply(popt[2], np.exp(np.multiply(popt[3], X))))

plt.plot(X, Z, 'o')
plt.plot(X, Z_fit, '-', color="red")

plt.show()

'''
