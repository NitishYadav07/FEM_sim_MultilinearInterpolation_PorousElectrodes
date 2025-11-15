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
directory_name = "Conc_x_y_t"
# Create the directory Conc_x_y_t
try:
    os.mkdir(path+directory_name)
    print(f"Directory '{directory_name}' created successfully.")
except FileExistsError:
    print(f"Directory '{directory_name}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{directory_name}'.")
except Exception as e:
    print(f"An error occurred: {e}")
    
# Create the directory Conc_x_t
directory_name2 = "Conc_x_t"
try:
    os.mkdir(path+directory_name2)
    print(f"Directory '{directory_name2}' created successfully.")
except FileExistsError:
    print(f"Directory '{directory_name2}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{directory_name2}'.")
except Exception as e:
    print(f"An error occurred: {e}")

'''
for l in range(0,5):
    for j in range(0, 2**l):
        MP = EstimateInitialMP(RegArr, j)
        #err = Error_(RegArr, j, MP, dt)
        err, coordx, coordy = find_MinErrorMP(RegArr, MP, j, dt)
        PathArr.append([coordx, coordy])
        RegArr.insert(j, [coordx, coordy])
        print(PathArr)
        #print(RegArr)
'''
'''
#plt.plot(PathArr[:][0], PathArr[:][1])
plt.plot(RegArr[:][0], RegArr[:][1])
plt.show()
plt.close()
'''

for t in range(0, numsteps_):
    df = pd.read_csv(path+directory_name + '/' +str(t)+'.csv')
    sorted_df = df.sort_values(by=":0", ascending=True)
    #conc, x, y= np.loadtxt(path+directory_name + '/' +str(t)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[1, 2, 3])
    conc = sorted_df["f_0"]
    x=sorted_df[":0"]
    Conc    = []
    X = []
    Conc.append(conc[0])
    X.append(x[0])
    for cnt in range(1, len(conc)):
        c   =   0
        if x[cnt] == x[cnt -1 ]:
            c   +=  conc[cnt]
        if c != 0:
            Conc.append(np.average(  c   )    )  #conc at time t, for all x
            X.append(   x[cnt] )
 

    f = open(path+directory_name2+"/"+str(t)+".csv", "w")
    f.write("{},{}\n".format("\"X\"","\"Conc\""))
    for x in zip(X, Conc):
        f.write("{},{}\n".format(x[0],x[1]))
    f.close()
    

#plt.plot(X, Z, 'o')

#ax.plot(X, Y, Z, label='rho', linestyle='-', lw=1.5, color='blue', zorder=-1)
#ax.plot(X, Y, Z, label='rho', linestyle='dotted', lw=1.5, color='blue', zorder=-1)
#surf = ax.scatter(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, label='A')
#surf = ax.scatter(xB, yB, zB, cmap=cm.coolwarm, linewidth=0, antialiased=False, label='B')
#surf = ax.scatter(X, Y, A, cmap=cm.coolwarm, linewidth=0, antialiased=False, label='A')
#surf = ax.scatter(X, Y, B, cmap=cm.coolwarm, linewidth=0, antialiased=False, label='B')
#ax.plot_surface(X2, Y2, Z2, alpha=0.1, color="blue", zorder=0)
#ax.set_xlabel('x axis')
#ax.set_ylabel('y axis')
#ax.set_zlabel('z axis')
#ax.legend()
#ax.set_xlim(-75.5, -65.5)
#ax.set_ylim(-10, 0)
#ax.set_zlim(0, 10)
#plt.axis('on')
#ax.set_aspect('equal')

#fig. savefig ("Cone.png", dpi=100, transparent = False)
#plt.show()
