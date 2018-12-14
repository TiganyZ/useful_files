import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import subprocess
import shlex
import math
import time
import sys
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import copy
import random
from matplotlib import rc, cm
import os
rc('font', **{'family': 'serif', 'serif': ['Palatino'],  'size': 18})
rc('text', usetex=True)


def pickle_data(xx, yy, etot, sf_type):

    xf = open("hom_gamma_surface_%s_xx.pkl" % (sf_type), "wb")
    yf = open("hom_gamma_surface_%s_yy.pkl" % (sf_type), "wb")
    ef = open("hom_gamma_surface_%s_etot.pkl" % (sf_type), "wb")

    pickle.dump(xx,   xf)
    pickle.dump(yy,   yf)
    pickle.dump(etot, ef)

    xf.close()
    yf.close()
    ef.close()


fname = open('hom_gamma_surface_Basal_tbe_etot_unrelaxed_8-8-8.pkl', 'rb')
etot_basal = pickle.load(fname)
fname.close()

fname = open('hom_gamma_surface_Basal_tbe_yy_unrelaxed_8-8-8.pkl', 'rb')
yy_basal = pickle.load(fname)
fname.close()

fname = open('hom_gamma_surface_Basal_tbe_xx_unrelaxed_8-8-8.pkl', 'rb')
xx_basal = pickle.load(fname)
fname.close


print(xx_basal, yy_basal, etot_basal)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r'$\gamma$ surface: Basal plane')
ax.set_xlabel(r' $1/2[10\bar{1}0]$')
ax.set_ylabel(r' $1/3[\bar{1}2\bar{1}0]$')
ax.plot_surface(xx_basal, yy_basal, etot_basal - etot_basal[0],
                rstride=1, cstride=1, cmap=cm.coolwarm)
e_b = etot_basal - etot_basal[0]
ax.contourf(xx_basal, yy_basal, e_b,
            zdir='z', offset=np.min(e_b), cmap=cm.coolwarm)
plt.savefig('basal_gamma_surface_sept_model.png')

"""    
fname = open('hsbc_gs_prismatic_tbe_2-2-2_xx.pkl', 'rb' )
xx_pris = pickle.load( fname ); fname.close()

fname = open('hsbc_gs_prismatic_tbe_2-2-2_yy.pkl', 'rb' )
yy_pris = pickle.load( fname ); fname.close()

fname = open('hsbc_gs_prismatic_tbe_2-2-2_etot.pkl', 'rb' )
etot_pris = pickle.load( fname ); fname.close

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_title(   r'$\gamma$ surface: Prismatic plane' )
ax1.set_xlabel(  r' $1/2[10\bar{1}0]$' )
ax1.set_ylabel(  r' $[0001]$' )
e_p = etot_pris - etot_pris[0]
ax1.plot_surface( xx_pris, yy_pris, etot_pris - etot_pris[0],
                 rstride=1, cstride=1, cmap=cm.coolwarm )
ax1.contourf( xx_pris, yy_pris, etot_pris - etot_pris[0],
                 zdir='z', offset=np.min(e_p), cmap=cm.coolwarm )


"""

# #  Data from Basal run initially.

# yv = np.linspace(0.0, 1.0, 11)

# basal_energy = np.array( [ -36.62637967,
#                            -36.6552219,
#                            -36.65917547,
#                            -36.66610954,
#                            -36.66165154,
#                            -36.68010682,
#                            -36.69190936,
#                            -36.65166395,
#                            -36.62917958,
#                            -36.5886075,
#                            -36.54256141 ])

# # This is from the second run where the y vector is along 1/3[1-210]
# basal_energy = np.array( [ -36.62637967,
#                            -36.65308724,
#                            -36.64440945,
#                            -36.62750537,
#                            -36.60947574,
#                            -36.60450655,
#                            -36.60945831,
#                            -36.62750903,
#                            -36.64440651,
#                            -36.65269534,
#                            -36.62665853 ])


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title(r'$\gamma$ surface: Basal plane along 1/3[1\bar{2}10]$')
# ax.set_xlabel(r' $1/3[1\bar{2}10]$')
# ax.set_ylabel(r' Energy (Ryd)')
# ax.plot(yv, basal_energy, 'bo', yv, basal_energy, 'k')


# # plt.plot_wireframe(xx,yy,etot)
plt.show()
