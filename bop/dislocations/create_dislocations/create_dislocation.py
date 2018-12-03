from simple_dislocation import Dislocation
from create_cells import Disl_supercell
import scipy as sci
import os
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from matplotlib import rc
rcParams["figure.figsize"] = 4, 3
rcParams["font.family"] = "serif"
rcParams["font.size"] = 8
rcParams["font.serif"] = ["DejaVu Serif"]
rc("text", usetex=True)
sci.set_printoptions(linewidth=200, precision=4)


# Elastic constants in units of 10^{9} Pa
c11 = sci.dtype(sci.float128)
c11 = 1.761e2
c12 = sci.dtype(sci.float128)
c12 = 0.868e2
c13 = sci.dtype(sci.float128)
c13 = 0.682e2
c33 = sci.dtype(sci.float128)
c33 = 1.905e2
c44 = sci.dtype(sci.float128)
c44 = 0.508e2
c66 = sci.dtype(sci.float128)
c66 = 0.450e2

C = (c11, c33, c44, c12, c13, c66)


# Types of dislocation
pure = True
screw = True
plot = True

# a lattice parameter
alat = 0.29012e1

# Burger's Vector
b = sci.array([0., 0., alat])

# Dislocation Coordinate System
disl_coord = np.eye(3)

# Instantiating class
dis = Dislocation(C, b=b, a=alat,
                  pure=pure, screw=screw, plot=plot, T=disl_coord)
# Generating dislocation
#dis = d.gen_disl_u()
#dis = d.u_screw()


# Specification of where the dislocation should be

#############    Example of square hcp unit cell  #############

# Length array
lengths = np.array([5.026674058492405,
                    2.9021516207990987,
                    4.679881023538525])

# Lattice Vectors
plat = np.array([[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]])

# Unit cell
unit_cell = np.array([[0.5, 1.0, 0.0],
                      [0.0, 0.5, 0.0],
                      [0.166666666667, 1.0, 0.5],
                      [0.666666666667, 0.5, 0.5]])

luc = len(unit_cell)

# Number of periodic images
nx = 2  # 4
ny = 2  # 2
nz = 2  # 5

nxyz = (2, 2, 2)

# Number of unit cells before inert atoms appear
ninertx = np.array([1, 11])
ninerty = np.array([0, 2])
ninertz = np.array([1, 12])
ninert = (ninertx, ninerty, ninertz)
rcore = None
rcphi = 90. * np.pi / 180.


# Working directory (If not specified will use current working directory)
cwd = os.getcwd()
gen_disl_path = cwd + '/generated_dislocations'

species = "Ti"
cell_file = "prismatic_screw_"

radii = 20, 25
ninert = radii
alat = 5.575
print("Writing disl supercell")
ds = Disl_supercell(unit_cell, lengths, alat, plat, nxyz,  # geometry='square',
                    ninert=ninert, disl=dis, n_disl=1, cwd=cwd)
ds.write_cell_with_dislocation()
ds.write_cell_with_dislocation(output='bop')

# def __init__(self, unit_cell, lengths, alat, plat, nxyz, ninert=None, disl=None, n_disl=1,
#              rcore=None, rcphi=0., rotation=np.eye(3), species="Ti",
#              cwd='./', output_path='./generated_dislocations', output='bop',
#              filename='cell_gen', geometry='circle', labels=['tih', 'hcp']):
