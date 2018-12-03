import create_simple_straight_dislocation as create_disl
from create_cells import create_disl_supercell as cds
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


# Types of dislocation
pure = True
screw = True
plot = True

# a lattice parameter
a = 0.29012e1

# Burger's Vector
b = sci.array([0., 0., a])

# Dislocation Coordinate System
disl_coord = np.eye(3)

# Instantiating class
d = create_disl.Dislocation(C, b, a,
                            pure, screw, plot,
                            hexagonal, disl_coord)
# Generating dislocation
dis = d.gen_disl_u()


# Specification of where the dislocation should be

#############    Example of square hcp unit cell  #############

# Length array
l = np.array([5.026674058492405,
              2.9021516207990987,
              4.679881023538525])

# Lattice Vectors
a = np.array([[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]])

# Unit cell
unit_cell = np.array([[0.5, 1.0, 0.0],
                      [0.0, 0.5, 0.0],
                      [0.166666666667, 1.0, 0.5],
                      [0.666666666667, 0.5, 0.5]])

luc = len(uc)

# Number of periodic images
nx = 13  # 4
ny = 2  # 2
nz = 14  # 5

# Number of unit cells before inert atoms appear
ninertx = np.array([1, 11])
ninerty = np.array([0, 2])
ninertz = np.array([1, 12])

ninert = nx * ny * nz * luc - ninertx[-1] * ninerty[-1] * ninertz[-1] * luc

n_atoms_tot = luc * nx * ny * nz


flen = np.zeros(l.shape)
flen[0] = nx * l[0]
flen[1] = ny * l[1]
flen[2] = nz * l[2]

rcore = 0.5 * flen
rcore[1] = 0.
rcphi = 90. * np.pi / 180.

disl_supercell = cds(unit_cell, lengths, alat, plat, nxyz, ninert, disl)


# Working directory (If not specified will use current working directory)
cwd = os.getcwd()
gen_disl_path = cwd + '/generated_dislocations'

species = "Ti"
cell_file = "prismatic_screw_"
