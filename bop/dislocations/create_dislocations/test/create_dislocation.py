from simple_dislocation import Dislocation
from create_cells import Disl_supercell
from write_input_files import WriteFiles
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

################################################################
####################     Define Unit cell  #####################


####################   Basal square hcp unit cell  #############

# Length array

alat_tbe = 5.575
clat_tbe = 4.683
lengths_tbe = np.array([ 3**(0.5) * alat_tbe,
                         alat_tbe,
                         clat_tbe])

alat_bop = 2.9021516207990987
lengths_bop = np.array([5.026674058492405,
                    2.9021516207990987,
                    4.679881023538525])

# Lattice Vectors
plat = np.array([[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]])

# Unit cell
## This is in units of the *lengths* of each axes. 
unit_cell = np.array([[0.5, 1.0, 0.0],
                      [0.0, 0.5, 0.0],
                      [0.166666666667, 1.0, 0.5],
                      [0.666666666667, 0.5, 0.5]])



# Elastic constants in units of 10^{9} Pa (Can actually be any units)
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


#####   Specifying parameters for dislocation    #####

# Number of periodic images
# These can be truncated depending on the geometry chosen. 
nxyz = (15, 15, 15)

# Number of unit cells before inert atoms appear
## For Square geometry
ninertx = np.array( [ 0, 15 ] )
ninerty = np.array( [ 0, 15 ] )
ninertz = np.array( [ 1, 15 ] )
n_inert = (ninertx, ninerty, ninertz)

# Where the dislocation is situated
rcore = None
rcphi = 90. * np.pi / 180.


radii = 20, 25
ninert = radii
#ninert = n_inert
#####################  BOP #########################

# Burger's Vector (in dislocation coordinate system)
b = sci.array([0., 0., alat_bop])

# Dislocation Coordinate System
## Identity is the same reference frame as the elastic constants
## Prismatic screw means that the dislocation is along y, so z maps to y etc

disl_coord = np.array( [ [ 1., 0.,  0. ],
                         [ 0., 0., 1. ],
                         [ 0., -1.,  0.  ]  ] )

disl_axis = 1
## Where 0 is x, 1 is y and z is 2

dis_bop = Dislocation(C, b=b, a=alat_bop,
                  pure=pure, screw=screw, plot=plot, T=disl_coord)
#disl_disp_bop = dis_bop.displacement()


print("Writing disl supercell for bop")
ds = Disl_supercell(unit_cell, lengths_bop, alat_bop, plat,
                    nxyz, ninert=ninert, #geometry='square',
                    rcphi=90. * np.pi/180,
                    disl=dis_bop, n_disl=1, disl_axis=disl_axis)
cwd = os.getcwd()
ds.write_cell_with_dislocation(output='bop', axis=disl_axis)
os.chdir(cwd)



#####################  TBE #########################

# Burger's Vector
b = sci.array([0., 0., alat_tbe])

# Dislocation Coordinate System
#disl_coord = np.eye(3)

dis_tbe = Dislocation(C, b=b, a=alat_tbe,
                  pure=pure, screw=screw, plot=plot, T=disl_coord)

#ninert = (50, 55)

print("Writing disl supercell")
ds = Disl_supercell(unit_cell, lengths_tbe, alat_tbe, plat, nxyz, #  geometry='square',
                    rcphi=90. * np.pi/180,
                    ninert=ninert, disl=dis_tbe, n_disl=1, disl_axis=disl_axis)

cwd = os.getcwd()
ds.write_cell_with_dislocation(axis=disl_axis)
os.chdir(cwd)




# # Working directory (If not specified will use current working directory)
# cwd = os.getcwd()
# gen_disl_path = cwd + '/generated_dislocations'

# species = "Ti"
# cell_file = "prismatic_screw_"

# radii = 20, 25
# ninert = radii

# alat = 5.575
# print("Writing disl supercell")
# ds = Disl_supercell(unit_cell, lengths, alat, plat, nxyz,  # geometry='square',
#                     ninert=ninert, disl=disl_disp, n_disl=1)

# ds.write_cell_with_dislocation()


# ds.alat = 2.9012
# ds.b 
# ds.lengths = np.array([5.026674058492405,
#                     2.9021516207990987,
#                     4.679881023538525])

# ds.write_cell_with_dislocation(output='bop')

# def __init__(self, unit_cell, lengths, alat, plat, nxyz, ninert=None, disl=None, n_disl=1,
#              rcore=None, rcphi=0., rotation=np.eye(3), species="Ti",
#              cwd='./', output_path='./generated_dislocations', output='bop',
#              filename='cell_gen', geometry='circle', labels=['tih', 'hcp']):

# Specification of where the dislocation should be

# luc = len(unit_cell)


# # Number of unit cells before inert atoms appear
# ninertx = np.array([1, 11])
# ninerty = np.array([0, 2])
# ninertz = np.array([1, 12])
# ninert = (ninertx, ninerty, ninertz)
# rcore = None
# rcphi = 90. * np.pi / 180.

