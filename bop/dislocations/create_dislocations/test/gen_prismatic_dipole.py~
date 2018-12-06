from write_input_files import WriteFiles as wf
from simple_dislocation import Dislocation
import anisotropic_dislocation as anis_dis
from create_cells import Disl_supercell
from write_input_files import WriteFiles
import scipy as sci
import os
import copy
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
#################       Elastic Constants      #################

# Elastic constants in units of 10^{9} Pa (Can actually be in any units)
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
nxyz = (9, 9, 8)


# Number of unit cells before inert atoms appear

# For Square geometry
ninertx = np.array([0, 9])
ninerty = np.array([0, 9])
ninertz = np.array([0, 8])
n_inert = (ninertx, ninerty, ninertz)


# Where the dislocation is situated
rcore = None


# Rotation of where the cut is about the dislocation axis.
rcphi = 90. * np.pi / 180.


# Radial boundary conditions:
# r1 is the radius after which there are only inert atoms
# r2 is the radius after which there are no atoms.

radii = 20, 25
ninert = radii
#ninert = n_inert

# Dislocation Coordinate System
# Identity is the same reference frame as the elastic constants
# Prismatic screw means that the dislocation is along -x with primitive unit cell, so z maps to -x etc
# Then must rotate so dislocation is prismatic screw, so z maps to x


# Rotation about z to line up [11-20] with x axis
disl_coord = np.array([[np.sqrt(3)/2., 0.5,  0.],
                       [-0.5, np.sqrt(3)/2., 0.],
                       [0., 0.,  1.]])
# Rotation about y such that z' is x
disl_coord = disl_coord.dot(np.array([[0., 0.,  -1.],
                                      [0., 1., 0.],
                                      [1., 0.,  0.]]))




disl_axis = 0
# Where 0 is x, 1 is y and z is 2

#####################  TBE #########################


alat_tbe = 5.575
clat_tbe = 8.849586805  # 4.683
q = clat_tbe / alat_tbe


rotation= np.array([[np.sqrt(3)/2., 0.5,  0.],
                       [-0.5, np.sqrt(3)/2., 0.],
                       [0., 0.,  1.]])


lengths_tbe = np.array([alat_tbe,
                        alat_tbe,
                        alat_tbe])

plat = np.array([[0.,           -1, 0.],
                 [3**(0.5)/2., 0.5, 0.],
                 [0.,            0, q]])

plat_inv = np.linalg.inv(plat)

unit_cell_prim_hcp = np.array([[0.,  0., 0.],
                               [1./(2.*3**(0.5)), -1/2.,  q/2]])

# for i, p in enumerate(  unit_cell_prim_hcp ):
#     unit_cell_prim_hcp[i] = plat_inv.dot( p )

unit_cell = unit_cell_prim_hcp
print("unit cell prim", unit_cell_prim_hcp)


# Burger's Vector in Dislocation coordinate system
b = sci.array([0., 0., alat_tbe])

# Dislocation Coordinate System
#disl_coord = np.eye(3)

dis_tbe1 = Dislocation(C, b=b, a=alat_tbe,
                       pure=pure, screw=screw, plot=plot, T=disl_coord)

dis_tbe2 = Dislocation(C, b=-b, a=alat_tbe,
                       pure=pure, screw=screw, plot=plot, T=disl_coord)

dis_tbe = [dis_tbe1, dis_tbe2]
ndis = 2

rcore1 = np.array([(1./3.) * (plat[0] + plat[2])]) * \
    lengths_tbe * np.asarray(nxyz)
rcore2 = np.array([(2./3.) * (plat[0] + plat[2])]) * \
    lengths_tbe * np.asarray(nxyz)

rcore = [rcore1, rcore2]

#ninert = (50, 55)
ninert = n_inert
print("Writing disl supercell")

dc_rot = np.array([[0., 0.,  -1.],
                    [0., 1., 0.],
                    [1., 0.,  0.]])

m = disl_coord[1]
n = disl_coord[2]

m = dc_rot[1]
n = dc_rot[2]


#m = np.array([1,0,0])
#n = np.array([0,1,0])

dis_tbe_anis1 = anis_dis.anis_dislocation(
    alat_tbe, b, C, m, n, pure, screw, plot)#, trans=disl_coord)
dis_tbe_anis2 = anis_dis.anis_dislocation(
    alat_tbe, -b, C, m, n, pure, screw, plot)# , trans=disl_coord)
ndis = 2
dis_tbe_anis = [dis_tbe_anis1, dis_tbe_anis2]


rcore_1d = np.array( [  (1./4.) * ( rotation.dot( plat[0] ) * lengths_tbe[0] * nxyz[0] +
                                    rotation.dot( plat[1] ) * lengths_tbe[1] * nxyz[1] +
                                    rotation.dot( plat[2] ) * lengths_tbe[2] * nxyz[2]) ])

rcore_2d = np.array( [  (3./4.) * ( rotation.dot( plat[0] ) * lengths_tbe[0] * nxyz[0] +
                                    rotation.dot( plat[1] ) * lengths_tbe[1] * nxyz[1] +
                                    rotation.dot( plat[2] ) * lengths_tbe[2] * nxyz[2]) ])

print("rcore 1d", rcore_1d)

rcore = [rcore_1d, rcore_2d]
disl_axis = 0
ds = Disl_supercell(unit_cell, lengths_tbe, alat_tbe, plat, nxyz,   geometry='square',
                    rcphi=[-90. * np.pi/180, -90. * np.pi/180],
                    full_anis=True,
                    rcore=rcore,
                    ninert=ninert,
                    disl=[dis_tbe_anis1,dis_tbe_anis2],
                    n_disl=2, disl_axis=disl_axis)


# ds = Disl_supercell(unit_cell, lengths_tbe, alat_tbe, plat, nxyz,   geometry='square',
#                     rcphi=[0,0],#[90. * np.pi/180, 90. * np.pi/180],
#                     full_anis=True,
#                     rcore=rcore,
#                     ninert=ninert, disl=dis_tbe_anis, n_disl=ndis, disl_axis=disl_axis)

cwd = os.getcwd()
ds.write_cell_with_dislocation(axis=disl_axis, add_name="anis_prim_multi", rotation=rotation)
os.chdir(cwd)

# print("Writing cell with no dislocations")
# print("lengths tbe", lengths_tbe)
# print("plat", plat)

# plat_t = copy.copy(plat)
# ndis = 0
# ds1 = Disl_supercell(unit_cell, lengths_tbe, alat_tbe, plat, nxyz,   geometry='square',
#                     rcphi=[90. * np.pi/180, 90. * np.pi/180],
#                     rcore=rcore,
#                     ninert=ninert, disl=dis_tbe, n_disl=ndis, disl_axis=disl_axis)

# print("ds1 is ds",ds1 is ds)

# atoms, inert_atoms = ds1.get_atoms()

# print("\n post instantiation")
# print("lengths tbe", lengths_tbe)
# print("plat", plat)

# print(atoms)
# file_ext = "trial_sol"
# cwd = os.getcwd()
# wfile = wf( filename="site.ti" + file_ext,  cwd='./', output_path='./generated_dislocations', write_to_dir=True  )
# wfile.write_site_file( (atoms, inert_atoms), ds1.species, ds1.alat, ds1.plat )
# os.chdir(cwd)
