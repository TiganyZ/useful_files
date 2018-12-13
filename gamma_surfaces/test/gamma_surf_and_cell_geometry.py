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
from matplotlib import rc
import os
rc('font', **{'family': 'serif', 'serif': ['Palatino'],  'size': 18})
rc('text', usetex=True)


def remove_bad_syntax(values, syntax):
    return values.replace(syntax, " ")


def cmd_result(cmd):
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result, err = proc.communicate()
    result = result.decode("utf-8")
    return result


def cmd_write_to_file(cmd, filename):
    output_file = open(filename, mode='w')
    retval = subprocess.call(cmd, shell=True, stdout=output_file)
    output_file.close()


########################   Convert file to XYZ   ###############################

def convert_file_to_xyz(plat, atom_pos, species, filename):

    if type(atom_pos) == tuple:
        atom_pos, inert_atom_pos = atom_pos
        n_inert = len(inert_atom_pos)
        n_at = len(atom_pos)
        n_tot = n_at + n_inert
    else:
        n_tot = len(atom_pos)
        n_at = n_tot

    n_cell_filename = filename + "_hom_convert.xyz"

    global iterations

    if iterations == 0:
        out_xyz_file = open(n_cell_filename, mode='w+')
    else:
        out_xyz_file = open(n_cell_filename, mode='a')

    out_xyz_file.write(str(n_tot) + "\n")
    out_xyz_file.write('Lattice=" ' + ' '.join(
        [str(x) for x in plat.flatten()]) + '" Properties=species:S:1:pos:R:3 \n')

    for i in range(n_at):
        out_xyz_file.write(" " + species
                           + " " + '{:<12.8f}'.format(atom_pos[i, 0])
                           + " " + '{:<12.8f}'.format(atom_pos[i, 1])
                           + " " + '{:<12.8f}'.format(atom_pos[i, 2])
                           + " \n")
    for i in range(n_inert):
        out_xyz_file.write(" " + species + "n"
                           + " " + '{:<12.8f}'.format(inert_atom_pos[i, 0])
                           + " " + '{:<12.8f}'.format(inert_atom_pos[i, 1])
                           + " " + '{:<12.8f}'.format(inert_atom_pos[i, 2])
                           + " \n")

    # for i in range(n_d, n_d + n_inert):
    #     out_xyz_file.write(  " " + species + " " + str( atom_pos[i,0]   )
    #                                   + " " + str( atom_pos[i,1]   )
    #                                   + " " + str( atom_pos[i,2]   )
    #                                   + " \n"                              )
    out_xyz_file.close()


def find_energy(LMarg, args, filename, ename="ebind", from_file=False, tail=1):

    cmd = LMarg + ' ' + args

    # print(cmd)

    if from_file != True:
        cmd_write_to_file(cmd, filename)

    if 'bop' in LMarg:
        cmd = "grep '" + str(ename) + "' " + filename
        if tail != 0:
            if tail > 0:
                cmd += " | tail -" + str(int(tail)) + " | awk '{print$2}' "
            elif tail < 0:
                cmd += " | head " + str(int(tail)) + " | awk '{print$2}' "

    elif 'lmf' in LMarg:
        cmd = "grep 'ehk' " + filename + \
            " | tail -2 | grep 'From last iter' | awk '{print $5}'"

    elif 'tbe' in LMarg:
        cmd = "grep 'total energy' " + filename + \
            " | tail -1 | awk '{print $4}'"

    etot = cmd_result(cmd)

    try:
        etot = float(etot[0:-1])
    except ValueError:
        cmd = "grep 'Exit' " + filename + " "
        error = cmd_result(cmd)
        print(str(error))
        print(' Error: \n       ' + str(error) +
              ' From file ' + filename + ' \n Exiting...')
        etot = 'error'

    return etot


def get_unit_cell(a, c, sf_type='Basal'):

    if sf_type == 'Basal':

        # Length array
        l = np.array([3**(0.5) * a,
                      a,
                      c])

        # Lattice Vectors
        plat = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])

        # Unit cell (In multiples of the lattice vectors)
        uc = np.array([[0.5, 1.0, 0.0],
                       [0.0, 0.5, 0.0],
                       [0.166666666667, 1.0, 0.5],
                       [0.666666666667, 0.5, 0.5]])

        x_vec = a * np.array([a, 0., 0.])
        y_vec = a * np.array([(3**(0.5) / 2.) * a, 0.5, 0.])

        # # Using Dimitar's cell such that the stacking faults are oriented like other gamma surfaces.

        # l = np.array([ a,
        #                3**(0.5) *a,
        #                c            ])

        # # Lattice Vectors
        # plat = np.array( [  [  1.0, 0.0, 0.0 ],
        #                  [  0.0, 1.0, 0.0 ],
        #                  [  0.0, 0.0, 1.0 ]  ] )

        # # Unit cell (In multiples of the lattice vectors)
        # uc = np.array(  [ [ 0.500000000000, 0.000000000000, 0.000000000000 ],
        #                   [ 0.000000000000, 0.500000000000, 0.000000000000],
        #                   [ 0.000000000000, 0.166666666667, 0.500000000000 ],
        #                   [ 0.500000000000, 0.666666666667, 0.500000000000 ]   ]  )

        fault_plane = 0.0

        in_units_of_len = True

    if sf_type == 'Pyramidal':
        # Using the formulae from F.C. Frank as proposed by of Adrian Sutton's PhD student.

        q = c/a

        plat = np.array([[(-9 + 11*3**(0.5))/3., 0, 4*q],
                         [0.,        1.,     0.],
                         [3*3**(0.5),    0.,   -2*q]])

        zp = plat[2]
        ang = np.arccos(np.array([3*3**(0.5),    0.,   2*q]).dot(
            np.array([0., 0., 1.]))/np.linalg.norm(np.array([3*3**(0.5),    0.,   2*q])))

        R = np.array([[np.cos(ang),     0.,  np.sin(ang)],
                      [0.,              1,         0.],
                      [-np.sin(ang),    0,   np.cos(ang)]])

        atom_pos_alat = np.zeros((64, 3))

        atom_pos_alat[:, 0] = np.array([10., 15., 13., 15., 13., 15., 16., 12., 16., 16., 19., 21., 19.,
                                        21., 19., 21., 19., 21., 19., 21., 22., 18., 22., 18., 22., 18.,
                                        22., 18., 22., 27., 25., 27., 25., 27., 25., 27., 25., 27., 25.,
                                        24., 28., 24., 28., 24., 28., 24., 28., 24., 28., 24., 31., 33.,
                                        31., 33., 31., 33., 30., 30., 34., 30., 34., 30., 37., 36.]) * (3**(0.5) / 6.)

        atom_pos_alat[:, 1] = np.array([1.5, 1., 1., 1., 1., 1., 1.5, 1.5, 1.5, 1.5, 1., 1., 1.,
                                        1., 1., 1., 1., 1., 1., 1., 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                                        1.5, 1.5, 1.5, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1., 1.,
                                        1., 1., 1., 1., 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1., 1.5])

        atom_pos_alat[:, 2] = np.array([0.5,  0.,  0.5,  1.,  1.5,  2.,  0.5,  1.,  1.5,  2.5, -0.5,
                                        0.,  0.5,  1.,  1.5,  2.,  2.5,  3.,  3.5,  4., -0.5,  0.,
                                        0.5,  1.,  1.5,  2.,  2.5,  3.,  3.5, -1., -0.5,  0.,  0.5,
                                        1.,  1.5,  2.,  2.5,  3.,  3.5, -1., -0.5,  0.,  0.5,  1.,
                                        1.5,  2.,  2.5,  3.,  3.5,  4.,  0.5,  1.,  1.5,  2.,  2.5,
                                        3.,  0.,  1.,  1.5,  2.,  2.5,  3.,  2.5,  2.]) * q

        # This cell is not centred, so we must make it so.
        atom_pos_alat -= np.array([(25./6)*np.sqrt(3), 1., 1.5 * q])

        pyramidal_inert = np.zeros((9, 3))

        pyramidal_inert[:, 0] = np.array([4.66666672, 5.16666677, 5.50000026, 5.00000021,
                                          5.66666682, 6.16666652, 6.50000001, 5.99999996, 6.66666694]) * 3**(0.5)

        pyramidal_inert[:, 1] = np.array(
            [4.425000, 2.95, 2.95, 4.4250002, 4.4250002, 2.95, 2.95, 4.4250002, 4.4250002]) / 2.95

        pyramidal_inert[:, 2] = np.array(
            [1.5, -0.5, 0.0, -1.0, 0.5, 1.5, 2.0, 1.0, 2.5]) * q

        # Rotate so that we have pyramidal plane normal along z
        Rinv = np.linalg.inv(R)
        opt_atom_pos = Rinv.dot(atom_pos_alat.T) * a

        plat = Rinv.dot(plat) * a

        l = np.array([1., 1., 1.])
        uc = opt_atom_pos, pyramidal_inert
        fault_plane = 0.0
        in_units_of_len = False

    if sf_type == 'Prismatic':
        # The prismatic plane of this cell is half of the cell.
        q = c/a
        # Length array
        l = np.array([a,
                      c,
                      3**(0.5) * a])

        # Lattice Vectors
        plat = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])

        # Unit cell (In multiples of the lattice vectors)
        uc = np.array([[0.5, 0.0, 0.0],
                       [0.0, 0.0, 0.5],
                       [0.0, 0.5, 1./6.],
                       [0.5, 0.5, 2./3.]])

        fault_plane = 0.0
        in_units_of_len = True

    return l, plat, uc, fault_plane, in_units_of_len


def write_bop_file(l, plat, uc, nxyz, atom_pos, inert_atom_pos, sf_type):

    conv_to_ang = 1. / 1.8897259886

    #plat *= conv_to_ang

    nx, ny, nz = nxyz

    flen = l * np.array([nx, ny, nz])

    fname = "cell"
    cell_file = fname + ".in"
    cell_inert_file = fname + "_inert.in"
    xyz_file = fname + "_conv.xyz"

    species = "Ti"
    out_file = open(cell_file, mode='w+')
    out_inert_file = open(cell_inert_file, mode='w+')
    out_xyz_file = open(xyz_file, mode='w+')

    n_at = len(atom_pos)
    n_inert = len(inert_atom_pos)
    n_atoms_tot = n_at + n_inert

    plat_inv = np.linalg.inv(plat)

    plat_b = plat / flen

    out_file.write(" " + fname + "   hcp\n")
    out_file.write(
        " " + ' '.join(['{:<12.10f}'.format(x) for x in plat_b[0]]) + "\n")
    out_file.write(
        " " + ' '.join(['{:<12.10f}'.format(x) for x in plat_b[1]]) + "\n")
    out_file.write(
        " " + ' '.join(['{:<12.10f}'.format(x) for x in plat_b[2]]) + "\n")

    out_file.write(" len " + '{:<12.10f}'.format(flen[0])
                   + " " + '{:<12.10f}'.format(flen[1])
                   + " " + '{:<12.10f}'.format(flen[2]) + "\n")

    out_file.write(" latpar 1.0\n")
    out_file.write(" nd " + str(n_at) + "\n")

    for i in range(n_at):
        pl_inv_a = plat_inv.dot(atom_pos[i])
        out_file.write(" " + species
                       + " " + '{:<12.10f}'.format(pl_inv_a[0])
                         + " " + '{:<12.10f}'.format(pl_inv_a[1])
                         + " " + '{:<12.10f}'.format(pl_inv_a[2])
                         + ' 0.0 0.0 \n')
    if n_inert > 0:
        out_file.write(" ninert " + str(n_inert) + "\n")
        for i in range(n_inert):
            pl_inv_a = plat_inv.dot(inert_atom_pos[i])
            out_file.write(" " + species
                           + " " + '{:<12.10f}'.format(pl_inv_a[0])
                           + " " + '{:<12.10f}'.format(pl_inv_a[1])
                           + " " + '{:<12.10f}'.format(pl_inv_a[2])
                           + ' 0.0 0.0 \n')

    out_file.write("\n nullheight       0.000000000000\n\n")

    out_file.write(" kspace_conf\n")
    out_file.write("   0 t         ! symmode symetry_enabled\n")
    out_file.write("   12 10 12    ! vnkpts\n")
    out_file.write("    t t t       ! offcentre\n\n")

    out_file.write("dislocation_conf\n")
    out_file.write("   0    ! gfbcon\n")
    out_file.write("   0.0  ! radiusi\n")
    out_file.write("   -100.0    100.0    -100.0  100.0\n")
    out_file.write("    0    ! gfbcon\n")
    out_file.write("   0.0  ! radiusi\n")
    out_file.write("   -100.0    100.0    -100.0  100.0\n")
    out_file.write("    0    ! gfbcon\n")
    out_file.write("   0.0  ! radiusi\n")
    out_file.write("   -100.0    100.0    -100.0  100.0\n")
    out_file.write("   0    ! gfbcon\n")
    out_file.write("   0.0  ! radiusi\n")
    out_file.write("   -100.0    100.0    -100.0  100.0\n")

    out_file.close()

    atom_pos_bop = np.zeros(atom_pos.shape)
    inert_atom_pos_bop = np.zeros(inert_atom_pos.shape)

    for i, a in enumerate(atom_pos):

        atom_pos_bop[i, 0] = (plat.dot(plat_inv.dot(atom_pos[i])))[0]
        atom_pos_bop[i, 1] = (plat.dot(plat_inv.dot(atom_pos[i])))[1]
        atom_pos_bop[i, 2] = (plat.dot(plat_inv.dot(atom_pos[i])))[2]

    for i, a in enumerate(inert_atom_pos):
        inert_atom_pos_bop[i, 0] = (
            plat.dot(plat_inv.dot(inert_atom_pos[i])))[0]
        inert_atom_pos_bop[i, 1] = (
            plat.dot(plat_inv.dot(inert_atom_pos[i])))[1]
        inert_atom_pos_bop[i, 2] = (
            plat.dot(plat_inv.dot(inert_atom_pos[i])))[2]

    convert_file_to_xyz(
        plat, (atom_pos_bop, inert_atom_pos_bop), species, 'cell_' + sf_type)


def write_site_file(s_file, tot_atom_pos, species, alat, plat, sf_type):
    convert_to_ryd = 1.8897259886

    atom_pos, inert_atom_pos = tot_atom_pos
    n_inert = len(inert_atom_pos)
    n_at = len(atom_pos)

    atom_pos = (atom_pos / alat).round(12)
    inert_atom_pos = (inert_atom_pos / alat).round(12)

    print("site plat", plat.flatten()/alat)

    site_file = open(s_file, mode='w')
    site_info = ('% site-data vn=3.0 fast io=63 nbas=' + str(n_at + n_inert) + ' alat=' + str(alat)
                 + ' plat=' + ' '.join([str(x) for x in plat.flatten()/alat]) + '\n')
    site_file.write(site_info)
    site_file.write(
        '#                        pos vel                                    eula                   vshft PL rlx\n')

    for i in range(n_at):
        site_file.write(" " + species
                        + " " + '{:<12.10f}'.format(atom_pos[i, 0])
                        + " " + '{:<12.10f}'.format(atom_pos[i, 1])
                        + " " + '{:<12.10f}'.format(atom_pos[i, 2])
                        + " 0.0000000 0.0000000 0.0000000    0.0000000    0.0000000    0.0000000 0.000000  0 001"
                        + " \n")
    for i in range(n_inert):
        site_file.write(" " + species
                        + " " + '{:<12.10f}'.format(inert_atom_pos[i, 0])
                        + " " + '{:<12.10f}'.format(inert_atom_pos[i, 1])
                        + " " + '{:<12.10f}'.format(inert_atom_pos[i, 2])
                        + " 0.0000000 0.0000000 0.0000000    0.0000000    0.0000000    0.0000000 0.000000  0 000"
                        + " \n")
    site_file.close()

    tot_atom_pos = (tot_atom_pos[0] / convert_to_ryd,
                    tot_atom_pos[1] / convert_to_ryd)

    convert_file_to_xyz(plat / convert_to_ryd, tot_atom_pos,
                        species, 'site_' + sf_type)


def gen_cell(l, plat, uc, n_rep_xyz, inert_bounds, sf_type, in_units_of_len=True):

    luc = len(uc)

    # Number of periodic images
    nx, ny, nz = tuple(n_rep_xyz)

    if sf_type == 'Pyramidal':
        pyr_atoms, inert_pyr_atoms = uc

        n_atoms = nx * ny * len(pyr_atoms)

        n_inert = nx * ny * len(inert_pyr_atoms)

        n_atoms_tot = n_atoms + n_inert

        print(sf_type, '\, n_atoms = %s\n n_inert = %s\n n_tot = %s' %
              (n_atoms, n_inert, n_atoms_tot))

    else:

        # Number of unit cells before inert atoms appear
        if type(inert_bounds) is tuple:
            inert_bound_x, inert_bound_y, inert_bound_z = inert_bounds
            print("inert bounds", inert_bound_x, inert_bound_y, inert_bound_z)

        n_inert = nx * ny * nz * luc - ((inert_bound_x[1] - inert_bound_x[0]) * (
            inert_bound_y[1] - inert_bound_y[0]) * (inert_bound_z[1] - inert_bound_z[0])) * luc
        print("n_inert", n_inert)

        flen = np.zeros(l.shape)
        flen[0] = nx * l[0]
        flen[1] = ny * l[1]
        flen[2] = nz * l[2]

        n_atoms_tot = luc * nx * ny * nz

        n_atoms = n_atoms_tot - n_inert
        print("n_atoms", n_atoms)

        atoms = np.zeros((n_atoms, 3))
        inert_atoms = np.zeros((n_inert, 3))
        # atoms[:luc,:] = uc
        na = 0
        nin = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for p in range(luc):

                        if not in_units_of_len:
                            # i*plat[0] + j*plat[1] + k*plat[2]
                            r = uc[p, :] + np.array([i, j, k]) * plat
                            r1, r2, r3 = r[0], r[1], r[2]
                        else:
                            r1 = (uc[p, 0] + i) * l[0]
                            r2 = (uc[p, 1] + j) * l[1]
                            r3 = (uc[p, 2] + k) * l[2]

                        #print(r1, r2, r3)
                        c0 = i < inert_bound_x[0] or j < inert_bound_y[0] or k < inert_bound_z[0]
                        c1 = i >= inert_bound_x[1] or j >= inert_bound_y[1] or k >= inert_bound_z[1]
                        if c0 or c1:
                            inert_atoms[nin, 0] = r1
                            inert_atoms[nin, 1] = r2
                            inert_atoms[nin, 2] = r3
                            nin += 1
                        else:
                            atoms[na, 0] = r1
                            atoms[na, 1] = r2
                            atoms[na, 2] = r3
                            na += 1

        plat = plat * flen
    return atoms, inert_atoms,  plat


def pickle_data(xx, yy, etot, sf_type):

    out_file = open("hom_gam_surf_%s_k_unrelaxed_8-8-8.dat" % (sf_type), 'w')
    out_file.write("#     x          y         etot ")
    for i in range(len(xx)):
        for j in range(len(yy)):
            out_file.write("{}  {}  {}\n".format(
                str(xx[i, j]), str(yy[i, j]), str(etot[i, j])))
    out_file.close()

    xf = open("hom_gamma_surface_%s_xx_unrelaxed_8-8-8.pkl" % (sf_type), "wb")
    yf = open("hom_gamma_surface_%s_yy_unrelaxed_8-8-8.pkl" % (sf_type), "wb")
    ef = open("hom_gamma_surface_%s_etot_unrelaxed_8-8-8.pkl" %
              (sf_type), "wb")

    pickle.dump(xx,   xf)
    pickle.dump(yy,   yf)
    pickle.dump(etot, ef)

    xf.close()
    yf.close()
    ef.close()


def gamma_surface(LMarg, ext, a, c, x_vec, y_vec,
                  steps, nxyz, inert_bound, sf_type, species,
                  fault_plane=0.0, plot=True):

    global iterations
    iterations = 0

    l, plat, uc, fault_plane, in_len = get_unit_cell(a, c, sf_type)

    # atom_pos are in units of alat
    atom_pos, inert_atom_pos, plat = gen_cell(
        l, plat, uc, nxyz, inert_bound, sf_type, in_units_of_len=in_len)

    # Fault plane is by default the plane at which z = fault_plane.
    if fault_plane == 0.0:
        fault_plane = plat[2, 2]/2. + 1e-4

    print("fault plane", fault_plane)

    nx, ny = tuple(steps)
    nx = np.linspace(0.0, 1.0, nx)
    ny = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(nx, ny)

    # Normalise vectors along plane.
    x_hat = x_vec  # /np.linalg.norm(x_vec)
    y_hat = y_vec  # /np.linalg.norm(y_vec)

    init_z = copy.copy(plat[2])

    etot = np.zeros(tuple(steps))
    xx = np.zeros(tuple(steps))
    yy = np.zeros(tuple(steps))

    for j, fy in enumerate(ny):
        for i, fx in enumerate(nx):

            print("Proportion of:\n   x_vec = %.4f \n   y_vec = %.4f " % (fx, fy))

            # Changing the z vector for periodicity:
            # This creates the homogeneous shear boundary conditions.
            xx[i, j] = (x_hat * fx + y_hat * fy)[0]
            yy[i, j] = (x_hat * fx + y_hat * fy)[1]

            print("   x_vec translation = %s\n   y_vec translation = %s " %
                  (x_hat * fx, y_hat * fy))

            plat[2] = init_z + x_hat * fx + y_hat * fy
            print("Old Z vector = ", init_z)
            print("New Z vector = ", plat[2])

            if 'tbe' in LMarg:
                # Write positions to tbe site file
                write_site_file('site.' + ext, (atom_pos, inert_atom_pos),
                                species, a, plat, sf_type)
            elif 'bop' in LMarg:
                # Write positions to cell.in bop file
                write_bop_file(l, plat, uc, nxyz, atom_pos,
                               inert_atom_pos, sf_type)
                cmd_result('rm timm.gs')
            print("lmarg", LMarg)
            etot[i, j] = find_energy(LMarg, ' ', 'gamma_test')

            if i == 0 and j == 0:
                e0 = etot[0, 0]

            #etot[i,j] -= e0

            print("Energy = %s\n" % (etot[i, j]))

            iterations += 1

    if 'tbe' in LMarg:
        extension = '_tbe'
    elif 'lmf' in LMarg:
        extension = '_lmf'
    else:
        extension = '_bop'

    pickle_data(xx, yy, etot, sf_type + extension)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')
        plt.contourf(xx, yy, etot)
        plt.show()
        plt.plot_wireframe(xx, yy, etot)
        plt.show()

    return xx, yy, etot


def run_bop_relax():
    """
    rm nnb.lst
    rm block.xbs

    locpath="/usr/tialsbop"
    potdir="${locpath}/data/fs/tial"
    exedir="${locpath}/bin"
    rundir="${locpath}/tial"

    ln -s  ${rundir}/in/ucell.in.relax      fort.8
    ln -s  ${rundir}/bl/ucell.block         fort.31

    ln -s  ${rundir}/out                    fort.9
    ln -s  ${rundir}/block.out              fort.32
    ln -s  ${rundir}/rl                     fort.72

    ${exedir}/penn_bop
    #/usr/tialsbop/bin/penn_bop
    #/usr/test

    rm fort.*
    rm bec.*
    rm core
    """


def get_all_gamma_surfaces(tbe_command, a_hcp, c_hcp):

    ext = 'ti'
    species = 'Ti'

    ############################     bop     #################################

    bop_a = 5.804303241598197 / 2.
    bop_c = 46.79881023538525 / 10.

    bop_command = '~/BOP/pb5/bld/mi/npbc/bin/bop '

    ############################     tbe     #################################
    ## Objective = 1010

    # a_hcp, c_hcp = 5.503722649302711,  8.735375609762187

    # tbe_command = ( 'tbe ti -vrfile=1 -vnkx=30 -vnky=30 -vnkz=30 '
    #                 + '-vfddtt=0.4589353025120264 -vqddstt=0.7118332818816158 -vb0tt=88.56120896145649 -vp0tt=1.2803107031586398 '
    #                 + '-vb1tt=-14.214028253751056 -vp1tt=0.9885838205722217 -vndt=1.9323918054338258 -vewtol=1d-14 '
    #                 + '-vrelax=5 -vgtol=1d-4 -vxtol=1d-4 -vhess=100 -vforces=1 -vrmaxh={:<12.8f} -vahcp={} -vq={} '.format(  6.7, a_hcp, c_hcp / a_hcp  ) )

    for LMarg in (tbe_command, bop_command):

        print(LMarg)

        if 'tbe' in LMarg:
            a = a_hcp
            c = c_hcp
        elif 'bop' in LMarg:
            a = bop_a
            c = bop_c

        # x_vec = a * np.array( [  1.0, 0.0, 0.0 ])
        # y_vec = a * np.array( [  0.0, 1.0, 0.0 ])

        y_vec = a * np.array([(3**(0.5) / 2.),  -0.5,   0.0])
        x_vec = a * np.array([0.0,        -1.0,   0.0])

        nxyz = (2, 2, 8)

        inert_bound_x = np.array([0, 2])
        inert_bound_y = np.array([0, 2])
        inert_bound_z = np.array([0, 8])

        inert_bound = (inert_bound_x, inert_bound_y, inert_bound_z)

        steps = 21, 21

        plot = False
        sf_type = 'Basal'

        kx = 9
        ky = 15
        kz = 4

        LMarg_b = LMarg + ' -vnkx={} -vnky={} -vnkz={}  '.format(kx, ky, kz)

        print("\n LMarg_b \n ", LMarg_b, "\n")

        xx, yy, etot = gamma_surface(LMarg_b, ext,
                                     a, c, x_vec, y_vec, steps,
                                     nxyz, inert_bound, sf_type, species,
                                     fault_plane=0.0, plot=plot)

        sf_type = 'Prismatic'

        kx = 15
        ky = 15
        kz = 3

        x_vec = a * np.array([1.0, 0.0, 0.0])
        y_vec = c * np.array([0.0, 1.0, 0.0])

        LMarg_p = LMarg + ' -vnkx={} -vnky={} -vnkz={}  '.format(kx, ky, kz)

        print("\n LMarg_p \n ", LMarg_p, "\n")

        xx, yy, etot = gamma_surface(LMarg_p, ext,
                                     a, c, x_vec, y_vec, steps,
                                     nxyz, inert_bound, sf_type, species,
                                     fault_plane=0.0, plot=plot)

        nxyz = (2, 2, 1)

        inert_bound_x = np.array([0, 2])
        inert_bound_y = np.array([0, 2])
        inert_bound_z = np.array([0, 1])

        inert_bound = (inert_bound_x, inert_bound_y, inert_bound_z)

        sf_type = 'Pyramidal'

        xx, yy, etot = gamma_surface(LMarg, ext,
                                     a, c, x_vec, y_vec, steps,
                                     nxyz, inert_bound, sf_type, species,
                                     fault_plane=0.0, plot=plot)


def lp_hcp_energy(x):
    # x is a vector of the lattice parameters x = [a, c]
    global LMarg_a, rmaxh_hcp

    etot = find_energy(LMarg_a + ' ', (' -vahcp=' + str(x[0]) + ' -vchcp=' + str(x[1])
                                       + ' -vrmaxh=' + str(rmaxh_hcp) + '  '),
                       'lpmin')
    print(" a_hcp={:<10.8f}, c_hcp={:<10.8f}, E_tot={:<10.8f}".format(
        x[0], x[1], etot))
    return etot


def lp_omega_energy(x):
    # x is a vector of the lattice parameters x = [a, q, u]
    global LMarg_a, rmaxh_omega
    etot = find_energy(LMarg_a + ' ', (' -vomega=1 -vhcp=0 -vaomega=' + str(x[0])
                                       + ' -vqomega=' +
                                       str(x[1]) + ' -vuomega=' + str(x[2])
                                       + ' -vrmaxh=' + str(rmaxh_omega) + '  '),
                       'lpmin')


def get_min_lp(minimiserf='Nelder-Mead', phase='hcp'):

    a_hcp_exp, c_hcp_exp = 5.57678969, 8.85210082
    q_hcp_exp = c_hcp_exp / a_hcp_exp

    a_omega_exp, c_omega_exp, u_omega_exp = 4.621/0.52917, 2.817/0.52917, 1.0
    q_omega_exp = c_omega_exp / a_omega_exp

    if phase == 'hcp':
        x0 = np.array([a_hcp_exp, c_hcp_exp])
        bnds = ((5.0, 6.0), (8.0, 9.0))
        fnc = lp_hcp_energy

    elif phase == "omega":
        x0 = np.array([a_omega_exp, q_omega_exp, u_omega_exp])
        bnds = ((8.0, 9.0), (0.2, 1.0), (0.0, 1.0))
        fnc = lp_omega_energy

    if minimiserf == "Nelder-Mead":
        print("Using %s\n" % (minimiserf))
        ret = minimize(fnc, x0, method='Nelder-Mead',
                       options={'disp': True, 'fatol': 1e-8})

    elif minimiserf == "SLSQP":
        print("Using SLSQP\n")
        if phase == "hcp":
            # ep = 0.0005
            ep = 0.0006443443443443443
        else:
            # ep = 0.0001
            ep = 0.0006443443443443443 / 10.
        ret = minimize(fnc, x0, method="SLSQP", bounds=bnds,
                       options={'disp': True, 'ftol': 1e-8, 'eps': ep, 'maxiter': 9999})
    elif minimiserf == "DiffEvol":
        # Options: ‘best1bin’‘best1exp’‘rand1exp’‘randtobest1exp’‘currenttobest1exp’‘best2exp’‘rand2exp’‘randtobest1bin’‘currenttobest1bin’‘best2bin’‘rand2bin’‘rand1bin’
        print("Using %s\n" % (minimiserf))
        ret = sci.optimize.differential_evolution(fnc, bounds=bnds, strategy='best1bin',
                                                  maxiter=1000, popsize=15, tol=1e-7, mutation=(0.5, 1), recombination=0.7,
                                                  seed=None, callback=None, disp=False, polish=True, init='latinhypercube', atol=0)
    elif minimiserf == "CG":
        print("Using %s\n" % (minimiserf))
        # Use Conjugate Gradient for lattice parameter minimisation within SciPy
        ret = minimize(fnc, x0, method=minimiserf,
                       options={'disp': True, 'gtol': 1e-10, 'eps': ep, 'maxiter': 9999})
    elif minimiserf == "BFGS":
        print("Using %s\n" % (minimiserf))
        # Use BFGS for lattice parameter minimisation within SciPy
        ret = minimize(fnc, x0, method=minimiserf,
                       options={'disp': True, 'gtol': 1e-6, 'eps': ep, 'maxiter': 9999})

    if phase == "hcp":
        ret = ret['x'][0], ret['x'][1], ret['fun']
    elif phase == "omega":
        ret = ret['x'][0], ret['x'][1], ret['x'][2], ret['fun']
    return ret


def get_nearest_neighbours(LMarg, args, filename):
    cmd = LMarg + ' ' + args + ' '
    cmd_write_to_file(cmd, filename)
    cmd = "grep 'pairc,' " + filename + "| tail -1 | awk '{print $4}'"
    res = cmd_result(cmd)
    return int(res) - 1


def check_rmaxh(LMarg, args, filename,
                rmx_name, rmaxh, nmax,
                find_bounds=False, nn_change_on_boundary=6):

    maxiter = 20

    # Lists used for finding the bounds
    rmh_list, nn_list = [], []
    rmaxh_u = 3 * rmaxh
    rmaxh_l = 0.2 * rmaxh

    res = get_nearest_neighbours(
        LMarg, args + ' -v' + rmx_name + '=' + str(rmaxh), filename)
    cond = res == nmax

    if cond == False:
        rmaxh_u = 3 * rmaxh
        rmaxh_l = 0.2 * rmaxh
        resu = get_nearest_neighbours(
            LMarg, args + ' -v' + rmx_name + '=' + str(rmaxh_u), filename)
        resl = get_nearest_neighbours(
            LMarg, args + ' -v' + rmx_name + '=' + str(rmaxh_l), filename)
        print('\n Initial Neighbours\n   rmaxh_l = %s, rmaxh_u = %s \n    nn_l = %s,     nn_u = %s' % (
            rmaxh_l, rmaxh_u, res, resu))

        rmaxh, rmh_list, nn_list, rmaxh_l, rmaxh_u = rmax_binary_search(LMarg, args, filename,
                                                                        nmax, maxiter, cond,
                                                                        rmaxh_l, rmaxh_u)
    print("\n Found rmaxh = %s, with number of neighbours = %s" % (rmaxh, nmax))
    if find_bounds:
        # Have now found rmaxh in bounds of rmaxhl and rmaxhu.
        # Now need to find the boundaries where the number of neighbours changes
        # up and down, such that the optimum rmaxh is found betwixt said boundaries.

        rmhl = np.asarray(rmh_list)
        nnl = np.asarray(nn_list)
        mnp = nmax + nn_change_on_boundary
        mnm = nmax - nn_change_on_boundary

        nn_m, nn_p = (nnl == mnm, nnl == mnp)
        rmh_m, rmh_p = rmhl[nn_m], rmhl[nn_p]

        rmaxh_u = 4 * rmaxh
        rmaxh_l = 0.2 * rmaxh

        rmaxh_pn, rmh_lpn, nn_lpn, rmaxh_l_pn, rmaxh_u_pn = rmax_binary_search(LMarg, args, filename,
                                                                               mnp,
                                                                               maxiter, False,
                                                                               rmaxh_l, rmaxh_u)
        rmaxh_u = 4 * rmaxh
        rmaxh_l = 0.2 * rmaxh
        rmaxh_mn, rmh_lmn, nn_lmn, rmaxh_l_mn, rmaxh_u_mn = rmax_binary_search(LMarg, args, filename,
                                                                               mnm,
                                                                               maxiter, False,
                                                                               rmaxh_l, rmaxh_u)
        if len(rmh_p) != 0:
            rmh_p_max = np.max(rmh_p)
            if rmh_p_max > rmaxh_pn:
                rmaxh_pn = rmh_p_max

        if len(rmh_m) != 0:
            rmh_m_min = np.min(rmh_m)
            if rmh_p_min < rmaxh_mn:
                rmaxh_pm = rmh_p_min

        print("\n Found rmaxh = %s,\n   within bounds of %.3f -- %.3f\n   with number of neighbours = %s, nmax = %s" %
              (rmaxh, rmaxh_mn, rmaxh_pn, res, nmax))
        # rmaxh = (rmaxh, rmaxh_mn, rmaxh_pn)
        rmaxh_l = rmaxh_mn
        rmaxh_u = rmaxh_pn

    return rmaxh, rmaxh_l, rmaxh_u


def energy_vs_rmaxh(LMarg, args, rmx_name, rmaxh_l, rmaxh_u, n_pts=100, plot=True):

    filename = 'rmaxh_energy'
    rmaxh = np.linspace(rmaxh_l, rmaxh_u, n_pts)
    energy = np.zeros(n_pts)
    n_neigh = np.zeros(n_pts)

    for i, rmh in enumerate(rmaxh):
        energy[i] = find_energy(LMarg, args, filename)
        n_neigh[i] = get_nearest_neighbours(
            LMarg, args + ' -v' + rmx_name + '=' + str(rmh), filename)
        print(energy[i], rmh, n_neigh[i])

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(rmaxh, energy, 'bo')
        ax.set_title(r' Variation of energy with rmaxh ')
        ax.set_xlabel(r' rmaxh (Bohr) ')
        ax.set_ylabel(r' Energy (Ryd) ')
        fig.savefig('Energy_vs_rmaxh_check.png')

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(rmaxh, n_neigh, 'g^')
        ax1.set_title(r' Variation of neighbours with rmaxh ')
        ax1.set_xlabel(r' rmaxh (Bohr) ')
        ax1.set_ylabel(r' Number of neighbours ')
        fig1.savefig('n_neighbours_vs_rmaxh_check.png')

        fname = open('energy_for_energy_vs_rmaxh.pkl', 'wb')
        pickle.dump(energy, fname)
        fname.close()

        fname = open('rmaxh_for_energy_or_n_neighbours_vs_rmaxh.pkl', 'wb')
        pickle.dump(rmaxh, fname)
        fname.close()

        fname = open('n_neighbours_for_n_neighbours_vs_rmaxh.pkl', 'wb')
        pickle.dump(energy, fname)
        fname.close()

        plt.show()


def get_mesh_criteria(filename):

    e_tot_cmd = " grep 'total energy' " + \
        filename + " | tail -1 | awk '{print$4}'"
    e_bs_cmd = " grep 'band structure energy' " + \
        filename + " | tail -1 | awk '{print$5}'"
    e_pp_cmd = " grep 'pair potential energy' " + \
        filename + " | tail -1 | awk '{print$5}'"
    mf_cmd = " grep 'Maximum force=' " + \
        filename + " | tail -1 | awk '{print$3}'"
    nb_cmd = " grep 'nbas' " + filename + " | tail -1 | awk '{print$4}'"

    e_tot = float(cmd_result(e_tot_cmd))
    e_bs = float(cmd_result(e_bs_cmd))
    e_pp = float(cmd_result(e_pp_cmd))
    nb = float(cmd_result(nb_cmd))

    mfc = cmd_result(mf_cmd)

    try:
        mf = float(mfc)
    except ValueError:
        mf = 0.0

    return e_tot/nb, e_bs/nb, e_pp/nb, mf/nb, int(nb)


def check_k_dec(kxx, i):
    if abs(i) >= kxx:
        if i < 0:
            i = - (kxx - 1)
        else:
            i = kxx - 1
        if kxx == 1:
            i = 0
    return i


def find_best_k_mesh(LMarg, kxx, kyy, kzz, filename):
    lim_list = [-2, -1, 0, 1, 2]
    for i in lim_list:
        for j in lim_list:
            for k in lim_list:
                i = check_k_dec(kxx, i)
                j = check_k_dec(kyy, j)
                k = check_k_dec(kzz, k)

                klist.append((kxx+i, kyy+j, kzz+k))

                cmd = LMarg + \
                    ' -vnkx={} -vnky={} -vnkz={} -vrfile=1 '.format(
                        kxx+i, kyy+j, kzz+k)
                cmd_write_to_file(cmd,  unit_cell_to_extend + '_' + filename)
                e_tot, e_bs, e_pp, mf, nb = get_mesh_criteria(
                    unit_cell_to_extend + '_' + filename)

                e_list.append(e_tot)
                bs_list.append(e_bs)
                pp_list.append(e_pp)
                mf_list.append(mf)
                print(e_tot)
                print(e_bs)
                print(e_pp)
                print(mf)


def find_best_k_mesh_and_structure(LMarg, ext, k0, plat0, plat,
                                   x_lim, y_lim, z_lim,
                                   unit_cell_reference,
                                   unit_cell_to_extend):

    # k0 is the starting tuple of k vectors for the initial unit cell of optimisation.
    kx, ky, kz = k0
    filename = "trial_k_mesh_struc"
    LMarg += ' -vforces=1 '
    x_l, x_u = x_lim
    y_l, y_u = y_lim
    z_l, z_u = z_lim

    len0 = (plat0**2).sum(1)
    len1 = (plat1**2).sum(1)

    length_ratios = len1/len0

    # Initial reference run.
    cmd = "cp site." + unit_cell_reference + " site." + ext
    cmd_result(cmd)

    cmd = LMarg + ' -vnkx={} -vnky={} -vnkz={} -vrfile=1 '.format(kx, ky, kz)
    cmd_write_to_file(cmd, unit_cell_reference + '_' + filename)
    e_tot0, e_bs0, e_pp0, mf0, nb0 = get_mesh_criteria(
        unit_cell_reference + '_' + filename)
    print(e_tot0)
    print(e_bs0)
    print(e_pp0)
    print(mf0)

    e_list = []
    bs_list = []
    pp_list = []
    mf_list = []
    k_list = []
    xyz_list = []

    lim_list = [-2, -1, 0, 1, 2]
    u_plat = np.eye(3)

    for x in range(x_l, x_u):
        for y in range(y_l, y_u):
            for z in range(z_l, z_u):

                # Copy cell to extend to site file.
                cmd = "cp site." + unit_cell_to_extend + " site." + ext
                cmd_result(cmd)

                # Extend unit cell
                n_p = u_plat * np.array([x, y, z])
                cmd = ('echo "m ' + ' '.join([str(it) for it in n_p.flatten()])
                       + '"  | lmscell --wsite -vrfile=1 ' + ext)

                kxx = int(round(float(kx) / (x * length_ratios[0]), 0))
                kyy = int(round(float(ky) / (y * length_ratios[1]), 0))
                kzz = int(round(float(kz) / (z * length_ratios[2]), 0))
                lim_list = [-2, -1, 0, 1, 2]
                for i in lim_list:
                    for j in lim_list:
                        for k in lim_list:
                            i = check_k_dec(kxx, i)
                            j = check_k_dec(kyy, j)
                            k = check_k_dec(kzz, k)

                            xyz_list.append((x, y, z))
                            k_list.append((kxx+i, kyy+j, kzz+k))

                            cmd = LMarg + \
                                ' -vnkx={} -vnky={} -vnkz={} -vrfile=1 '.format(
                                    kxx+i, kyy+j, kzz+k)
                            cmd_write_to_file(
                                cmd,  unit_cell_to_extend + '_' + filename)
                            e_tot, e_bs, e_pp, mf, nb = get_mesh_criteria(
                                unit_cell_to_extend + '_' + filename)

                            e_list.append(e_tot)
                            bs_list.append(e_bs)
                            pp_list.append(e_pp)
                            mf_list.append(mf)
                            print('\n Total Energy: %s' % (e_tot))
                            print('\n Band Structure Energy: %s' % (e_bs))
                            print('\n Pair potential energy: %s' % (e_pp))
                            print('\n Maximum Force: %s\n' % (mf))

    min_force_ind = np.argmin(np.array(mf))
    best_k = k_list[min_force_ind]
    best_xyz = xyz_list[min_force_ind]
    print("Best Structure:", best_xyz)
    print("Best k mesh:", best_k)
    return best_k, best_xyz


def lattice_parameters_vs_k_mesh(LMarg, minimiserf='Nelder-Mead', plot=True, data=False):
    """Function to evaluate how the lattice parameters change with k points. """
    nk = np.arange(4, 30, 2)
    a_lp = np.zeros(nk.shape)
    c_lp = np.zeros(nk.shape)
    e_lp = np.zeros(nk.shape)
    if data is False:
        for i, k in enumerate(nk):
            global LMarg_a, rmaxh_hcp, rmaxh_omega
            rmaxh_hcp = 20.

            LMarg_a = LMarg + ' -vnkx={} -vnky={} -vnkz={} '.format(k, k, k)

            a_lp[i], c_lp[i], e_lp[i] = get_min_lp(
                minimiserf='Nelder-Mead', phase='hcp')
            print("Minimum lattice parameters for Nelder-Mead:\n   a = {}\n   c = {}\n   e = {} ".format(
                a_lp[i], c_lp[i], e_lp[i]))

            # a_s, c_s, e_s = get_min_lp( minimiserf='SLSQP', phase='hcp')
            # print("Minimum lattice parameter for SLSQP:\n   a = {}\n   c = {}\n   e = {}  ".format(  a_s, c_s, e_s  ) )

        fname = open('a_hcp_vs_nk_rmaxh_large.pkl', 'wb')
        pickle.dump(a_lp, fname)
        fname.close()
        fname = open('c_hcp_vs_nk_rmaxh_large.pkl', 'wb')
        pickle.dump(c_lp, fname)
        fname.close()

        fname = open('e_hcp_vs_nk_rmaxh_large.pkl', 'wb')
        pickle.dump(e_lp, fname)
        fname.close()
    else:
        a_lp, c_lp, e_lp = data

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(nk, a_lp, 'bo', label=r'$a_{hcp}$ vs nk')
        ax.plot([np.min(nk), np.max(nk)], [5.57, 5.57],
                'r--', label=r'$a_{hcp exp}$')
        ax.set_title(r' Variation of $a_{hcp}$ with nk ')
        ax.set_xlabel(r' nk ')
        ax.set_ylabel(r' a$_{hcp}$ (Bohr) ')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig.savefig('a_hcp_vs_nk_large_rmaxh.png')

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(nk, c_lp, 'go', label=r'$c_{hcp}$ vs nk')
        ax1.plot([np.min(nk), np.max(nk)], [8.852, 8.852],
                 'r--', label=r'$c_{hcp exp}$')
        ax1.set_title(r' Variation of c$_{hcp}$ with nk ')
        ax1.set_xlabel(r' nk ')
        ax1.set_ylabel(r' c$_{hcp}$ (Bohr) ')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig1.savefig('c_hcp_vs_nk_large_rmaxh.png')

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(nk, e_lp, 'mo', label=r'$e_{tot}$ vs nk')
        ax2.set_title(
            r' Variation of total energy of hcp at minimum lattice parameters with nk ')
        ax2.set_xlabel(r' nk ')
        ax2.set_ylabel(r' Total Energy (Ryd) ')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig2.savefig('e_hcp_vs_nk_large_rmaxh.png')

        plt.show()


LMarg = 'tbe ti -vhcp=1 '
args = ' '
rmh_check_file = 'checkrmaxh'
rmx_name = 'rmaxh'
nmax = 12
rmaxh_l = 0.5
rmaxh_u = 20.

############################     tbe     #################################
## Objective = 1010


a_hcp, c_hcp = 5.503722649302711,  8.735375609762187

a_hcp = 5.5125
c_hcp = 8.8090


a_hcp = 5.5118
c_hcp = 8.7970

ahcp = 5.5118
chcp = 8.7970

tbe_command = 'tbe ti -vhcp=1 '
#                + 'fddtt=0.4635719996209968 qddstt=0.7047980050392473 b0tt=91.2550281067963 p0tt=1.2499569879391594 b1tt=-15.827072545564256 p1tt=0.965755869608115 ndt=1.9350513317847982 '
#                + '-vfddtt=0.4589353025120264 -vqddstt=0.7118332818816158 -vb0tt=88.56120896145649 -vp0tt=1.2803107031586398 '
#                + '-vb1tt=-14.214028253751056 -vp1tt=0.9885838205722217 -vndt=1.9323918054338258 '
# + ' -vewtol=1d-14 ')  # -vrmaxh={:<12.8f} -vforces=1 '.format( 6.7 ) )

# -vrmaxh={:<12.8f} -vahcp={} -vq={} '.format(  6.7, a_hcp, c_hcp / a_hcp  ) )
relax_command = (
    '-vrelax=5 -vgtol=1d-4 -vxtol=1d-4 -vhess=100 -vforces=1 -vtetra=0 ')


# See how the number of k-points affects the lattice parameters.

# fname = open('a_hcp_vs_nk.pkl', 'rb' )
# a_lp = pickle.load( fname ); fname.close()

# fname = open('c_hcp_vs_nk.pkl', 'rb' )
# c_lp = pickle.load( fname ); fname.close()

# fname = open('e_hcp_vs_nk.pkl', 'rb' )
# e_lp = pickle.load( fname ); fname.close

# data = (a_lp, c_lp, e_lp)


# data = False
# lattice_parameters_vs_k_mesh( tbe_command, minimiserf='Nelder-Mead', plot=True, data=data)


global LMarg_a, rmaxh_hcp, rmaxh_omega
rmaxh_hcp = 20.
k = 8
LMarg_a = tbe_command + \
    ' -vrfile=0 -vnkx={} -vnky={} -vnkz={} '.format(k, k, k)

#a_lp, c_lp, e_lp = get_min_lp( minimiserf='Nelder-Mead', phase='hcp')
#print("Minimum lattice parameters for Nelder-Mead:\n   a = {}\n   c = {}\n   e = {} ".format( a_lp, c_lp, e_lp ) )


# Check how the energy varies with different values of rmaxh.
#energy_vs_rmaxh(tbe_command, args, rmx_name, rmaxh_l, rmaxh_u)
ext = 'ti'
k0 = (30, 30, 30)


plat0 = np.array([[0.,        -1.,     0.],
                  [3**(0.5)/2.,  0.5,     0.],
                  [0.,            0., chcp/ahcp]])

plat1 = np.array([[3**(0.5),      0.,     0.],
                  [0.,         1.,     0.],
                  [0.,          0., chcp/ahcp]])

x_lim = (1, 2)
y_lim = (1, 2)
z_lim = (6, 11)

unit_cell_reference = 'hcp_primitive_uc'
unit_cell_to_extend = 'basal_right_ca'

# kp, xyz = find_best_k_mesh_and_structure(LMarg, ext, k0, plat0, plat1,
#                                          x_lim, y_lim, z_lim,
#                                          unit_cell_reference,
#                                          unit_cell_to_extend)


# Best Structure: (1, 1, 6)
# Best k mesh: (8, 28, 3)


# ahcp=5.5125; chcp=8.8090

relax_command = ' '

get_all_gamma_surfaces(tbe_command + relax_command + ' -vrfile=1 ',  # ,  a_lp,  c_lp / a_lp ),
                       a_hcp, c_hcp)

"""
Particularly good objective functions:
    
obj = 1010.9506173873626
a_hcp, c_hcp = 5.503722649302711,  8.735375609762187
c11, c33, c44, c12, c13 = 179.25051281715676, 186.57059141284216, 57.79417272253099, 61.63177733062568, 73.3759135930488
a_omega, c_omega, u_omega = 8.549705808402102, 5.3662812146761665, 1.1735550699737716
dE_omega_hcp = -2.036551666666692
a_bcc = 6.191084422483171
bandwidth = 0.37889999999999996

-vfddtt=0.4589353025120264 -vqddstt=0.7118332818816158 -vb0tt=88.56120896145649 -vp0tt=1.2803107031586398 -vb1tt=-14.214028253751056 -vp1tt=0.9885838205722217 -vndt=1.9323918054338258 






994.1611682760212 
5.534996145438653 8.795578260179248 185.4519339741608 189.83557009502255 58.51382565329746 63.67107055703468 75.35583248028877 8.612574932412254 5.508770505034845 1.0000286951961137 1.6837849999999932 6.2214190181306215 0.3883    -vfddtt=0.4685290447605113 -vqddstt=0.7066885641073046 -vb0tt=80.43424453244602 -vp0tt=1.2150123002234887 -vb1tt=-15.436019114183193 -vp1tt=0.943122985594374 -vndt=1.9353141306781194

"""
