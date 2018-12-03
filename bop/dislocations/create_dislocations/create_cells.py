import os
import scipy as sci
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


class create_disl_supercell:
    def __init__(self, unit_cell, lengths, alat, plat, nxyz, ninert, disl, n_disl=1,
                 rcore=None, rcphi=0., rotation=np.eye(3), species="Ti",
                 cwd='./', output_path='./generated_dislocations', output='bop',
                 filename='cell_gen', geometry='circle', labels=['tih', 'hcp']):

        self.disl = disl
        self.alat = alat
        self.plat = plat
        self.n_disl = n_disl
        self.labels = labels
        self.lengths = lengths
        self.rotation = rotation
        self.geometry = geometry
        self.unit_cell = unit_cell
        self.final_lengths = lengths * nxyz

        ##############################################################################
        #######################   Specifying dislocation coords   ####################

        if rcore == None:
            self.rcore = np.zeros((n_disl, 3))
            if n_disl == 1:
                # Single dislocation in cen
                self.rcore[0] = 0.5 * self.final_lengths
            elif n_disl == 2:
                # Dipole
                self.rcore[0] = 0.5 * self.final_lengths - \
                    0.5 * np.array([self.alat, 0., 0.])
                self.rcore[1] = 0.5 * self.final_lengths + \
                    0.5 * np.array([self.alat, 0., 0.])
            elif n_disl == 3:
                # Triangular configuration
                disl_coord1 = np.array(
                    [-0.5, -1./(2 * 3**(0.5)), 0.]) * self.alat
                disl_coord2 = np.array(
                    [0.0,  1./(3**(0.5)),    0.]) * self.alat
                disl_coord3 = np.array(
                    [0.5, -1./(2 * 3**(0.5)), 0.]) * self.alat
                self.rcore[0] = 0.5 * self.final_lengths + disl_coord1
                self.rcore[1] = 0.5 * self.final_lengths + disl_coord2
                self.rcore[2] = 0.5 * self.final_lengths + disl_coord3
            elif n_disl == 4:
                # Quadrupole
                disl_coord1 = np.array([-0.5, -0.5, 0.]) * self.alat
                disl_coord2 = np.array([0.5, -0.5, 0.]) * self.alat
                disl_coord3 = np.array([0.5,  0.5, 0.]) * self.alat
                disl_coord4 = np.array([-0.5, 0.5, 0.]) * self.alat
                self.rcore[0] = 0.5 * self.final_lengths + disl_coord1
                self.rcore[1] = 0.5 * self.final_lengths + disl_coord2
                self.rcore[2] = 0.5 * self.final_lengths + disl_coord3
                self.rcore[3] = 0.5 * self.final_lengths + disl_coord3
        else:
            self.rcore = rcore

        ##############################################################################
        #######################   Write output to directory  #########################

        self.cwd = cwd
        if os.path.isdir(output_path) is False:
            os.mkdir(output_path)

        try:
            os.chdir(output_path)
            print("Changed directory from {} to {}".format(cwd, output_path))
        except TypeError:
            print("Could not change directory from {} to {}".format(cwd, output_path))

        ##############################################################################
        #######################   Geometry dependent inert atoms  ####################
        if self.geometry == 'circle':
            self.inert_rad1, self.inert_rad2 = ninert

            def inert_cond(i, j, k):
                r = np.sqrt(i**2 + j**2 + k**2)
                c0 = r > self.inert_rad1
                c1 = r > self.inert_rad2
                if c1:
                    c0 = None
                return c0
        elif self.geometry == 'square':
            def inert_cond(i, j, k):
                c0 = i < ninertx[0] or j < ninerty[0] or k < ninertz[0]
                c1 = i > ninertx[1] or j > ninerty[1] or k > ninertz[1]
                return c0 or c1
        else:
            def inert_cond(i, j, k):
                return False

    def add_dislocation(self, disl=self.disl, atoms=self.atoms, rcore=self.rcore, n_disl=self.n_disl,
                        rcphi=self.rcphi, screw=True, edge=False, plot=False, axis=2):

        # axis is the displacement in z direction for screw dislocation by default.
        # For edge dislocation, axis is the one that is *not* displaced.

        for j in range(n_disl):
            if n_disl == 1:
                dis = disl
            else:
                dis = self.disl[j]
            for position in atoms:
                r1, r2, r3 = tuple(position)
                if screw:
                    r12 = np.sqrt(
                        (r1 - rcore[0]) ** 2 + (r3 - rcore[2]) ** 2)
                    s = self.dis.u_screw((r12 * np.cos(self.rcphi +
                                                       np.arctan2(r3-rcore[2], r1-rcore[0]))),
                                         (r12 * np.sin(self.rcphi +
                                                       np.arctan2(r3-rcore[2], r1-rcore[0]))))
                    position[axis] += s
                elif edge:
                    position[axis - 2] += self.dis.u_edge(r1 -
                                                          rcore[0], r2 - rcore[1])
                    position[axis - 1] += self.dis.u_edge(r1 -
                                                          rcore[0], r2 - rcore[1])
        return atoms

    def get_atoms(self, l=self.lengths):
        atom_counter = 0
        inert_counter = 0
        atoms = np.zeros((len(self.unit_cell), 3))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for p in range(luc):
                        r1 = (uc[p, 0] + i) * l[0]
                        r2 = (uc[p, 1] + j) * l[1]
                        r3 = (uc[p, 2] + k) * l[2]

                        r1, r2, r3 = tuple(
                            self.add_dislocation(atoms=(self.rotation.dot(np.array([r1, r2, r3]))).reshape(1, 3)))

                        inert_condition = self.inert_cond(r1, r2, r3)

                        if inert_condition and inert_condition is not None:
                            inert_counter += 1
                            if inert_counter == 1:
                                inert_atoms = np.array(
                                    [r1, r2, r3]).reshape(1, 3)
                            else:
                                inert_atoms = np.append(inert_atoms, np.array(
                                    [r1, r2, r3])).reshape(inert_counter, 3)
                        if not inert_condition:
                            atom_counter += 1
                            if atom_counter == 1:
                                atoms = np.array([r1, r2, r3]).reshape(1, 3)
                            else:
                                atoms = np.append(atoms, np.array(
                                    [r1, r2, r3])).reshape(atom_counter, 3)

        return atoms, inert_atoms

    def write_bop_cell(self, atoms=self.atoms, inert_atoms=None):

        l = self.lengths
        flen = self.final_lengths

        file_ext = "_{}x_{}y_{}z_{}_{}_disl".format(
            self.nxyz[0], self.nxyz[1], self.nxyz[2], self.geometry, self.ndisl)

        cell_file = self.filename + file_ext + ".in"
        xyz_file = self.filename + file_ext + ".xyz"

        out_file = open(cell_file, mode='w+')
        out_xyz_file = open(xyz_file, mode='w+')

        out_file.write(" " + self.labels[0] + '  ' + self.labels[1] + "\n")
        out_file.write(" 1.0 0.0 0.0\n")
        out_file.write(" 0.0 1.0 0.0\n")
        out_file.write(" 0.0 0.0 1.0\n")
        out_file.write(" len " + str(flen[0])
                       + " " + str(flen[1])
                         + " " + str(flen[2]) + "\n")
        out_file.write(" latpar 1.0\n")
        out_file.write(" nd " + str(n_atoms) + "\n")

        out_xyz_file.write(str(n_atoms_tot) + "\n")
        out_xyz_file.write('Lattice=" ' + str(flen[0]) + ' 0.0 0.0   0.0 ' + str(
            flen[1]) + ' 0.0   0.0 0.0 ' + str(flen[2]) + '" Properties=species:S:1:pos:R:3 \n')

        for ind, a in enumerate([atoms, inert_atoms]):
            if a is not None:
                for i in a:
                    r1, r2, r3 = tuple(i)
                    out_file.write(" " + self.species
                                   + " {:<12.10f} {:<12.10f} {:<12.10f} ".format(
                                       r1 / flen[0], r2 / flen[1], r3 / flen[2])
                                   + " 0.0 0.0"
                                   + " \n")
                    if ind > 0:
                        spec = self.species
                    else:
                        spec = self.species + "n"
                    out_xyz_file.write(" " + spec
                                       + " {:<12.10f} {:<12.10f} {:<12.10f} ".format(
                                           r1, r2, r3)
                                       + " \n")

        out_file.write(" ninert " + str(ninert) + "\n")
        out_file.write(out_inert_file.read())
        out_inert_file.close()

        out_file.write("\n nullheight       0.000000000000\n\n")

        out_file.write(" kspace_conf\n")
        out_file.write("   0 t         ! symmode symetry_enabled\n")
        out_file.write("   12 10 12    ! vnkpts\n")
        out_file.write("    t t t       ! offcentre\n\n")

        out_file.write("dislocation_conf\n")
        for i in range(self.n_disl):
            out_file.write("   0    ! gfbcon\n")
            out_file.write("   0.0  ! radiusi\n")
            out_file.write("   -100.0    100.0    -100.0  100.0\n")

        out_file.close()
        out_xyz_file.close()
