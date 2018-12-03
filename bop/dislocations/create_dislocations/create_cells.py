import types
from matplotlib import rc
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import matplotlib.pyplot as pp
import numpy as np
import scipy as sci
import os
rcParams["figure.figsize"] = 4, 3
rcParams["font.family"] = "serif"
rcParams["font.size"] = 8
rcParams["font.serif"] = ["DejaVu Serif"]
rc("text", usetex=True)
sci.set_printoptions(linewidth=200, precision=4)


class Disl_supercell:
    def __init__(self, unit_cell, lengths, alat, plat, nxyz, ninert=(20., 30.), disl=None, n_disl=1,
                 rcore=None, rcphi=0., rotation=np.eye(3), species="Ti", screw=True, pure=True,
                 cwd='./', output_path='./generated_dislocations', output='bop',
                 filename='cell_gen', geometry='circle', labels=['tih', 'hcp']):

        self.disl = disl
        self.alat = alat
        self.plat = plat
        self.nxyz = nxyz
        self.pure = pure
        self.screw = screw
        self.rcphi = rcphi
        self.n_disl = n_disl
        self.ninert = ninert
        self.labels = labels
        self.species = species
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
                # Single dislocation in cell
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
                self.rcore[3] = 0.5 * self.final_lengths + disl_coord4
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
            print("In self.geometry circle statement")
            self.inert_rad1, self.inert_rad2 = ninert

            def inert_cond(self, i, j, k):
                i -= 0.5 * self.final_lengths[0]
                j -= 0.5 * self.final_lengths[1]
                k -= 0.5 * self.final_lengths[2]
                r = np.sqrt(i**2 + j**2)
                c0 = r > self.inert_rad1
                c1 = r > self.inert_rad2
                if c1:
                    c0 = 'out of bounds'
                return c0
            self.inert_cond = types.MethodType(inert_cond, self)
            print("Trialling out inert_cond")
            print("self.inert_cond(1,2,3)", self.inert_cond(1, 2, 3))
        elif self.geometry == 'square':
            ninertx, ninerty, ninertz = self.ninert

            def inert_cond(self, i, j, k):
                c0 = i < ninertx[0]*self.lengths[0] or j < ninerty[0] * \
                    self.lengths[1] or k < ninertz[0]*self.lengths[2]
                c1 = i > ninertx[1]*self.lengths[0] or j > ninerty[1] * \
                    self.lengths[1] or k > ninertz[1]*self.lengths[2]
                return c0 or c1
        else:
            def inert_cond(self, i, j, k):
                return False

        self.inert_cond = types.MethodType(inert_cond, self)

    ########################   Convert file to XYZ   ###############################

    def convert_file_to_xyz(self, plat, atom_pos, species, filename, iterations=0):

        # print(plat, atom_pos, species)

        if type(atom_pos) == tuple:
            atom_pos, inert_atom_pos = atom_pos
            if inert_atom_pos is not None:
                n_inert = len(inert_atom_pos)
            else:
                n_inert = 0
            n_at = len(atom_pos)
            n_tot = n_at + n_inert
        else:
            n_tot = len(atom_pos)
            n_at = n_tot

        n_cell_filename = filename + "_convert.xyz"

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

        out_xyz_file.close()

    def write_bop_file(self,  tot_atom_pos, file_ext=""):
        convert_to_ryd = 1.8897259886
        species = self.species
        alat = self.alat
        plat = self.plat
        l = self.lengths
        flen = self.final_lengths
        file_ext = "cell" + file_ext
        atom_pos, inert_atom_pos = tot_atom_pos

        cell_file = file_ext + ".in"
        xyz_file = file_ext + ".xyz"

        out_file = open(cell_file, mode='w+')
        out_xyz_file = open(xyz_file, mode='w+')

        n_at = len(atom_pos)
        if inert_atom_pos is not None:
            n_inert = len(inert_atom_pos)
        else:
            n_inert = 0
        n_atoms_tot = n_at + n_inert

        plat_inv = np.linalg.inv(plat)

        plat_b = plat / flen

        out_file.write(" " + self.labels[0] + '  ' + self.labels[1] + "\n")
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
        for i in range(self.n_disl):
            out_file.write("   0    ! gfbcon\n")
            out_file.write("   0.0  ! radiusi\n")
            out_file.write("   -100.0    100.0    -100.0  100.0\n")

        out_file.close()

        atom_pos_bop = np.zeros(atom_pos.shape)

        for i, a in enumerate(atom_pos):

            atom_pos_bop[i, 0] = (plat.dot(plat_inv.dot(atom_pos[i])))[0]
            atom_pos_bop[i, 1] = (plat.dot(plat_inv.dot(atom_pos[i])))[1]
            atom_pos_bop[i, 2] = (plat.dot(plat_inv.dot(atom_pos[i])))[2]

        if n_inert > 0:
            inert_atom_pos_bop = np.zeros(inert_atom_pos.shape)
            for i, a in enumerate(inert_atom_pos):
                inert_atom_pos_bop[i, 0] = (
                    plat.dot(plat_inv.dot(inert_atom_pos[i])))[0]
                inert_atom_pos_bop[i, 1] = (
                    plat.dot(plat_inv.dot(inert_atom_pos[i])))[1]
                inert_atom_pos_bop[i, 2] = (
                    plat.dot(plat_inv.dot(inert_atom_pos[i])))[2]
            else:
                inert_atom_pos_bop = None

        self.convert_file_to_xyz(
            plat, (atom_pos_bop, inert_atom_pos_bop), species, file_ext)

    def write_site_file(self, tot_atom_pos, file_ext="site"):
        convert_to_ryd = 1.8897259886
        species = self.species
        alat = self.alat

        if self.geometry == 'circle':
            self.plat[0] = np.array([self.inert_rad2, 0., 0.])
            self.plat[1] = np.array([0., self.inert_rad2, 0.])
            plat = self.plat
        else:
            plat = self.plat

        atom_pos, inert_atom_pos = tot_atom_pos
        if inert_atom_pos is not None:
            n_inert = len(inert_atom_pos)
            inert_atom_pos = (inert_atom_pos / alat).round(12)
        else:
            n_inert = 0
        n_at = len(atom_pos)

        atom_pos = (atom_pos / alat).round(12)

        site_file = open(file_ext, mode='w')
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

        tot_atom_pos_xyz = (tot_atom_pos[0] / convert_to_ryd, None)
        if tot_atom_pos[1] is not None:
            tot_atom_pos_xyz = (
                tot_atom_pos[0] / convert_to_ryd, tot_atom_pos[1] / convert_to_ryd)

        self.convert_file_to_xyz(
            plat, tot_atom_pos_xyz, species, 'site_' + file_ext)

    def add_dislocation(self, atoms, axis=2):

        rcores = self.rcore
        rcphi = self.rcphi
        screw = self.screw
        pure = self.pure

        print(rcores)

        # axis is the displacement in z direction for screw dislocation by default.
        # For edge dislocation, axis is the one that is *not* displaced.

        for j in range(self.n_disl):
            if self.n_disl == 1:
                dis = self.disl
                rcore = rcores[0]
            else:
                dis = self.disl[j]
                rcore = rcores[j]
            for position in atoms:
                r1, r2, r3 = tuple(position)
                if screw:
                    r12 = np.sqrt(
                        (r1 - rcore[0]) ** 2 + (r2 - rcore[1]) ** 2)
                    s = dis.u_screw((r12 * np.cos(self.rcphi +
                                                  np.arctan2(r2-rcore[1], r1-rcore[0]))),
                                    (r12 * np.sin(self.rcphi +
                                                  np.arctan2(r2-rcore[1], r1-rcore[0]))))
                    position[axis] += s
                elif pure:
                    # Pure Edge dislocation
                    position[axis - 2] += dis.u_edge(r1 -
                                                     rcore[0], r2 - rcore[1])
                    position[axis - 1] += dis.u_edge(r1 -
                                                     rcore[0], r2 - rcore[1])
        return atoms

    def get_atoms(self):
        l = self.lengths
        uc = self.unit_cell
        atom_counter = 0
        inert_counter = 0
        nx, ny, nz = self.nxyz
        luc = len(self.unit_cell)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for p in range(luc):
                        r1 = (uc[p, 0] + i) * l[0]
                        r2 = (uc[p, 1] + j) * l[1]
                        r3 = (uc[p, 2] + k) * l[2]

                        r1, r2, r3 = tuple(
                            self.rotation.dot(np.array([r1, r2, r3])))
                        print(r1, r2, r3)
                        inert_condition = self.inert_cond(r1, r2, r3)
                        if inert_condition != 'out of bounds':
                            if inert_condition:
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
                                    atoms = np.array(
                                        [r1, r2, r3]).reshape(1, 3)
                                else:
                                    atoms = np.append(atoms, np.array(
                                        [r1, r2, r3])).reshape(atom_counter, 3)
        self.plat = self.plat * self.lengths * np.asarray(self.nxyz)
        if inert_counter == 0:
            inert_atoms = None
        print("plat", self.plat)
        return atoms, inert_atoms

    def write_cell_with_dislocation(self, output='tbe'):

        atoms, inert_atoms = self.get_atoms()
        atoms_with_disl = self.add_dislocation(atoms)
        if inert_atoms is not None:
            inert_with_disl = self.add_dislocation(inert_atoms)
        else:
            inert_with_disl = None
        all_atoms = (atoms_with_disl, inert_with_disl)

        file_ext = "_{}x_{}y_{}z_{}_{}_disl".format(
            self.nxyz[0], self.nxyz[1], self.nxyz[2], self.geometry, self.n_disl)

        if output == 'tbe':
            self.write_site_file(tot_atom_pos=all_atoms, file_ext=file_ext)
        if output == 'bop':
            self.write_bop_file(tot_atom_pos=all_atoms, file_ext=file_ext)
