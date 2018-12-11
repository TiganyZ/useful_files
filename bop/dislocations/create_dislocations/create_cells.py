from write_input_files import WriteFiles as wf
import types
from matplotlib import rc
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import matplotlib.pyplot as pp
import numpy as np
import scipy as sci
import os
import copy
rcParams["figure.figsize"] = 4, 3
rcParams["font.family"] = "serif"
rcParams["font.size"] = 8
rcParams["font.serif"] = ["DejaVu Serif"]
rc("text", usetex=True)
sci.set_printoptions(linewidth=200, precision=4)


class Disl_supercell:
    def __init__(self, unit_cell, lengths, alat, plat, nxyz, ninert=(20., 30.), disl=None, n_disl=1,
                 rcore=None, rcphi=0., rotation=np.eye(3), species="Ti", pure=True, screw=True, disl_axis=2,
                 output='bop', type_plat='square', full_anis=False, in_units_of_len=False,
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
        self.full_anis = full_anis
        self.disl_axis = disl_axis
        self.type_plat = type_plat
        self.final_lengths = lengths * nxyz
        self.in_units_of_len = in_units_of_len
        ##############################################################################
        #######################   Specifying dislocation coords   ####################

        if rcore is None:
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
        #######################   Geometry dependent inert atoms  ####################
        if self.geometry == 'circle':
            #print("In self.geometry circle statement")
            self.inert_rad1, self.inert_rad2 = ninert

            def inert_cond(self, i, j, k):

                i -= 0.5 * self.final_lengths[0]
                j -= 0.5 * self.final_lengths[1]
                k -= 0.5 * self.final_lengths[2]
                ijk = np.array([i, j, k])
                ijk[self.disl_axis] = 0.
                r = np.linalg.norm(ijk)
                c0 = r > self.inert_rad1
                c1 = r > self.inert_rad2
                if c1:
                    c0 = 'out of bounds'
                return c0
            self.inert_cond = types.MethodType(inert_cond, self)

        elif self.geometry == 'square':
            ninertx, ninerty, ninertz = self.ninert
            nx, ny, nz = self.nxyz

            def inert_cond(self, i, j, k):
                c0 = i < ninertx[0] or j < ninerty[0] or k < ninertz[0]
                c1 = i > ninertx[1] or j > ninerty[1] or k > ninertz[1]
                c2 = i > nx or j > ny or k > nz

                if c2:
                    return 'out of bounds'
                else:
                    return c0 or c1
        else:
            def inert_cond(self, i, j, k):
                return False

        self.inert_cond = types.MethodType(inert_cond, self)

    def calculate_dis_pbc(self, u, plat, atoms, disp_atoms, trunc=(20, 20), axis=2,
                          xy=[(0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)]):
        # Default xy is to measure at the corners of the simulation cell.

        # Calculate the error with regards to the non-periodicity of the dislocation dipole displacement
        # Need to do the conditionally convergent sum and then calculate the error in displacement. x
        # plat is the plat of the /supercell/
        # trunc is the limit for the truncation of the sum over periodic images.
         # Recipe to remove the spurious non-periodic part of the displacement field:
         # 1. Evaluate the conditionally convergent sum
         #     u_sum = np.sum( [u(x + Rx, y + Ry) for Rx, Ry in \
         #    [ (plat[0]*i, plat[1]*j) for i, j in zip(range(1,i+1),range(1,j+1))]  )
         #    $u_{z}^{\text{sum}}(\mathbf{r})$, using an arbitrary truncation.
         # 2. "Measure" the linear spurious part of the resulting field, using
         #    u_err = s.dot(r)
         #    equation below, by comparing its values at four points in the
         #    periodic supercell from the equation
         #            u_sum = u_pbc + s.dot(r) + u_0
         #            u_sum_p_ci - u_sum = s.dot(ci)
         #    \[ u_{z}^{\text{err}}(\mathbf{r}) =  \mathbf{s}\cdot\mathbf{r},  \]
         #    \[ u_{z}^{\text{sum}}(\mathbf{r} + \mathbf{c}_{i})  -
         #    u_{z}^{\text{sum}}(\mathbf{r}) = \mathbf{s}\cdot\mathbf{c}_{i}, \]
         #    where $i=1,2$.
         # 3. Finally, subtract the linear term $u_{z}^{\text{err}}(\mathbf{r})$ from
         #    $u_{z}^{\text{sum}}(\mathbf{r})$ to obtain the corrected solution
         #    $u_{z}^{\text{PBC}}(\mathbf{r})$

        us = np.zeros(len(xy))
        usm = np.zeros(len(xy))
        usx = np.zeros(len(xy))
        usxm = np.zeros(len(xy))
        usy = np.zeros(len(xy))
        usym = np.zeros(len(xy))

        for l in range(len(xy)):
            x, y = xy[l]
            x *= plat[axis-2]
            y *= plat[axis-1]
            u_sum, u_sum_m = 0., 0.
            u_sum_cx, u_sum_cxm = 0., 0.
            u_sum_cy, u_sum_cym = 0., 0.
            for i in range(trunc[0]):
                for j in range(trunc[1]):

                    Rx, Ry = plat[axis-2]*i, plat[axis-1]*j
                    u_sum += u(x + Rx, y + Ry)
                    u_sum_m += u(-x + Rx, -y + Ry)

                    Rx, Ry = plat[axis-2]*(i + 1), plat[axis-1]*j
                    u_sum_cx += u(x + Rx, y + Ry)
                    u_sum_cx_m += u(-x + Rx, -y + Ry)

                    Rx, Ry = plat[axis-2]*i, plat[axis-1]*(j + 1)
                    u_sum_cy += u(x + Rx, y + Ry)
                    u_sum_cy_m += u(-x + Rx, -y + Ry)

            us[l] = u_sum
            usx[l] = u_sum_cx
            usy[l] = u_sum_cy
            usm[l] = u_sum_m
            usxm[l] = u_sum_cx_m
            usym[l] = u_sum_cy_m

        # Measure the gradient of the error in the displacement
        s11s21 = usm - us

        u_err_x = usx - us  # = s.dot( c1 ) # This is definitely true
        u_err_y = usy - us  # = s.dot( c2 )

        u_err_x = usx - us
        u_err_y = usy - us

        # c1x, c1y = tuple( plat[ axis - 2  ][:-1] )
        # c2x, c2y = tuple( plat[ axis - 1  ][:-1] )

        s12 = (u_err_x[0] - c1x * u_err_y[0]) / (c2x * c1y - c2y * c1x)
        s11 = (u_err_x[0] - s12 * c1y) / c1x

        s22 = (u_err_x[1] - c1x * u_err_y[1]) / (c2x * c1y - c2y * c1x)
        s21 = (u_err_x[1] - s22 * c1y) / c1x

        s = np.array([[s11, s12],
                      [s21, s22]])

        return s

    def modify_plat_for_strain(self, plat, a, b, axis=2):
        # Area normal to z of the periodic cell
        A0 = np.det(np.outer(plat[axis-2],  plat[axis-2]))
        # Area swept out between dislocation dipoles normal to z of the periodic cell
        A = np.det(np.outer(a,  plat[axis-2]))

        return

    def add_dislocation_anis(self, atoms, axis=2, rotation=None):

        rcores = self.rcore
        rcphi = self.rcphi
        screw = self.screw
        pure = self.pure
        xil = np.zeros((len(atoms), 2))
        print("rcores", rcores)
        new_atom_pos = np.zeros(atoms.shape)
        disp = np.zeros(len(atoms))
        # axis is the displacement in z direction for screw dislocation by default.
        # For edge dislocation, axis is the one that is *not* displaced.

        if rotation is not None:
            for i in range(3):
                self.plat[i] = rotation.dot(self.plat[i])
            print("Rotated the lattice vectors")
            print(self.plat)

        for j in range(self.n_disl):
            if self.n_disl == 1:
                dis = self.disl
                rcphi = self.rcphi
                rcore = rcores[0]
                print("rcore and axis", rcore, rcore[axis-2], rcore[axis-1])
            else:
                dis = self.disl[j]
                rcore = rcores[j][0]
                rcphi = self.rcphi[j]
                print(rcore, rcores)
            if j > 0:
                rotation = None
            for indx, position in enumerate(atoms):
                if rotation is not None:
                    position = rotation.dot(position)
                if screw:
                    r12 = np.sqrt(
                        (position[axis-2] - rcore[axis-2]) ** 2
                        + (position[axis-1] - rcore[axis-1]) ** 2)

                    xi = r12 * np.cos(rcphi + np.arctan2(position[axis-2] - rcore[axis-2],
                                                         position[axis-1] - rcore[axis-1]))

                    yi = r12 * np.sin(rcphi + np.arctan2(position[axis-2] - rcore[axis-2],
                                                         position[axis-1] - rcore[axis-1]))

                    s = dis(xi, yi)
                    xil[indx, :] = np.array([xi, yi])
                    position[axis] += s
                    disp[indx] += s
                    # Displacement only along the axis specified for screw dislocations.

                    new_atom_pos[indx, :] = position
                    atoms[indx, :] = position
                elif pure:
                    # Pure Edge dislocation
                    position[axis - 2] += dis(positon[axis-2] - rcore[axis-2],
                                              positon[axis-1] - rcore[axis-1], k=0)
                    position[axis - 1] += dis(positon[axis-2] - rcore[axis-2],
                                              positon[axis-1] - rcore[axis-1], k=1)
                    new_atom_pos[indx, :] = position
        minxil = np.argmin(np.sum(xil**2, axis=1))
        print(minxil)
        print("min xil ", xil[minxil], atoms[minxil])
        return new_atom_pos, disp

    def add_dislocation(self, atoms, axis=2, rotation=None):

        rcores = self.rcore
        rcphi = self.rcphi
        screw = self.screw
        pure = self.pure

        atom_copy = copy.copy(atoms)
        new_atom_pos = np.zeros(atoms.shape)
        print(rcores)

        # axis is the displacement in z direction for screw dislocation by default.
        # For edge dislocation, axis is the one that is *not* displaced.

        for j in range(self.n_disl):
            if self.n_disl == 1:
                dis = self.disl
                pure = dis.pure
                screw = dis.screw
                rcphi = self.rcphi
                rcore = rcores[0]
            else:
                dis = self.disl[j]
                rcore = rcores[j][0]
                rcphi = self.rcphi[j]
                pure = dis.pure
                print(rcore, rcores)
                screw = dis.screw
            for indx, position in enumerate(atoms):
                if rotation is not None:
                    position = rotation.dot(position)
                if screw:
                    r12 = np.sqrt(
                        (position[axis-2] - rcore[axis-2]) ** 2
                        + (position[axis-1] - rcore[axis-1]) ** 2)

                    s = dis.u_screw((r12 * np.cos(rcphi +
                                                  np.arctan2(position[axis-2] - rcore[axis-2],
                                                             position[axis-1] - rcore[axis-1]))),
                                    (r12 * np.sin(rcphi +
                                                  np.arctan2(position[axis-2] - rcore[axis-2],
                                                             position[axis-1] - rcore[axis-1]))))
                    position[axis] += s
                    new_atom_pos[indx, :] = position
                elif pure:
                    # Pure Edge dislocation
                    position[axis - 2] += dis.u_edge(
                        positon[axis-2] - rcore[axis-2], positon[axis-1] - rcore[axis-1])
                    position[axis - 1] += dis.u_edge(
                        positon[axis-2] - rcore[axis-2], positon[axis-1] - rcore[axis-1])
                    new_atom_pos[indx, :] = position
        print(atom_copy-new_atom_pos)
        return new_atom_pos

    def get_atoms(self):
        l = self.lengths
        uc = self.unit_cell
        atom_counter = 0
        inert_counter = 0
        nx, ny, nz = self.nxyz
        luc = len(self.unit_cell)
        print("lengths\n", self.lengths)
        # , self.plat * self.lengths * np.asarray(self.nxyz))
        print("plat\n", self.plat, '\n')
        print("unit cell\n", uc)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for p in range(luc):

                        if not self.in_units_of_len:
                            dr = i * self.plat[0] * self.lengths[0] + \
                                j * self.plat[1] * self.lengths[1] + \
                                k * self.plat[2] * self.lengths[2]
                            r = dr + uc[p, :] * self.alat
                            r1, r2, r3 = tuple(r)
                        else:
                            print("In units of plat lengths")
                            r1 = (uc[p, 0] + i) * l[0]
                            r2 = (uc[p, 1] + j) * l[1]
                            r3 = (uc[p, 2] + k) * l[2]
                            r = np.array([r1, r2, r3])

                        r1, r2, r3 = tuple(self.rotation.dot(r))

                        if self.geometry == 'square':
                            inert_condition = self.inert_cond(i, j, k)
                        else:
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

        self.plat[0] = nx * self.plat[0] * self.lengths[0]
        self.plat[1] = ny * self.plat[1] * self.lengths[1]
        self.plat[2] = nz * self.plat[2] * self.lengths[2]

        if inert_counter == 0:
            inert_atoms = None
        print("plat", self.plat)
        return atoms, inert_atoms

    def write_cell_with_dislocation(self, output='tbe', add_name="", axis=2, rotation=None):

        atoms, inert_atoms = self.get_atoms()

        if self.full_anis:
            atoms_with_disl = self.add_dislocation_anis(
                atoms, axis=axis, rotation=rotation)
        else:
            atoms_with_disl = self.add_dislocation(
                atoms, axis=axis, rotation=rotation)

        if inert_atoms is not None:
            if self.full_anis:
                inert_with_disl = self.add_dislocation_anis(
                    inert_atoms, axis=axis, rotation=rotation)
            else:
                inert_with_disl = self.add_dislocation(
                    inert_atoms, axis=axis, rotation=rotation)
        else:
            inert_with_disl = None

        all_atoms = (atoms_with_disl, inert_with_disl)

        file_ext = "_{}x_{}y_{}z_{}_{}_disl".format(
            self.nxyz[0], self.nxyz[1], self.nxyz[2], self.geometry, self.n_disl) + add_name

        if output == 'tbe':
            wfiles = wf(filename="site.ti" + file_ext,  cwd='./',
                        output_path='./generated_dislocations', write_to_dir=True)
            wfiles.write_site_file(
                all_atoms, self.species, self.alat, self.plat)
        if output == 'bop':
            wfiles = wf(filename="cell.in" + file_ext,  cwd='./',
                        output_path='./generated_dislocations', write_to_dir=True)
            wfiles.write_bop_file(self.plat, all_atoms, self.species,
                                  self.final_lengths, n_disl=self.n_disl)
