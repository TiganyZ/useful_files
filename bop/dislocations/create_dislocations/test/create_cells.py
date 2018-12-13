from write_input_files import WriteFiles as wf
import types
import functools
from matplotlib import rc
from matplotlib import cm
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
                 output='bop', type_plat='square', full_anis=False, in_units_of_len=False, b=None,
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

        if b is None:
            self.b = alat * np.array([0., 0., 1.])
        else:
            self.b = b

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

    def get_xy_dis(self, xt, yt, rcorei, rcphi):
        axis = self.disl_axis

        xi = xt + yt
        xt, yt = xi[axis-2], xi[axis-1]
        r12 = np.sqrt(
            (xt - rcorei[axis-2]) ** 2 + (yt - rcorei[axis-1]) ** 2)
        xi = r12 * np.cos(rcphi + np.arctan2(xt - rcorei[axis-2],
                                             yt - rcorei[axis-1]))
        yi = r12 * np.sin(rcphi + np.arctan2(xt - rcorei[axis-2],
                                             yt - rcorei[axis-1]))
        return xi, yi

    def get_u_sum_term(self, plat, axis, i, j, dis, xt, yt, rcore, rcphi, edge):
        Rx, Ry, Rz = tuple(plat[axis-2]*i + plat[axis-1]*j)
        s = 0
        for ind, u in enumerate(dis):
            x, y = self.get_xy_dis(xt, yt, rcore[ind], rcphi[ind])
            if edge:
                s += u(x + Rx, y + Ry, k=coord)
            else:
                st = u(x + Rx, y + Ry)
                print("st", st)
                s += st
        return s

    def obtain_u_pbc(self, dis, plat, atoms, disp, rcore, rcphi,
                     trunc=(20, 20), axis=2, coord=2, edge=False,
                     xy=[(1.0, 0.0), (1.0, 1.0), (0.0, 0.0), (0.0, 1.0)], plot=True):
        # This routine corrects a particular component of the displacement field, thus making it periodic.

        us = np.zeros(len(xy))
        usx = np.zeros(len(xy))
        usy = np.zeros(len(xy))

        for l in range(len(xy)):
            x, y = xy[l]
            xt = x * plat[axis-2]
            yt = y * plat[axis-1]

            u_sum = 0.
            u_sum_cx = 0.
            u_sum_cy = 0.
            for i in range(trunc[0]):
                for j in range(trunc[1]):

                    u_sum += self.get_u_sum_term(plat, axis,
                                                 i,     j,     dis, xt, yt, rcore, rcphi, edge)

                    u_sum_cx += self.get_u_sum_term(plat, axis,
                                                    i + 1, j,     dis, xt, yt, rcore, rcphi, edge)

                    u_sum_cy += self.get_u_sum_term(plat, axis,
                                                    i,     j + 1, dis, xt, yt, rcore, rcphi, edge)

            us[l] = u_sum
            usx[l] = u_sum_cx
            usy[l] = u_sum_cy

        # Measure the gradient of the error in the displacement

        u_err_x = usx - us  # = s.dot( c1 ) # This is definitely true
        u_err_y = usy - us  # = s.dot( c2 )

        print("\nu_sum(r)")
        print(us)
        print("u_sum(r + cx)")
        print(usx)
        print("u_sum(r + cy)")
        print(usy)
        print("S.cx")
        print(u_err_x)
        print("S.cy")
        print(u_err_y)

        c1x, c1y, c1z = tuple(plat[axis - 2])
        c2x, c2y, c2z = tuple(plat[axis - 1])

        sy = (u_err_y * c1x - u_err_x) / (c2y * c1x - c1y * c2x)
        sx = (u_err_x - sy * c1y) / c1x

        sx = (c2y * u_err_x - c1y * u_err_y) / (c1x * c2y - c2x * c1y)
        sy = (-c2x * u_err_x + c1x * u_err_y) / (c1y * c2x - c2y * c1x)

        # print("Before the average of the S matrix")
        # print(s11s21)
        # print(s12s22)
        # s11, s21 = np.mean( s11s21)
        # s12, s22 = np.mean( s12s22)

        # s = np.array([[s11, s12],
        #               [s21, s22]])

        # print(" After the average of the S matrix")
        # print(s)

        s = np.array([sx, sy])
        s = np.mean(s, axis=1)
        print("Vector s")
        print(s)

        print("changing the displacement so it is periodic")
        u_err_arr = np.zeros(len(atoms))
        new_atoms = np.zeros(atoms.shape)
        new_disp = np.zeros(atoms.shape)
        for i, position in enumerate(atoms):
            # print("rcore", rcore)
            u_err = s.dot((position - sum(rcore)/len(rcore))[:-1])
            #print("u_err", u_err)
            u_err_arr[i] = u_err
            disp[i, coord] -= u_err
            new_disp[i, :] = disp[i, :]
            position[coord] -= u_err
            new_atoms[i, :] = position

        if plot:

            centred_atoms = atoms - sum(rcore)/len(rcore)
            x = centred_atoms[:, 0]
            y = centred_atoms[:, 1]
            z = centred_atoms[:, 2]
            # x = atoms[:, axis-2 ] - rcore
            # y = atoms[:, axis-1 ]

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            z = u_err
            ax.set_title(r"$u_{err}$")
            ax.scatter(x, y, z)
            plt.show()

        print(atoms - new_atoms)

        if plot and self.n_disl > 1:
            self.plot_displacement(new_atoms, new_disp, rcore, axis)

        return new_atoms

    def modify_plat_for_strain(self, axis=2):
        # With introduction of dislocation dipole the lattice vectors have to be modified
        # to account for the elastic strain that would counteract plastic strain, with fixed periodic lattice vectors.

        full_plat_disl = 2 * (sum(self.rcore) / self.n_disl)
        a = self.rcore[1] - self.rcore[0]
        b = self.b
        # Area of plane containing the dislocations, to bounds of the periodic cell
        A0 = np.linalg.norm(np.cross(full_plat_disl,  self.plat[axis]))

        # Area swept out between dislocation dipoles
        normal = np.cross(a,  self.plat[axis])
        n = normal / np.linalg.norm(normal)
        A = np.linalg.norm(np.cross(a,  self.plat[axis]))
        # In direction of normal
        tilt = b * (A / A0)
        print("tilt", tilt)
        print("unit normal", n)

        for i, p in enumerate(self.plat):
            print("pre-strain plat", i)
            print(p)
            dp = p * n * np.linalg.norm(abs(tilt))
            self.plat[i] = p + dp
            print("New plat")
            print(p)

    def add_dislocation_anis(self, atoms, axis=2, rotation=None, plotdis=True):

        rcores = self.rcore
        rcphi = self.rcphi
        screw = self.screw
        pure = self.pure
        xil = np.zeros((len(atoms), 2))
        print("rcores", rcores)
        new_atom_pos = np.zeros(atoms.shape)
        disp = np.zeros((len(atoms), 3))
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
                print("rcore and axis", rcore, rcore[axis-2], rcore[axis-1])
            else:
                dis = self.disl[j]
                rcore = rcores[j]
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
                    disp[indx, axis] += s
                    # Displacement only along the axis specified for screw dislocations.

                    new_atom_pos[indx, :] = position
                    atoms[indx, :] = position
                elif pure:
                    # Pure Edge dislocation

                    sx = dis(position[axis-2] - rcore[axis-2],
                             position[axis-1] - rcore[axis-1], k=0)

                    sy = dis(position[axis-2] - rcore[axis-2],
                             position[axis-1] - rcore[axis-1], k=1)

                    position[axis - 2] += sx
                    position[axis - 1] += sy

                    disp[indx, axis-2] += sx
                    disp[indx, axis-1] += sy

                    new_atom_pos[indx, :] = position

        minxil = np.argmin(np.sum(xil**2, axis=1))
        print(minxil)
        print("min xil ", xil[minxil], atoms[minxil])

        if plotdis:
            self.plot_displacement(atoms, disp, rcore, axis)

        if self.n_disl > 1:
            # If there is more than one dislocation
            # for j in range(self.n_disl):
            #     dis   = self.disl[j]
            #     rcore = rcores[j][0]
            #     rcphi = self.rcphi[j]
            #     print(rcore, rcores)
            if screw:
                new_atom_pos = self.obtain_u_pbc(
                    self.disl, self.plat, new_atom_pos, disp, self.rcore, self.rcphi)
            else:
                for i in range(2):
                    new_atom_pos = self.obtain_u_pbc(self.disl,  self.plat, new_atom_pos,
                                                     disp, self.rcore, self.rcphi, edge=True, coord=i)
            # Change the lattice vectors to account for plastic strain of dislocation dipole.
            self.modify_plat_for_strain()
        return new_atom_pos

    def plot_displacement(self, atoms, disp, rcore, axis, fig=None, d3=True, scaledis=False):
        # plot the displacement in each coordinate system
        x = atoms[:, axis-2]  # - rcore[axis-2]
        y = atoms[:, axis-1]  # - rcore[axis-1]

        print("plotting displacement")
        if fig is None:
            fig = plt.figure(figsize=(15, 5))
        if scaledis:
            scale = np.floor(np.log10(np.max(np.absolute(disp))))
        else:
            scale = 0.

        if d3:
            ax = fig.add_subplot(1, 3, 1, projection='3d')
            z = disp[:, 0] / (10 ** scale)
        else:
            ax = fig.add_subplot(1, 3, 1)
            z = disp[:, 0] / (10 ** scale)

        ax.set_title(r"$x$")

        if d3:
            ax.scatter(x, y, z)
            # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
            #                        linewidth=0, antialiased=False)
        else:
            imx = ax.imshow(z, cmap='coolwarm')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            bc = fig.colorbar(imz, cax=cax, format="%1.3f")
            bc.solids.set_edgecolor("face")

        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.set_title(r"$y$")
        z = disp[:, 1] / (10 ** scale)
        if d3:
            ax.scatter(x, y, z)
            # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
            #                        linewidth=0, antialiased=False)
        else:
            imy = ax.imshow(z, cmap='coolwarm')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            bc = fig.colorbar(imy, cax=cax, format="%1.3f")
            bc.solids.set_edgecolor("face")

        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.set_title(r"$z$")
        z = disp[:, 2] / (10 ** scale)
        if d3:
            ax.scatter(x, y, z)
            # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
            #                        linewidth=0, antialiased=False   )
        else:
            imz = ax.imshow(z, cmap='coolwarm')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            bc = fig.colorbar(imz, cax=cax, format="%1.3f")
            bc.solids.set_edgecolor("face")

        fig.suptitle(
            r"Displacement field $\,\times\,10^{" + str(int(-scale)) + "}$")
        plt.show()
        plt.close()

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
