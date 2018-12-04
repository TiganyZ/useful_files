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


class Dislocation:

    def __init__(self, C, b=np.array([1., 0, 0]), a=5.575,
                 pure=True, screw=True, plot=True, hexagonal=True,
                 T=np.array([[1.,    0.,    0.],      # Dislocation Coordinate system
                             [0.,    1.,    0.],
                             [0.,    0.,    1.]])):
        self.C = C
        self.b = b
        self.a = a
        self.pure = pure
        self.screw = screw
        self.plot = plot

        if hexagonal:
            c11, c33, c44, c12, c13, c66 = C

            C = sci.array(
                [
                    [c11,  c12,  c13,  0.,  0.,  0.,  0.,  0.,  0.],
                    [c12,  c11,  c13,  0.,  0.,  0.,  0.,  0.,  0.],
                    [c13,  c13,  c33,  0.,  0.,  0.,  0.,  0.,  0.],
                    [0.,   0.,   0.,  c44, 0.,  0., c44,  0.,  0.],
                    [0.,   0.,   0.,  0., c44,  0.,  0., c44,  0.],
                    [0.,   0.,   0.,  0.,  0., c66,  0.,  0., c66],
                    [0.,   0.,   0., c44,  0.,  0., c44,  0.,  0.],
                    [0.,   0.,   0.,  0., c44,  0.,  0., c44,  0.],
                    [0.,   0.,   0.,  0.,  0., c66,  0.,  0., c66]
                ]
            )
        print("Untransformed C matrix \n")
        print(C, "\n")
        self.C = self.c_transform(C, T)
        print("Transformed C matrix \n")
        print(self.C, "\n")

    def contract_index(self,i, j):

        if i == j:
            if i == 1 - 1:
                i1 = 1 - 1
            elif i == 2 - 1:
                i1 = 2 - 1
            elif i == 3 - 1:
                i1 = 3 - 1
        elif i == 1 - 1:
            if j == 2 - 1:
                i1 = 6 - 1
            elif j == 3 - 1:
                i1 = 8 - 1
        elif i == 2 - 1:
            if j == 3 - 1:
                i1 = 4 - 1
            elif j == 1 - 1:
                i1 = 9 - 1
        elif i == 3 - 1:
            if j == 1 - 1:
                i1 = 5 - 1
            elif j == 2 - 1:
                i1 = 7 - 1
        return i1

    def get_Q_rot(self,a):
        b = np.zeros((9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        i1 = self.contract_index(i, j)
                        i2 = self.contract_index(k, l)
                        b[i1][i2] = a[k][i] * a[l][j]
        return b

    def c_transform(self,C, a):
        Q = self.get_Q_rot(a)
        C_t = Q.T.dot(C.dot(Q))
        return C_t

    def u_edge(self,x, y):
        C = self.C
        b = self.b
        C11bar = (C[1][1] * C[2][2])**(0.5)
        phi = 0.5 * np.arccos((C[1][2]**2 + 2 * C[1][2]
                               * C[6][6] - C11bar**2) / (2 * C11bar * C[6][6]))
        lam = (C[1][1] / C[2][2])**(0.25)

        q = np.sqrt(x**2 + 2*x*y * lam * np.cos(phi) + y**2 * lam**2)
        t = np.sqrt(x**2 - 2*x*y * lam * np.cos(phi) + y**2 * lam**2)

        ux = - (b[0] / (4. * np.pi)) * (np.arctan2((2*x*y * lam * np.sin(phi)),     (x**2 - lam**2 * y**2))
                                        + (C11bar**2 - C[1][2]**2) * np.log(q / t) / (2. * C11bar * C[6][6] * np.sin(2. * phi)))

        ux += - (b[1] / (4. * np.pi * lam * C11bar * np.sin(2. * phi))) * (
            (C11bar - C[1][2]) * np.cos(phi) * np.log(q * t) - ((C11bar + C[1][2]) * np.sin(phi)
                                                                * np.arctan2((x**2 * np.sin(2. * phi)),    (lam**2 * y**2 - x**2 * np.cos(2. * phi)))))

        uy = - (b[1] / (4. * np.pi)) * (np.arctan2((2*x*y * lam * np.sin(phi)), (x**2 - lam**2 * y**2))
                                        - ((C11bar**2 - C[1][2]**2) * np.log(q / t)) / (2. * C11bar * C[6][6] * np.sin(2. * phi)))

        uy += (lam * b[0] / (4. * np.pi * C11bar * np.sin(2. * phi))) * (
            (C11bar - C[1][2]) * np.cos(phi) * np.log(q * t) - ((C11bar + C[1][2]) * np.sin(phi)
                                                                * np.arctan2((y**2 * lam**2 * np.sin(2. * phi)), (x**2 - y**2 * lam**2 * np.cos(2. * phi)))))

        return ux, uy

    def u_screw(self,x, y):
        bz = self.b[-1]
        # Displacement is only in the z directon
        C = self.C
        uz = -(bz/(2. * np.pi)) * \
            np.arctan2(np.sqrt(C[4][4]*C[5][5] - C[4][5])
                       * y,  (C[4][4] * x - C[4][5] * y))

        return uz

    def get_Disl_edge(self):
        length = 200
        a = self.a
        u, v = sci.meshgrid(sci.linspace(-5 * a, 5 * a, length), sci.linspace(-5 * a, 5 * a, length))
        dis = [ sci.zeros((length, length), dtype=np.float64),
                sci.zeros((length, length), dtype=np.float64) ]

        for i in range(length):
            for j in range(length):
                x = u[i][j]
                y = v[i][j]
                ux, uy = self.u_edge(x, y)
                dis[0][i][j] = ux
                dis[1][i][j] = uy

        return dis

    def get_Disl_screw(self):
        length = 200
        a = self.a
        u, v = sci.meshgrid(sci.linspace(-5 * a, 5 * a, length), sci.linspace(-5 * a, 5 * a, length))
        dis = sci.zeros((length, length), dtype=np.float64)

        for i in range(length):
            for j in range(length):
                x = u[i][j]
                y = v[i][j]
                dis[i][j] = self.u_screw(x, y)    

        return dis


    def plot_dis_edge(self, dis):
        fig = pp.figure(1, figsize=(12, 6), dpi = 100)
        for k in range(2):
            scale = np.floor(np.log10(np.max(np.absolute(dis[k]))))
            ax = fig.add_subplot(1, 2, k + 1)
            ax.set_title(r" $ x^{" + str(k + 1) + "} $ Displacement field $ \\times 10^{" + str(int(-scale)) + "}$: Edge ")
            im = ax.imshow(dis[k] / (10 ** scale), extent=(-5, 5, -5, 5), cmap = 'coolwarm')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            bc = fig.colorbar(im, cax=cax, format="%1.3f")
            bc.solids.set_edgecolor("face")
        pp.show()

    def plot_dis_screw(self, dis):
        fig = pp.figure(1, figsize=(12, 6), dpi = 100)
        scale = np.floor(np.log10(np.max(np.absolute(dis))))
        print("scale",scale)
        ax = fig.add_subplot(111)
        ax.set_title(r"Displacement field $\,\times\,10^{" + str(int(-scale)) + "}$")
        im = ax.imshow(dis / (10 ** scale), extent=(-5, 5, -5, 5), cmap = 'coolwarm')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        bc = fig.colorbar(im, cax=cax, format="%1.3f")
        bc.solids.set_edgecolor("face")
        pp.show()

    def gen_disl_u(self):
        if self.pure and self.screw:
            print("This is a pure Screw: b = %s" %(b[2]))
            dis  = get_Disl_screw()
            if self.plot:
                 self.plot_dis_screw(dis)
        elif self.pure and not self.screw:
            dis = self.get_Disl_edge()
            if self.plot:
                self.plot_dis_edge(dis)
        else:
            dis_z  = self.get_Disl_screw()
            dis_xy = self.get_Disl_edge()
            if self.plot:
                self.plot_dis_screw(dis)
                self.plot_dis_edge(dis)
        return  dis

    def displacement(self):
        if self.pure and self.screw:
            print("This is a pure Screw: b = %s" %(self.b[2]))
            dis  = self.u_screw
            if self.plot:
                pl = self.get_Disl_screw()
                self.plot_dis_screw(pl)
        elif self.pure and not self.screw:
            dis = self.u_edge
            if self.plot:
                pl = self.get_Disl_edge()
                self.plot_dis_edge(pl)
        else:
            dis = ( self.u_edge, self.u_screw )
            if self.plot:
                dis_z  = self.get_Disl_screw()
                dis_xy = self.get_Disl_edge()
                self.plot_dis_screw(dis)
                self.plot_dis_edge(dis)
        return  dis
