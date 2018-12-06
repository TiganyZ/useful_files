from scipy.linalg import eig, det
import scipy as sci
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from matplotlib import rc
import functools
rcParams["figure.figsize"] = 4, 3
rcParams["font.family"] = "serif"
rcParams["font.size"] = 8
rcParams["font.serif"] = ["DejaVu Serif"]
rc("text", usetex=True)

#import sympy as sp

# sp.init_printing()
sci.set_printoptions(linewidth=200, precision=4)


################################################################################
#################    Definition of Cubic HCP Elastic constants     #############


def contract_index(i, j):

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


def get_Q_rot(a):
    b = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    i1 = contract_index(i, j)
                    i2 = contract_index(k, l)
                    b[i1][i2] = a[k][i] * a[l][j]
    return b


def c_transform(C, a):
    Q = get_Q_rot(a)
    C_t = Q.T.dot(C.dot(Q))
    return C_t


def C_hex(C, a=False):
    c11 = sci.dtype(sci.float128)
    c12 = sci.dtype(sci.float128)
    c13 = sci.dtype(sci.float128)
    c33 = sci.dtype(sci.float128)
    c44 = sci.dtype(sci.float128)

    # Elastic constants in units of 10^{9} Pa
    c11, c33, c44, c12, c13, c66 = C

    C_arr = sci.array(
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

    print("Untransformed  Matrix Hex")
    print("--------------------------------------------------------------------------------") 
    print(C_arr)
    if a is not False:
        print("C rotation")
        print(a)
        C_arr = c_transform(C_arr, a)

    # Map the indicies of the second order tensor to contracted representation
    n_dic = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 0): 4, (0, 1): 5, (2, 1): 6, (0, 2): 7, (1, 0): 8}

    
    C = lambda i, j, k, l: C_arr[  n_dic[i,j] ][  n_dic[k,l] ]

  
    print("Stiffness Matrix Hex")
    print("--------------------------------------------------------------------------------") 
    print(C_arr)
    print("\n")
    return C, C_arr



################################################################################
#################    Definition of Cubic Elastic constants     #################

def C_cubic():
    c11 = sci.dtype(sci.float128)
    c12 = sci.dtype(sci.float128)
    c44 = sci.dtype(sci.float128)

    c11 = 2.431e-3
    c12 = 1.381e-3
    c44 = 1.219e-3


    E = sci.dtype(sci.float128)
    nu = sci.dtype(sci.float128)

    ##  Isotropic case
    
    E = 123-3
    nu = 0.35
    
    c11 = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    c12 = E * (nu) / ((1 + nu) * (1 - 2 * nu))
    c44 = E / 2 / (1 + nu)

    H = sci.dtype(sci.float128)
    H = 2.0 * c44 + c12 - c11
    C = lambda i, j, k, l: c44 * ((i == k) * (j == l) + (i == l) * (j == k)) + \
    c12 * (i == j) * (k == l) - H * (i == j) * (k == l) * (i == k)

    ## Map the indicies of the second order tensr to contracted representation
    n_dic = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 0): 4, (0, 1): 5, (2, 1): 6, (0, 2): 7, (1, 0): 8}
    n_dic_inv = {n_dic[k]: k for k in n_dic}


    C_contracted = sci.zeros((9, 9))

    for i in range(9):
        for j in range(9):
            C_contracted[i, j] = C(*(n_dic_inv[i] + n_dic_inv[j]))

    print("Stiffness Matrix")
    print("--------------------------------------------------------------------------------")
    print(C_contracted)
    print("\n")
    return C, C_contracted




################################################################################
#################     Definition of a_ik and Stroh Vectors     #################


def get_Chrid_tensors(C, n, m):
    ##  Define Chridtoffe tensors

    nn = lambda i, k: sum(sum(C(i, j, k, l) * n[j] * n[l] for j in range(3)) for l in range(3))
    mn = lambda i, k: sum(sum(C(i, j, k, l) * m[j] * n[l] for j in range(3)) for l in range(3))
    nm = lambda i, k: sum(sum(C(i, j, k, l) * n[j] * m[l] for j in range(3)) for l in range(3))
    mm = lambda i, k: sum(sum(C(i, j, k, l) * m[j] * m[l] for j in range(3)) for l in range(3))

    NN = sci.empty((3, 3), dtype=sci.float128)
    MN = sci.empty((3, 3), dtype=sci.float128)
    NM = sci.empty((3, 3), dtype=sci.float128)
    MM = sci.empty((3, 3), dtype=sci.float128)

    for i in range(3):
        for k in range(3):
            NN[i, k] = nn(i, k)
            MN[i, k] = mn(i, k)
            NM[i, k] = nm(i, k)
            MM[i, k] = mm(i, k)
    return NN, MN, NM, MM


def get_stroh_and_ps(NN, MN, NM, MM):
    ## From the Chridtoffe tensors we can define an eigen problem such that one can obtain the p_a's 

    NN_inv = sci.linalg.inv(NN)
    R = sci.dot(NN_inv, NM)
    H = NN_inv
    F = sci.dot(np.dot(MN, NN_inv), NM) - MM
    G = sci.dot(MN, NN_inv)

    print(  "  Matrix to determine eigenvalues of is N\n  [ -(nn)^{-1} (nm)                   -(nn)^{-1} ]\n  [ -{ (mn)(nn)^{-1}(nm) - (mm)}   (mm)(nn)^{-1} ]"
)


    N1 = sci.concatenate((R, H), axis=1)
    N2 = sci.concatenate((F, G), axis=1)
    N = -sci.concatenate((N1, N2), axis=0)

    ##  Matrix to determine eigenvalues of is N
    ##  [ -(nn)^{-1} (nm)                   -(nn)^{-1} ]
    ##  [ -{ (mn)(nn)^{-1}(nm) - (mm)}   (mm)(nn)^{-1} ]

 

    eigen, stroh_vectors = eig(N)

    print("Determinant versus eigenvalues")
    print("--------------------------------------------------------------------------------")
    print("Eigenvalues: ", sci.sort(eigen))
    p =  np.asarray(eigen)
    print("Roots: ", sci.sort(p))
    print("Stroh Vectors:")
    

    A = []
    L = []
    for i in range(6):
        A.append(stroh_vectors[:3, i])
        L.append(stroh_vectors[3:, i])

    print("\n")
    print(" A: ")
    print(  A  )
    print(" L: ")
    print(  L  , "\n")
    return p, A, L




################################################################################
#################   Orthogonality and Normalisation Checks     #################

def orthog_check(A, L, eigen):
    ##  Verify orthogonality
    ##  A_{αi} L_{βi}  +  L_{αi} A_{βi} = 0 if α != β (11)

    print("Orthogonality check")
    print("--------------------------------------------------------------------------------")
    for alpha in range(6):
        for beta in range(6):
            s = sum(A[alpha][i] * L[beta][i] + L[alpha][i] * A[beta][i]

    for i in range(3))
        print("{:d} | {:d} | {:+.5f} | {:+.5f} | {:+.16f} |".format(alpha,
        beta, eigen[alpha], eigen[beta], s))
    print("\n")

def normalise_check(A, L):
    ##  Normalise Stroh vectors
    ##  2 * A_{αi} L_{αi} = 1 

    for alpha in range(6):
        scale = 1.0 / sci.sqrt(sum(2.0 * A[alpha][i] * L[alpha][i] for i in range(3)))
        L[alpha] = scale * L[alpha]
        A[alpha] = scale * A[alpha]

    print("Normality check")
    print("--------------------------------------------------------------------------------")
    for alpha in range(6):
        print(sum(2.0 * A[alpha][i] * L[alpha][i] for i in range(3)))
        print("\n")
    return A, L




################################################################################
#################    Functions to define displacement field    #################

def pm(p_alpha):
    return sci.sign(np.imag(p_alpha))

def f(x, m, n, p_alpha):
    return sci.log(sci.dot(m, x) + sci.dot(n, x) * p_alpha)

def mx_pnx(x, m, n, p_alpha):
    return (sci.dot(m, x) + sci.dot(n, x) * p_alpha)

def df(x, m, n, p_alpha):
    return 1./ ( sci.dot(m, x) + sci.dot(n, x) * p_alpha )

def D(L_alpha, b):
    return sum(b[i] * L_alpha[i] for i in range(3))

def ui(x, k, A, L, b, m, n, p):
    ui = 0.0
    for alphai, p_alpha in enumerate(p):
        uit = pm(p_alpha) * A[alphai][k] * D(L[alphai], b) * f(x, m, n, p_alpha)
        ui += uit
        #print("ui", uit)
    ui = ui / 2.0 / sci.pi / 1.0j
    #print("ui total", ui)
    return ui

def distortion_kl(x, k, l, A, L, b, m, n, p):
    ui = 0.0
    for alphai, p_alpha in enumerate(p):
        uit = pm(p_alpha) * A[alphai][k] * D(L[alphai], b) * df(x, m, n, p_alpha) * ( m[l] + p_alpha * n[l] )
        ui += uit
        #print("ui", uit)
    ui = ui / 2.0 / sci.pi / 1.0j
    #print("ui total", ui)
    return ui

def strain_ij(x, i, j, A, L, b, m, n, p):
    dui_dxj = distortion_kl(x, i, j, A, L, b, m, n, p)
    duj_dxi = distortion_kl(x, j, i, A, L, b, m, n, p)
    strain = 0.5 * ( dui_dxj + duj_dxi  )
    return strain
    

def stress_ij(x, k, l,  A, L, b, m, n, p, C):
    ui = 0.0
    for alphai, p_alpha in enumerate(p):
        for ii in range(3):
            for jj in range(3):
                uit = pm(p_alpha) * A[alphai][k] * D(L[alphai], b) * df(x, m, n, p_alpha) * ( m[l] + p_alpha * n[l] ) * C(ii, jj, k, l)
                ui += uit
        #print("ui", uit)
    ui = ui / 2.0 / sci.pi / 1.0j
    #print("ui total", ui)
    return ui

def ui2(x, k, A, L, b, m, n, p):
    ui2 = sum(A[alpha][k] * D(L[alpha], b) * f(x, m, n, p_alpha)  
        for alpha, p_alpha in enumerate(p) if sci.imag(p_alpha) < 0)
    ui2 = -ui2 / 2.0 / np.pi / 1.0j
    return ui2




################################################################################
#################   Solutions to Anisotropic Elastic Theory    #################

def trial_sol(x, a, A, L, b, m, n, eigen):
    x = sci.array([.1 * a, .1 * a, .1 * a])

    print("Trial solution")
    print("--------------------------------------------------------------------------------")
    print("\n x = %s,\n\n a = %s,\n\n A = %s,\n\n L = %s,\n\n b = %s,\n\n m = %s,\n\n n = %s,\n\n eigen = %s \\nn"%(x, a, A, L, b, m, n, eigen))

    #for indx, p_val in enumerate(eigen):
    #    print("index %s, p_val = %s"%(indx, p_val))

    #print(ui(x, a, A, L, b, m, n, eigen))
    #print(ui2(x, a, A, L, b, m, n, eigen))
    print("\n")


def get_Disl_edge(a, A, L, b, m, n, eigen ):
    length = 200
    u, v = sci.meshgrid(sci.linspace(-5 * a, 5 * a, length), sci.linspace(-5 * a, 5 * a, length))
    DIS = [sci.zeros((length, length), dtype=sci.complex256), sci.zeros((length, length), dtype=sci.complex256)]
    dis = [sci.zeros((length, length), dtype=np.float64),
    sci.zeros((length, length), dtype=np.float64)]

    x = np.zeros(3)
    for k in range(2):
        for i in range(length):
            for j in range(length):
                x[0] = u[i, j]
                x[1] = v[i, j]
                DIS[k][i, j] = ui(x, k, A, L, b, m, n, eigen)
                dis[k][i, j] = sci.real(DIS[k][i, j])

    return DIS, dis

def get_Disl_screw(a, A, L, b, m, n, eigen ):
    length = 200
    u, v = sci.meshgrid(sci.linspace(-5 * a, 5 * a, length), sci.linspace(-5 * a, 5 * a, length))
    DIS = sci.zeros((length, length), dtype=sci.complex256)
    dis = sci.zeros((length, length), dtype=np.float64)

    x = np.zeros(3)
    k = 2
    for i in range(length):
        for j in range(length):
            x[0] = u[i, j]
            x[1] = v[i, j]
            DIS[i, j] = ui(x, k, A, L, b, m, n, eigen)
            dis[i, j] = sci.real(DIS[i, j])    

    return DIS, dis

def get_Disl_full(a, A, L, b, m, n, eigen ):
    length = 200
    u, v = sci.meshgrid(sci.linspace(-5 * a, 5 * a, length), sci.linspace(-5 * a, 5 * a, length))
    DIS = [sci.zeros((length, length), dtype=sci.complex256), sci.zeros((length, length), dtype=sci.complex256), sci.zeros((length, length), dtype=sci.complex256)]
    dis = [sci.zeros((length, length), dtype=np.float64), sci.zeros((length, length), dtype=np.float64),
    sci.zeros((length, length), dtype=np.float64)]

    x = np.zeros(3)
    for k in range(3):
        for i in range(length):
            for j in range(length):
                x[0] = u[i, j]
                x[1] = v[i, j]
                DIS[k][i, j] = ui(x, k, A, L, b, m, n, eigen)
                dis[k][i, j] = sci.real(DIS[k][i, j])

    return DIS, dis


def get_strain_full(a, A, L, b, m, n, eigen ):
    length = 200
    u, v = sci.meshgrid(sci.linspace(-5 * a, 5 * a, length), sci.linspace(-5 * a, 5 * a, length))
    DIS = [sci.zeros((length, length), dtype=sci.complex256),
           sci.zeros((length, length), dtype=sci.complex256),
           sci.zeros((length, length), dtype=sci.complex256), 
           sci.zeros((length, length), dtype=sci.complex256), 
           sci.zeros((length, length), dtype=sci.complex256),
           sci.zeros((length, length), dtype=sci.complex256),
           sci.zeros((length, length), dtype=sci.complex256),
           sci.zeros((length, length), dtype=sci.complex256),
           sci.zeros((length, length), dtype=sci.complex256)]

    dis = [sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64)]
    
    x = np.zeros(3)
    for k in range(3):
        for l in range(3):
            for i in range(length):
                for j in range(length):
                    x[0] = u[i, j]
                    x[1] = v[i, j]
                    DIS[k*3 + l][i, j] = strain_ij(x, k, l, A, L, b, m, n, eigen)
                    dis[k*3 + l][i, j] = sci.real(DIS[k][i, j])

    return DIS, dis

def get_stress_full(a, A, L, b, m, n, eigen, C ):
    length = 200
    u, v = sci.meshgrid(sci.linspace(-5 * a, 5 * a, length), sci.linspace(-5 * a, 5 * a, length))
    DIS = [sci.zeros((length, length), dtype=sci.complex256),
           sci.zeros((length, length), dtype=sci.complex256),
           sci.zeros((length, length), dtype=sci.complex256), 
           sci.zeros((length, length), dtype=sci.complex256), 
           sci.zeros((length, length), dtype=sci.complex256),
           sci.zeros((length, length), dtype=sci.complex256),
           sci.zeros((length, length), dtype=sci.complex256),
           sci.zeros((length, length), dtype=sci.complex256),
           sci.zeros((length, length), dtype=sci.complex256)]

    dis = [sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64),\
           sci.zeros((length, length), dtype=np.float64)]
    
    x = np.zeros(3)
    for k in range(3):
        for l in range(3):
            for i in range(length):
                for j in range(length):
                    x[0] = u[i, j]
                    x[1] = v[i, j]
                    DIS[k*3 + l][i, j] = stress_ij(x, k, l, A, L, b, m, n, eigen, C)
                    dis[k*3 + l][i, j] = sci.real(DIS[k][i, j])

    return DIS, dis

################################################################################
#################        Dislocation plotting functions        #################


def plot_dis_strain_full(dis):
    fig = pp.figure(1, figsize=(18, 18), dpi = 100)
    for k in range(3):
        for l in range(3):
            scale = np.floor(np.log10(np.max(np.absolute(dis[k]))))
            ax = fig.add_subplot(3, 3, (3*k + l) + 1)
            ax.set_title(r" $ \epsilon_{" + str(k + 1) + str(l + 1)  + "} $ Strain field $ \\times 10^{" + str(int(-scale)) + "}$ ")
            im = ax.imshow(dis[k] / (10 ** scale), extent=(-5, 5, -5, 5), cmap = 'coolwarm')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            bc = fig.colorbar(im, cax=cax, format="%1.3f")
            bc.solids.set_edgecolor("face")
    pp.show()


def plot_dis_stress_full(dis):
    fig = pp.figure(1, figsize=(18, 18), dpi = 100)
    for k in range(3):
        for l in range(3):
            scale = 0#np.floor(np.log10(np.max(np.absolute(dis[k])))) - 1
            ax = fig.add_subplot(3, 3, (3*k + l) + 1)
            ax.set_title(r" $ \sigma_{" + str(k + 1) + str(l + 1)  + "} $ Stress field $ \\times 10^{" + str(int(-scale)) + "}$ ")
            im = ax.imshow(np.log10(dis[k]) / (10 ** scale), extent=(-5, 5, -5, 5), cmap = 'coolwarm')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            bc = fig.colorbar(im, cax=cax, format="%1.3f")
            bc.solids.set_edgecolor("face")
    pp.show()

def plot_dis_full(dis):
    fig = pp.figure(1, figsize=(18, 6), dpi = 100)
    for k in range(3):
        scale = np.floor(np.log10(np.max(np.absolute(dis[k]))))
        ax = fig.add_subplot(1, 3, k + 1)
        ax.set_title(r" $ x^{" + str(k + 1) + "} $ Displacement field $ \\times 10^{" + str(int(-scale)) + "}$ ")
        im = ax.imshow(dis[k] / (10 ** scale), extent=(-5, 5, -5, 5), cmap = 'coolwarm')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        bc = fig.colorbar(im, cax=cax, format="%1.3f")
        bc.solids.set_edgecolor("face")
    pp.show()


def plot_dis_edge(dis):
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

def plot_dis_screw(dis):
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

def b_demonstration(DIS, dis, m, n, a, b, A, L, eigen):
    ##  DEMONSTRATION

    x_, y_ = np.meshgrid( np.linspace(-5*a, 5*a, 20), np.linspace(-5*a, 5*a, 20) )
    x = np.zeros(3)
    for k in range(2):
        for i in range(20):
            for j in range(20):
                x[0] = x_[i, j]
                x[1] = y_[i, j]
                DIS[k][i, j] = ui(x, k, A, L, b, m, n, eigen)


    for k in range(2):
        for i in range(20):
            for j in range(20):
                dis[k][i, j] = sci.real(DIS[k][i, j])

    for i in range(20):
        for j in range(20):
            x_[i,j] = x_[i,j] + dis[0][i,j]
            y_[i,j] = y_[i,j] + dis[1][i,j]

    fig = pp.figure(1, figsize = (6, 6))
    ax = fig.add_subplot(1,1,1)

    for k in range(19):
        for l in range(19):
            ax.plot([x_[k,l], x_[k,l+1]], [y_[k,l], y_[k,l+1]], "k-")
            ax.plot([x_[l,k], x_[l+1,k]], [y_[l,k], y_[l+1,k]], "k-")

    #ax.plot([-5*a,0.25*a], [0,0], "k-")
    ax.set_xlim(-4.5*a, 4.5*a)
    ax.set_ylim(-4.5*a, 4.5*a)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    pp.show()




##############################################################################################
#################    MAIN: Generate dislocation in Anisotropic equations     #################

def uij(x1, x2, k, A, L, b, m, n, p):

    ui = 0.0
    x = np.array([x1,x2, 0])
    for alphai, p_alpha in enumerate(p):
        uit = pm(p_alpha) * A[alphai][k] * D(L[alphai], b) * f(x, m, n, p_alpha)
        ui += uit
        #print("ui", uit)
    ui = ui / 2.0 / sci.pi / 1.0j
    #print("ui total", ui)
    return sci.real(ui)

def anis_dislocation(a, b, C, m, n, pure, screw, plot, trans=False):

    print("transform?", trans)
    C, C_arr = C_hex(C, a=trans)

    NN, MN, NM, MM = get_Chrid_tensors(C, n, m)
    
    p, A, L        = get_stroh_and_ps(NN, MN, NM, MM)
    
    orthog_check(A, L, p)
    normalise_check(A, L)

    if pure and screw:
        return functools.partial(uij, k=2, A=A, L=L, b=b, m=m, n=n, p=p )
    
    elif pure and not screw:
        # have extra variable k such that one can get the x and y displacement
        return functools.partial(uij, A=A, L=L, b=b, m=m, n=n, p=p )
    else:
        # have extra variable k such that one can get the x and y displacement
        return functools.partial(uij, A=A, L=L, b=b, m=m, n=n, p=p )


def gen_disl_u(a, b, C, m, n, pure, screw, plot):

     #aik, ps        = get_aik(C)
     
     NN, MN, NM, MM = get_Chrid_tensors(C, n, m)
     
     p, A, L        = get_stroh_and_ps(NN, MN, NM, MM)

     orthog_check(A, L, p)
     normalise_check(A, L)

     if pure and screw:
         DIS, dis       = get_Disl_screw(a, A, L, b, m, n, p)
         if plot:
             plot_dis_screw(dis)
     elif pure and not screw:
         DIS, dis       = get_Disl_edge(a, A, L, b, m, n, p)
         if plot:
             plot_dis_edge(dis)
     else:
         DIS, dis       = get_Disl_full(a, A, L, b, m, n, p)
         if plot:
             plot_dis_full(dis)

     if plot:
         ## Plot Dislocation displacement heatmap
         b_demonstration(DIS, dis, m, n, a, b, A, L, p)
             
     return DIS, dis

def gen_disl_strain(a, b, C, m, n):

     #aik, ps        = get_aik(C)
     
     NN, MN, NM, MM = get_Chrid_tensors(C, n, m)
     
     p, A, L        = get_stroh_and_ps(NN, MN, NM, MM)

     orthog_check(A, L, p)
     normalise_check(A, L)


     DIS, dis       = get_strain_full(a, A, L, b, m, n, p )
     if plot:
         plot_dis_strain_full(dis)

     if plot:
         ## Plot Dislocation displacement heatmap
         b_demonstration(DIS, dis, m, n, a, b, A, L, p)
             
     return DIS, dis

def gen_disl_stress(a, b, C, m, n):

     #aik, ps        = get_aik(C)
     
     NN, MN, NM, MM = get_Chrid_tensors(C, n, m)
     
     p, A, L        = get_stroh_and_ps(NN, MN, NM, MM)

     orthog_check(A, L, p)
     normalise_check(A, L)


     DIS, dis       = get_stress_full(a, A, L, b, m, n, p, C )
     if plot:
         plot_dis_stress_full(dis)

     if plot:
         ## Plot Dislocation displacement heatmap
         b_demonstration(DIS, dis, m, n, a, b, A, L, p)
             
     return DIS, dis

     
###############################################################################################
###############################################################################################



# # Dislocation reference frame 
# m   = sci.array([   1.0,       0.0,    0.0 ], dtype=sci.float128)
# n   = sci.array([   0.0,       1.0,     0.0 ], dtype=sci.float128)
# tau = sci.array([   0.0,       0.0,     1.0 ], dtype=sci.float128)

# ##  Rotated reference frame for prismatic dislocation
# #m   = sci.array([ 0.8660254,    0.5,      0.0 ], dtype=sci.float128)
# #n   = sci.array([    0.5,   -0.8660254,   0.0 ], dtype=sci.float128)
# #tau = sci.array([    0.0,       0.0,     -1.0 ], dtype=sci.float128)

# ##  Burgers Vector
# a = sci.dtype(sci.float128)
# b = sci.dtype(sci.float128)

# """
# a = 0.286e-3
# b = sci.array([a, a, a]) * (1. / ( a * 3**(0.5)))
# #b = sci.array([0.,0., a])
# b = sci.array([a, 0.,0.])
# ## Elastic Constants

# """
# ## Cubic System
# #C, C_contracted =  C_cubic()


# ## Hexagonal System

# #a = 0.295e-3

# a  = 0.29012e-3


# #b = (1./3.) * sci.array([a, a, 0.])  ## This is 1./3 * [1,1,-2,0] a_ dislocation


# ## For dislocation in the basal plane
# """
# C_arr = sci.array(
#     [
#         [ c11,  c12,  c13,  0.,  0.,  0.,  0.,  0.,  0. ],
#         [ c12,  c12,  c13,  0.,  0.,  0.,  0.,  0.,  0. ],
#         [ c13,  c13,  c33,  0.,  0.,  0.,  0.,  0.,  0. ],
#         [  0.,   0.,   0.,  c44, 0.,  0., c44,  0.,  0. ],
#         [  0.,   0.,   0.,  0., c44,  0.,  0., c44,  0. ],
#         [  0.,   0.,   0.,  0.,  0., c66,  0.,  0., c66 ],
#         [  0.,   0.,   0., c44,  0.,  0., c44,  0.,  0. ],
#         [  0.,   0.,   0.,  0., c44,  0.,  0., c44,  0. ],
#         [  0.,   0.,   0.,  0.,  0., c66,  0.,  0., c66 ]
#     ]
# )

# """

# ##  Transformation matrix for the dislocation reference frame
# ##  Transformation for pure screw dislocation with that burgers vector: k' on j axis, 
# a_trans = (2**(-0.5)) * sci.array([ [-1,1,0],  [0,0,2**(0.5)],  [1,1,0]]  )


# ##  Transformation for a pure edge dislocation along z
# #a_trans = sci.array([ [1, 0, 0],  [0.,1.,0.],  [0.,0.,1.]]  )
# a_trans = False


# C, C_arr =  C_hex(a_trans)





# b = sci.array([0., a, 0.])
# b = sci.array([0., a, 0.])
# ## Get Dislocation

# #"""
# ##  Dislocation reference frame 
# #m   = sci.array([1., -1., 0.], dtype=sci.float128) * (1. / 2**(0.5))
# #n   = sci.array([0., 0., 1.], dtype=sci.float128) 
# #tau = sci.array([1., 1., 0.], dtype=sci.float128) * (1. / 2**(0.5))


# #b = (1./3.) * sci.array([a, a, 0.])  ## This is 1./3 * [1,1,-2,0] a_ dislocation
# #"""
# pure  = True
# screw = False
# plot  = True


# #gen_disl_stress(a, b, C, m, n)

# #gen_disl_strain(a, b, C, m, n)


# ## Get Dislocation

# pure  = True
# screw = False
# plot  = True

# DIS, dis = gen_disl_u(a, b, C, m, n, pure, screw, plot)

