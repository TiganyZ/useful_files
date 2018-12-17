
""" 
 This is a file which is to evaluate the elastic constants in both relaxed and unrelaxed configurations
 According to Clouet (2012) and Cousins (1979), in a strained hcp lattice there are internal degrees of freedom
 that are not accounted for when applying a homogeneous strain.
 This is necessary for C_11, C_12 and C_66 elastic constants.
 Two of these inner elastic constants, e11, e33, are related to the phonon frequencies of the optical branches at the gamma point.
 omega_i = s * np.sqrt( Omega * e_ii / m )
 Where Omega = a**2 * c * 3**(0.5) / 2 (The atomic volume), and m is the mass
 The inner elastic constants d_21 couples the internal degrees of freedom to the homogeneous strain, leading to a contribution:
 delta * C_12 = d_21**2 / e_11
 C^0_ij are the unrelaxed elastic constants
 The true elastic constants are then given by 
 C_{11} = C^0_{11} - delta * C_{12} 
 C_{12} = C^0_{11} + delta * C_{12} 
 C_{66} = C^0_{66} - delta * C_{12} 
 With all others being unchanged 


To get an elastic constant we have to make use of the relation that defines them 

stress_ij = C_ijkl * strain_kl

strain_kl = 0.5 * ( du_i/dx_j + du_j/dx_i )

Know that 

C_ijkl = 0.5 * (  d^2E/de_ijde_kl   )

Or we can use the formula from Sutton's book 



C_ijkl = -( 0.5 / V_prim_unit_cell ) * ( sum_{p != n}  ( X_k_(p) - X_k_(n)} ) S_ij_(np)  \big( X_l^{(p) - X_l_(n)   )

S_{ij}^{(np)} =  \frac{\partial E}{\partial u_i^{(n)} \partial u_j^{(p)} } 


 C_{ikjl} = -\frac{1}{8\Omega}  \Big\{ 
    &\sum_{p\neq n}\big( X_k^{(p)} - X_k^{(n)} \big) S_{ij}^{(np)}  \big( X_l^{(p)} - X_l^{(n)}  \big) \\
  + &\sum_{p\neq n}\big( X_i^{(p)} - X_i^{(n)} \big) S_{kj}^{(np)}  \big( X_l^{(p)} - X_l^{(n)}  \big) \\
  + &\sum_{p\neq n}\big( X_k^{(p)} - X_k^{(n)} \big) S_{il}^{(np)}  \big( X_j^{(p)} - X_j^{(n)}  \big) \\
  + &\sum_{p\neq n}\big( X_i^{(p)} - X_i^{(n)} \big) S_{kl}^{(np)}  \big( X_j^{(p)} - X_j^{(n)}  \big)  \Big\}


"""

import numpy as np
import scipy as sci
from scipy.optimize import minimize
import subprocess
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Palatino'],  'size': 18})
rc('text', usetex=True)
sci.set_printoptions(linewidth=200, precision=4)


def cmd_result(cmd):
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result, err = proc.communicate()
    result = result.decode("utf-8")
    return result


def cmd_write_to_file(cmd, filename):
    output_file = open(filename, mode='w')
    retval = subprocess.call(cmd, shell=True, stdout=output_file)
    output_file.close()


def Cijkl(X_p, X_n, S_np, V_prim_uc):
    # X_n, X_p are the original positons of the atoms.
    # S_np is the force constant matrix: S_ij^(np) =  d^2 E/d u_i^(n) d u_j^(p)
    C = np.zeros((3, 3, 3, 3))
    if X_p is not X_n:
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    for l in range(3):
                        C[i, k, j, l] = -(0.5 / V_prim_uc) * \
                            ((X_p[k] - X_n[k]) * S_np[i, j] * (X_p[l] - X_n[l]))
                        print(
                            "C_{:d}{:d}{:d}{:d} = {:.10f} Ryd/bohr^3".format(i, k, j, l, C[i, k, j, l]))
    return C


def S_np(LMarg, h, check_pos_def=True,
         use_forces=False, second_order=False):
    # Force constant matrix defined to be Phi_ij(M,N) = change in Energy with respect to displacements U_i(M) and U_j(N).
    # h is degree of displacements.

    Phi0 = find_energy(LMarg, args, 'forceconstant')

    # First atom displacements
    ai = ("ai", 0)
    aj = ("aj", 0)
    ak = ("ak", 0)
    arr = [ai, aj, ak]

    # Second atom displacements
    aii = ("aii", 0)
    ajj = ("ajj", 0)
    akk = ("akk", 0)
    arr2 = [aii, ajj, akk]

    Phi = np.zeros((3, 3))
    # split into 4, 3x3 matrices
    # [ (1,1) (1,2) ]
    # [ (2,1) (2,2) ]

    if use_forces:
        m_11 = np.zeros((3, 3))
        m_12 = np.zeros((3, 3))
        m_21 = np.zeros((3, 3))
        m_22 = np.zeros((3, 3))

        for i in range(3):
            a = arr[i]
            print("Derivatives using ", a)
            da1, da2 = cds_second_order(args, a, h)
            m_11[:, i] = -da1
            m_21[:, i] = -da2

        for i in range(3):
            a = arr2[i]
            print("Derivatives using ", a)
            da1, da2 = cds_second_order(args, a, h)
            m_22[:, i] = -da1
            m_12[:, i] = -da2

        Phi[:3, :3] = m_11
        Phi[3:, :3] = m_21
        Phi[:3, 3:] = m_12
        Phi[3:, 3:] = m_22

    else:
        a = arr + arr2
        for i in range(3):
            a1 = arr[i]
            for j in range(3):
                a2 = arr2[j]
                Phi[i, j] = cds_fourth_order(
                    args, a1, a2, h, second_order=second_order)
                print(Phi[i, j])

    print(Phi)

    return S


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


def C_hex(a=False):
    c11 = sci.dtype(sci.float128)
    c12 = sci.dtype(sci.float128)
    c13 = sci.dtype(sci.float128)
    c33 = sci.dtype(sci.float128)
    c44 = sci.dtype(sci.float128)

    # Elastic constants in units of 10^{9} Pa
    c11 = 1.761e2
    c12 = 0.868e2
    c13 = 0.682e2
    c33 = 1.905e2
    c44 = 0.508e2
    c66 = 0.450e2

    #c66 = 0.5 * ( c11 - c12 )

    C_arr = sci.array(
        [
            [c11,  c12,  c13,  0.,  0.,  0.,  0.,  0.,  0.],
            [c12,  c12,  c13,  0.,  0.,  0.,  0.,  0.,  0.],
            [c13,  c13,  c33,  0.,  0.,  0.,  0.,  0.,  0.],
            [0.,   0.,   0.,  c44, 0.,  0., c44,  0.,  0.],
            [0.,   0.,   0.,  0., c44,  0.,  0., c44,  0.],
            [0.,   0.,   0.,  0.,  0., c66,  0.,  0., c66],
            [0.,   0.,   0., c44,  0.,  0., c44,  0.,  0.],
            [0.,   0.,   0.,  0., c44,  0.,  0., c44,  0.],
            [0.,   0.,   0.,  0.,  0., c66,  0.,  0., c66]
        ]
    )
    if a is not False:
        C_arr = c_transform(C_arr, a)

    # Map the indicies of the second order tensor to contracted representation
    n_dic = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 0)
              : 4, (0, 1): 5, (2, 1): 6, (0, 2): 7, (1, 0): 8}

    def C(i, j, k, l): return C_arr[n_dic[i, j]][n_dic[k, l]]

    print("Stiffness Matrix Hex")
    print("--------------------------------------------------------------------------------")
    print(C_arr)
    print("\n")
    return C, C_arr


def get_strains(strain):
    strain_names = [' -vexx=', ' -veyy=', ' -vezz=',
                    ' -veyz=', ' -vexz=', ' -vexy=']
    strain_command = ' '
    for name, s in zip(strain_names, strain):
        strain_command += name + s + ' '
    return strain_command


def strains():

    s_11 = np.array([1,  0, 0, 0, 0, 0])
    s_112 = np.array([0,  1, 0, 0, 0, 0])
    s_33 = np.array([0,  0, 1, 0, 0, 0])
    s_2C11_2C12 = np.array([1,  1, 0, 0, 0, 0])
    s_2_C11_2_C22 = np.array([1,  1, 0, 0, 0, 0])
    s_5o4_C11_C12 = np.array([0.5, 1, 0, 0, 0, 0])
    s_C11_C33_2_C13 = np.array([1,  0, 1, 0, 0, 0])
    s_C11_C33_2_C13_2 = np.array([0,  1, 1, 0, 0, 0])
    s_4_C44 = np.array([0,  0, 0, 0, 1, 0])
    s_4_C44_2 = np.array([0,  0, 0, 1, 0, 0])
    s_8_C44 = np.array([0,  0, 0, 1, 1, 1])
    s_4_C66 = np.array([1, -1, 0, 0, 0, 0])
    s_4C662 = np.array([0,  0, 0, 0, 0, 1])

    c11 = 0.5 * (curvature[1-1] + curvature[2-1])  # / (10**(9)) #Correct
    c33 = 1.0 * curvature[3-1]  # / (10**(9)) #Correct
    c12 = 1./3. * (curvature[4-1] + curvature[5-1] -
                   3.25*c11)  # / (10**(9)) #Correct
    c13 = 0.25 * (curvature[6-1] + curvature[7-1] -
                  2*c11 - 2*c33)  # / (10**(9)) #Correct
    c66 = 0.125 * (curvature[12-1] + curvature[13-1])  # / (10**(9))
    c442 = 0.0625 * (curvature[9-1] + curvature[10-1] +
                     curvature[11-1] - 4*c66)  # / (10**(9))
    c44 = 0.0625 * (curvature[9-1] + curvature[10-1] +
                    curvature[11-1])  # / (10**(9))

    kk = (1./9.) * (2 * c11 + c33 + 2 * c12 + 4 * c13)

    rr = (1./3.) * (0.5 * c11 + c33 + 0.5 * c12 - 2 * c13)

    hh = (1./3.) * (1.2 * c11 + 0.25 * c33 - c12 - 0.5 * c13)

    #kk  = curvature[  8 -1 ] / 9. / (10**(9))
    #rr  = curvature[ 14 -1 ] / 3. / (10**(9))
    #hh  = curvature[ 15 -1 ] / 3.  / (10**(9))

    print('\n Elastic Constants: Girshick Routine \n')
    print('\n C11 = {:< .10f},   C11_FR = {:< .10f}' .format(
        c11,  ec_exp_arr[0]))
    print(' C33 = {:< .10f},   C33_FR = {:< .10f}' .format(
        c33,  ec_exp_arr[1]))
    print(' C44 = {:< .10f},   C44_FR = {:< .10f}' .format(
        c44,  ec_exp_arr[2]))
    print(' C44 = {:< .10f},   C44_FR = {:< .10f}' .format(
        c442, ec_exp_arr[2]))
    print(' C66 = {:< .10f},   C66_FR = {:< .10f}' .format(
        c66,  ec_exp_arr[3]))
    print(' C12 = {:< .10f},   C12_FR = {:< .10f}' .format(
        c12,  ec_exp_arr[4]))
    print(' C13 = {:< .10f},   C13_FR = {:< .10f}' .format(
        c13,  ec_exp_arr[5]))

    print(' K = {:< .10f},   K_FR = {:< .10f}' .format(kk, ec_exp_arr[6]))
    print(' R = {:< .10f},   R_FR = {:< .10f}' .format(rr, ec_exp_arr[7]))
    print(' H = {:< .10f},   H_FR = {:< .10f} \n ' .format(hh, ec_exp_arr[8]))
    print('C66 - 0.5(C11 - C12) = {:< .10f},   C66_FR = {:< .10f}'.format(
        c66 - 0.5 * (c11 - c12), ec_exp_arr[3]))


def get_elastic_constant(BIN, ext, vargs,
                         ec, ahcp, chcp):
    global rmaxh_hcp
    AHCP = '%.10f' % (ahcp)
    CHCP = '%.10f' % (chcp)
    subprocess.call('rm -f save.' + ext, shell=True)
    alpha_inc = 0.002
    alpha = -0.01
    while alpha < 0.011:
        ALPHA = '%.3f' % (alpha)
        arg = '-valpha=' + ALPHA + ' -vahcp=' + AHCP + ' ' + \
            ' -vchcp=' + CHCP + ' -vrmaxh=' + str(rmaxh_hcp) + ' '
        cmd = BIN + 'tbe -vbcc=0 -vhcp=1 -vnk=30 -v' + \
            ec + '=T ' + arg + ' ' + vargs + ext
        cmd_write_to_file(cmd, 'out-' + ec + '-' + ALPHA)
        alpha += alpha_inc

    cmd_write_to_file(BIN + 'vextract t alpha etot < save.' + ext, 'pt-' + ec)

    cmd = "echo '0\nmin' | " + BIN + "pfit -nc=2 5 pt-" + \
        ec + " | grep min= | awk '{print $7}'"
    MINIMUM = try_value(cmd, 'minimum for ' + ec)

    if abs(MINIMUM) > 0.1:
        print(sys.argv[0], 'minimum for ' + ec + ' not found, leaving ...')
        panic()

    cmd = "echo " + str(MINIMUM) + " | " + BIN + "pfit -nc=2 3 pt-" + \
        str(ec) + " | grep f= | awk '{print $5}'"
    Epp = try_value(cmd, ec)
    return Epp


##########################################################################################
########################     Elastic Constant Strain Routines      #######################

    s_2_C11_2_C22 = np.array([1.0, 1, 0, 0, 0, 0])
    # 4

    s_5o4_C11_C12 = np.array([0.5, 1, 0, 0, 0, 0])

    s_C11_C33_2_C13 = np.array([1, 0, 1, 0, 0, 0])

    # 6

    s_C11_C33_2_C13_2 = np.array([0, 1, 1, 0, 0, 0])
    # 7

    s_4_C44 = np.array([0, 0, 0, 0, 1, 0])

    s_4_C44_2 = np.array([0, 0, 0, 1, 0, 0])
    # 10

    s_8_C44 = np.array([0, 0, 0, 1, 1, 1])

    # 11

    s_4_C66 = np.array([1, -1, 0, 0, 0, 0])

    s_4C662 = np.array([0, 0, 0, 0, 0, 1])

    c11 = 0.5 * (curvature[1-1] + curvature[2-1])  # / (10**(9)) #Correct
    c33 = 1.0 * curvature[3-1]  # / (10**(9)) #Correct
    c12 = 1./3. * (curvature[4-1] + curvature[5-1] -
                   3.25*c11)  # / (10**(9)) #Correct
    c13 = 0.25 * (curvature[6-1] + curvature[7-1] -
                  2*c11 - 2*c33)  # / (10**(9)) #Correct
    c66 = 0.125 * (curvature[12-1] + curvature[13-1])  # / (10**(9))
    c442 = 0.0625 * (curvature[9-1] + curvature[10-1] +
                     curvature[11-1] - 4*c66)  # / (10**(9))
    c44 = 0.0625 * (curvature[9-1] + curvature[10-1] +
                    curvature[11-1])  # / (10**(9))

    kk = (1./9.) * (2 * c11 + c33 + 2 * c12 + 4 * c13)

    rr = (1./3.) * (0.5 * c11 + c33 + 0.5 * c12 - 2 * c13)

    hh = (1./3.) * (1.2 * c11 + 0.25 * c33 - c12 - 0.5 * c13)

    #kk  = curvature[  8 -1 ] / 9. / (10**(9))
    #rr  = curvature[ 14 -1 ] / 3. / (10**(9))
    #hh  = curvature[ 15 -1 ] / 3.  / (10**(9))

    print('\n Elastic Constants: Girshick Routine \n')
    print('\n C11 = {:< .10f},   C11_FR = {:< .10f}' .format(
        c11,  ec_exp_arr[0]))
    print(' C33 = {:< .10f},   C33_FR = {:< .10f}' .format(
        c33,  ec_exp_arr[1]))
    print(' C44 = {:< .10f},   C44_FR = {:< .10f}' .format(
        c44,  ec_exp_arr[2]))
    print(' C44 = {:< .10f},   C44_FR = {:< .10f}' .format(
        c442, ec_exp_arr[2]))
    print(' C66 = {:< .10f},   C66_FR = {:< .10f}' .format(
        c66,  ec_exp_arr[3]))
    print(' C12 = {:< .10f},   C12_FR = {:< .10f}' .format(
        c12,  ec_exp_arr[4]))
    print(' C13 = {:< .10f},   C13_FR = {:< .10f}' .format(
        c13,  ec_exp_arr[5]))

    print(' K = {:< .10f},   K_FR = {:< .10f}' .format(kk, ec_exp_arr[6]))
    print(' R = {:< .10f},   R_FR = {:< .10f}' .format(rr, ec_exp_arr[7]))
    print(' H = {:< .10f},   H_FR = {:< .10f} \n ' .format(hh, ec_exp_arr[8]))
    print('C66 - 0.5(C11 - C12) = {:< .10f},   C66_FR = {:< .10f}'.format(
        c66 - 0.5 * (c11 - c12), ec_exp_arr[3]))

    return np.array([c11, c33, c44, c66, c12, c13]), np.array([c11, c33, c44, c66, c12, c13]) - np.asarray(ec_exp_arr[:-3])


def lp_hcp_energy(x):
    ext = "ti"
    vargs = ' '
    etot = find_energy(' tbe ' + ext + ' ', (vargs + ' -vahcp=' + str(x[0]) + ' -vchcp=' + str(x[1])
                                             + ' -vnk=30 '),
                       'lpmin')
    return etot


def lp_omega_energy(x):
    ext = "ti"
    vargs = ' '
    etot = find_energy(' tbe ' + ext + ' ', (vargs + ' -vomega=1 -vhcp=0 -vaomega=' + str(x[0])
                                             + ' -vqomega=' +
                                             str(x[1]) + ' -vuomega=' +
                                             str(x[2])
                                             + ' -vnk=30 '),
                       'lpmin')
    return etot


def find_energy(LMarg, args, filename):
    cmd = LMarg + ' ' + args
    cmd_write_to_file(cmd, filename)
    if 'lmf' in LMarg:
        cmd = "grep 'ehk' " + filename + \
            " | tail -2 | grep 'From last iter' | awk '{print $5}'"
    elif 'tbe' in LMarg:
        cmd = "grep 'total energy' " + filename + \
            " | tail -1 | awk '{print $4}'"
    etot = cmd_result(cmd)
    print(etot)
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


def get_min_lp(phase="hcp"):

    if phase == "hcp":
        x0 = np.array([a_hcp_exp, c_hcp_exp])
        fnc = lp_hcp_energy
    elif phase == "omega":
        x0 = np.array([a_omega_exp, q_omega_exp, u_omega_exp])
        fnc = lp_omega_energy

    try:
        ret = minimize(fnc, x0, method='Nelder-Mead',
                       options={'disp': True, 'fatol': 1e-10})
    except TypeError:
        print("Could not find the minimum lattice parameter, exiting...")
        ret = 0
    if phase == "hcp":
        ret = ret['x'][0], ret['x'][1], ret['fun']
    elif phase == "omega":
        ret = ret['x'][0], ret['x'][1], ret['x'][2], ret['fun']
    return ret


def is_positive_definite(c11, c33, c44, c12, c13):
    C_arr = sci.array(
        [
            [c11,  c12,  c13,  0.,  0.,  0.],
            [c12,  c12,  c13,  0.,  0.,  0.],
            [c13,  c13,  c33,  0.,  0.,  0.],
            [0.,   0.,   0.,  c44, 0.,  0.],
            [0.,   0.,   0.,  0., c44,  0.],
            [0.,   0.,   0.,  0.,  0., c66]]
    )

    is_stable = np.all(np.linalg.eigvals(C_arr) > 0)

    print("is positive definite = ", is_stable)
    return np.all(np.linalg.eigvals(C_arr) > 0)


def is_stability_satisfied(C_11, C_33, C_44, C_12, C_13):
    print("\n   Criteria for stability:\n")

    c1 = C_11 - C_12 > 0
    print("C_11 - C_12 > 0 \n", c1)

    c2 = C_11 + C_12 + C_33 > 0
    print(" C_11 + C_12 + C_33 > 0 \n", c2)

    c3 = (C_11 + C_12) * C_33 - 2 * C_13**2 > 0
    print("( C_11 + C_12 ) * C_33 - 2 * C_13**2 > 0 \n", c3)

    c4 = C_44 > 0
    print("C_44 > 0 \n", c4)

    c5 = (C_11 - C_12) > 0
    print("(C_11 - C_12) > 0\n", c5)

    c6 = (C_11 + C_12)*C_33 > 0
    print("( C_11 + C_12 )*C_33 > 0 \n", c6)

    c7 = C_11 + C_12 > 0
    print("C_11 + C_12 > 0\n ", c7)

    c8 = C_33 > 0
    print("C_33 > 0\n", c8)

    c9 = C_11 > 0
    print("C_11 > 0\n", c9)


def cds_fourth_order(args, ai, aj, h, second_order=False):

    ai_name, ai_val = ai
    aj_name, aj_val = aj

    n_disp = np.array([-2, -1, 1, 2])
    f_arr = np.zeros((len(n_disp), len(n_disp)))
    del_h_dict = {}

    if second_order:
        n_disp = np.array([-1, 1])
    for i, ni in enumerate(n_disp):
        for j, nj in enumerate(n_disp):
            xargs = args + " -v" + ai_name + "=" + \
                str(ni * h) + " -v" + aj_name + "=" + str(nj * h)
            print(xargs)
            f_arr[i, j] = find_energy(xargs, '',  'cds_fourth_order')
            del_h_dict[(ni, nj)] = (i, j)

    def fa(ni, nj): return f_arr[del_h_dict[(ni, nj)]]
    if second_order:
        mixed_derivative = (1./(4. * h**2)) * (fa(-1, -1) +
                                               fa(1, 1) - fa(1, -1) - fa(-1, 1))
    else:
        mixed_derivative = (1. / (144. * h**2)) * (
            8. * (fa(1, -2) + fa(2, -1) + fa(-2, 1) + fa(-1, 2))
            - 8. * (fa(-1, -2) + fa(-2, -1) + fa(1, 2) + fa(2, 1))
            - 1. * (fa(2, -2) + fa(-2, 2) - fa(-2, -2) - fa(2, 2))
            + 64. * (fa(-1, -1) + fa(1, 1) - fa(1, -1) - fa(-1, 1)))

    print("\nDerivative = %s" % (mixed_derivative))

    return mixed_derivative


def cds_second_order(args, ai, h):

    ai_name, ai_val = ai

    filename = 'force_derivatives'
    forces_atom_p1 = np.zeros(3)
    forces_atom_p2 = np.zeros(3)
    forces_atom_m1 = np.zeros(3)
    forces_atom_m2 = np.zeros(3)

    for j in [-1, 1]:
        cmd_write_to_file(args + " -v" + ai_name + "=" + str(j * h), filename)
        #print( cmd_result(" grep -A4 'Forces on atom' " + filename  ) )
        for i in range(3):
            greparg = " grep -A4 'Forces on atom' " + filename + \
                " | grep 'Total' | awk '{print$" + str(i + 3) + "}'"
            if j == -1:
                forces_atom_m1[i], forces_atom_m2[i] = tuple(
                    [float(x) for x in (cmd_result(greparg).strip('\n')).split()])
            else:
                forces_atom_p1[i], forces_atom_p2[i] = tuple(
                    [float(x) for x in (cmd_result(greparg).strip('\n')).split()])

    derivatives1 = (forces_atom_p1 - forces_atom_m1) / (2. * h)
    derivatives2 = (forces_atom_p2 - forces_atom_m2) / (2. * h)
    print("For", ai)
    print("forces atom 1: - h = %s" % (forces_atom_m1))
    print("forces atom 2: - h = %s" % (forces_atom_m2))
    print("forces atom 1: + h = %s" % (forces_atom_p1))
    print("forces atom 2: + h = %s" % (forces_atom_p2))
    print("Derivative1 = %s" % (derivatives1))
    print("Derivative2 = %s" % (derivatives2))

    return derivatives1, derivatives2


def build_force_constant_matrix(LMarg, h, check_pos_def=True,
                                use_forces=False, second_order=False):
    # Force constant matrix defined to be Phi_ij(M,N) = change in phi with respect to displacements U_i(M) and U_j(N).
    # h is degree of displacements.

    Phi0 = find_energy(LMarg, args, 'forceconstant')

    # First atom displacements
    ai = ("ai", 0)
    aj = ("aj", 0)
    ak = ("ak", 0)
    arr = [ai, aj, ak]

    # Second atom displacements
    aii = ("aii", 0)
    ajj = ("ajj", 0)
    akk = ("akk", 0)
    arr2 = [aii, ajj, akk]

    Phi = np.zeros((6, 6))
    # split into 4, 3x3 matrices
    # [ (1,1) (1,2) ]
    # [ (2,1) (2,2) ]

    if use_forces:
        m_11 = np.zeros((3, 3))
        m_12 = np.zeros((3, 3))
        m_21 = np.zeros((3, 3))
        m_22 = np.zeros((3, 3))

        for i in range(3):
            a = arr[i]
            print("Derivatives using ", a)
            da1, da2 = cds_second_order(args, a, h)
            m_11[:, i] = -da1
            m_21[:, i] = -da2

        for i in range(3):
            a = arr2[i]
            print("Derivatives using ", a)
            da1, da2 = cds_second_order(args, a, h)
            m_22[:, i] = -da1
            m_12[:, i] = -da2

        Phi[:3, :3] = m_11
        Phi[3:, :3] = m_21
        Phi[:3, 3:] = m_12
        Phi[3:, 3:] = m_22

    else:
        a = arr + arr2
        for i in range(6):
            a1 = a[i]
            for j in range(6):
                a2 = a[j]
                Phi[i, j] = cds_fourth_order(
                    args, a1, a2, h, second_order=second_order)
                print(Phi[i, j])

    print(Phi)

    if check_pos_def:
        # Check if the force constant matrix is positive definite.
        print("Eigenvalues")
        print(np.linalg.eigvals(Phi))
        is_stable = np.all(np.linalg.eigvals(Phi) > 0)
        print("\n Is force constant matrix positive definite?", is_stable)

    return Phi


def make_symm_file(args, a, c, npts, hcp=True, bands=True):
    # G,K,M,G,A,H,L,A,H
    G = np.array([0.,               0.,                   0.])
    A = np.array([0.,               0.,                   np.pi/c])
    K = np.array([4*np.pi/(3.*a),   0.,                   0.])
    H = np.array([4*np.pi/(3.*a),   0.,                   np.pi/c])
    M = np.array([np.pi/a,         -np.pi/(3**(0.5)*a),   0.])
    L = np.array([np.pi/a,         -np.pi/(3**(0.5)*a),   np.pi/c])

    labels = ['G', 'K', 'M', 'G', 'A', 'H', 'L', 'A', 'H']
    values = [G, K, M, G, A, H, L, A, H]

    k_point_dict = {label: value for label, value in zip(labels, values)}

    sym_file = open("syml.ti", 'w')
    for i, label in enumerate(labels):
        if i == len(labels) - 1:
            sym_file.write("0 0 0 0 0 0 0")
            sym_file.close()
            break
        k1, k2 = values[i], values[i+1]
        print(k1, k2, label, labels[i+1])
        sym_file.write("{:d}  {:<12.8f} {:<12.8f} {:<12.8f}  {:<12.8f} {:<12.8f} {:<12.8f}  {} to {}\n".format(
            npts, k1[0], k1[1], k1[2], k2[0], k2[1], k2[2], label, labels[i+1]))

    ef = '-' + cmd_result(args + " | grep  'E_F' | awk '{print$3}' ")[5:]
    print("Fermi energy", ef)
    cmd_result(args + "   --band~fn=syml" + " -vef0=" + ef)
    cmd_result(
        "echo -13.6,13.6 / | plbnds -fplot -ef=0 -scl=13.6 -lbl=G,K,M,G,A,H,L,A,H bnds.ti")
    cmd_result("fplot -f plot.plbnds")


# Elastic constants from Tony's model

c_11, c_11_exp = 184.207498,   176.100000
c_33, c_33_exp = 192.992932,   190.500000
c_44, c_44_exp = 41.016848,    50.800000
c_12, c_12_exp = 61.028170,    86.900000
c_13, c_13_exp = 56.351377,    68.300000

a_hcp, a_hcp_exp = 5.512530,       5.576790
c_hcp, c_hcp_exp = 8.809030,      8.852101


c_11_diff_min = 185.869078
c_33_diff_min = 191.509023
c_44_diff_min = 39.728293
c_12_diff_min = 57.030221
c_13_diff_min = 55.910451


# Elastic constants in units of 10^{9} Pa
c11 = 1.761e2
c12 = 0.868e2
c13 = 0.682e2
c33 = 1.905e2
c44 = 0.508e2
c66 = 0.450e2


###################################################################################################
############################     Obtaining Force Constant Matrix      #############################
varg = ("-vfddtt=0.4668210290518986 -vqddstt=0.6661853205248066  -vb0tt=94.6065975095372  -vp0tt=1.1904247675736155 -vb1tt=-26.730183186922584  -vp1tt=0.9999763795267558 -vcr1=-6.160555487840057  -vcr2=3.9496959741413162  -vcr3=-1.0282981668933682  -vndt=1.9923522610256579 -vrmaxh=8.51")

varg = (" -vfddtt=0.4668418806546737 -vqddstt=0.6660968695540497 -vb0tt=94.4011791926749 -vp0tt=1.1902574670213237 -vb1tt=-26.704816810939302 -vp1tt=0.9999600888309667 -vcr1=-6.158653986495596 -vcr2=3.9496749559495172 -vcr3=-1.0282840982939534 -vndt=1.992406298332605 -vahcp=5.5274  -vqq=1.5997394796830335 -vrmaxh=8.51 -vnk=30 ")

varg = ("  -vfddtt=0.2939228243 -vqddstt=0.5832592246 -vb0tt=112.9050409 -vp0tt=1.507105391 -vndti=2.051341768 ")

args = ' tbe ti -vhcp=1 '  # + varg
ahcp = 5.5125
chcp = 8.8090
npts = 80
# make_symm_file( args, ahcp, chcp, npts, hcp=True, bands=True)


h = 0.01
Phi = build_force_constant_matrix(args, h, use_forces=True, second_order=True)
print("Force Constant Matrix")
print(Phi)

# [[ 7.7099e-13  2.3901e-11 -2.3901e-11]
#  [-7.7099e-13  0.0000e+00  0.0000e+00]
#  [ 7.7099e-13  0.0000e+00  0.0000e+00]]


# Eigenvalues
# [-0.3173  0.3173  2.5963 -0.3185  0.3185 -2.5963]

#  Is force constant matrix positive definite? False
# Force Constant Matrix
# [[ 7.7099e-13  2.3901e-11 -2.3901e-11 -3.1729e-01  2.3901e-11 -2.3901e-11]
#  [-7.7099e-13  0.0000e+00  0.0000e+00 -7.7099e-13 -3.1847e-01  0.0000e+00]
#  [ 7.7099e-13  0.0000e+00  0.0000e+00  7.7099e-13  0.0000e+00  2.5963e+00]
#  [-3.1729e-01 -2.5443e-11  2.5443e-11  2.5443e-11 -2.5443e-11  2.5443e-11]
#  [-7.7099e-13 -3.1847e-01  0.0000e+00 -7.7099e-13  0.0000e+00  0.0000e+00]
#  [ 7.7099e-13  0.0000e+00  2.5963e+00  7.7099e-13  0.0000e+00  0.0000e+00]]


###################################################################################################
############################     Obtaining Minimum hcp lattice pars   #############################
global count
count = 100
ahcp, chcp, E_tot_hcp = get_min_lp(phase='hcp')
q = chcp / ahcp

V0 = 0.5 * np.sqrt(3.0) * q * ahcp**3


print('Got a, c : a=%s, c=%s c/a=%s' % (ahcp, chcp, q))
print('Targets  : a=%.4f, c=%.4f c/a=%.4f' %
      (a_hcp_exp, c_hcp_exp, c_hcp_exp/a_hcp_exp))

###################################################################################################
############################     Obtaining Elastic Shear Constants    #############################

print('\nGetting hcp shear constants ...\n')

# get c_11
V = V0
BIN = ' '
ext = 'ti'
vargs = ' '
c_11 = (1.0/V) * 14700 * get_elastic_constant('c11', ahcp, chcp)

c_33 = (1.0/V) * 14700 * get_elastic_constant('c33', ahcp, chcp)
# C' = 1/2(c_11 - c_12)
c_p = (0.25/V) * 14700 * get_elastic_constant('cp',  ahcp, chcp)
# C'' = 1/4 (c_11 + c_33 - 2c_13)
c_pp = (0.25/V) * 14700 * get_elastic_constant('cpp', ahcp, chcp)

c_44 = (1.0/V) * 14700 * get_elastic_constant('c44', ahcp, chcp)

c_12 = c_11 - 2.0 * c_p
c_13 = 0.5 * (c_11 + c_33) - 2.0 * c_pp
S = 0.5 * (c_p + 2.0 * c_44)
R = (0.5*c_11 + c_33 + 0.5*c_12 - 2.0*c_13)
H = ((5.0/4.0)*c_11 + 0.25*c_33 - c_12 - 0.5*c_13)

print("shear constants: c_11=%5.1f, c_33=%5.1f, c_44=%5.1f, c_12=%5.1f, c_13=%5.1f, c_p= %5.1f, c_pp=%5.1f, S=%5.1f, R=%5.1f, H=%5.1f " %
      (c_11, c_33, c_44, c_12, c_13, c_p, c_pp, S, R, H))

K_hcp = (1.0/9.0) * (2.0 * c_11 + c_33 + 2.0 * c_12 + 4.0 * c_13)
print("   bulk modulus: %.0f " % (K_hcp))


###################################################################################################
############################        Checking Stability Criteria       #############################


print("Checking Stability for tbe elastic constants. \n")
print(is_positive_definite(c_11, c_33, c_44, c_12, c_13))
print(is_stability_satisfied(c_11, c_33, c_44, c_12, c_13))

print("Checking Stability for tbe elastic constants with different min lattice parameters. \n")
print(is_positive_definite(c_11_diff_min, c_33_diff_min,
                           c_44_diff_min, c_12_diff_min, c_13_diff_min))
print(is_stability_satisfied(c_11_diff_min, c_33_diff_min,
                             c_44_diff_min, c_12_diff_min, c_13_diff_min))

print("Checking Stability for experimental elastic constants. \n")
print(is_positive_definite(c_11_exp, c_33_exp, c_44_exp, c_12_exp, c_13_exp))
print(is_stability_satisfied(c_11_exp, c_33_exp, c_44_exp, c_12_exp, c_13_exp))

print("Checking Stability for experimental elastic constants. \n")
print(is_positive_definite(c11, c33, c44, c12, c13))
print(is_stability_satisfied(c11, c33, c44, c12, c13))


# #c66 = 0.5 * ( c11 - c12 )

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
#     )
# if a is not False:
#     C_arr = c_transform(C_arr, a)


# ## Map the indicies of the second order tensor to contracted representation
# n_dic = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 0): 4, (0, 1): 5, (2, 1): 6, (0, 2): 7, (1, 0): 8}


# C = lambda i, j, k, l: C_arr[  n_dic[i,j] ][  n_dic[k,l] ]

"""

def panic(filename='fmin.val'):
    global count
    ObjectiveFunction=100000 * count 
    write_and_exit (ObjectiveFunction, filename)
    
def try_value(cmd, val_name, *filename, write=False, alt_val=False ):
    global count
    count -= 1
    if write:
        for f in filename:
            cmd_write_to_file(cmd, f)
    res = cmd_result(cmd)
    try:
        res = float(res[0:-1])
    except ValueError:
        print(sys.argv[0])
        cmd = "grep 'Exit' " + str(filename) + " "
        error = cmd_result(cmd)
        print( ' Error: \n       ' + str(error) + ' From file ' + str(filename) +  ' when obtaining ' + str(val_name) + ' \n Exiting...' )
        if alt_val is False:
            panic()
        else:
            res = alt_val
    print('  ' + str(val_name) + ' = ' + str(res) )
    return res

def get_elastic_constant( ec, ahcp, chcp ):
    BIN, ext, vargs  = ' ', 'ti', ' '
    AHCP = '%.6f' % (ahcp)
    CHCP = '%.6f' % (chcp)
    subprocess.call('rm -f save.' + ext, shell=True)
    alpha_inc = 0.002
    alpha = -0.01
    while alpha < 0.011:
        ALPHA = '%.3f' % (alpha)
        arg = '-valpha=' + ALPHA + ' -vahcp=' + AHCP + ' ' + ' -vchcp=' + CHCP  + ' '
        cmd = BIN + 'tbe -vbcc=0 -vhcp=1 -vnk=30 -v' + ec + '=T ' + arg +  ' ' + vargs + ext
        cmd_write_to_file(cmd, 'out-' + ec + '-' + ALPHA)
        alpha += alpha_inc

    cmd_write_to_file(BIN + 'vextract t alpha etot < save.' + ext, 'pt-' + ec )

    cmd = "echo '0\nmin' | " + BIN + "pfit -nc=2 5 pt-" + ec + " | grep min= | awk '{print $7}'"
    MINIMUM = try_value(cmd, 'minimum for ' + ec)

    if abs(MINIMUM) > 0.1:
        print(sys.argv[0],'minimum for ' + ec + ' not found, leaving ...')
        panic()

    cmd = "echo " + str(MINIMUM) + " | " + BIN + "pfit -nc=2 3 pt-" + str(ec) + " | grep f= | awk '{print $5}'"
    Epp = try_value(cmd, ec)
    return Epp
"""
