
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
import copy
rc('font', **{'family': 'serif', 'serif': ['Palatino'],  'size': 18})
rc('text', usetex=True)
sci.set_printoptions(linewidth=200, precision=4)

#==========================================================================
############################    General    ################################
#==========================================================================
def cmd_result(cmd):
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result, err = proc.communicate()
    result = result.decode("utf-8")
    return result


def cmd_write_to_file(cmd, filename):
    output_file = open(filename, mode='w')
    retval = subprocess.call(cmd, shell=True, stdout=output_file)
    output_file.close()



#========================================================================================
############################    Get Elastic Constants    ################################
#========================================================================================

def Cijkl(X_p, X_n, S_np, V_prim_uc):
    # X_n, X_p are the original positons of the atoms.
    # S_np is the force constant matrix: S_ij^(np) =  d^2 E/d u_i^(n) d u_j^(p)
    C = np.zeros((3, 3, 3, 3))
    C_sym = np.zeros((3, 3, 3, 3))
    if X_p is not X_n:
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    for l in range(3):
                        C[i, k, j, l] = -(0.5 / V_prim_uc) * \
                            ((X_p[k] - X_n[k]) * S_np[i, j] * (X_p[l] - X_n[l]))
                        C_sym[i, k, j, l] = -(1. / ( 8. * V_prim_uc ) ) * \
                            ( ( X_p[k] - X_n[k] ) * S_np[i,j] * ( X_p[l] - X_n[l] ) + \
                              ( X_p[i] - X_n[i] ) * S_np[k,j] * ( X_p[l] - X_n[l] ) + \
                              ( X_p[k] - X_n[k] ) * S_np[i,l] * ( X_p[j] - X_n[j] ) + \
                              ( X_p[i] - X_n[i] ) * S_np[k,l] * ( X_p[j] - X_n[j] )   )
    print_full_cij(C, extra_args='')
    print_full_cij(C_sym, extra_args='sym ')
    return C, C_sym

def print_full_cij(C, extra_args=''):
    bohr_to_ang = 0.529177
    convert = (13.606 / bohr_to_ang**3) * 160.21766208
    # for i in range(3):
    #     for j in range(3):
    #         for k in range(3):
    #             for l in range(3):
    #                 print("C_{:d}{:d} {} = {: .10f} GPa".format(
    #                             contract_index(i,k, within_six=True)+1,
    #                             contract_index(j,l, within_six=True)+1, extra_args, C[i, k, j, l] * convert))

    for ii in range(6):
        for jj in range(6):
            i, j = contract_index(ii,0, expand=True)
            k, l = contract_index(jj,0, expand=True)
            #print(i,j,k,l)
            print(
                "C_{:d}{:d} {} = {: .10f} GPa C_{:d}{:d} {} = {: .10f} GPa".format(
                    ii+1, jj+1,
                    extra_args, C[i, j, k, l] * convert,
                    jj+1, ii+1,
                    extra_args, C[k, l, i, j] * convert))

def S_np(LMarg, h, ahcp, X_n, X_p, check_pos_def=True,
         use_forces=False, second_order=False):
    # Force constant matrix defined to be Phi_ij(M,N) = change in Energy with respect to displacements U_i(M) and U_j(N).
    # h is degree of displacements.

    # First atom displacements
    ai = ("ai", X_n[0])
    aj = ("aj", X_n[1])
    ak = ("ak", X_n[2])
    arr = [ai, aj, ak]

    # Second atom displacements
    aii = ("aii", X_p[0])
    ajj = ("ajj", X_p[1])
    akk = ("akk", X_p[2])
    arr2 = [aii, ajj, akk]

    Phi = np.zeros((3, 3))
    Phi_nn = np.zeros((3,3))

    if use_forces:
        for i in range(3):
            a = arr[i]
            print("Derivatives using ", a)
            da1, da2 = cds_second_order(args, a, h, alat=ahcp)
            Phi[:, i] = -da2

    else:
        for i in range(3):
            a1 = arr[i]
            for j in range(3):
                a2 = arr2[j]
                print(a1, a2)
                Phi[i, j] = cds_fourth_order(
                    args, a1, a2, h, second_order=second_order, alat=ahcp)
                print(Phi[i, j])
                Phi_nn[i, j] = cds_fourth_order(
                    args, a1, a1, h, second_order=second_order, alat=ahcp)
                print("\nChecking if equilibrium condition holds:")
                print("S_nn_ij = - sum_{n != p}( S_np_ij  ) ")
                print(Phi[i, j], -Phi_nn[i,j])
                print(Phi[i, j] == -Phi_nn[i,j] , '\n')

    print(Phi)
    
    return Phi


#==========================================================================================
#################################     Derivatives     #####################################
#==========================================================================================


def cds_second_order(args, ai, h, alat=5.57):
    ai_name, ai_val = ai

    print("Moving atom ", ai_name)
    
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

    h *= alat
    derivatives1 = (forces_atom_p1 - forces_atom_m1) / (2. * h)
    derivatives2 = (forces_atom_p2 - forces_atom_m2) / (2. * h)
    print("For", ai)
    print("forces atom 1: - h = %s" % (forces_atom_m1))
    print("forces atom 2: - h = %s" % (forces_atom_m2))
    print("forces atom 1: + h = %s" % (forces_atom_p1))
    print("forces atom 2: + h = %s" % (forces_atom_p2))
    print("df^1/du_%s = %s" % (ai_name, derivatives1))
    print("df^2/du_%s = %s" % (ai_name, derivatives2))

    return derivatives1, derivatives2

def cds_fourth_order(args, ai, aj, h, alat=5.57, second_order=False, use_bind=False):

    ai_name, ai_val = ai
    aj_name, aj_val = aj

    n_disp = np.array([-2, -1, 1, 2])
    f_arr = np.zeros((len(n_disp), len(n_disp)))
    del_h_dict = {}

    if second_order:
        n_disp = np.array([-1, 1])
    for i, ni in enumerate(n_disp):
        for j, nj in enumerate(n_disp):
            xargs = args \
                + " -v" + ai_name + "=" \
                + str(ni * h + ai_val) \
                + " -v" + aj_name + "=" \
                + str(nj * h + aj_val)
            print(xargs)
            f_arr[i, j] = find_energy(xargs, '',  'cds_fourth_order')
            del_h_dict[(ni, nj)] = (i, j)

    def fa(ni, nj): return f_arr[del_h_dict[(ni, nj)]]

    h *= alat
    print("new h = ", h)
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



#==========================================================================================
#################################     Contracting Indices     #############################
#==========================================================================================


def contract_index(i, j, within_six=False, expand=False):

    if not expand:
        contract_dict = {
            (0,0) : 0,
            (1,1) : 1,
            (2,2) : 2,
            (1,2) : 3,
            (2,0) : 4,
            (0,1) : 5,
            (2,1) : 6,
            (0,2) : 7,
            (1,0) : 8
        }
        i1 = contract_dict[(i,j)]
    else:
        expand_dict = {
            0 : (0,0),
            1 : (1,1),
            2 : (2,2),
            3 : (1,2),
            4 : (2,0),        
            5 : (0,1),
            6 : (2,1),
            7 : (0,2),
            8 : (1,0),
        }
        i1 = expand_dict[i]

    if within_six:
        if i1 == 6:
            i1 = 3
        if i1 == 7:
            i1 = 4
        if i1 == 8:
            i1 = 5
            
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

def get_Cij(C):
    new_C = np.zeros((9,9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    i1 = contract_index(i, j)
                    i2 = contract_index(k, l)
                    print(i, j, " --> ", i1)
                    print(k, l, " --> ", i2)                    
                    new_C[i1,i2] = C[i,j,k,l]
    return new_C

def c_transform(C, a):
    Q = get_Q_rot(a)
    C_t = Q.T.dot(C.dot(Q))
    return C_t

def get_elastic_constants_from_energy(LMarg, X_n, X_p, V_prim_uc, h, alat, 
                                      use_forces=False, second_order=False, rot=False):
    np.set_printoptions(precision=5)
    
    S = S_np(LMarg, h, alat,  X_n, X_p, check_pos_def=True,
         use_forces=use_forces, second_order=second_order)
    C, C_sym = Cijkl(X_p, X_n, S, V_prim_uc/alat**2 )

    if rot is not False:
         C = c_transform(C, rot)
         C_sym = c_transform(C_sym, rot)
    
    C = get_Cij(C) 

    print("\nElastic constant matrix Ryd/bohr**3:\n",C)
    C_sym = get_Cij(C_sym)
    print("\nElastic constant matrix Ryd/bohr**3 sym:\n",C_sym)
    bohr_to_ang =  0.529177
    convert = (13.606 / bohr_to_ang**3) * 160.21766208
    C *= convert
    C_sym *= convert

    print("\nElastic constant matrix GPa:\n",C)
    print("\nElastic constant matrix GPa: sym\n",Csym)    

    print("\n Checking Stability for tbe elastic constants. \n")
    is_stable = np.all(np.linalg.eigvals( C ) > 0)
    print("is positive definite = ", is_stable)
    print("\nEigenvalues are ", np.linalg.eigvals(C))
    print(is_stability_satisfied(C[0,0], C[2,2], C[3,3], C[0,1], C[0,2]))

    print("\n Checking Stability for tbe sym elastic constants. \n")
    is_stable = np.all(np.linalg.eigvals( C_sym ) > 0)
    print("is positive definite = ", is_stable)
    print("\nEigenvalues are ", np.linalg.eigvals(C_sym))
    print(is_stability_satisfied(C_sym[0,0], C_sym[2,2], C_sym[3,3], C_sym[0,1], C_sym[0,2]))
    return C

#========================================================================================
############################    Get min lp and energy    ################################
#========================================================================================

def lp_hcp_energy(x):
    global lpargs
    ext = "ti"
    vargs = ' '
    etot = find_energy(' tbe ' + ext + ' ', (lpargs + ' -vahcp=' + str(x[0]) + ' -vchcp=' + str(x[1])
                                             + ' -vnk=30 '),
                       'lpmin')
    return etot


def lp_omega_energy(x):
    global lpargs 
    ext = "ti"
    vargs = ' '
    etot = find_energy(' tbe ' + ext + ' ', (lpargs + ' -vomega=1 -vhcp=0 -vaomega=' + str(x[0])
                                             + ' -vqomega=' +
                                             str(x[1]) + ' -vuomega=' +
                                             str(x[2])
                                             + ' -vnk=30 '),
                       'lpmin')
    return etot



def find_energy(LMarg, args, filename, pipe=True):
    cmd = LMarg + ' ' + args
    if not pipe:
        cmd_write_to_file(cmd, filename)
        if 'lmf' in LMarg:
            cmd = " grep 'ehk' " \
                " | tail -2 | grep 'From last iter' | awk '{print $5}'"
        elif 'tbe' in LMarg:
            cmd = " grep 'total energy' " + filename + \
                " | tail -1 | awk '{print $4}'"
    else:
        if 'lmf' in LMarg:
            cmd += " | grep 'ehk' " + \
                " | tail -2 | grep 'From last iter' | awk '{print $5}'"
        elif 'tbe' in LMarg:
            cmd += " | grep 'total energy' " + \
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
        x0 = np.array([5.57, 8.683])
        fnc = lp_hcp_energy
    elif phase == "omega":
        x0 = np.array([1, 1, 1])
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



#==========================================================================================================
############################    Check if ec matrix is positive definite    ################################
#==========================================================================================================

def is_positive_definite(c11, c33, c44, c12, c13, C=None):

    if C is None:
        C_arr = sci.array(
        [
            [c11,  c12,  c13,  0.,  0.,  0.],
            [c12,  c12,  c13,  0.,  0.,  0.],
            [c13,  c13,  c33,  0.,  0.,  0.],
            [0.,   0.,   0.,  c44, 0.,  0.],
            [0.,   0.,   0.,  0., c44,  0.],
            [0.,   0.,   0.,  0.,  0., c66]]
        )
    else:
        C_arr = C

    is_stable = np.all(np.linalg.eigvals(C_arr) > 0)

    print("is positive definite = ", is_stable)
    print("Eigenvalues are ", np.linalg.eigvals(C_arr))
    print(C_arr)
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


args = ' tbe ti -vhcp=1 -vspecpos=1 -vspecplat=1 -vforces=1 '  # + varg
# ahcp = 5.5125
# chcp = 8.8090
# ahcp = 5.5118
# chcp = 8.7970
ahcp=5.6361
chcp=9.0429
q = chcp/ahcp

h = 0.0001
X_n = np.array([0.,0.,0.])
X_p = np.array( [ 1./(2*np.sqrt(3)) , -1/2., q/2 ] ) 
V_prim_uc = (3**(0.5) / 2) * ahcp**2 * chcp

rotation = np.array([[np.sqrt(3)/2., 0.5,  0.],
                     [-0.5, np.sqrt(3)/2., 0.],
                     [0., 0.,  1.]])

plat = np.array([ [     0,         -1,  0 ],
                   [np.sqrt(3)/2,  0.5,  0 ],
                   [     0,         0,   q ] ] )

new_plat = np.zeros(plat.shape)
rotate_plat = True
rotation = np.eye(3)
plat_str = [ [ '-vplxa=', '-vplxb=', '-vplxc='  ],
             [ '-vplya=', '-vplyb=', '-vplyc='  ],
             [ '-vplza=', '-vplzb=', '-vplzc='  ] ]
plat_comm = ' '
if rotate_plat:
    X_n = rotation.dot(X_n)
    X_p = rotation.dot(X_p)    
    for i in range(3):
        new_plat[i,:] = rotation.dot( plat[i] )
        n_p_str = plat_str[i]
        for k, var in enumerate( n_p_str ):
            n_p_str[k] = var + str( new_plat[i,k].round(10) )
        plat_comm += ' '.join( n_p_str ) + ' '

print("rotated plat")
print(new_plat)
print("New rotated plat command")
print(plat_comm)
args += plat_comm

print("Volume", V_prim_uc)

Cij =  get_elastic_constants_from_energy(args, X_n, X_p, V_prim_uc, h, ahcp, 
                                         use_forces=True, second_order=True)

Cij =  get_elastic_constants_from_energy(args, X_n, X_p, V_prim_uc, h, ahcp, 
                                         use_forces=False, second_order=True)


#==========================================================================================
############################     Find min lps and see difference     ######################
#==========================================================================================

print("Finding Minimum lattice parameters to see if there is a difference." )

# ahcp = 5.5118
# chcp = 8.7970
global lpargs
lpargs = ' -vhcp=1 -vspecpos=1 -vspecplat=1 -vforces=1 ' + plat_comm

ahcp, chcp, etot = get_min_lp(phase="hcp")
print("Minimum lattice parameters\n a = {:.8f}\n c = {:.8f}".format(ahcp, chcp) )
q = chcp/ahcp
h = 0.0001
X_n = np.array([0.,0.,0.])
X_p = np.array( [ 1./(2*np.sqrt(3)) , -1/2., q/2 ] ) 
V_prim_uc = (3**(0.5) / 2) * ahcp**2 * chcp

args_mn = args + ' -vahcp={:.8f} -vchcp={:.8f}'.format(ahcp, chcp)
Cij =  get_elastic_constants_from_energy(args_mn, X_n, X_p, V_prim_uc, h, ahcp, 
                                         use_forces=False, second_order=True)






# #==========================================================================================
# #############################          Add rotation          ##############################
# #==========================================================================================

# print("Adding rotation." )
# rotation = np.array([[np.sqrt(3)/2., 0.5,  0.],
#                      [-0.5, np.sqrt(3)/2., 0.],
#                      [0., 0.,  1.]])

# ahcp = 5.5118
# chcp = 8.7970
# q = chcp/ahcp

# h = 0.002
# X_n = np.array([0.,0.,0.])
# X_p = np.array( [ 1./(2*np.sqrt(3)) , -1/2., q/2 ] ) 
# V_prim_uc = (3**(0.5) / 2) * ahcp**2 * chcp
# Cij =  get_elastic_constants_from_energy(args, X_n, X_p, V_prim_uc, h, ahcp, 
#                                          use_forces=False, second_order=True, rot=rotation)


alphas = np.linspace(-0.01, 0.01, 11  )
energy_array = np.array( [
#Strain 0
np.array( [ -0.12754049, -0.12791091, -0.12827002, -0.12861785, -0.12895441, -0.12927973,  -0.12959383, -0.12989675, -0.13018851, -0.13046914, -0.13073870 ]),
#Strain 1                                                                                                                                                    ,
np.array( [ -0.12753988, -0.12791047, -0.12826973, -0.12861767, -0.12895433, -0.12927973,  -0.12959391, -0.12989691, -0.13018876, -0.13046952, -0.13073923 ]),
#Strain 2                                                                                                                                                    ,
np.array( [ -0.12787715, -0.12817992, -0.12847157, -0.12875209, -0.12902148, -0.12927973,  -0.12952685, -0.12976285, -0.12998773, -0.13020151, -0.13040420 ]),
#Strain 3                                                                                                                                                    ,
np.array( [ -0.12569477, -0.12647399, -0.12722201, -0.12793893, -0.12862480, -0.12927973,  -0.12990382, -0.13049718, -0.13105992, -0.13159220, -0.13209415 ]),
#Strain 4                                                                                                                                                    ,
np.array( [ -0.12665271, -0.12721484, -0.12775857, -0.12828393, -0.12879097, -0.12927973,  -0.12975026, -0.13020262, -0.13063687, -0.13105306, -0.13145128 ]),
#Strain 5                                                                                                                                                    ,
np.array( [ -0.12604406, -0.12675100, -0.12742804, -0.12807517, -0.12869240, -0.12927973,  -0.12983719, -0.13036482, -0.13086264, -0.13133071, -0.13176910 ]),
#Strain 6                                                                                                                                                    ,
np.array( [ -0.12604352, -0.12675061, -0.12742777, -0.12807500, -0.12869232, -0.12927973,  -0.12983728, -0.13036499, -0.13086292, -0.13133113, -0.13176970 ]),
#Strain 7                                                                                                                                                    ,
np.array( [ -0.12410528, -0.12525441, -0.12634641, -0.12738127, -0.12835904, -0.12927973,  -0.13014341, -0.13095015, -0.13170003, -0.13239316, -0.13302968 ]),
#Strain 8                                                                                                                                                    ,
np.array( [ -0.12915029, -0.12919697, -0.12923321, -0.12925907, -0.12927457, -0.12927973,  -0.12927457, -0.12925907, -0.12923321, -0.12919697, -0.12915029 ]),
#Strain 9                                                                                                                                                    ,
np.array( [ -0.12915030, -0.12919697, -0.12923321, -0.12925907, -0.12927457, -0.12927973,  -0.12927457, -0.12925907, -0.12923321, -0.12919697, -0.12915030 ]),
#Strain 10                                                                                                                                                   ,
np.array( [ -0.12902027, -0.12911394, -0.12918660, -0.12923838, -0.12926940, -0.12927973,  -0.12926940, -0.12923838, -0.12918660, -0.12911394, -0.12902027 ]),
#Strain 11                                                                                                                                                   ,
np.array( [ -0.12910554, -0.12916818, -0.12921698, -0.12925187, -0.12927280, -0.12927973,  -0.12927263, -0.12925146, -0.12921621, -0.12916685, -0.12910338 ]),
#Strain 12                                                                                                                                                   ,
np.array( [ -0.12910376, -0.12916695, -0.12921616, -0.12925138, -0.12927257, -0.12927973,  -0.12927285, -0.12925193, -0.12921697, -0.12916800, -0.12910503 ]),
#Strain 13                                                                                                                                                   ,
np.array( [ -0.12947342, -0.12945734, -0.12942994, -0.12939120, -0.12934113, -0.12927973,  -0.12920701, -0.12912298, -0.12902764, -0.12892100, -0.12880308 ]),
#Strain 14
np.array( [ -0.12897867, -0.12906023, -0.12913114, -0.12919137, -0.12924090, -0.12927973,  -0.12930784, -0.12932521, -0.12933185, -0.12932774, -0.12931289 ])
])



fig, axes = plt.subplots(3, 5)
fig.suptitle('Girshick Strains')
for i, en in enumerate( energy_array ):
    print("plotting")
    print(en)
    j = i//5
    k = i % 5
    print(j,k)
    ax = axes[j,k]
    print(ax)
    ax.set_title('{:d}'.format(i) )
    ax.plot( alphas, en, 'bo' )
    poly_coeffs = np.polyfit( alphas, en, 5 )
    poly = np.poly1d(poly_coeffs)
    pl = [ poly(ai) for ai in np.linspace(alphas[0], alphas[-1], 100)   ]
    ax.plot( np.linspace(alphas[0], alphas[-1], 100), pl  )




 # Elastic Constants: Girshick Routine Applied Strains 


 # C11 =  165.8623476695,   C11_exp =  176.1000000000
 # C33 =  164.5632628633,   C33_exp =  190.5000000000
 # C44 =  38.1796557562,   C44_exp =  50.8000000000
 # C66 =  51.8860924371,   C66_exp =  44.6000000000
 # C12 = -10383.7436103689,   C12_exp =  86.9000000000
 # C13 = -9548.9962292795,   C13_exp =  68.3000000000
 # K =  93.6561928893,   K_FR =  109.9666666667
 # R =  55.7940512318,   R_FR =  61.8000000000
 # H =  52.8336415119,   H_FR =  45.9650000000 
 
energy_array = np.array( [
#Strain 0
np.array( [ -0.12754049, -0.12791091, -0.12827002, -0.12861785, -0.12895441, -0.12927973,  -0.12959383, -0.12989675, -0.13018851, -0.13046914, -0.13073870 ]),
#Strain 1                                                                                                                                                    ,
np.array( [ -0.12787715, -0.12817992, -0.12847157, -0.12875209, -0.12902148, -0.12927973,  -0.12952685, -0.12976285, -0.12998773, -0.13020151, -0.13040420 ]),
#Strain 2                                                                                                                                                    ,
np.array( [ -0.12910554, -0.12916818, -0.12921698, -0.12925187, -0.12927280, -0.12927973,  -0.12927263, -0.12925146, -0.12921621, -0.12916685, -0.12910338 ]),
#Strain 3                                                                                                                                                    ,
np.array( [ -0.12875957, -0.12889317, -0.12901201, -0.12911606, -0.12920530, -0.12927973,  -0.12933934, -0.12938411, -0.12941406, -0.12942918, -0.12942948 ]),
#Strain 4                                                                                                                                                    ,
np.array( [ -0.12924743, -0.12925907, -0.12926811, -0.12927457, -0.12927844, -0.12927973,  -0.12927844, -0.12927457, -0.12926811, -0.12925907, -0.12924743 ]),
#Strain 5                                                                                                                                                    ,
np.array( [ -0.12947342, -0.12945734, -0.12942994, -0.12939120, -0.12934113, -0.12927973,  -0.12920701, -0.12912298, -0.12902764, -0.12892100, -0.12880308 ]),
#Strain 6                                                                                                                                                    ,
np.array( [ -0.12897867, -0.12906023, -0.12913114, -0.12919137, -0.12924090, -0.12927973,  -0.12930784, -0.12932521, -0.12933185, -0.12932774, -0.12931289 ]),
#Strain 7
    np.array( [ -0.12917137, -0.12921028, -0.12924059, -0.12926227, -0.12927533, -0.12927973,  -0.12927546, -0.12926248, -0.12924078, -0.12921031, -0.12917104 ]) ])

 # Constants: Girshick Routine Applied Strains 


 # C11 =  165.8231485457,   C11_exp =  176.1000000000
 # C33 =  164.5632628633,   C33_exp =  190.5000000000
 # C44 =  38.1494487390,   C44_exp =  50.8000000000
 # C66 =  207.5028445231,   C66_exp =  44.6000000000
 # C12 = -249.1825405004,   C12_exp =  86.9000000000
 # C13 = -273.1296728747,   C13_exp =  68.3000000000
 # K = -1094.6742125447,   K_FR =  109.9666666667
 # R =  669.1429126353,   R_FR =  61.8000000000
 # H =  634.1671283357,   H_FR =  45.9650000000 


 
energy_array = np.array( [
np.array( [ -0.12754049, -0.12527027, -0.12637182,  -0.12739325, -0.12836281, -0.12927973, -0.13014341, -0.13095326, -0.13170870, -0.13240917, -0.13305411 ])
np.array( [ -0.12787715, -0.12526749, -0.12637184, -0.12739325, -0.12836281, -0.12927973, -0.13014341, -0.13095326, -0.13170870, -0.13240917, -0.13305411])
np.array( [ -0.12910554, -0.12525440, -0.12637191, -0.12739325, -0.12836281, -0.12927973, -0.13014341, -0.13095326, -0.13170870, -0.13240917, -0.13305411])
np.array( [ -0.12875957, -0.12525719, -0.12637190, -0.12739325, -0.12836281, -0.12927973, -0.13014341, -0.13095326, -0.13170870, -0.13240917, -0.13305411])
np.array( [ -0.12924743, -0.12525441, -0.12637191, -0.12739325, -0.12836281, -0.12927973, -0.13014341, -0.13095326, -0.13170870, -0.13240917, -0.13305411])
np.array( [ -0.13078237, -0.12523854, -0.12637200, -0.12739325, -0.12836281, -0.12927973, -0.13014341, -0.13095326, -0.13170870, -0.13240917, -0.13305411])
np.array( [ -0.12897867, -0.12525580, -0.12637191, -0.12739325, -0.12836281, -0.12927973, -0.13014341, -0.13095326, -0.13170870, -0.13240917, -0.13305411])
np.array( [ -0.12917137, -0.12525441, -0.12637191, -0.12739325, -0.12836281, -0.12927973, -0.13014341, -0.13095326, -0.13170870, -0.13240917, -0.13305411])
])
 
fig, axes = plt.subplots(2, 4)
fig.suptitle('Tony Strains')
for i, en in enumerate( energy_array ):
    print("plotting")
    print(en)
    j = i//4
    k = i % 4
    print(j,k)
    ax = axes[j,k]
    print(ax)
    ax.set_title('{:d}'.format(i) )    
    ax.plot( alphas, en, 'bo' )
    poly_coeffs = np.polyfit( alphas, en, 5 )
    poly = np.poly1d(poly_coeffs)
    pl = [ poly(ai) for ai in np.linspace(alphas[0], alphas[-1], 100)   ]
    ax.plot( np.linspace(alphas[0], alphas[-1], 100), pl  )

plt.show()
