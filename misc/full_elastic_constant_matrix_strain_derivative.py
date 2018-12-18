
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
    for ii in range(6):
        for jj in range(6):
            i, j = contract_index(ii,0, expand=True)
            k, l = contract_index(jj,0, expand=True)
            print("C_{:d}{:d} {} = {: .10f} GPa C_{:d}{:d} {} = {: .10f} GPa".format(
                    ii+1, jj+1,
                    extra_args, C[i, j, k, l] * convert,
                    jj+1, ii+1,
                    extra_args, C[k, l, i, j] * convert))

            
def apply_strain(LMarg, h, alat, X_n, X_p, check_pos_def=True,
         use_forces=False, second_order=False):
    # Force constant matrix defined to be Phi_ij(M,N) = change in Energy with respect to displacements U_i(M) and U_j(N).
    # h is degree of displacements.

    # Homogeneous strain: u = e.dot( X )
    
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

    for i in range(3):
        a1 = arr[i]
        for j in range(3):
            a2 = arr2[j]
            print(a1, a2)
            Phi[i, j] = cds_fourth_order(
                args, a1, a2, h, second_order=second_order, alat=alat)
            print(Phi[i, j])
            Phi_nn[i, j] = cds_fourth_order(
                args, a1, a1, h, second_order=second_order, alat=alat)
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

    h = h * alat
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

    S = S_np(LMarg, h, alat,  X_n, X_p, check_pos_def=True,
         use_forces=use_forces, second_order=second_order)
    C, C_sym = Cijkl(X_p, X_n, S, V_prim_uc)

    if rot is not False:
         C = c_transform(C, rot)
         C_sym = c_transform(C_sym, rot)
    
    C = get_Cij(C)
    np.set_printoptions(precision=5)
    print("\nElastic constant matrix Ryd/bohr**3:\n",C)
    C_sym = get_Cij(C_sym)
    print("\nElastic constant matrix Ryd/bohr**3 sym:\n",C_sym)
    bohr_to_ang =  0.529177
    convert = (13.606 / bohr_to_ang**3) * 160.21766208
    C *= convert
    C_sym *= convert
    np.set_printoptions(precision=3)
    Cpr = copy.copy(C)
    Cpr[Cpr < 0.001] = 0
    print("\nElastic constant matrix GPa:\n",Cpr)
    Cpr = copy.copy(C_sym)
    Cpr[Cpr < 0.001] = 0
    print("\nElastic constant matrix GPa: sym\n",Cpr)    

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

def check_old_elastic_constants(args, alpha, strain):
    return
    
def tony_strains(ec):
    if ec == 'c11':
        strain = np.array( [ 1, 0, 0, 0, 0, 0 ] )
    elif ec == 'c33':
        strain = np.array( [ 0, 0, 1, 0, 0, 0 ] )
    
    elif ec == 'cp':
        strain = np.array( [ 1 ,-1, 0, 0, 0, 0 ] )
    
    elif ec == 'cpp':
        strain = np.array( [ 1, 0, -1, 0, 0, 0 ] )
    
    elif ec == 'c44':
        strain = np.array( [ 0, 0, 0, 0, 0.5, 0 ] )
    
    elif ec == 'R':
        strain = np.array( [ -0.5, -0.5, 1, 0, 0, 0 ] )
    
    elif ec == 'H':
        strain = np.array( [ 1, -0.5 ,-0.5, 0, 0, 0 ] )
    
    elif ec == 'S':
        strain = np.array( [  0, 0, 0, 0.5, 0.5, 0.5 ] )
    return strain



args = ' tbe ti -vhcp=1 -vspecpos=1 -vspecplat=1 -vforces=1 '  # + varg
# ahcp = 5.5125
# chcp = 8.8090
# ahcp = 5.5118
# chcp = 8.7970
ahcp=5.6361
chcp=9.0429
q = chcp/ahcp

h = 0.002
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
plat_str = [ [ '-vplxa=', '-vplya=', '-vplza='  ],
             [ '-vplxb=', '-vplyb=', '-vplzb='  ],
             [ '-vplxc=', '-vplyc=', '-vplzc='  ] ]
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

ahcp = 5.5118
chcp = 8.7970
global lpargs
lpargs = ' -vhcp=1 -vspecpos=1 -vspecplat=1 -vforces=1 ' + plat_comm

ahcp, chcp, etot = get_min_lp(phase="hcp")
print("Minimum lattice parameters\n a = {:.8f}\n c = {:.8f}".format(ahcp, chcp) )
q = chcp/ahcp
h = 0.002
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
