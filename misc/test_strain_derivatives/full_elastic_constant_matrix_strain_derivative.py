
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
############################    Argument manipulation    ################################
#========================================================================================

def get_strained_configuration(h, e, alat, positions, position_names, plat, plat_names=None):
    # Homogeneous strain: u = e.dot( X )
    print("Applying strain")
    print(e)
    
    positions_strained = np.zeros(positions.shape)
    position_args = ''
    for i, X in enumerate( positions ):
        X_strained = h * e.dot( X * alat ) + X * alat
        for j, name in enumerate( position_names[i] ):
            positions_strained[i,j] = X_strained[j] / alat
            position_args += ' -v{}={:.10f}'.format(name, X_strained[j] / alat )

    plat_strained = np.asarray( [ ( pl * alat + h * e.dot( pl * alat ) ) / alat  for pl in plat ] )
    plat_args = get_plat_command(plat_strained, plat_names)
    command = plat_args + position_args

    print("Strained positions")
    print(positions_strained)
    
    print("Strained lattice vectors")
    print(plat_strained)
    
    return positions_strained, plat_strained, command


def get_plat_command(plat, rotation=None, plat_str=None):
    if plat_str is None:
        plat_str = [ [ '-vplxa=', '-vplya=', '-vplza='  ],
                     [ '-vplxb=', '-vplyb=', '-vplzb='  ],
                     [ '-vplxc=', '-vplyc=', '-vplzc='  ] ]
    
    plat_comm = ' '
    new_plat = np.zeros(plat.shape)
    for i in range(3):
        if rotation is not None:
            new_plat[i,:] = rotation.dot( plat[i] )
        else:
            new_plat[i,:] = plat[i,:]
        n_p_str = plat_str[i]
        for k, var in enumerate( n_p_str ):
            n_p_str[k] = var + str( new_plat[i,k].round(10) )
        plat_comm += ' '.join( n_p_str ) + ' '

    return plat_comm

#========================================================================================
############################    Get Elastic Constants    ################################
#========================================================================================

def Cijkl(args, h, alat, plat, positions, position_names, second_order=False):
    # We can obtain the elastic constants by differentiating the elastic energy density with respect to two strains
    # C_ijkl = 0.5 * d^2 E / de_ij de_kl
    C = np.zeros((3, 3, 3, 3))
    if X_p is not X_n:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        e_ij = np.zeros((3,3)); e_ij[i,j] = 1
                        e_kl = np.zeros((3,3)); e_kl[k,l] = 1
                        C[i, j, k, l] = 0.5 * cds_fourth_order(args, h, e_ij, e_kl, positions, alat=alat,
                                                               position_names=position_names,
                                                               second_order=second_order, plat=plat)

    print_full_cij(C, extra_args='')
    return C

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


#==========================================================================================
#################################     Derivatives     #####################################
#==========================================================================================


def cds_fourth_order(args, h, e1, e2, positions, position_names=None,  alat=5.57,
                     second_order=False, plat=None):


    n_disp = np.array([-2, -1, 1, 2])
    f_arr = np.zeros((len(n_disp), len(n_disp)))
    del_h_dict = {}

    if second_order:
        n_disp = np.array([-1, 1])
    for i, ni in enumerate(n_disp):
        for j, nj in enumerate(n_disp):
            positions_strained, plat_strained, command = get_strained_configuration( h * ni , e1, alat, 
                                                                                     positions, position_names,
                                                                                     plat, plat_names=None)
            positions_strained, plat_strained, command = get_strained_configuration( h * nj , e2, alat, 
                                                                                     positions_strained, position_names,
                                                                                     plat_strained, plat_names=None)
            f_arr[i, j] = find_energy(args + command, '',  'cds_fourth_order')
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
                    print(i+1, j+1, " --> ", i1+1)
                    print(k+1, l+1, " --> ", i2+1)                    
                    new_C[i1,i2] = C[i,j,k,l]
    return new_C

def c_transform(C, a):
    Q = get_Q_rot(a)
    C_t = Q.T.dot(C.dot(Q))
    return C_t

#=========================================================================================================
#=======================================          MAIN         ===========================================
#=========================================================================================================

def get_elastic_constants_strain_derivative(args, h, alat, plat, positions, position_names, second_order):
    C = Cijkl(args, h, alat, plat, positions, position_names, second_order=second_order)
    print("Untransformed elastic constant matrix")
    print(C)
    C = get_Cij(C)
    
    np.set_printoptions(precision=5)
    print("\nElastic constant matrix Ryd/bohr**3:\n",C)

    bohr_to_ang =  0.529177
    convert = (13.606 / bohr_to_ang**3) * 160.21766208
    C *= convert
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


args = ' tbe ti -vhcp=1 -vspecpos=1 -vspecplat=1 -vforces=1 ' 
ahcp=5.6361
chcp=9.0429
q = chcp/ahcp

rotation = np.array([[np.sqrt(3)/2., 0.5,  0.],
                     [-0.5, np.sqrt(3)/2., 0.],
                     [0., 0.,  1.]])

h = 0.0001
X_n = rotation.dot( np.array([0.,0.,0.]) ) 
X_p = rotation.dot( np.array( [ 1./(2*np.sqrt(3)) , -1/2., q/2 ] ) ) 

positions = np.asarray( [ X_n, X_p ] )
position_names = [ [ 'ai',  'aj',  'ak'  ],
                   [ 'aii', 'ajj', 'akk' ] ]
second_order = False

plat = np.array([ [     0,         -1,  0 ],
                   [np.sqrt(3)/2,  0.5,  0 ],
                   [     0,         0,   q ] ] )

plat_comm = get_plat_command(plat, rotation=rotation)


C = get_elastic_constants_strain_derivative(args, h, ahcp, plat, positions, position_names, second_order)

#==========================================================================================
############################     Find min lps and see difference     ######################
#==========================================================================================

# print("Finding Minimum lattice parameters to see if there is a difference." )

# ahcp = 5.5118
# chcp = 8.7970
# global lpargs
# lpargs = ' -vhcp=1 -vspecpos=1 -vspecplat=1 -vforces=1 ' + plat_comm

# ahcp, chcp, etot = get_min_lp(phase="hcp")
# print("Minimum lattice parameters\n a = {:.8f}\n c = {:.8f}".format(ahcp, chcp) )
# q = chcp/ahcp
# h = 0.002
# X_n = np.array([0.,0.,0.])
# X_p = np.array( [ 1./(2*np.sqrt(3)) , -1/2., q/2 ] ) 
# V_prim_uc = (3**(0.5) / 2) * ahcp**2 * chcp

# args_mn = args + ' -vahcp={:.8f} -vchcp={:.8f}'.format(ahcp, chcp)
# Cij =  get_elastic_constants_from_energy(args_mn, X_n, X_p, V_prim_uc, h, ahcp, 
#                                          use_forces=False, second_order=True)



# # #==========================================================================================
# # #############################          Add rotation          ##############################
# # #==========================================================================================

# # print("Adding rotation." )
# # rotation = np.array([[np.sqrt(3)/2., 0.5,  0.],
# #                      [-0.5, np.sqrt(3)/2., 0.],
# #                      [0., 0.,  1.]])

# # ahcp = 5.5118
# # chcp = 8.7970
# # q = chcp/ahcp

# # h = 0.002
# # X_n = np.array([0.,0.,0.])
# # X_p = np.array( [ 1./(2*np.sqrt(3)) , -1/2., q/2 ] ) 
# # V_prim_uc = (3**(0.5) / 2) * ahcp**2 * chcp
# # Cij =  get_elastic_constants_from_energy(args, X_n, X_p, V_prim_uc, h, ahcp, 
# #                                          use_forces=False, second_order=True, rot=rotation)


# ec = np.array( 
#  [[ 2.2272,   0.27192,  0.80071,  0.16768, -0.07893,  0.72153,  0.0575,   0.17501, -0.00938],
#  [ 0.27192,  0.49531,  0.10444, -0.20127,  0.10815, -0.07809,  0.02675, -0.0278,  -0.15076],
#  [ 0.80071,  0.10444,  1.15753,  0.00964, -0.01775, -0.1803,   0.21149,  0.31484,  0.00912],
#  [ 0.16768, -0.22401,  0.03253,  0.67398, -0.07726, -0.20069, -0.08181,  0.04262,  0.0558 ],
#  [-0.09106,  0.10815, -0.00513, -0.07438,  0.3322,   0.08569, -0.24375,  0.61174, -0.03012],
#  [ 0.64789, -0.00461, -0.1803,  -0.15911,  0.06787,  0.51815,  0.00377, -0.07379,  0.11769],
#  [ 0.0575,   0.04526,  0.19313,  0.05313, -0.24375,  0.00377,  0.7419,  -0.84346, -0.00724],
#  [ 0.13481, -0.0278,   0.35608,  0.04262,  0.77688, -0.07379, -0.76929,  2.24564, -0.00662],
#  [-0.0117,  -0.14795,  0.00912,  0.0558,  -0.03012, -0.18063,  0.0053,   0.01619,  0.0823 ]] )
