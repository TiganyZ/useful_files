
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
    positions_strained = np.zeros(positions.shape)
    position_args = ''
    e *= h
    e += np.eye(3)
    for i, X in enumerate( positions ):
        X_strained = e.dot( X ) 
        for j, name in enumerate( position_names[i] ):
            positions_strained[i,j] = X_strained[j] 
            position_args += ' -v{}={:.10f}'.format(name, X_strained[j]  )

    plat_strained = np.asarray( [ (  e.dot( pl ) )   for pl in plat ] )
    plat_args = get_plat_command(plat_strained, plat_names)
    command = plat_args + position_args
    return positions_strained, plat_strained, command


def get_plat_command(plat, rotation=None, plat_str=None):
    if plat_str is None:
        plat_str = [ [ '-vplxa=', '-vplxb=', '-vplxc='  ],
                     [ '-vplya=', '-vplyb=', '-vplyc='  ],
                     [ '-vplza=', '-vplzb=', '-vplzc='  ] ]
    
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
    checked_indices = []
    if X_p is not X_n:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        if (k,l,i,j) not in checked_indices:
                            checked_indices.append( (i,j,k,l) )
                            # Using symmetry C_ijkl==C_klij, to hasten computation                    
                            e_ij = np.zeros((3,3)); e_ij[i,j] = 1
                            e_kl = np.zeros((3,3)); e_kl[k,l] = 1
                            print("Applying strains:")
                            print(e_ij)
                            print(e_kl)                                        
                            C[i, j, k, l] = 0.5 * cds_fourth_order(args, h, e_ij, e_kl, positions, alat=alat,
                                                                   position_names=position_names,
                                                                   second_order=second_order, plat=plat)
                            C[k,l,i,j] = C[i,j,k,l]
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

def get_elastic_constants_strain_derivative(args, h, alat, clat, plat, positions, position_names, second_order):
    C = Cijkl(args, h, alat, plat, positions, position_names, second_order=second_order)
    print("Untransformed elastic constant matrix")
    print(C)
    C = get_Cij(C)
    
    np.set_printoptions(precision=5)
    print("\nElastic constant matrix Ryd/bohr**3:\n",C)

    V = alat**2 * clat * ( 3**(0.5) / 2. ) 
    bohr_to_ang =  0.529177
    convert = (13.606 / bohr_to_ang**3) * 160.21766208 / V
    C *= convert
    print("\nElastic constant matrix GPa:\n",C)

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

    print( "C_11", C_11 )
    print( "C_33", C_33 )
    print( "C_44", C_44 ) 
    print( "C_12", C_12 )
    print( "C_13", C_13 )    

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


# args = ' tbe ti -vhcp=1 -vspecpos=1 -vspecplat=1 -vforces=1 -vfdd=0.1525402687652677 -vqdds=0.4515866553638987 -vqddp=0.5410907176556236 -vqddd=0.61454484045679 -vb0=250.83759285526475 -vp0=1.754628603563741 -vb1=0.0 -vp1=0.3186096185888274 -vcr1=-6.115256911742206 -vcr2=4.305593645542065 -vcr3=-1.0630734624316684 -vndt=2.004546000855847  -vahcp=5.3909715246 -vchcp=8.9010503600 -vq=1.6511032046 -vrmaxh=6.74791552 -vr1dd=6.2460048 -vrcdd=6.692148 -vr1pp=6.2460048 -vrcpp=6.692148' 
# ahcp=5.6361
# chcp=9.0429
# ahcp=5.3909715246
# chcp=8.9010503600



q = chcp/ahcp

rotation = np.array([[np.sqrt(3)/2., 0.5,  0.],
                     [-0.5, np.sqrt(3)/2., 0.],
                     [0., 0.,  1.]])

rotation = np.eye(3)

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


C = get_elastic_constants_strain_derivative(args, h, ahcp, chcp, plat, positions, position_names, second_order)

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


# ec = np.array( [[ 2.11323e+00,  9.12153e-02,  1.05778e+00,  1.68333e-01, -5.05208e-02,  1.17253e+00, -5.05556e-02,  1.70139e-03, -4.11806e-02],
# [ 9.12153e-02,  3.02292e-01, -9.47569e-02, -2.53542e-01,  1.37500e-01, -2.96528e-01,  1.04965e-01, -7.29167e-04, -1.00208e-01],
# [ 1.05778e+00, -9.47569e-02,  2.04892e+00,  3.61806e-01, -2.34236e-01,  5.70313e-01, -1.50972e-01, -1.89931e-02,  3.07986e-02],
# [ 1.68333e-01, -3.27222e-01,  4.38229e-01,  3.67535e-01, -2.13958e-01,  9.40278e-02, -3.57743e-01,  6.19792e-02,  6.45833e-03],
# [-9.07639e-02,  1.37500e-01, -1.96285e-01, -2.10729e-01,  2.04410e-01, -4.77778e-02, -5.02431e-02,  5.56806e-01, -4.23611e-03],
# [ 1.02149e+00, -1.48194e-01,  5.70313e-01,  9.32986e-02, -2.86806e-02, -3.77951e-01,  2.78993e-01,  4.92535e-01,  1.03542e-01],
# [-5.05556e-02,  7.87153e-02, -1.24965e-01, -9.75000e-02, -5.02431e-02,  2.78993e-01,  6.28194e-01, -2.02653e+00,  1.25694e-02],
# [ 2.77778e-04, -7.29167e-04, -2.00000e-02,  6.19792e-02,  6.21493e-01,  4.92535e-01, -1.87181e+00,  5.76163e+00, -5.04514e-02],
# [-4.30556e-02, -9.64583e-02,  3.07986e-02,  6.45833e-03, -4.23611e-03, -2.16979e-01,  5.30556e-02,  2.62153e-02,  3.32292e-02]] )

# anec= np.array(  [[ 2.25000e+00,  2.50000e-01,  9.47917e-01,  2.95139e-01,  4.51389e-02,  5.52083e-01, -2.08333e-02, -1.73611e-01, -6.94444e-03],
#   [ 2.50000e-01,  5.55556e-01,  1.49306e-01, -4.86111e-02,  3.75000e-01, -1.54198e-10, -2.08333e-02, -1.94444e-01, -2.98611e-01],
#   [ 9.47917e-01,  1.49306e-01,  1.29514e+00, -2.43056e-02,  2.81250e-01, -3.54167e-01,  3.02083e-01,  1.77083e-01,  4.81868e-12],
#   [ 2.95139e-01, -4.86111e-02, -2.43056e-02,  7.43056e-01, -1.77083e-01, -2.98611e-01, -2.01389e-01, -5.55556e-02, -7.63889e-02],
#   [ 4.51389e-02,  3.75000e-01,  2.81250e-01, -1.77083e-01,  4.27083e-01,  1.73611e-01, -4.44444e-01,  9.44444e-01, -1.70139e-01],
#   [ 5.52083e-01, -1.54198e-10, -3.54167e-01, -2.98611e-01,  1.73611e-01,  7.29167e-01,  3.13214e-10,  7.63889e-02,  1.25000e-01],
#   [-2.08333e-02, -2.08333e-02,  3.02083e-01, -2.01389e-01, -4.44444e-01,  3.13214e-10,  9.72222e-01, -9.20139e-01, -4.81868e-12],
#   [-1.73611e-01, -1.94444e-01,  1.77083e-01, -5.55556e-02,  9.44444e-01,  7.63889e-02, -9.20139e-01,  2.25000e+00, -3.47222e-03],
#  [-6.94444e-03, -2.98611e-01,  4.81868e-12, -7.63889e-02, -1.70139e-01,  1.25000e-01, -4.81868e-12, -3.47222e-03,  4.58333e-01]] )





# r1dd = 6.24600480000
# rcdd = 6.692148

# -vfdd=0.1525402687652677 -vqdds=0.4515866553638987 -vqddp=0.5410907176556236 -vqddd=0.61454484045679 -vb0=250.83759285526475 -vp0=1.754628603563741 -vb1=0.0 -vp1=0.3186096185888274 -vcr1=-6.115256911742206 -vcr2=4.305593645542065 -vcr3=-1.0630734624316684 -vndt=2.004546000855847  -vahcp=5.3909715246 -vchcp=8.9010503600 -vq=1.6511032046 -vrmaxh=6.74791552 -vr1dd=6.2460048 -vrcdd=6.692148 -vr1pp=6.2460048 -vrcpp=6.692148


#  rmaxh for hcp 
#    rmaxh_hcp = 6.74791552
#              = 1.21000000 a_hcp_exp 
#  rmaxh for omega 
#    rmaxh_omega = 6.74791552
#                = 0.77273198 a_omega_exp 

# Getting hcp c/a ...
# Using Nelder-Mead

# Optimization terminated successfully.
#          Current function value: -0.591731
#          Iterations: 29
#          Function evaluations: 62
# Got a, c : a=5.3909715246, c=8.9010503600 c/a=1.6511032046. Volume per atom=112.0149449917
# Targets  : a=5.5767896900, c=8.8521008200 c/a=1.5873112152. Volume per atom=119.2107777334

# Obtaining Bandwidth 

#   eval 1 for bandwidth = -0.2507
#   eval 2 for bandwidth = 0.2075
# bandwidth: 0.458 (target: 0.426)

# Getting hcp shear constants ...

#   minimum for c11 = -1.64977e-06
#   c11 = 2.76986
#   minimum for c33 = 5.49534e-07
#   c33 = 2.85488
#   minimum for cp = 4.45804e-07
#   cp = 3.12997
#   minimum for cpp = 1.93473e-07
#   cpp = 3.55478
#   minimum for c44 = -3.56927e-21
#   c44 = 0.646853
# shear constants: c_11=181.7, c_33=187.3, c_44= 42.4, c_12= 79.1, c_13= 67.9, c_p=  51.3, c_pp= 58.3, S= 68.1, R=181.9, H=161.0 
#          target: c_11=176.1, c_33=190.5, c_44= 50.8, c_12= 86.9, c_13= 68.3, c_p=  44.6, c_pp= 57.5, S= 73.1, R=185.4, H=146.7 
#    bulk modulus: 109; target: 110 

# Obtaining bcc Ti quantities

#   trial bcc output from pfit = 0.0
#   VF = 0.954288
#   Epp bcc = 3.96211

# Getting omega phase lattice constants and internal parameter ...

# Using Nelder-Mead

# Optimization terminated successfully.
#          Current function value: -0.881989
#          Iterations: 59
#          Function evaluations: 125

# Got omega : a=8.6000, c=5.3337 c/a=0.6202, u=1.0000. Volume per atom=113.8749
# Targets   : a=8.7325, c=5.3234 c/a=0.6096, u=1.0000. Volume per atom=117.1878
# E_omega - E_hcp = 1.869mRy per atom 
#       GGA Target: -0.735
# bcc:     a=  6.10, K=466 Volume per atom=114
# target:  a=  6.18, K=118,                     
#         E_bcc - E_hcp = 46.680mRy per atom 


#  Build Objective Function
#      ...with L1 norm
#                     predicted       target      squared diff.        p_norm         weight       objective
#   a_hcp       :   5.39097152     5.57678969     0.03452839       0.18581817     1000.00000000      220.34655604 
#   c_hcp       :   8.90105036     8.85210082     0.00239606       0.04894954     1000.00000000       51.34559751 
#   c_11        : 181.74781054   176.10000000    31.89776390       5.64781054       1.00000000       37.54557444 
#   c_33        : 187.32650363   190.50000000    10.07107919       3.17349637       1.00000000       13.24457555 
#   c_44        :  42.44406450    50.80000000    69.82165811       8.35593550       1.00000000       78.17759361 
#   c_12        :  79.05937240    86.90000000    61.47544123       7.84062760       1.00000000       69.31606883 
#   c_13        :  67.91150056    68.30000000     0.15093181       0.38849944       1.00000000        0.53943125 
#   a_omega     :   8.59996086     8.73254342     0.01757814       0.13258256     250.00000000       37.54017428 
#   c_omega     :   5.33366487     5.32343103     0.00010473       0.01023384     250.00000000        2.58464249 
#   u_omega     :   1.00002992     1.00000000     0.00000000       0.00002992       1.00000000        0.00002992 
#   DE (o, hcp) :   1.86909667    -0.73475386     6.78003755       2.60385052       1.00000000        9.38388808 
#   a_bcc       :   6.10483666     6.17948863     0.00557292       0.07465197     500.00000000       40.11244309 
#   bandwidth   :   0.45820000     0.42600000     0.00103684       0.03220000     1000.00000000       33.23684000 
