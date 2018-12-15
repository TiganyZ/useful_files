
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



C_ijkl = -( 0.5 / V_prim_unit_cell ) * ( sum_{p\ne n} )

 C_{ikjl} = -\frac{1}{8\Omega}  \Big\{ 
    &\sum_{p\neq n}\big( X_k^{(p)} - X_k^{(n)} \big) S_{ij}^{(np)}  \big( X_l^{(p)} - X_l^{(n)}  \big) \\
  + &\sum_{p\neq n}\big( X_i^{(p)} - X_i^{(n)} \big) S_{kj}^{(np)}  \big( X_l^{(p)} - X_l^{(n)}  \big) \\
  + &\sum_{p\neq n}\big( X_k^{(p)} - X_k^{(n)} \big) S_{il}^{(np)}  \big( X_j^{(p)} - X_j^{(n)}  \big) \\
  + &\sum_{p\neq n}\big( X_i^{(p)} - X_i^{(n)} \big) S_{kl}^{(np)}  \big( X_j^{(p)} - X_j^{(n)}  \big)  \Big\}


"""

def get_strains(strain):
    strain_names = [' -vexx=', ' -veyy=', ' -vezz=',
                  ' -veyz=', ' -vexz=', ' -vexy=' ]
    strain_command = ' '
    for name, s in zip(strain_names, strain):
        strain_command += name + s + ' '
    return strain_command
    

def strains():

    s_11 = np.array([1, 0, 0, 0, 0, 0])
    s_112 = np.array([0, 1, 0, 0, 0, 0])
    s_33 = np.array([0, 0, 1, 0, 0, 0])
    s_2C11_2C12 = np.array([1, 1, 0, 0, 0, 0])
    s_2_C11_2_C22 =np.array( [1.0,1,0,0,0,0] )
    s_5o4_C11_C12 = np.array( [0.5,1,0,0,0,0] )
    s_C11_C33_2_C13 = np.array( [1,0,1,0,0,0] )
    s_C11_C33_2_C13_2 = np.array( [0,1,1,0,0,0] )
    s_4_C44 = np.array( [0,0,0,0,1,0] )
    s_4_C44_2 = np.array( [0,0,0,1,0,0] )
    s_8_C44 = np.array( [0,0,0,1,1,1] )
    s_4_C66 = np.array( [1,-1,0,0,0,0] )
    s_4C662 = np.array([0, 0, 0, 0, 0, 1])

    

    s_11 = ' -vexx=1' + ' -veyy=0' + ' -vezz=0' +\
        ' -veyz=0' + ' -vexz=0' + ' -vexy=0 '

    s_112 = ' -vexx=0' + ' -veyy=1' + ' -vezz=0' +\
        ' -veyz=0' + ' -vexz=0' + ' -vexy=0 '

    s_11 = ' -vexx=0' + ' -veyy=0' + ' -vezz=0' +\
        ' -veyz=0' + ' -vexz=0' + ' -vexy=0 '

    s_11 = ' -vexx=1' + ' -veyy=0' + ' -vezz=0' +\
        ' -veyz=0' + ' -vexz=0' + ' -vexy=0 '

    s_11 = ' -vexx=1' + ' -veyy=0' + ' -vezz=0' +\
        ' -veyz=0' + ' -vexz=0' + ' -vexy=0 '

    s_11 = ' -vexx=1' + ' -veyy=0' + ' -vezz=0' +\
        ' -veyz=0' + ' -vexz=0' + ' -vexy=0 '

    s_11 = ' -vexx=1' + ' -veyy=0' + ' -vezz=0' +\
        ' -veyz=0' + ' -vexz=0' + ' -vexy=0 '

    s_11 = ' -vexx=1' + ' -veyy=0' + ' -vezz=0' +\
        ' -veyz=0' + ' -vexz=0' + ' -vexy=0 '

    s_11 = ' -vexx=1' + ' -veyy=0' + ' -vezz=0' +\
        ' -veyz=0' + ' -vexz=0' + ' -vexy=0 '

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

    s_2_C11_2_C22 =np.array( [1.0,1,0,0,0,0] )
    # 4

    s_5o4_C11_C12 = np.array( [0.5,1,0,0,0,0] )

    s_C11_C33_2_C13 = np.array( [1,0,1,0,0,0] )

    # 6

    s_C11_C33_2_C13_2 = np.array( [0,1,1,0,0,0] )
    # 7

    s_4_C44 = np.array( [0,0,0,0,1,0] )

    s_4_C44_2 = np.array( [0,0,0,1,0,0] )
    # 10

    s_8_C44 = np.array( [0,0,0,1,1,1] )

    # 11

    s_4_C66 = np.array( [1,-1,0,0,0,0] )
    
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
