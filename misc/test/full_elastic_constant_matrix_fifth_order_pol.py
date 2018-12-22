# This calculates the elastic constants by fitting a fifth order polynomial to the change in energy with respect to strains.
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
    print(cmd)
    print("Energy", etot)
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

#===========================================================================================
############################    Elastic Constant Strains    ################################
#===========================================================================================

def get_strains(strain):
    strain_names = [' -vexx=', ' -veyy=', ' -vezz=',
                    ' -veyz=', ' -vexz=', ' -vexy=']
    strain_command = ' '
    for name, s in zip(strain_names, strain):
        strain_command += name + '{:.1f} '.format(s)
    return strain_command

def get_strain_tensor(strain):
    strain_tensor = np.zeros((3,3))
    for i in range(3):
        strain_tensor[i,i] = strain[i]
    strain_tensor[2,3] = strain[3]
    strain_tensor[3,2] = strain[3]
    strain_tensor[1,3] = strain[4]
    strain_tensor[3,1] = strain[4]
    strain_tensor[1,2] = strain[5]
    strain_tensor[2,1] = strain[5]
    return strain_tensor


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

    plat_strained = np.asarray( [ (  e.dot( pl * alat ) ) / alat  for pl in plat ] )

    np.printoptions(linewidth=200,precision=8)
    print("\nplat strained")
    for i, p in enumerate( plat_strained ):
        print('{: .10f} {: .10f} {: .10f}'.format(p[0], p[1], p[2]) )
    print("\npositions strained")
    for i, p in enumerate( positions_strained ):
        print('{: .10f} {: .10f} {: .10f}'.format(p[0], p[1], p[2]) )
    
    plat_args = get_plat_command(plat_strained, plat_names)
    command = plat_args + position_args
    return positions_strained, plat_strained, command

def fit_poly(alphas, energies, poly_order):
    # Now need to fit a fifth order polynomial to the points. 
    poly_coeffs = np.polyfit( alphas, energies, poly_order )
    # degree  5  4  3  2  1  0
    # coeffs p0 p1 p2 p3 p4 p5
    # slope 5 * p0 * x**4  +  4 * p1 * x**3  +  3 * p2 * x**2  +  2 * p3 * x  +  p4
    # curv  5 * 4 * p0 * x**3  +  4 * 3 * p1 * x**2  +  3 * 2 * p2 * x  +  2 * p3
    value = poly_coeffs[-1]
    slope = poly_coeffs[4]
    curvature = 2. * poly_coeffs[3]
    return value, slope, curvature, poly_coeffs

def get_elast(args, e, alphas, plat, alat, positions, position_names, poly_order=5 ):

    energies = np.zeros(len(alphas))
    for i, h in enumerate( alphas ):
        positions_strained, plat_strained, command = get_strained_configuration(h, e, alat, positions, position_names, plat )
        energies[i] = find_energy(args + command, '',  'get_elast')
    print("Energies")
    print(energies)
    value, slope, curvature, poly_coeffs = fit_poly(alphas, energies, poly_order)
    return value, slope, curvature, energies

def get_Girshick_strains():
    
    strain = np.zeros( (15, 3, 3) )
    tbe_strain = np.zeros((15, 6))

    # 'C11'
    strain[  1-1, 1-1, 1-1 ] = 1.
    strain[  1-1, 1-1, 2-1 ] = 0.
    strain[  1-1, 1-1, 3-1 ] = 0.

    strain[  1-1, 2-1, 1-1 ] = 0.
    strain[  1-1, 2-1, 2-1 ] = 0.
    strain[  1-1, 2-1, 3-1 ] = 0.

    strain[  1-1, 3-1, 1-1 ] = 0.
    strain[  1-1, 3-1, 2-1 ] = 0.
    strain[  1-1, 3-1, 3-1 ] = 0.
    
    tbe_strain[0,:]=[1, 0, 0, 0, 0, 0]


    # 'C11'
    strain[  2-1, 1-1, 1-1 ] = 0.
    strain[  2-1, 1-1, 2-1 ] = 0.
    strain[  2-1, 1-1, 3-1 ] = 0.

    strain[  2-1, 2-1, 1-1 ] = 0.
    strain[  2-1, 2-1, 2-1 ] = 1.
    strain[  2-1, 2-1, 3-1 ] = 0.

    strain[  2-1, 3-1, 1-1 ] = 0.
    strain[  2-1, 3-1, 2-1 ] = 0.
    strain[  2-1, 3-1, 3-1 ] = 0.

    tbe_strain[1,:]=[0, 1, 0, 0, 0, 0]

    
    # 'C33'
    strain[  3-1, 1-1, 1-1 ] = 0.
    strain[  3-1, 1-1, 2-1 ] = 0.
    strain[  3-1, 1-1, 3-1 ] = 0.

    strain[  3-1, 2-1, 1-1 ] = 0.
    strain[  3-1, 2-1, 2-1 ] = 0.
    strain[  3-1, 2-1, 3-1 ] = 0.

    strain[  3-1, 3-1, 1-1 ] = 0.
    strain[  3-1, 3-1, 2-1 ] = 0.
    strain[  3-1, 3-1, 3-1 ] = 1.

    tbe_strain[2,:]=[0, 0, 1, 0, 0, 0]
    
    # '2C11+2C12'
    strain[  4-1, 1-1, 1-1 ] = 1.
    strain[  4-1, 1-1, 2-1 ] = 0.
    strain[  4-1, 1-1, 3-1 ] = 0.

    strain[  4-1, 2-1, 1-1 ] = 0.
    strain[  4-1, 2-1, 2-1 ] = 1.
    strain[  4-1, 2-1, 3-1 ] = 0.

    strain[  4-1, 3-1, 1-1 ] = 0.
    strain[  4-1, 3-1, 2-1 ] = 0.
    strain[  4-1, 3-1, 3-1 ] = 0.

    tbe_strain[3,:]=[1, 1, 0, 0, 0, 0]
    
    # '1.25C11+C12'
    strain[  5-1, 1-1, 1-1 ] = 0.5
    strain[  5-1, 1-1, 2-1 ] = 0.
    strain[  5-1, 1-1, 3-1 ] = 0.

    strain[  5-1, 2-1, 1-1 ] = 0.
    strain[  5-1, 2-1, 2-1 ] = 1.
    strain[  5-1, 2-1, 3-1 ] = 0.

    strain[  5-1, 3-1, 1-1 ] = 0.
    strain[  5-1, 3-1, 2-1 ] = 0.
    strain[  5-1, 3-1, 3-1 ] = 0.

    tbe_strain[4,:]=[0.5, 1, 0, 0, 0, 0]
    
    # 'C11+C33+2C13'
    strain[  6-1, 1-1, 1-1 ] = 1.
    strain[  6-1, 1-1, 2-1 ] = 0.
    strain[  6-1, 1-1, 3-1 ] = 0.

    strain[  6-1, 2-1, 1-1 ] = 0.
    strain[  6-1, 2-1, 2-1 ] = 0.
    strain[  6-1, 2-1, 3-1 ] = 0.

    strain[  6-1, 3-1, 1-1 ] = 0.
    strain[  6-1, 3-1, 2-1 ] = 0.
    strain[  6-1, 3-1, 3-1 ] = 1.

    tbe_strain[5,:]=[1, 0, 1, 0, 0, 0]
    
    # 'C11+C33+2C13'
    strain[  7-1, 1-1, 1-1 ] = 0.
    strain[  7-1, 1-1, 2-1 ] = 0.
    strain[  7-1, 1-1, 3-1 ] = 0.

    strain[  7-1, 2-1, 1-1 ] = 0.
    strain[  7-1, 2-1, 2-1 ] = 1.
    strain[  7-1, 2-1, 3-1 ] = 0.

    strain[  7-1, 3-1, 1-1 ] = 0.
    strain[  7-1, 3-1, 2-1 ] = 0.
    strain[  7-1, 3-1, 3-1 ] = 1.

    tbe_strain[6,:]=[0, 1, 1, 0, 0, 0]
    
    # '2C11+2C12+C33+4C13'
    strain[  8-1, 1-1, 1-1 ] = 1.
    strain[  8-1, 1-1, 2-1 ] = 0.
    strain[  8-1, 1-1, 3-1 ] = 0.

    strain[  8-1, 2-1, 1-1 ] = 0.
    strain[  8-1, 2-1, 2-1 ] = 1.
    strain[  8-1, 2-1, 3-1 ] = 0.

    strain[  8-1, 3-1, 1-1 ] = 0.
    strain[  8-1, 3-1, 2-1 ] = 0.
    strain[  8-1, 3-1, 3-1 ] = 1.

    tbe_strain[7,:]=[1, 1, 1, 0, 0, 0]
    
    # '4C44'
    strain[  9-1, 1-1, 1-1 ] = 0.
    strain[  9-1, 1-1, 2-1 ] = 0.
    strain[  9-1, 1-1, 3-1 ] = 1.

    strain[  9-1, 2-1, 1-1 ] = 0.
    strain[  9-1, 2-1, 2-1 ] = 0.
    strain[  9-1, 2-1, 3-1 ] = 0.

    strain[  9-1, 3-1, 1-1 ] = 1.
    strain[  9-1, 3-1, 2-1 ] = 0.
    strain[  9-1, 3-1, 3-1 ] = 0.

    tbe_strain[8,:]=[0, 0, 0, 0, 1, 0]
    
    # '4C44'
    strain[ 10-1, 1-1, 1-1 ] = 0.
    strain[ 10-1, 1-1, 2-1 ] = 0.
    strain[ 10-1, 1-1, 3-1 ] = 0.

    strain[ 10-1, 2-1, 1-1 ] = 0.
    strain[ 10-1, 2-1, 2-1 ] = 0.
    strain[ 10-1, 2-1, 3-1 ] = 1.

    strain[ 10-1, 3-1, 1-1 ] = 0.
    strain[ 10-1, 3-1, 2-1 ] = 1.
    strain[ 10-1, 3-1, 3-1 ] = 0.

    tbe_strain[9,:]=[0, 0, 0, 1, 0, 0]
    
    # '8C44'
    strain[ 11-1, 1-1, 1-1 ] = 0.
    strain[ 11-1, 1-1, 2-1 ] = 0.
    strain[ 11-1, 1-1, 3-1 ] = 1.

    strain[ 11-1, 2-1, 1-1 ] = 0.
    strain[ 11-1, 2-1, 2-1 ] = 0.
    strain[ 11-1, 2-1, 3-1 ] = 1.

    strain[ 11-1, 3-1, 1-1 ] = 1.
    strain[ 11-1, 3-1, 2-1 ] = 1.
    strain[ 11-1, 3-1, 3-1 ] = 0.

    tbe_strain[10,:]=[0, 0, 0, 1, 1, 0]
    
    # '4C66'
    strain[ 12-1, 1-1, 1-1 ] = 1.
    strain[ 12-1, 1-1, 2-1 ] = 0.
    strain[ 12-1, 1-1, 3-1 ] = 0.

    strain[ 12-1, 2-1, 1-1 ] = 0.
    strain[ 12-1, 2-1, 2-1 ] = -1
    strain[ 12-1, 2-1, 3-1 ] = 0

    strain[ 12-1, 3-1, 1-1 ] = 0
    strain[ 12-1, 3-1, 2-1 ] = 0
    strain[ 12-1, 3-1, 3-1 ] = 0

    tbe_strain[11,:]=[1, -1, 0, 0, 0, 0]


    # '4C66'
    strain[ 13-1, 1-1, 1-1 ] = 0
    strain[ 13-1, 1-1, 2-1 ] = 1
    strain[ 13-1, 1-1, 3-1 ] = 0

    strain[ 13-1, 2-1, 1-1 ] = 1
    strain[ 13-1, 2-1, 2-1 ] = 0
    strain[ 13-1, 2-1, 3-1 ] = 0

    strain[ 13-1, 3-1, 1-1 ] = 0
    strain[ 13-1, 3-1, 2-1 ] = 0
    strain[ 13-1, 3-1, 3-1 ] = 0

    tbe_strain[12,:]=[0, 0, 0, 0, 0, 1]
    
    # '3R'
    strain[ 14-1, 1-1, 1-1 ] = -0.5
    strain[ 14-1, 1-1, 2-1 ] = 0.
    strain[ 14-1, 1-1, 3-1 ] = 0.

    strain[ 14-1, 2-1, 1-1 ] = 0.
    strain[ 14-1, 2-1, 2-1 ] = -0.5
    strain[ 14-1, 2-1, 3-1 ] = 0.

    strain[ 14-1, 3-1, 1-1 ] = 0.
    strain[ 14-1, 3-1, 2-1 ] = 0.
    strain[ 14-1, 3-1, 3-1 ] = 1.

    tbe_strain[13,:]=[-0.5, -0.5, 1, 0, 0, 0]
    
    # '3H'
    strain[ 15-1, 1-1, 1-1 ] = 1.
    strain[ 15-1, 1-1, 2-1 ] = 0.
    strain[ 15-1, 1-1, 3-1 ] = 0.

    strain[ 15-1, 2-1, 1-1 ] = 0.
    strain[ 15-1, 2-1, 2-1 ] = -0.5
    strain[ 15-1, 2-1, 3-1 ] = 0.

    strain[ 15-1, 3-1, 1-1 ] = 0.
    strain[ 15-1, 3-1, 2-1 ] = 0.
    strain[ 15-1, 3-1, 3-1 ] = -0.5

    tbe_strain[14,:]=[1, -0.5, -0.5, 0, 0, 0]
    
    number_of_strains = 15
    
    return strain, tbe_strain, number_of_strains

def get_tbe_strains():
    number_of_strains = 8
    strain = np.zeros((8, 3, 3))
    tbe_strain = np.zeros((number_of_strains, 6))
    # c11:
    strain[0,:,:] = np.array([ [1,0,0], [0,0,0], [0,0,0]  ])
    tbe_strain[0,:]=[1, 0, 0, 0, 0, 0] 
    # c33:
    strain[1,:,:] = np.array([ [0,0,0], [0,0,0], [0,0,1]  ])
    tbe_strain[1,:] = [0, 0, 1, 0, 0, 0]
    # cp:
    strain[2,:,:] = np.array([ [1,0,0], [0,-1,0], [0,0,0]  ])
    tbe_strain[2,:] = [1,-1, 0, 0, 0, 0]
    # cpp:
    strain[3,:,:] = np.array([ [1,0,0], [0,0,0], [0,0,-1]  ])
    tbe_strain[3,:] = [1, 0, -1, 0, 0, 0]
    # c44:
    strain[4,:,:] = np.array([ [0,0,0.5], [0,0,0], [0.5,0,0]  ])
    tbe_strain[4,:] = [0, 0, 0, 0, 1/2., 0 ]
    # R:
    strain[5,:,:] = np.array([ [-0.5,0,0], [0,-0.5,0], [0,0,0]  ])
    tbe_strain[5,:] = [ -1/2., -1/2., 1, 0, 0, 0]
    # H:
    strain[6,:,:] = np.array([ [1,0,0], [0,-0.5,0], [0,0,-0.5]  ])
    tbe_strain[6,:] = [1, -1/2., -1/2., 0, 0, 0]
    # S:
    strain[7,:,:] = np.array([ [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]  ])
    tbe_strain[7,:] = [0, 0, 0, 1/2., 1/2., 1/2.]
    return strain, tbe_strain, number_of_strains

def Girshick_strains(args, alphas, plat, alat, positions, position_names, V  ):

    strain, tbe_strain, number_of_strains = get_Girshick_strains()

    curvature=[0.]
    tmatrix = np.zeros((3,3))
    energy_array = np.zeros( (number_of_strains, len(alphas) ) )
    for i  in range( number_of_strains ):
        for j in range(3):
            for k in range(3):
                tmatrix[ j, k ] = strain[ i, j, k ]
        value, slope, curv, energies = get_elast( args, tmatrix, alphas, plat, alat, positions, position_names, poly_order=5 )
        curvature.append(  curv  )
        energy_array[i, : ] = energies

    c11, c33, c44, c66, c12, c13, K, R, H = get_ec_from_curvature(curvature, V, 'girshick')

    return c11, c33, c44, c66, c12, c13, K, R, H, energy_array

def get_ec_from_curvature(curvature, V, set_of_strains):
    bohr_to_ang =  0.529177
    convert = (13.606 / bohr_to_ang**3) * 160.21766208 / V

    if set_of_strains == 'tony':
        c11 = convert * curvature[1]
        c33 = convert * curvature[2]
        cp  = convert * curvature[3]
        cpp = convert * curvature[4]        
        c44 = convert * curvature[5]
        rr  = convert * curvature[6]
        hh  = convert * curvature[7]        
        ss  = convert * curvature[8]

        c12 = c11 - 2.0 * cp
        c13 = 0.5 * (c11 + c33) - 2.0 * cpp
        c66 = 0.5 * ( c11 - c12 )
        S   = 0.5 * (cp + 2.0 * c44)
        R   = (0.5*c11 + c33 + 0.5*c12 - 2.0*c13)
        H   = ((5.0/4.0)*c11 + 0.25*c33 - c12 - 0.5*c13)
        K   = 2.*c11 + 2.*c12 + c33 + 4.*c13

        print_elastic_constants(c11, c33, c44, c66, c12, c13, K, R, H)

    else:
        c11 = convert * 0.5   * ( curvature[1] + curvature[2] )
        c33 = convert * 1.0   *   curvature[3]
        c12 = convert * 1./3  * ( curvature[4] + curvature[5] - 3.25*c11 )
        c13 = convert * 0.25  * ( curvature[6] + curvature[7] - 2*c11 - 2*c33 )
        c44 = convert * 0.0625* ( curvature[9] + curvature[10] + curvature[11] )
        c66 = convert * 0.125 * ( curvature[ 12 ] + curvature[ 13 ] )
        kk  = convert * curvature[  8 ] / 9.
        rr  = convert * curvature[ 14 ] / 3.
        hh  = convert * curvature[ 15 ] / 3.

        print_elastic_constants(c11, c33, c44, c66, c12, c13, kk, rr, hh)
        K = kk
        R = rr
        H = hh

    return c11, c33, c44, c66, c12, c13, K, R, H

def tony_strains(args, alphas, plat, alat, positions, position_names, V  ):

    strain, tbe_strain, number_of_strains = get_tbe_strains()

    curvature=[0.]
    tmatrix = np.zeros((3,3))
    energy_array = np.zeros( (number_of_strains, len(alphas) ) )
    for i  in range( number_of_strains ):
        for j in range(3):
            for k in range(3):
                tmatrix[ j, k ] = strain[ i, j, k ]
        value, slope, curv, energies = get_elast( args, tmatrix, alphas, plat, alat, positions, position_names, poly_order=5 )
        curvature.append(  curv  )
        energy_array[i, : ] = energies

    c11, c33, c44, c66, c12, c13, K, R, H =  get_ec_from_curvature(curvature, V, 'tony')

    return c11, c33, c44, c66, c12, c13, K, R, H, energy_array

def get_tbe_elastic_constants(args, alphas, plat, alat, clat, positions, position_names, V, girshick=True ):

    if girshick:
        strain, tbe_strain, number_of_strains = get_Girshick_strains()
        set_of_strains = 'girshick'
    else:
        strain, tbe_strain, number_of_strains = get_tbe_strains()
        set_of_strains = 'tony'
        
    curvature=[0.]
    energy_array = np.zeros( (number_of_strains, len(alphas) ) )
    for i, s in enumerate( tbe_strain ):
        print("Strain", i)
        for j, a in enumerate( alphas ):
            # do tbe strains
            strainco = get_strains(s) + ' -vecstr=1 '
            strain_command = ' -valpha=' + str(a) + ' ' +  strainco
            energy_array[i,j] = find_energy(args+strain_command, '', 'tbe_strains')
        value, slope, curv, poly_coeffs = fit_poly(alphas, energy_array[i], 5)
        curvature.append(  curv  )
        
    c11, c33, c44, c66, c12, c13, K, R, H =  get_ec_from_curvature(curvature, V, set_of_strains=set_of_strains)
    return c11, c33, c44, c66, c12, c13, K, R, H, energy_array

def print_elastic_constants(c11, c33, c44, c66, c12, c13, kk, rr, hh, routine='Girshick Routine'):
    c11_exp = 176.1000000
    c33_exp = 190.50000000
    c44_exp = 50.80000000
    c12_exp = 86.90000000
    c13_exp = 68.30000000
    c66_exp = 0.5 * ( c11_exp - c12_exp  )
    kk_exp = (1./9.) * (2 *   c11_exp + c33_exp + 2 *    c12_exp + 4 *   c13_exp)
    rr_exp = (1./3.) * (0.5 * c11_exp + c33_exp + 0.5 *  c12_exp - 2 *   c13_exp)
    hh_exp = (1./3.) * (1.2 * c11_exp + 0.25 * c33_exp - c12_exp - 0.5 * c13_exp)


    print('\n Elastic Constants: {} Applied Strains \n'.format( routine  ))
    print('\n C11 = {:< .10f},   C11_exp = {:< .10f}' .format(
        c11,  c11_exp))
    print(' C33 = {:< .10f},   C33_exp = {:< .10f}' .format(
        c33,  c33_exp))           
    print(' C44 = {:< .10f},   C44_exp = {:< .10f}' .format(
        c44,  c44_exp))           
    print(' C66 = {:< .10f},   C66_exp = {:< .10f}' .format(
        c66,  c66_exp))           
    print(' C12 = {:< .10f},   C12_exp = {:< .10f}' .format(
        c12,  c12_exp))           
    print(' C13 = {:< .10f},   C13_exp = {:< .10f}' .format(
        c13, c13_exp))

    print(' K = {:< .10f},   K_FR = {:< .10f}' .format(kk, kk_exp))
    print(' R = {:< .10f},   R_FR = {:< .10f}' .format(rr, rr_exp))
    print(' H = {:< .10f},   H_FR = {:< .10f} \n ' .format(hh, hh_exp))


args = ' tbe ti -vhcp=1 '
specstrainargs = ' -vspecpos=1 -vspecplat=1 -vforces=1 '
ahcp=5.6361
chcp=9.0429

# args = ' tbe ti -vhcp=1 -vspecpos=1 -vspecplat=1 -vforces=1 -vfdd=0.1525402687652677 -vqdds=0.4515866553638987 -vqddp=0.5410907176556236 -vqddd=0.61454484045679 -vb0=250.83759285526475 -vp0=1.754628603563741 -vb1=0.0 -vp1=0.3186096185888274 -vcr1=-6.115256911742206 -vcr2=4.305593645542065 -vcr3=-1.0630734624316684 -vndt=2.004546000855847  -vahcp=5.3909715246 -vchcp=8.9010503600 -vq=1.6511032046 -vrmaxh=6.74791552 -vr1dd=6.2460048 -vrcdd=6.692148 -vr1pp=6.2460048 -vrcpp=6.692148' 
# ahcp=5.3909715246
# chcp=8.9010503600


q = chcp/ahcp


rotation = np.array([[np.sqrt(3)/2., 0.5,  0.],
                     [-0.5, np.sqrt(3)/2., 0.],
                     [0., 0.,  1.]])
rotation = np.eye(3)

h = 0.0001
alphas = np.linspace(-h, h, 9  )
alphas = np.linspace(-0.01, 0.01, 11  )

X_n = rotation.dot( np.array([0.,0.,0.]) ) 
X_p = rotation.dot( np.array( [ 1./(2*np.sqrt(3)) , -1/2., q/2 ] ) ) 

positions = np.asarray( [ X_n, X_p ] )
position_names = [ [ 'ai',  'aj',  'ak'  ],
                   [ 'aii', 'ajj', 'akk' ] ]
second_order = True

plat = np.array([ [     0,         -1,  0 ],
                   [np.sqrt(3)/2,  0.5,  0 ],
                   [     0,         0,   q ] ] )

plat_comm = get_plat_command(plat, rotation=rotation)

print( plat_comm ) 

V = (3**(0.5) / 2.) * ahcp**2 * chcp


print("\nplat")
for i, p in enumerate( plat ):
    print('{: .10f} {: .10f} {: .10f}'.format(p[0], p[1], p[2]) )
print("\npositions")
for i, p in enumerate( positions ):
    print('{: .10f} {: .10f} {: .10f}'.format(p[0], p[1], p[2]) )


# c11, c33, c44, c66, c12, c13, K, R, H, energy_array= get_tbe_elastic_constants(args, alphas, plat, ahcp, chcp, positions, position_names, V, girshick=True )

# fig, axes = plt.subplots(3, 5)
# fig.suptitle('Girshick tbe Strains')
# for i, en in enumerate( energy_array ):
#     print("plotting")
#     print(en)
#     j = i//5
#     k = i % 5
#     print(j,k)
#     ax = axes[j,k]
#     print(ax)
#     ax.set_title('{:d}'.format(i) )    
#     ax.plot( alphas, en, 'bo' )
#     poly_coeffs = np.polyfit( alphas, en, 5 )
#     poly = np.poly1d(poly_coeffs)
#     pl = [ poly(ai) for ai in np.linspace(alphas[0], alphas[-1], 100)   ]
#     ax.plot( np.linspace(alphas[0], alphas[-1], 100), pl  )


# c11, c33, c44, c66, c12, c13, K, R, H, energy_array = get_tbe_elastic_constants(args, alphas, plat, ahcp, chcp, positions, position_names, V, girshick=False )
# fig, axes = plt.subplots(2, 4)
# fig.suptitle('Tony tbe Strains')
# for i, en in enumerate( energy_array ):
#     print("plotting")
#     print(en)
#     j = i//4
#     k = i % 4
#     print(j,k)
#     ax = axes[j,k]
#     print(ax)
#     ax.set_title('{:d}'.format(i) )    
#     ax.plot( alphas, en, 'bo' )
#     poly_coeffs = np.polyfit( alphas, en, 5 )
#     poly = np.poly1d(poly_coeffs)
#     pl = [ poly(ai) for ai in np.linspace(alphas[0], alphas[-1], 100)   ]
#     ax.plot( np.linspace(alphas[0], alphas[-1], 100), pl  )

c11, c33, c44, c66, c12, c13, kk, rr, hh, energy_array = tony_strains(args + specstrainargs, alphas, plat, ahcp, positions, position_names, V  )

fig, axes = plt.subplots(2, 4)
fig.suptitle('tony Strains')
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


# c11, c33, c44, c66, c12, c13, kk, rr, hh, energy_array = Girshick_strains(args + specstrainargs, alphas, plat, ahcp, positions, position_names, V  )


# fig, axes = plt.subplots(3, 5)
# fig.suptitle('Girshick Strains')
# for i, en in enumerate( energy_array ):
#     print("plotting")
#     print(en)
#     j = i//5
#     k = i % 5
#     print(j,k)
#     ax = axes[j,k]
#     print(ax)
#     ax.set_title('{:d}'.format(i) )
#     ax.plot( alphas, en, 'bo' )
#     poly_coeffs = np.polyfit( alphas, en, 5 )
#     poly = np.poly1d(poly_coeffs)
#     pl = [ poly(ai) for ai in np.linspace(alphas[0], alphas[-1], 100)   ]
#     ax.plot( np.linspace(alphas[0], alphas[-1], 100), pl  )
plt.show()



# ect = np.array( [ [30443.30153,  1379.14285, 10726.66665,  1532.38095,  3984.19047, 16243.23806,  1532.38095,  3626.63491,   766.19047],
#                   [ 1379.14285,  3013.68253,     0.,          0.,       1481.30158,     0.,       2656.12698,  3269.07936,  1481.30158],
#                   [10726.66665,     0.,      32792.95232,  8785.65078,     0.,       7048.95236,     0.,          0.,          0.     ],
#                   [ 1532.38095,     0.,       8070.53967,  5465.49205,     0.,       2196.4127,      0.,          0.,          0.     ],
#                   [    0.,       1481.30158,     0.,          0.,       1123.74603,   715.11111,   715.11111,  9909.3968,    408.63493],
#                   [14302.22219,     0.,       7048.95236,  2196.41269,   306.47619,     0.,       3728.79364,  7304.34919,     0.     ],
#                   [ 1532.38095,     0.,       1430.22222,     0.,        715.11111,  3728.79364,  5056.85714,     0.,          0.     ],
#                   [ 3626.63491,  3269.07936,     0.,          0.,       8734.57141,  7304.34919,     0.,      81318.34904,   306.4762 ],
#                   [ 1174.82539,  1072.66666,     0.,          0.,        408.63492,     0.,          0.,        102.15873,     0.     ] ] )

# en1 = np.array( [ -0.08900295,-0.08913350,-0.08926374,-0.08939367,-0.08952329,-0.08965259,-0.08978159,-0.08991028,-0.09003866 ] )
# en2 = np.array( [ -0.08960353,-0.08958355,-0.08956352,-0.08954343,-0.08952329,-0.08950308,-0.08948282,-0.08946251,-0.08944214 ] )
# en3 = np.array( [ -0.08933400,-0.08938156,-0.08942896,-0.08947620,-0.08952329,-0.08957022,-0.08961699,-0.08966361,-0.08971008 ] )
# en4 = np.array( [ -0.08908264,-0.08919346,-0.08930384,-0.08941378,-0.08952329,-0.08963236,-0.08974099,-0.08984919,-0.08995696 ] )
# en5 = np.array( [ -0.08934371,-0.08938886,-0.08943384,-0.08947865,-0.08952329,-0.08956776,-0.08961206,-0.08965620,-0.08970017 ] )
# en6 = np.array( [ -0.08881205,-0.08899087,-0.08916901,-0.08934648,-0.08952329,-0.08969943,-0.08987490,-0.09004970,-0.09022385 ] )
# en7 = np.array( [ -0.08941403,-0.08944170,-0.08946914,-0.08949633,-0.08952329,-0.08955000,-0.08957648,-0.08960272,-0.08962872 ] )
# en8 = np.array( [ -0.08889153,-0.08905071,-0.08920906,-0.08936658,-0.08952329,-0.08967917,-0.08983425,-0.08998850,-0.09014195 ] )
# en9 = np.array( [ -0.08946144,-0.08947772,-0.08949345,-0.08950864,-0.08952329,-0.08953740,-0.08955097,-0.08956400,-0.08957649 ] )
# en10 = np.array( [ -0.08943942,-0.08946065,-0.08948171,-0.08950259,-0.08952329,-0.08954381,-0.08956414,-0.08958430,-0.08960428])
# en11 = np.array( [ -0.08937967,-0.08941626,-0.08945240,-0.08948807,-0.08952329,-0.08955805,-0.08959235,-0.08962619,-0.08965957 ] ) 
# en12 = np.array( [ -0.08892234,-0.08907303,-0.08922341,-0.08937350,-0.08952329,-0.08967277,-0.08982196,-0.08997085,-0.09011944 ] ) 
# en13 = np.array( [ -0.08938150,-0.08941709,-0.08945259,-0.08948798,-0.08952329,-0.08955850,-0.08959361,-0.08962863,-0.08966356 ] ) 
# en14 = np.array( [-0.08955261,-0.08954551,-0.08953825,-0.08953085,-0.08952329,-0.08951558,-0.08950771,-0.08949970,-0.08949154])
# en15 = np.array( [ -0.08905721,-0.08917409,-0.08929072,-0.08940712,-0.08952329,-0.08963921,-0.08975491,-0.08987036,-0.08998558 ] )

# enlist = [en1 ,
#           en2 ,
#           en3 ,
#           en4 ,
#           en5 ,
#           en6 ,
#           en7 ,
#           en8 ,
#           en9 ,
#           en10,
#           en11,
#           en12,
#           en13,
#           en14,
#           en15 ]

# energy_array = np.array( [ 
# # Strain 0
# np.array( [ -0.12754049,-0.12791091,-0.12827002,-0.12861785,-0.12895441,-0.12927973,-0.12959383,-0.12989675,-0.13018851,-0.13046914,-0.13073870 ] ),
# # Strain 1                                                                                                                                         ,
# np.array( [ -0.12753988,-0.12791047,-0.12826973,-0.12861767,-0.12895433,-0.12927973,-0.12959391,-0.12989691,-0.13018876,-0.13046952,-0.13073923 ] ),
# # Strain 2                                                                                                                                         ,
# np.array( [ -0.12787715,-0.12817992,-0.12847157,-0.12875209,-0.12902148,-0.12927973,-0.12952685,-0.12976285,-0.12998773,-0.13020151,-0.13040420 ] ),
# # Strain 3                                                                                                                                         ,
# np.array( [ -0.12569477,-0.12647399,-0.12722201,-0.12793893,-0.12862480,-0.12927973,-0.12990382,-0.13049718,-0.13105992,-0.13159220,-0.13209415 ] ),
# # Strain 4                                                                                                                                         ,
# np.array( [ -0.12665271,-0.12721484,-0.12775857,-0.12828393,-0.12879097,-0.12927973,-0.12975026,-0.13020262,-0.13063687,-0.13105306,-0.13145128 ] ),
# # Strain 5                                                                                                                                         ,
# np.array( [ -0.12604406,-0.12675100,-0.12742804,-0.12807517,-0.12869240,-0.12927973,-0.12983719,-0.13036482,-0.13086264,-0.13133071,-0.13176910 ] ),
# # Strain 6                                                                                                                                         ,
# np.array( [ -0.12604352,-0.12675061,-0.12742777,-0.12807500,-0.12869232,-0.12927973,-0.12983728,-0.13036499,-0.13086292,-0.13133113,-0.13176970 ] ),
# # Strain 7                                                                                                                                         ,
# np.array( [ -0.12410528,-0.12525441,-0.12634641,-0.12738127,-0.12835904,-0.12927973,-0.13014341,-0.13095015,-0.13170003,-0.13239316,-0.13302968 ] ),
# # Strain 8                                                                                                                                         ,
# np.array( [ -0.12915029,-0.12919697,-0.12923321,-0.12925907,-0.12927457,-0.12927973,-0.12927457,-0.12925907,-0.12923321,-0.12919697,-0.12915029 ] ),
# # Strain 9                                                                                                                                         ,
# np.array( [ -0.12915030,-0.12919697,-0.12923321,-0.12925907,-0.12927457,-0.12927973,-0.12927457,-0.12925907,-0.12923321,-0.12919697,-0.12915030 ] ),
# # Strain 10                                                                                                                                        ,
# np.array( [ -0.12902027,-0.12911394,-0.12918660,-0.12923838,-0.12926940,-0.12927973,-0.12926940,-0.12923838,-0.12918660,-0.12911394,-0.12902027 ] ),
# # Strain 11                                                                                                                                        ,
# np.array( [ -0.12910554,-0.12916818,-0.12921698,-0.12925187,-0.12927280,-0.12927973,-0.12927263,-0.12925146,-0.12921621,-0.12916685,-0.12910338 ] ),
# # Strain 12                                                                                                                                        ,
# np.array( [ -0.12910376,-0.12916695,-0.12921616,-0.12925138,-0.12927257,-0.12927973,-0.12927285,-0.12925193,-0.12921697,-0.12916800,-0.12910503 ] ),
# # Strain 13                                                                                                                                        ,
# np.array( [ -0.12947342,-0.12945734,-0.12942994,-0.12939120,-0.12934113,-0.12927973,-0.12920701,-0.12912298,-0.12902764,-0.12892100,-0.12880308 ] ),
# # Strain 14
# np.array( [ -0.12897867,-0.12906023,-0.12913114,-0.12919137,-0.12924090,-0.12927973,-0.12930784,-0.12932521,-0.12933185,-0.12932774,-0.12931289 ] ) ] )




# energy_array = np.array( [ 
#  np.array( [-0.1275, -0.1279, -0.1283, -0.1286, -0.129,  -0.1293, -0.1296, -0.1299, -0.1302, -0.1305, -0.1307] ),
#  np.array( [-0.1275, -0.1279, -0.1283, -0.1286, -0.129,  -0.1293, -0.1296, -0.1299, -0.1302, -0.1305, -0.1307] ),
#  np.array( [-0.1279, -0.1282, -0.1285, -0.1288, -0.129,  -0.1293, -0.1295, -0.1298, -0.13,   -0.1302, -0.1304] ),
#  np.array( [-0.1257, -0.1265, -0.1272, -0.1279, -0.1286, -0.1293, -0.1299, -0.1305, -0.1311, -0.1316, -0.1321] ),
#  np.array( [-0.1267, -0.1272, -0.1278, -0.1283, -0.1288, -0.1293, -0.1298, -0.1302, -0.1306, -0.1311, -0.1315] ),
#  np.array( [-0.126,  -0.1268, -0.1274, -0.1281, -0.1287, -0.1293, -0.1298, -0.1304, -0.1309, -0.1313, -0.1318] ),
#  np.array( [-0.126,  -0.1268, -0.1274, -0.1281, -0.1287, -0.1293, -0.1298, -0.1304, -0.1309, -0.1313, -0.1318] ),
#  np.array( [-0.1241, -0.1253, -0.1263, -0.1274, -0.1284, -0.1293, -0.1301, -0.131,  -0.1317, -0.1324, -0.133 ] ),
#  np.array( [-0.1292, -0.1292, -0.1292, -0.1293, -0.1293, -0.1293, -0.1293, -0.1293, -0.1292, -0.1292, -0.1292] ),
#  np.array( [-0.1292, -0.1292, -0.1292, -0.1293, -0.1293, -0.1293, -0.1293, -0.1293, -0.1292, -0.1292, -0.1292] ),
#  np.array( [-0.129,  -0.1291, -0.1292, -0.1292, -0.1293, -0.1293, -0.1293, -0.1292, -0.1292, -0.1291, -0.129 ] ),
#  np.array( [-0.1291, -0.1292, -0.1292, -0.1293, -0.1293, -0.1293, -0.1293, -0.1293, -0.1292, -0.1292, -0.1291] ),
#  np.array( [-0.1291, -0.1292, -0.1292, -0.1293, -0.1293, -0.1293, -0.1293, -0.1293, -0.1292, -0.1292, -0.1291] ),
#  np.array( [-0.1295, -0.1295, -0.1294, -0.1294, -0.1293, -0.1293, -0.1292, -0.1291, -0.129,  -0.1289, -0.1288] ),
#  np.array( [-0.129,  -0.1291, -0.1291, -0.1292, -0.1292, -0.1293, -0.1293, -0.1293, -0.1293, -0.1293, -0.1293] ) ] )



# ###    This is the array of energeies using the same strains as above but using tony's strains
# energy_array2 = np.array( [ 
# np.array( [-0.1275, -0.1279, -0.1283, -0.1286, -0.129,  -0.1293, -0.1296, -0.1299, -0.1302, -0.1305, -0.1307] )

# np.array( [-0.1279, -0.1282, -0.1285, -0.1288, -0.129,  -0.1293, -0.1295, -0.1298, -0.13,   -0.1302, -0.1304] )

# np.array( [-0.1291, -0.1292, -0.1292, -0.1293, -0.1293, -0.1293, -0.1293, -0.1293, -0.1292, -0.1292, -0.1291] )

# np.array( [-0.1288, -0.1289, -0.129,  -0.1291, -0.1292, -0.1293, -0.1293, -0.1294, -0.1294, -0.1294, -0.1294] )

# np.array( [-0.1292, -0.1293, -0.1293, -0.1293, -0.1293, -0.1293, -0.1293, -0.1293, -0.1293, -0.1293, -0.1292] )

# np.array( [-0.1295, -0.1295, -0.1294, -0.1294, -0.1293, -0.1293, -0.1292, -0.1291, -0.129,  -0.1289, -0.1288] )

# np.array( [-0.129,  -0.1291, -0.1291, -0.1292, -0.1292, -0.1293, -0.1293, -0.1293, -0.1293, -0.1293, -0.1293] )

# np.array( [-0.1292, -0.1292, -0.1292, -0.1293, -0.1293, -0.1293, -0.1293, -0.1293, -0.1292, -0.1292, -0.1292] ) ] ) 
