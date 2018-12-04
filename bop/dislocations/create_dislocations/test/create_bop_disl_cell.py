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
rc("text", usetex = True)
import os
import scipy as sci
sci.set_printoptions(linewidth=200, precision = 4)



def contract_index(i, j):
    
    if i == j:  
        if i == 1- 1:
            i1 = 1- 1
        elif i == 2- 1:
            i1 = 2- 1
        elif i == 3- 1:
            i1 = 3- 1
    elif i == 1- 1:
        if j == 2- 1:
            i1 = 6- 1
        elif j == 3- 1:
            i1 = 8- 1
    elif i == 2- 1:
        if j == 3- 1:
            i1 = 4- 1
        elif j == 1- 1:
            i1 = 9- 1
    elif i == 3- 1:
        if j == 1- 1:
            i1 = 5- 1
        elif j == 2- 1:
            i1 = 7- 1
    return i1

def get_Q_rot(a):
    b = np.zeros((9,9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    i1 = contract_index(i,j)
                    i2 = contract_index(k,l)
                    b[i1][i2] = a[k][i] * a[l][j]
    return b

def c_transform(C, a):
    Q   = get_Q_rot(a)
    C_t = Q.T.dot( C.dot( Q ) )
    return C_t


def u_edge(C, b, x, y):
    C11bar = ( C[1][1] * C[2][2] )**(0.5)
    phi = 0.5 * np.arccos( ( C[1][2]**2   +   2 * C[1][2] * C[6][6]   -  C11bar**2  ) / (2 * C11bar * C[6][6] ) )
    lam = ( C[1][1] / C[2][2] )**(0.25)

    q = np.sqrt(  x**2  +   2*x*y * lam * np.cos(phi)   +   y**2 * lam**2  )
    t = np.sqrt(  x**2  -   2*x*y * lam * np.cos(phi)   +   y**2 * lam**2  ) 
    

    ux =  - ( b[0] / (4. * np.pi) ) * (    np.arctan2(   ( 2*x*y * lam * np.sin(phi)) ,     ( x**2 - lam**2 * y**2 )     )
                   +  ( C11bar**2 - C[1][2]**2 ) * np.log(  q / t ) / ( 2. * C11bar * C[6][6] * np.sin(2. * phi) ) )
    
    ux += - ( b[1] / ( 4. * np.pi * lam * C11bar * np.sin( 2. * phi )  ) )  * (
               ( C11bar - C[1][2] ) * np.cos(phi) * np.log(q * t)  -  ( ( C11bar + C[1][2] ) * np.sin(phi)
                                * np.arctan2(  ( x**2 * np.sin(2. * phi) ) ,    ( lam**2 * y**2 - x**2 * np.cos(2. * phi)  )   )    )    ) 

    
    uy = - (  b[1] / (4. * np.pi) ) * (    np.arctan2( ( 2*x*y * lam * np.sin(phi) ) , ( x**2 - lam**2 * y**2 ) )
                   -  ( ( C11bar**2 - C[1][2]**2 ) * np.log(  q / t ) ) / ( 2. * C11bar * C[6][6] * np.sin(2. * phi) ) )
    
    uy +=  ( lam *  b[0] / ( 4. * np.pi * C11bar * np.sin( 2. * phi )  ) )  * (
               ( C11bar - C[1][2] ) * np.cos(phi) * np.log(q * t)  -  ( ( C11bar + C[1][2] ) * np.sin(phi)
                   * np.arctan2( ( y**2 * lam**2 * np.sin(2. * phi) ) , (  x**2 - y**2 * lam**2 * np.cos(2. * phi)  ) ) )    )

    return ux, uy

def u_screw(C, bz, x, y):
    bz = bz[-1]
    # Displacement is only in the z directon
    uz = -(bz/(2. * np.pi)) * np.arctan2( np.sqrt( C[4][4]*C[5][5] - C[4][5] ) * y  ,  ( C[4][4] * x - C[4][5] * y  )  )

    return uz




def get_Disl_edge(C, b):
    length = 200
    u, v = sci.meshgrid(sci.linspace(-5 * a, 5 * a, length), sci.linspace(-5 * a, 5 * a, length))
    dis = [ sci.zeros((length, length), dtype=np.float64),
            sci.zeros((length, length), dtype=np.float64) ]

    for i in range(length):
        for j in range(length):
            x = u[i][j]
            y = v[i][j]
            ux, uy = u_edge(C, b, x, y)
            dis[0][i][j] = ux
            dis[1][i][j] = uy

    return dis

def get_Disl_screw(C, bz):
    length = 200
    u, v = sci.meshgrid(sci.linspace(-5 * a, 5 * a, length), sci.linspace(-5 * a, 5 * a, length))
    dis = sci.zeros((length, length), dtype=np.float64)

    for i in range(length):
        for j in range(length):
            x = u[i][j]
            y = v[i][j]
            dis[i][j] = u_screw(C, b, x, y)    

    return dis


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

def gen_disl_u(C, b, pure, screw, plot):
    if pure and screw:
        print("This is a pure Screw: b = %s" %(b[2]))
        dis  = get_Disl_screw(C, b)
        if plot:
             plot_dis_screw(dis)
    elif pure and not screw:
        dis = get_Disl_edge(C,b)
        if plot:
            plot_dis_edge(dis)
    else:
        dis_z  = get_Disl_screw(C,b)
        dis_xy = get_Disl_edge(C,b[2])
        if plot:
            plot_dis_screw(dis)
            plot_dis_edge(dis)
    return  dis

 
c11 = sci.dtype(sci.float128)
c12 = sci.dtype(sci.float128)
c13 = sci.dtype(sci.float128)
c33 = sci.dtype(sci.float128)
c44 = sci.dtype(sci.float128)


##  Elastic constants in units of 10^{9} Pa
c11 = 1.761e2
c12 = 0.868e2
c13 = 0.682e2
c33 = 1.905e2
c44 = 0.508e2
c66 = 0.450e2

C_arr = sci.array(
    [
        [ c11,  c12,  c13,  0.,  0.,  0.,  0.,  0.,  0. ],
        [ c12,  c11,  c13,  0.,  0.,  0.,  0.,  0.,  0. ],
        [ c13,  c13,  c33,  0.,  0.,  0.,  0.,  0.,  0. ],
        [  0.,   0.,   0.,  c44, 0.,  0., c44,  0.,  0. ],
        [  0.,   0.,   0.,  0., c44,  0.,  0., c44,  0. ],
        [  0.,   0.,   0.,  0.,  0., c66,  0.,  0., c66 ],
        [  0.,   0.,   0., c44,  0.,  0., c44,  0.,  0. ],
        [  0.,   0.,   0.,  0., c44,  0.,  0., c44,  0. ],
        [  0.,   0.,   0.,  0.,  0., c66,  0.,  0., c66 ]
    ]
)

T = np.array( [ [0.8660254,    -0.5,       0.0 ],      # Dislocation Coordinate system
                [  0.5,    -0.8660254,     0.0 ],
                [  0.0,          0.0,      -1.0 ]       ])


T = np.array( [ [  1.,    0.,    0. ],      # Dislocation Coordinate system
                [  0.,    0.,    1. ],
                [  0.,    -1.,    0. ]       ])


# T = np.array( [ [  1.,    0.,    0. ],      # Dislocation Coordinate system
#                 [  0.,    1.,    0. ],
#                 [  0.,    0.,    1. ]       ])
C = c_transform(C_arr, T)
#C = C_arr
print("Transformed C matrix \n")
print( C , "\n")
print(type(C))

pure  = True
screw = True
plot  = True


a = 0.29012e1

b = sci.array([0., 0., a])




dis = gen_disl_u(C, b, pure, screw, plot)
pp.show()





#Length array
l = np.array( [ 5.026674058492405,
                2.9021516207990987,
                4.679881023538525  ] )

# Lattice Vectors
a = np.array( [  [  1.0, 0.0, 0.0 ],
                 [  0.0, 1.0, 0.0 ],
                 [  0.0, 0.0, 1.0 ]  ] )

# Unit cell
uc = np.array(  [ [ 0.5, 1.0, 0.0 ],
                  [ 0.0, 0.5, 0.0 ],
                  [ 0.166666666667, 1.0, 0.5 ],
                  [ 0.666666666667, 0.5, 0.5 ]   ]  )

luc = len(uc)

# Number of periodic images
nx = 13#4
ny = 2#2
nz = 14#5

# Number of unit cells before inert atoms appear
ninertx = np.array( [ 1, 11 ] )
ninerty = np.array( [ 0, 2  ] )
ninertz = np.array( [ 1, 12 ] )

ninert = nx * ny * nz * luc -  ninertx[-1] * ninerty[-1] * ninertz[-1] * luc 

n_atoms_tot = luc * nx * ny * nz

n_atoms = n_atoms_tot - ninert

flen = np.zeros(l.shape)
flen[0] = nx * l[0]
flen[1] = ny * l[1]
flen[2] = nz * l[2]

rcore = 0.5 *  flen
rcore[1] = 0.
rcphi = 90. * np.pi / 180.

atoms = np.zeros( ( n_atoms , 3) )

atoms[:luc,:] = uc

cwd = os.getcwd()
gen_disl_path = './generated_dislocations'
if os.path.isdir(gen_disl_path) is False:
    os.mkdir( gen_disl_path )
    
try:
    os.chdir(gen_disl_path)
    print("Changed directory from {} to {}".format(cwd, gen_disl_path) )    
except TypeError:
    print("Could not change directory from {} to {}".format(cwd, gen_disl_path) )
    

cell_file       = "pris_scr_disl_3x_2y_14z_t.in"
cell_inert_file = "pris_scr_disl_13x_2y_14z_inert.in"
xyz_file        = "pris_scr_disl_13x_2y_14z_t.xyz"
species   = "Ti"
out_file       = open(cell_file,       mode='w+')
out_inert_file = open(cell_inert_file, mode='w+')
out_xyz_file   = open(xyz_file,        mode='w+')

out_file.write( " tig   hcp\n"  )
out_file.write( " 1.0 0.0 0.0\n"  )
out_file.write( " 0.0 1.0 0.0\n"  )
out_file.write( " 0.0 0.0 1.0\n"  )
out_file.write( " len " + str(flen[0])  
                  + " " + str(flen[1]) 
                  + " " + str(flen[2]) + "\n")
out_file.write( " latpar 1.0\n"  )
out_file.write( " nd " + str(n_atoms) + "\n"  )

out_xyz_file.write(  str(n_atoms_tot) + "\n"  )
out_xyz_file.write(  'Lattice=" ' + str(flen[0]) + ' 0.0 0.0   0.0 ' + str(flen[1]) + ' 0.0   0.0 0.0 ' + str(flen[2]) + '" Properties=species:S:1:pos:R:3 \n'  )

counter = 0
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            for p in range(luc):
                if screw:
                    r1 =  ( uc[p][0] + i ) * l[0] 
                    r2 =  ( uc[p][1] + j ) * l[1]
                    r3 =  ( uc[p][2] + k ) * l[2]
                    r12  = np.sqrt( ( r1 - rcore[0]) **2 + ( r3 - rcore[2] ) **2)
                    s  =   u_screw(C, b, ( r12 * np.cos(rcphi + np.arctan2(r3-rcore[2],r1-rcore[0]) ) ),  ( r12 * np.sin(rcphi + np.arctan2(r3-rcore[2],r1-rcore[0]) ) ) )

                    
                    r1 =  ( uc[p][0] + i ) * l[0] 
                    r2 =  ( uc[p][1] + j ) * l[1] +  s
                    r3 =  ( uc[p][2] + k ) * l[2] # + u_screw(C, b, ( r12 * np.cos(rcphi + np.arctan2(r2-rcore[1],r1-rcore[0]) ) ), #( r1 - rcore[0] )
#                                                                   ( r12 * np.sin(rcphi + np.arctan2(r2-rcore[1],r1-rcore[0]) ) ) ) 
                    #r3 =  ( uc[p][2] + k ) * l[2]  + u_screw(C, b, ( r1 - rcore[0]) , r2 - rcore[1])
                elif edge:
                    r1 =  ( uc[p][0] + i ) * l[0]  + u_edge(C, b, r1 - rcore[0], r2 - rcore[1])
                    r2 =  ( uc[p][1] + j ) * l[1]  + u_edge(C, b, r1 - rcore[0], r2 - rcore[1])
                    r3 =  ( uc[p][2] + k ) * l[2]
                else:
                    r1 =  ( uc[p][0] + i ) * l[0] 
                    r2 =  ( uc[p][1] + j ) * l[1] 
                    r3 =  ( uc[p][2] + k ) * l[2]

                #print("r1", r1, r1/flen[0], r1/l[0])
                #print("r2", r2, r1/flen[1], r2/l[1])
                #print("r3", r3, r1/flen[2], r3/l[2])
                c0 = i < ninertx[0] or j < ninerty[0] or k < ninertz[0]
                c1 = i > ninertx[1] or j > ninerty[1] or k > ninertz[1]
                if c0 or c1:
                    counter += 1
                    out_inert_file.write( " " + species
                                + " " + str( r1 / flen[0] )
                                + " " + str( r2 / flen[1] )
                                + " " + str( r3 / flen[2] )
                                + " 0.0 0.0"
                                + " \n"  )
                    out_xyz_file.write( " " + species + "n"
                                    + " " + str( r1 )
                                    + " " + str( r2 )
                                    + " " + str( r3 )
                                    + " \n"  )                    
                else:
                    out_file.write( " " + species
                                    + " " + str( r1 / flen[0] )
                                    + " " + str( r2 / flen[1] )
                                    + " " + str( r3 / flen[2] )
                                    + " 0.0 0.0"
                                    + " \n"  )
                    out_xyz_file.write( " " + species 
                                    + " " + str( r1 )
                                    + " " + str( r2 )
                                    + " " + str( r3 )
                                    + " \n"  )
                #print("Scaled Atomic Position: ", (uc[p][0] + i), (uc[p][1] + j), (uc[p][2] + k))
                #print("Cartesian Atomic Position: ", (uc[p][0] + i)*l[0], (uc[p][1] + j)*l[1], (uc[p][2] + k)*l[2])                
                
print("Number of Inert atoms = %s" %(counter))
print("Calculated number of Inert atoms = %s" %( nx * ny * nz *luc -  ninertx[-1] * ninerty[-1] * ninertz[-1] * luc ))
print("ninert x,y,z",nx, ny, nz, ninertx[-1] , ninerty[-1] , ninertz[-1], luc, nx * ny * nx *luc ,  ninertx[-1] * ninerty[-1] * ninertz[-1] * luc )
out_inert_file.close()
out_inert_file = open(cell_inert_file, mode='r+')

out_file.write(" ninert " + str(ninert) + "\n" )
out_file.write( out_inert_file.read() )
out_inert_file.close()

out_file.write("\n nullheight       0.000000000000\n\n"  )

out_file.write(" kspace_conf\n"  )
out_file.write("   0 t         ! symmode symetry_enabled\n"  )
out_file.write("   12 10 12    ! vnkpts\n"  )
out_file.write("    t t t       ! offcentre\n\n"  )

out_file.write("dislocation_conf\n"  )
out_file.write("   0    ! gfbcon\n"  )
out_file.write("   0.0  ! radiusi\n"  )
out_file.write("   -100.0    100.0    -100.0  100.0\n"  )
out_file.write("    0    ! gfbcon\n"  )
out_file.write("   0.0  ! radiusi\n"  )
out_file.write("   -100.0    100.0    -100.0  100.0\n"  )
out_file.write("    0    ! gfbcon\n"  )
out_file.write("   0.0  ! radiusi\n"  )
out_file.write("   -100.0    100.0    -100.0  100.0\n"  )
out_file.write("   0    ! gfbcon\n"  )
out_file.write("   0.0  ! radiusi\n"  )
out_file.write("   -100.0    100.0    -100.0  100.0\n"  )
    
    
out_file.close()
out_xyz_file.close()

os.chdir(cwd)
