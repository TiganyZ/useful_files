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
        [ c12,  c12,  c13,  0.,  0.,  0.,  0.,  0.,  0. ],
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


C = c_transform(C_arr, T)
#C = C_arr
print("Transformed C matrix \n")
print( C , "\n")
print(type(C))

pure  = True
screw = False
plot  = True


a = 0.29012e1

b = sci.array([0., a, 0.])


dis = gen_disl_u(C, b, pure, screw, plot)


# nrec 4
#        (2.907089643807246, 4.65853166251347, 99.11855345623607)
#     Find Latpars Grid:
#     n_grid = [30 30]
#      a_l---a_u = 2.9070891860435744---2.907090101570918
#      c_l---c_u = 4.658528610755658---4.658534714271282

# nrec 5
#        (2.921125404094828, 4.612634698275863, 99.5704106633822)
#     Find Latpars Grid:
#     n_grid = [30 30]
#      a_l---a_u = 2.9210765759698276---2.921174232219828
#      c_l---c_u = 4.612244073275862---4.613025323275863
