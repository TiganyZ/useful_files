import numpy as np
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cmd_result(cmd):
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result,err = proc.communicate() 
    result = result.decode("utf-8")
    return result

def cmd_write_to_file(cmd, filename):
    output_file = open(filename, mode='w')
    retval = subprocess.call(cmd, shell=True, stdout = output_file)
    output_file.close()



filename = 'gamma_test'


cmd = " grep 'Coordinates:' " + filename + " | awk '{print$2}'"; x = cmd_result(cmd).strip('\n')
cmd = " grep 'Coordinates:' " + filename + " | awk '{print$3}'"; y = cmd_result(cmd).strip('\n')
cmd = " grep 'Coordinates:' " + filename + " | awk '{print$4}'"; z = cmd_result(cmd).strip('\n')

x = np.asarray([float(xi) for xi in x.split()  ])
y = np.asarray([float(xi) for xi in y.split()  ])
z = np.asarray([float(xi) for xi in z.split()  ])

coords = np.zeros( x.shape + (3,) )
coords[:,0] = x
coords[:,1] = y
coords[:,2] = z

cd = coords.reshape(  (coords.shape[0], 1, coords.shape[1])  )

c = cd - coords

d = (c**2).sum( 2 )

i = np.arange( x.shape[0] )
d[i,i] = np.inf

ind = np.argmin(d, 1)

print(ind)

print(d[ind])




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)
ax.set_title('Coordinates when trying to relax.')
fig.savefig('atom_coordinates_when_trying_to_relax.png')
plt.show()


