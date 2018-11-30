import numpy as np 
import matplotlib.pyplot as plt
import subprocess, shlex, math, time, sys
from optparse import OptionParser
import random 
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
rc('font', **{'family':'serif','serif':['Palatino'],  'size'   : 18})
rc('text', usetex=True)


def cmd_result(cmd):
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result,err = proc.communicate() 
    result = result.decode("utf-8")
    return result

def cmd_write_to_file(cmd, filename):
    output_file = open(filename, mode='w')
    retval = subprocess.call(cmd, shell=True, stdout = output_file)
    output_file.close()


filename          = "pris_scr_disl_4x_2y_5z_rel.xyz"
cell_filename     = "pris_scr_disl_4x_2y_5z_rel.in"#"cell.in"
n_cell_filename   = "cell_xyz.in"
species           = "Ti"

cmd = " head -1  " +  filename
n_atoms = int(cmd_result(cmd).strip('\n'))

cmd = "wc -l < " + cell_filename
nlines = int(cmd_result(cmd).strip('\n'))
nl     = 0

lengths = np.zeros(3)
for i in range(3):
    cmd = " grep 'len' " +  cell_filename + "| awk '{print$" + str(i+2) + "}'"
    lengths[i] =  float( cmd_result(cmd).strip('\n') )

cmd = " grep -n '" + species + "' " +  cell_filename + "| head -1 | cut -d: -f1"
lin = int(cmd_result(cmd).strip('\n'))

cmd = " grep  'nd' " +  cell_filename + " | awk '{print$2}'"
n_at_cell = int(cmd_result(cmd).strip('\n'))

atom_pos  = np.zeros(  (3, n_atoms)  )
for i in range(3):
    cmd = " grep " + species + " " +  filename + "| awk '{print$" + str(i+2) + "}'"
    xi  = cmd_result(cmd)
    print(i, xi)
    atom_pos[i, : n_atoms] =  np.asarray( [ float( x ) for x in (xi.strip('\n')).split() ]  )


cell_file = open(cell_filename, mode='r')
out_file  = open(n_cell_filename, mode='w+')
    
for i in range(lin):
    nl += 1
    c = cell_file.readline()
    print(c)
    out_file.write( c  )
    # Just plopping what was in text file before the atoms.

for i in range(n_atoms):

    out_file.write( " " + species + " " + str( atom_pos[0, i] / lengths[0]  )
                                  + " " + str( atom_pos[1, i] / lengths[1]  )
                                  + " " + str( atom_pos[2, i] / lengths[2]  )
                                  + " 0.0 0.0 \n"                              )

for i in range(n_at_cell):
    cell_file.readline()

for i in range(nlines-nl):
    out_file.write( cell_file.readline()  )
    # Just plopping what was in text file after the atoms.

cell_file.close()
out_file.close()


cmd = 'sed "s/nd/nd "' + str(n_atoms) +  '"/g" ' + n_cell_filename
cmd_result(cmd)






