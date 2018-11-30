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


n_cell_filename   = "pris_scr_disl_13x_2y_14z_final.xyz"
filename          = "rel_pris_screw_cell_13_2_14.in"#"pris_scr_disl_13x_2y_14z_t.in"
species           = "Ti"

inert = True

lengths = np.zeros(3)
for i in range(3):
    cmd = " grep 'len' " +  filename + "| awk '{print$" + str(i+2) + "}'"
    lengths[i] =  float( cmd_result(cmd).strip('\n') )


cmd = " grep  'nd' " +  filename + " | awk '{print$2}'"
n_d       = int(cmd_result(cmd).strip('\n'))
n_at_cell = int(cmd_result(cmd).strip('\n'))

cmd = " grep  'ninert' " +  filename + " | awk '{print$2}'"
n_inert    = int(cmd_result(cmd).strip('\n'))
n_at_cell += int(cmd_result(cmd).strip('\n'))

atom_pos  = np.zeros(  (3, n_at_cell)  )
for i in range(3):
    cmd = " grep " + species + " " +  filename + "| awk '{print$" + str(i+2) + "}'"
    xi  = cmd_result(cmd)
    #print(i, xi)
    atom_pos[i, : n_at_cell] =  np.asarray( [ float( x ) * lengths[i]  for x in (xi.strip('\n')).split() ]  )


cell_file = open(filename, mode='r')
out_xyz_file  = open(n_cell_filename, mode='w+')

out_xyz_file.write(  str(n_at_cell) + "\n"  )
out_xyz_file.write(  'Lattice=" ' + str(lengths[0]) + ' 0.0 0.0   0.0 ' + str(lengths[1]) + ' 0.0   0.0 0.0 ' + str(lengths[2]) + '" Properties=species:S:1:pos:R:3 \n'  )



for i in range(n_d):

    out_xyz_file.write( " " + species + " " + str( atom_pos[0, i]   )
                                  + " " + str( atom_pos[1, i]   )
                                  + " " + str( atom_pos[2, i]   )
                                  + " \n"                              )


for i in range(n_d, n_d + n_inert):

    out_xyz_file.write( " " + species + "n " + str( atom_pos[0, i]   )
                                  + " " + str( atom_pos[1, i]   )
                                  + " " + str( atom_pos[2, i]   )
                                  + " \n"                              )


out_xyz_file.close()







