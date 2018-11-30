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


n_cell_filename   = "pris_scr_disl_13x_2y_14z_rel_trial.dat"  #"original_prism_screw.dat"
filename          = "pris_scr_disl_13x_2y_14z_rel_trial.in"
filename_u        = "gen_cell_13x_2y_14z.in"  
species           = "Ti"



lengths = np.zeros(3)
for i in range(3):
    cmd = " grep 'len' " +  filename + "| awk '{print$" + str(i+2) + "}'"
    lengths[i] =  float( cmd_result(cmd).strip('\n') )


cmd = " grep  'nd' " +  filename + " | awk '{print$2}'"
n_at_cell = int(cmd_result(cmd).strip('\n'))
cmd = " grep  'ninert' " +  filename + " | awk '{print$2}'"
n_at_cell += int(cmd_result(cmd).strip('\n'))


out_ddp_file  = open(n_cell_filename, mode='w+')

out_ddp_file.write(  str(n_at_cell) + "\n"  )


atom_pos_r  = np.zeros(  (3, n_at_cell)  )
atom_pos_u  = np.zeros(  (3, n_at_cell)  )
for i in range(3):
    cmd = " grep " + species + " " +  filename + "| awk '{print$" + str(i+2) + "}'"
    xi_r  = cmd_result(cmd)
    cmd = " grep " + species + " " +  filename_u + "| awk '{print$" + str(i+2) + "}'"
    xi_u  = cmd_result(cmd)
    #print(i, xi_r)
    atom_pos_r[i, : n_at_cell] =  np.asarray( [ float( x ) * lengths[i]  for x in (xi_r.strip('\n')).split() ]  )
    atom_pos_u[i, : n_at_cell] =  np.asarray( [ float( x ) * lengths[i]  for x in (xi_u.strip('\n')).split() ]  )    

for z in atom_pos_r[1,:]:
    out_ddp_file.write( str(z) + "\n" )
for x, y  in zip(atom_pos_r[0,:], atom_pos_r[2,:]):
    out_ddp_file.write( str(x) + " " + str(y) + " 0\n" )
    
out_ddp_file.write(  str(n_at_cell) + "\n"  )

for z in atom_pos_u[1,:]:
    out_ddp_file.write( str(z) + "\n" )
for x, y  in zip(atom_pos_u[0,:], atom_pos_u[2,:]):
    out_ddp_file.write( str(x) + " " + str(y) + "\n" )

out_ddp_file.write("0\n")
out_ddp_file.write(  str(lengths[0]) + "\n"  )
out_ddp_file.write(  str(lengths[1]) + "\n"  )
out_ddp_file.write(  str(lengths[2]) + "\n"  )

out_ddp_file.close()







