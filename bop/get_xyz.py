#!/usr/bin/env/python3.6
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
##########################
#######################################################################################
###########################     General routines      #################################


def cmd_result(cmd):
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result,err = proc.communicate() 
    result = result.decode("utf-8")
    return result

def cmd_write_to_file(cmd, filename):
    output_file = open(filename, mode='w')
    retval = subprocess.call(cmd, shell=True, stdout = output_file)
    output_file.close()


def construct_cmd_arg(arg_name, value):
    """arg_name is a string corresponding to the variable name with a value."""
    return ' -v' + arg_name + '=' + str(value) + ' ' 

filename = "sites.xyz"

a1 = np.array([  1.0,    0.0,   0.0 ])
a2 = np.array([  0.0,    1.0,   0.0 ])
a3 = np.array([  0.0 ,   0.0,   1.0 ])
a = 2.918939393939394
c = 4.609090909090909
a = 20.10669708
c = 4.67988110 
b = 17.41291046

l = np.array([5.026674058492405,          2.9021516207990987,          4.679881023538525])

a1 = a1 * l[0] * 1
a2 = a2 * l[1] * 1
a3 = a3 * l[2] * 1

lstr = " "
for i in a1:
    lstr += str(i) + ", "
for i in a2:
    lstr += str(i) + ", "
for i in a3[:-1]:
    lstr += str(i) + ", "
lstr += str(a3[-1])

latinfo  = ' Properties=species:S:1:pos:R:3 Lattice="' + lstr + '"' 

cmd  = "~/BOP/pb5/bld/mi/npbc/bin/bop | grep -A1 'Atomic Site' | sed -n '1~2!p' | sed '/Atomic/d' "
cmd_write_to_file(cmd, filename)

cmd = "sed -i -e  's/^/  Ti/' " + filename
cmd_result(cmd)

cmd = "wc -l < " + filename
na = cmd_result(cmd).strip('\n')


cmd = "sed -i -e '1i" + str(latinfo) + "\' " +  filename
res = cmd_result(cmd)
print(cmd)
print(res)

cmd = "sed -i -e '1i" + str(na) + "\' " +  filename
cmd_result(cmd)

print(cmd)


