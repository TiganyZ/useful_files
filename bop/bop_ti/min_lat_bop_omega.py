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

def get_hcp_sq_latpar(a, c):
    return 3**(0.5)*a, a , c  ##  This is for the unit cell of Tig

def change_bop_latpar(a, b, c, filename):
    value_name = "len"
    line = find_line_of_value(value_name, filename).strip('\n')
    new_line = "   len           " + str(a) + "          " + str(b) + "          " + str(c)
    cmd = "sed -i 's/%s/%s/g' %s " %(line, new_line, filename)
    cmd_result(cmd)



##################################################################################
###################    Modifying the BOP conf and cell files    ##################

def find_line_of_value(value_name, filename, *args):
    cmd = 'grep "' + str(value_name) + '" ' + str(filename) + ' | head -1 '
    for arg in args:
        cmd += ' | ' + arg + ' '
    result = cmd_result(cmd)
    return result

def find_value_at_field(field, line):
    """This routine uses awk to find value at field"""
    cmd = "awk '{print $" + int(field) + "}' "    
    return cmd_result(cmd)

def find_index_of_char(char, line):
    if char in line:
        char_index = line.index(str(char))
    else:
        print("No character in line")
        raise ValueError
    return char_index

def find_value_on_line(name, line, bop):
    """This routine finds the value of a quantity (denoted by name) on a given line. 
        If not bop, the convention is that the value is to the right of the name."""
    if str(name) in line:
        if bop:
            ##  Use comment to find values in line
            comment_idx = find_index_of_char("!", line)
            ##  Find first comment
            line_values = line[comment_idx + 1:].split()
            value_idx   = line_values.index(str(name))
            quantity    = line[:comment_idx][value_idx]
            field       = value_idx
            ## Quantity is in line before the comment at the value index
        else:
            ## Assumes that the value is to the right of the name
            field = 0
            line.replace("=", " = ")
            for i, c in enumerate(line.split()):
                if c == name:
                    field = i
                    break
            quantity = line.split()[field + 1]
        #print (name, quantity, field)
        return quantity, field
    else:
        print("\n   Error: name of quantity %s is not in line \n         %s" %(name, line))
        print("   Exiting...")
        raise ValueError

def remove_unwanted_char(line, unwanted_char):
    return line.replace(unwanted_char, "")

def replace_value_on_line(value_name, new_value, filename, bop):
    line         = find_line_of_value(value_name, filename).strip('\n')
    value, field = find_value_on_line(value_name, line, bop)
    new_line     = line.split(); new_line[field] = new_value 
    new_line     = new_line.join()
    print("line with !", line)
    line.replace("!", "\!")
    print("line without !", line)
    cmd = "sed -i 's/" + line + "/" + new_line + "/g'" + " " + str(filename)
    print(cmd)
    cmd_result(cmd)
    print("new line from file", find_line_of_value(value_name, filename).strip('\n'))
    return new_line


def remove_bad_syntax(values, unwanted_char):
    new_values=[]
    for i in values:
        if unwanted_char in i[1:]:
            temp = i[1:]
            temp = temp.replace(unwanted_char, " " + unwanted_char)
            i =  (i[0]+ temp).split()
            for j in i:
                new_values.append(float(j))
        else:
            new_values.append(float(i))
    return new_values



def find_energy(LMarg, args, ename, tail, filename, from_file):
    cmd =  LMarg + ' ' + args
    if from_file != True:
        cmd_write_to_file(cmd, filename)
    if 'bop' in LMarg:
        cmd = "grep '" + str(ename) + "' " + filename 
        if tail != 0:
            if   tail > 0:
                cmd += " | tail -" + str(int(tail)) + " "
            elif tail < 0:
                cmd += " | head "  + str(int(tail)) + " "

    etot = cmd_result(cmd)
    #print("etot", etot)
    etot, field = find_value_on_line(ename, etot, False)
    if etot == "NaN":
        print("Warning: Energy is NaN. Setting to zero")
        etot = 0.
    try:
        etot = float(etot)
    except ValueError:
        cmd = "grep 'Exit' " + filename + " "
        error = cmd_result(cmd)
        print( str(error) )
        print( ' Error: \n       ' + str(error) + ' From file ' + filename +  ' \n Exiting...' )
        etot = 'error'
    return etot


def plot_function(n_plots, x, y, colour, title, x_name, y_name):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        if n_plots > 1:
            for i in range(n_plots):
                ax.plot(x[i], y[i], colour[i])
        else:
            ax.plot(x, y, colour)
        plt.show() 

def find_latpars_grid(LMarg, args, n_lp, names_lp, limits_lp, n_grid, ax, plot):
    name_limits = '    Find Latpars Grid:\n    n_grid = %s\n'%(n_grid)  
    al_l = [] 
    sz   = ()
    if n_lp > 1:
        for i in range(n_lp):
            name_limits += '     %s_l---%s_u = %s---%s\n'%(names_lp[i], names_lp[i], limits_lp[i][0], limits_lp[i][1] )
            al = np.linspace(limits_lp[i][0], limits_lp[i][1], n_grid[i])
            al_l.append(al)
            sz += (n_grid[i],)
    else:
        name_limits += '     %s_l---%s_u = %s---%s\n'%(names_lp, names_lp, limits_lp[0], limits_lp[1] )
        al = np.linspace(limits_lp[0], limits_lp[1], n_grid)
        al_l.append(al)
        sz += (n_grid,)
    print( name_limits )
    
    etot_a  = np.zeros( sz )
    eband_a = np.zeros( sz )
    epair_a = np.zeros( sz )

    if n_lp == 1:
        for i in range(n_grid):
            xx_args = args + construct_cmd_arg(names_lp, al_l[0][i])
            etot  = find_energy(LMarg, xx_args, 'ebind', 2, 'boptest', False)
            epair = find_energy(LMarg, xx_args, 'eclas', 2, 'boptest', True)
            eband = find_energy(LMarg, xx_args, 'eband', 2, 'boptest', True)
            if etot is not str:
                etot_a[i] = etot
            if eband is not str:
                eband_a[i] = eband
            if epair is not str:
                epair_a[i] = epair
                

    elif n_lp == 2:
        for i in range(n_grid[0]):
            for j in range(n_grid[1]):

                a, b, c = get_hcp_sq_latpar(al_l[0][i], al_l[1][j])
                change_bop_latpar(a, b, c, "cell.in")

                etot  = find_energy(LMarg, args, 'ebind', 2, 'boptest', False)
                epair = find_energy(LMarg, args, 'eclas', 2, 'boptest', True)
                eband = find_energy(LMarg, args, 'eband', 2, 'boptest', True)
                if etot is not str:
                    etot_a[i][j] = etot
                if eband is not str:
                    eband_a[i][j] = eband
                if epair is not str:
                    epair_a[i][j] = epair
    else:
        for i in range(n_grid[0]):
            for j in range(n_grid[1]):
                for k in range(n_grid[2]):
                    xx_args = args + construct_cmd_arg(names_lp[0], al_l[0][i])  \
                                   + construct_cmd_arg(names_lp[1], al_l[1][j])  \
                                   + construct_cmd_arg(names_lp[2], al_l[2][k])
                    etot = find_energy(LMarg, xx_args, 'pptest')
                    if etot is not str:
                        etot_a[i][j][k] = etot


    if n_lp == 1 and plot == True:
        if ax == True:
            fig= plt.figure()
            ax = fig.add_subplot(111)
        xp =     al
        yp =     etot_a
        colour = 'b--'
        b_min = np.argmin(yp)
        yy = np.array([np.min(yp), np.max(yp)])
        #print('yy', yy)
        xx = [ xp[b_min] for i in yy]
        #print('xx', xx)
        ax.plot( al, eband_a, 'r--', label = r'BOP $E_{band}$')
        ax.plot( al, epair_a, 'b--', label = r'BOP $E_{pair}$')
        ax.plot( xp,    yp,   'k--', label = r'BOP $E_{bind}$')
        ax.plot( xx,    yy,   'm--', label = r'BOP min$(E_{bind})$')

        ax.set_xlim(0.14, 5)
        ax.set_ylim(-50, 50)
        ax.set_title(r"$E_{band}$ vs $c$ and $a$")
        ax.set_xlabel(r"$a$")
        ax.set_ylabel(r"$c$")
        ax.legend(bbox_to_anchor=(1., 1.05))
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    

    if n_lp == 2 and plot == True:
        if ax == True:
            fig= plt.figure()
            ax = fig.add_subplot(111)
        xp =     al
        yp =     etot_a
        colour = 'b--'
        """b_min = np.argmin(yp)
        yy = np.array([np.min(yp), np.max(yp)])
        #print('yy', yy)
        xx = [ xp[b_min] for i in yy]
        #print('xx', xx)
        ax.plot( al, eband_a, 'r-', label = r'BOP $E_{band}$')
        ax.plot( al, epair_a, 'b-', label = r'BOP $E_{pair}$')
        ax.plot( al,  etot_a, 'k-', label = r'BOP $E_{bind}$')
        #ax.plot( xx,      yy, 'm-', label = r'BOP min$E_{bind}$')

        ax.set_xlim(0.14, 5)
        ax.set_ylim(-50, 50)
        ax.set_title( r'O$_2$ py vs tbe' )
        ax.legend(bbox_to_anchor=(1., 1.05))
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))"""

        X, Y = np.meshgrid(al_l[0], al_l[1])
        print(X.shape, Y.shape, etot_a.shape)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, etot_a, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)  #Try coolwarm vs jet
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_title(r"$E_{bind}$ vs $c$ and $a$")
        ax.set_xlabel(r"$a$")
        ax.set_ylabel(r"$c$")
        ax.set_zlabel(r"$E_{bind}$")

        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d')
        surf = ax2.plot_surface(X, Y, eband_a, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)  #Try coolwarm vs jet
        ax2.zaxis.set_major_locator(LinearLocator(10))
        ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig2.colorbar(surf, shrink=0.5, aspect=5)
        ax2.set_title(r"$E_{band}$ vs $c$ and $a$")
        ax2.set_xlabel(r"$a$")
        ax2.set_ylabel(r"$c$")
        ax2.set_zlabel(r"$E_{band}$")

        fig3 = plt.figure()
        ax3 = fig3.gca(projection='3d')
        surf = ax3.plot_surface(X, Y, epair_a, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)  #Try coolwarm vs jet
        ax3.zaxis.set_major_locator(LinearLocator(10))
        ax3.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig3.colorbar(surf, shrink=0.5, aspect=5)
        ax3.set_title(r"$E_{pair}$ vs $c$ and $a$")
        ax3.set_xlabel(r"$a$")
        ax3.set_ylabel(r"$c$")
        ax3.set_zlabel(r"$E_{pair}$")
        plt.show()

    min_ind = np.unravel_index( np.argmin(etot_a), sz )
    print('\n Minimum lattice parameters')

    if n_lp == 1:
        al_min = al_l[0][min_ind]
        #min_ind = np.argmin(al)
        print('     %s = %s' %(names_lp[0], al_min))
        min_vol = ( 3**(0.5) / 2. ) * ( al_min**3 )
        ret = (al_min,)
        wv = calc_wv(al, etot_a, min_ind[0], 15.999 / 2.)
        #curv = (al_l[0][min_ind+1] - 2 * al_l[0][min_ind] )
        print('wavenumber = %s' %(wv))
    if n_lp > 1:
        al_min  = al_l[0][ min_ind[0] ]
        al2_min = al_l[1][ min_ind[1] ]
        print('     %s = %s' %(names_lp[0], al_min))
        print('     %s = %s' %(names_lp[1], al2_min))
        print('   %s/%s = %s' %(names_lp[1], names_lp[0],  al2_min/al_min))
        min_vol = ( 3**(0.5) / 2. ) * ( al_min**3 ) * al2_min
        ret = (al_min, al2_min)
    if n_lp > 2:
        al3_min = al_l[2][ min_ind[2] ]
        print('     %s = %s' %(names_lp[2], al3_min))
        min_vol = ( 3**(0.5) / 2. ) * ( al_min**3 ) * al2_min * al3_min
        ret += (al3_min,)

    print('     vol    = %s\n' %(min_vol))
    ret += (min_vol,)
    return ret




LMarg     = "~/BOP/pb5/bld/mi/npbc/bin/bop "
ext       = ""
args      = ""
#old_line  = "   len           2.95          5.10955          4.6835"
tony_len  = "   len           5.05191         2.917          4.6552"
tonylen   = False
tonya     = 2.917
tonyc     = 4.6552
ax        = False

n_lp      = 2
names_lp  = ["a", "c"]
limits_lp = [np.array([2.89, 2.94]), np.array([4.4,4.8])]
ng = 30
n_grid    = np.array([ng,ng])
plot = False
nt = 10
for i in range(nt):
    if i == nt-1:
        plot=True
    ret = find_latpars_grid(LMarg, args, n_lp, names_lp, limits_lp, n_grid, ax, plot)
    print('\n\nReturn: Find lattice parameters:\n\n      ' ,ret)
    plt.show()
    a = ret[0]#2.6052631578947367
    c = ret[1]#4.4
    dlp0 = (limits_lp[0][1] - limits_lp[0][0])/float(4)
    dlp1 = (limits_lp[1][1] - limits_lp[1][0])/float(4)
    limits_lp[0] = np.array([a - dlp0, a + dlp0  ])
    limits_lp[1] = np.array([c - dlp1, c + dlp1  ])

    change_bop_latpar(3**(0.5) * a, a,  c, "cell.in")
    if tonylen == True:
        change_bop_latpar(tonya, 3**(0.5) * tonya, tonyc, "cell.in")

