import types
from matplotlib import rc
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import matplotlib.pyplot as pp
import numpy as np
import scipy as sci
import os
rcParams["figure.figsize"] = 4, 3
rcParams["font.family"] = "serif"
rcParams["font.size"] = 8
rcParams["font.serif"] = ["DejaVu Serif"]
rc("text", usetex=True)
sci.set_printoptions(linewidth=200, precision=4)


class WriteFiles:

    def __init__(self, filename="ti",  cwd='./', output_path='./generated_dislocations', write_to_dir=True ):
        self.filename = filename

        #######################   Write output to directory  #########################        
        if write_to_dir:

            self.cwd = cwd
            if os.path.isdir(output_path) is False:
                os.mkdir(output_path)

            try:
                os.chdir(output_path)
                print("Changed directory from {} to {}".format(cwd, output_path))
            except TypeError:
                print("Could not change directory from {} to {}".format(cwd, output_path))
            

    def __doc__():
        """
        This writes files to tbe/bop/xyz files such that they can be used. 
        It converts Cartesian positons ONLY. 
        """

        

    ########################   Convert file to XYZ   ###############################

    def convert_file_to_xyz(self, plat, atom_pos, species, filename, append=False):

        if type(atom_pos) == tuple:
                atom_pos, inert_atom_pos = atom_pos
                n_inert = len(inert_atom_pos)
                n_at = len(atom_pos)
                n_tot = n_at + n_inert
        else:
            n_tot = len(atom_pos)
            n_at = n_tot


        n_cell_filename   = filename + "_convert.xyz"

        if append:
            out_xyz_file  = open(n_cell_filename, mode='a')
        else:
            out_xyz_file  = open(n_cell_filename, mode='w+')        

        out_xyz_file.write(  str(n_tot) + "\n"  )
        out_xyz_file.write(  'Lattice=" ' + ' '.join([str(x) for x in plat.flatten() ])  + '" Properties=species:S:1:pos:R:3 \n'  )


        for i in range(n_at):
            out_xyz_file.write( " " + species
                                + " " + '{:<12.8f}'.format( atom_pos[i,0] )   
                                + " " + '{:<12.8f}'.format( atom_pos[i,1] )
                                + " " + '{:<12.8f}'.format( atom_pos[i,2] )
                                + " \n"                              )
        for i in range(n_inert):
            out_xyz_file.write( " " + species + "n"
                                + " " + '{:<12.8f}'.format( inert_atom_pos[i,0] )   
                                + " " + '{:<12.8f}'.format( inert_atom_pos[i,1] )
                                + " " + '{:<12.8f}'.format( inert_atom_pos[i,2] )
                                          + " \n"                              )

        out_xyz_file.close()



    def write_bop_file(self, plat, atom_pos, species, flen, n_disl=4 ):

        if type(atom_pos) == tuple:
            atom_pos, inert_atom_pos = atom_pos
            n_inert = len(inert_atom_pos)
            n_at = len(atom_pos)
            n_tot = n_at + n_inert
        else:
            n_tot = len(atom_pos)
            n_at = n_tot

        conv_to_ang = 1. / 1.8897259886

        fname           = self.filename
        cell_file       = fname# + self.filename + ".in"
        out_file       = open(cell_file, mode='w+')

        plat_inv = np.linalg.inv(plat)

        plat_b = plat / flen


        out_file.write( " " + fname + "   hcp\n"  )
        out_file.write( " " + ' '.join(['{:<12.10f}'.format(x) for x in plat_b[0] ]) + "\n"  )
        out_file.write( " " + ' '.join(['{:<12.10f}'.format(x) for x in plat_b[1] ]) + "\n"   )
        out_file.write( " " + ' '.join(['{:<12.10f}'.format(x) for x in plat_b[2] ]) + "\n"   )


        out_file.write( " len " + '{:<12.10f}'.format(flen[0])  
                          + " " + '{:<12.10f}'.format(flen[1]) 
                          + " " + '{:<12.10f}'.format(flen[2]) + "\n")

        out_file.write( " latpar 1.0\n"  )
        out_file.write( " nd " + str(n_at) + "\n"  )



        for i in range(n_at):
            pl_inv_a = plat_inv.dot( atom_pos[i] )
            out_file.write( " " + species
                             + " " + '{:<12.10f}'.format( pl_inv_a[0] )  
                             + " " + '{:<12.10f}'.format( pl_inv_a[1] )
                             + " " + '{:<12.10f}'.format( pl_inv_a[2] )
                             + ' 0.0 0.0 \n'                              )
        if n_inert > 0:
            out_file.write(" ninert " + str(n_inert) + "\n" )
            for i in range(n_inert):
                pl_inv_a = plat_inv.dot( inert_atom_pos[i] )
                out_file.write( " " + species
                             + " " + '{:<12.10f}'.format( pl_inv_a[0] )   
                             + " " + '{:<12.10f}'.format( pl_inv_a[1] )
                             + " " + '{:<12.10f}'.format( pl_inv_a[2] )
                             + ' 0.0 0.0 \n'                              )



        out_file.write("\n nullheight       0.000000000000\n\n"  )

        out_file.write(" kspace_conf\n"  )
        out_file.write("   0 t         ! symmode symetry_enabled\n"  )
        out_file.write("   12 10 12    ! vnkpts\n"  )
        out_file.write("    t t t       ! offcentre\n\n"  )

        out_file.write("dislocation_conf\n"  )
        for _ in range(n_disl):
            out_file.write("   0    ! gfbcon\n"  )
            out_file.write("   0.0  ! radiusi\n"  )
            out_file.write("   -100.0    100.0    -100.0  100.0\n"  )

        out_file.close()


        atom_pos_bop = np.zeros( atom_pos.shape )
        inert_atom_pos_bop = np.zeros( inert_atom_pos.shape )

        for i, a in enumerate( atom_pos ):

            atom_pos_bop[i,0] = ( plat.dot( plat_inv.dot( atom_pos[i] ) ) )[0]
            atom_pos_bop[i,1] = ( plat.dot( plat_inv.dot( atom_pos[i] ) ) )[1]
            atom_pos_bop[i,2] = ( plat.dot( plat_inv.dot( atom_pos[i] ) ) )[2]

        for i, a in enumerate( inert_atom_pos ):
            inert_atom_pos_bop[i,0] = (  plat.dot( plat_inv.dot( inert_atom_pos[i] ) ) )[0]
            inert_atom_pos_bop[i,1] = (  plat.dot( plat_inv.dot( inert_atom_pos[i] ) ) )[1]
            inert_atom_pos_bop[i,2] = (  plat.dot( plat_inv.dot( inert_atom_pos[i] ) ) )[2] 


        self.convert_file_to_xyz( plat, (atom_pos_bop, inert_atom_pos_bop) , species, self.filename )


    def write_site_file(self,  tot_atom_pos, species, alat, plat ):

        if type(tot_atom_pos) == tuple:
            atom_pos, inert_atom_pos = tot_atom_pos
            n_inert = len(inert_atom_pos)
            n_at = len(atom_pos)
            n_tot = n_at + n_inert
            atom_pos = ( atom_pos / alat ).round(12)
            inert_atom_pos = ( inert_atom_pos / alat ).round(12)
        else:
            n_tot = len(atom_pos)
            n_at = n_tot
            n_inert = 0
            atom_pos = ( atom_pos / alat ).round(12)
        
        convert_to_ryd = 1.8897259886

        s_file = self.filename
        site_file = open( s_file, mode='w' )
        site_info = ( '% site-data vn=3.0 fast io=63 nbas='  + str(n_at + n_inert) + ' alat=' + str(alat)
                      + ' plat=' + ' '.join([str(x) for x in plat.flatten()/alat ])  + '\n' )
        site_file.write(site_info)
        site_file.write('#                        pos vel                                    eula                   vshft PL rlx\n')

        for i in range(n_at):
            site_file.write( " " + species
                             + " " + '{:<12.10f}'.format( atom_pos[i,0] )   
                             + " " + '{:<12.10f}'.format( atom_pos[i,1] )
                             + " " + '{:<12.10f}'.format( atom_pos[i,2] )
                             + " 0.0000000 0.0000000 0.0000000    0.0000000    0.0000000    0.0000000 0.000000  0 001"
                             + " \n"                              )
        for i in range(n_inert):
            site_file.write( " " + species
                             + " " + '{:<12.10f}'.format( inert_atom_pos[i,0] ) 
                             + " " + '{:<12.10f}'.format( inert_atom_pos[i,1] )
                             + " " + '{:<12.10f}'.format( inert_atom_pos[i,2] )
                             + " 0.0000000 0.0000000 0.0000000    0.0000000    0.0000000    0.0000000 0.000000  0 000"
                             + " \n"                              )
        site_file.close()

        tot_atom_pos = ( tot_atom_pos[0] / convert_to_ryd   , tot_atom_pos[1] / convert_to_ryd  )

        self.convert_file_to_xyz( plat / convert_to_ryd, tot_atom_pos, species, self.filename)










