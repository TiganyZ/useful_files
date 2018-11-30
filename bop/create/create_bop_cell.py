import numpy as np

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
nx = 13
ny = 2
nz = 14

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


atoms = np.zeros( ( n_atoms , 3) )

atoms[:luc,:] = uc                

cell_file       = "gen_cell_13x_2y_14z_ivo.in"
cell_inert_file = "gen_cell_13x_2y_14z_inert_ivo.in"
xyz_file        = "gen_xyz_13x_2y_14z_ivo.xyz"

species        = "Ti"
out_file       = open(cell_file, mode='w+')
out_inert_file = open(cell_inert_file, mode='w+')
out_xyz_file   = open(xyz_file, mode='w+')

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
                r1 =  ( uc[p][0] + i ) * l[0] 
                r2 =  ( uc[p][1] + j ) * l[1]
                r3 =  ( uc[p][2] + k ) * l[2]
                
                c0 = i < ninertx[0] or j < ninerty[0] or k < ninertz[0]
                c1 = i > ninertx[1] or j > ninerty[1] or k > ninertz[1]
                if c0 or c1:
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


out_inert_file.close()
out_inert_file = open(cell_inert_file, mode='r+')

out_file.write(" ninert " + str(ninert) + "\n" )
out_file.write( out_inert_file.read() )
out_inert_file.close()
                    
out_file.write(" nullheight       0.000000000000\n\n"  )

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
