#!/bin/bash

echo " "
read -p "Give site file name (leave empty for default 'site.ext' file):" sitename
if [ -z "$sitename" ]; then
   suffix=$(ls ctrl.* | awk '/^ctrl./{print substr($1,6)}')
   sitename="site.$suffix"
fi

nbas=$(awk -F "nbas" 'NR==1 {print substr($2,2,3)}' $sitename) # up to 999 atoms 

for j in {1..9}; do
   eval a$j=$(awk -F "plat=" 'NR==1 {print $2}' $sitename  | awk -v j=$j '{print $j}')
done

echo $nbas > $sitename.xyz
echo "Lattice=\"$a1 $a2 $a3 $a4 $a5 $a6 $a7 $a8 $a9\" Properties=species:S:1:pos:R:3" >> $sitename.xyz
awk 'NR >2 {print $1,$2,$3,$4}' $sitename >> $sitename.xyz

echo "Using '$sitename' to create '$sitename.xyz' (extended XYZ format)"
echo " "
