#!/bin/sh

\rm -r log
for j in 064 128 256
do
    mkdir log
    ../../src/ver23/cac_${j} 3 image.pgm mask_03.pgm cage_03.txt
    cd log
    ../../genimages 256 256 > file 
    cat file | gnuplot
    cd ..
    mv log log_${j}
done


