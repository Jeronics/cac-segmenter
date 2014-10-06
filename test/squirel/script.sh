#!/bin/sh

\rm -r log
for j in 1 2
do
  for i in 01 02 03 04 
  do
    mkdir log
    ../../src/ver23/cac $j image.pgm mask_01.pgm cage_${i}.txt
    cd log
    ../../genimages 527 380 > file
    cat file | gnuplot
    cd ..
    mv log log_${j}_${i}
  done
done


