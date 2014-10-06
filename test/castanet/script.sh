#!/bin/sh

\rm -r log
for j in 1 2 
do
  for i in 02 07 12 
  do
    mkdir log
    ../../src/ver25/cac $j image.pgm mask_${i}.pgm cage_${i}.txt
    cd log
    ../../genimages 640 480 > file
    cat file | gnuplot
    cd ..
    mv log log_${j}_${i}
  done
done


