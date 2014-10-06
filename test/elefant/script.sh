#!/bin/sh

\rm -r log
for j in 1 2 3
do
  for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15
  do
    mkdir log
    ../../verXX/cac $j image.pgm mask_${i}.pgm cage_${i}.txt
    cd log
    ../../genimages 640 480 > file
    cat file | gnuplot
    cd ..
    mv log log_${j}_${i}
  done
done


