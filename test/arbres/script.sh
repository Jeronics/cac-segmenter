#!/bin/sh

\rm -r log
for j in 3
do
  for i in 01 02 03
  do
    mkdir log
    ../../ver23_jeroni/cac $j image.pgm mask_${i}.pgm cage_${i}.txt
    cd log
    ../../genimages_full 481 321 > file
    cat file | gnuplot
    cd ..
    mv log log_${j}_${i}
  done
done


