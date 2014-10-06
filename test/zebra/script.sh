#!/bin/sh

\rm -r log
for j in 3
do
  for i in 01 02 03 04
  do
    mkdir log
    ../../src/ver23/cac $j image.pgm mask_${i}.pgm cage_${i}.txt
    cd log
    ../../genimages_full 586 391 > file
    cat file | gnuplot
    cd ..
    mv log log_${j}_${i}
  done
done


