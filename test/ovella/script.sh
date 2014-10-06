#!/bin/sh

\rm -r log
for j in 3
do
  for i in 01 
  do
    mkdir log
    ../../src/ver25/cac $j image.pgm mask${i}.pgm cage${i}.txt
    cd log
    ../../genimages 450 600 > file
    cat file | gnuplot
    cd ..
    mv log log_${j}_${i}
  done
done


