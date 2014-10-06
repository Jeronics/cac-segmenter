#!/bin/sh

\rm -r log
for j in 1 
do
  for i in 5 
  do
    mkdir log
    ../../src/ver23/cac $j image.pgm mask.pgm cage${i}.txt
    cd log
    ../../genimages 300 300 > file
    cat file | gnuplot
    cd ..
    mv log log_${j}_${i}
  done
done


