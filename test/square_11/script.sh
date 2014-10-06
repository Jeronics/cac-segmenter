#!/bin/sh

\rm -r log
for i in 01 02 
do
mkdir log
../../ver21/cac 1 image.pgm mask_01.pgm cage_${i}.txt
cd log
../../genimages 256 256 > file
cat file | gnuplot
cd ..
mv log log_1_${i}
done


