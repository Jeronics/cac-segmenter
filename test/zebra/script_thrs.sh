#!/bin/sh

\rm -r log
for i in `LC_ALL=C seq -f "%+1.8f" 0.0 0.0000001 0.00004`
do
mkdir log
../../src/ver24/cac $i 3 image.pgm mask_new_037.pgm cage_new_037.txt
cd log
../../genimages 586 391 > file
cat file | gnuplot
cd ..
mv log log_${i}
done


