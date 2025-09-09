#!/bin/bash
set -u -e

currdir=`python -c 'import os,sys;print(os.path.dirname(os.path.realpath(sys.argv[1])))' $0`
cd $currdir

rm -rf TEMP
mkdir TEMP
git clone --depth 1 https://github.com/sjdv1982/nucleotide-library TEMP/
mv TEMP/output output0
mv TEMP/library library0
rm output/ -rf
rm library/ -rf
mv output0 output
mv library0 library
rm -rf TEMP