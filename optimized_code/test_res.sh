#!/bin/bash

for i in {0..319}
do
   ./display --checksum output/resultat.raw $i >> output/check_resultat.txt
   ./display --checksum output/mire.raw $i >> output/check_mire.txt
done

diff output/check_resultat.txt output/check_mire.txt

rm -rf output/check_resultat.txt output/check_mire.txt