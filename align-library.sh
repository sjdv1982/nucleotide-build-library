#!/bin/bash

set -u -e
for motif in AA AC CA CC; do
    echo dinuc $motif
    for precision in 0.5 1.0 2.0; do
        for sub in primary extension; do
            python3 align-library.py dinuc $motif $precision $sub &
        done
    done
    wait
    for precision in 0.5 1.0 2.0; do
        python3 align-replacement-library.py dinuc $motif $precision &
    done
    wait
done

for motif in AAA AAC ACA ACC CAA CAC CCA CCC; do
    echo trinuc $motif
    for precision in 0.5 1.0 2.0; do
        for sub in primary extension; do
            python3 align-library.py trinuc $motif $precision $sub &
        done
    done
    wait
    for precision in 0.5 1.0 2.0; do
        python3 align-replacement-library.py trinuc $motif $precision &
    done
    wait
done
