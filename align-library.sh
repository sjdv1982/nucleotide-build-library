#!/bin/bash

set -u -e
for motif in AA AC CA CC; do
    for precision in 0.5 1.0 2.0; do
        for sub in primary replacement extension; do
            echo dinuc $motif $precision $sub
            python3 align-library.py dinuc $motif $precision $sub
        done
    done
done

for motif in AAA AAC ACA ACC CAA CAC CCA CCC; do
    for precision in 0.5 1.0 2.0; do
        for sub in primary replacement extension; do
            echo trinuc $motif $precision $sub
            python3 align-library.py trinuc $motif $precision $sub
        done
    done
done
