for motif in AA AC CA CC; do
    for precision in 0.5 1.0; do
        python3 build-library.py dinuc $motif $precision
    done
done

for motif in AAA AAC ACA ACC CAA CAC CCA CCC; do
    for precision in 0.5 1.0; do
        python3 build-library.py trinuc $motif $precision
    done
done
