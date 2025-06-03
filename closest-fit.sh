for motif in AA AC CA CC; do
    python3 closest-fit.py dinuc $motif
done

for motif in AAA AAC ACA ACC CAA CAC CCA CCC; do
    python3 closest-fit.py trinuc $motif
done
