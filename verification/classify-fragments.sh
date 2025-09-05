for motif in AA AC CA CC; do
    for precision in 0.5 1.0; do
        echo dinuc $motif $precision
        python3 classify-fragments.py dinuc $motif $precision > result/classify-fragments-dinuc-$motif-$precision.out
    done
done

for motif in AAA AAC ACA ACC CAA CAC CCA CCC; do
    for precision in 0.5 1.0; do
        echo trinuc $motif $precision
        python3 classify-fragments.py trinuc $motif $precision > result/classify-fragments-trinuc-$motif-$precision.out
    done
done
