for motif in AA AC CA CC; do
    for precision in 0.5 1.0 2.0; do
        python3 apply_clustering.py nucleotide-fragments/dinuc/origin/$motif.txt \
            output/lib-dinuc-$motif-$precision.all.clust output/lib-dinuc-$motif-$precision.all.origin.txt \
            --force --concat --sep // &
    done
done

for motif in AAA AAC ACA ACC CAA CAC CCA CCC; do
    for precision in 0.5 1.0 2.0; do
        python3 apply_clustering.py nucleotide-fragments/trinuc/origin/$motif.txt \
            output/lib-trinuc-$motif-$precision.all.clust output/lib-trinuc-$motif-$precision.all.origin.txt \
            --force --concat --sep // &
    done
done

wait