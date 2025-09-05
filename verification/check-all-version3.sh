# version of check-all.sh where all lists are checked with brute force
# no files from ../output/closest-fit/ are required, but CPU time is significant
run(){
    lib=$1
    motif=$2
    precision=$3
    echo $lib $motif $precision
    python3 check-singletons-brute-force.py result/lib-$lib-$motif-$precision.singletons.txt $lib $motif $precision --true > result/check-$lib-$motif-$precision.singletons.out
    python3 check-singletons-brute-force.py result/lib-$lib-$motif-$precision.putative_singletons.txt $lib $motif $precision --true > result/check-$lib-$motif-$precision.putative_singletons.out
    python3 check-singletons-brute-force.py result/lib-$lib-$motif-$precision.close-pairs.txt $lib $motif $precision --false > result/check-$lib-$motif-$precision.close-pairs.out
    python3 check-singletons-brute-force.py result/lib-$lib-$motif-$precision.putative_non_singletons.txt $lib $motif $precision --false > result/check-$lib-$motif-$precision.putative_non_singletons.out

}

for motif in AA AC CA CC; do
    for precision in 0.5 1.0; do
        run dinuc $motif $precision
    done
done

for motif in AAA AAC ACA ACC CAA CAC CCA CCC; do
    for precision in 0.5 1.0; do
        run trinuc $motif $precision
    done
done
