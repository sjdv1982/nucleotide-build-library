import sys
import numpy as np
from numpy.linalg import svd, det

lib = sys.argv[1]
assert lib in ("dinuc", "trinuc")
motif = sys.argv[2]
precision = sys.argv[3]


def err(*args):
    print(*args, file=sys.stderr)
    exit(1)


n_report = 0


def report(*args):
    global n_report
    n_report += 1
    print("#{:2d}".format(n_report), "", *args)
    return n_report


def superimpose_from_covar(covar, residuals1, residuals2):
    v, s, wt = svd(covar)
    reflect = det(v) * det(wt)
    s[:, -1] *= reflect
    sd = (residuals1 + residuals2) - 2 * s.sum(axis=1)
    sd = np.maximum(sd, 0)
    return sd


print(f"Library '{lib}', sequence motif '{motif}'")
print(
    f"Analyse how many unique conformations come from single PDBs at {precision} A precision."
)

origin_file = f"../nucleotide-fragments/{lib}/origin/{motif}.txt"
clusterfile = f"../output/lib-{lib}-{motif}-{precision}.all.clust"

clustering = []
with open(clusterfile) as f:
    for lnr, l in enumerate(f.readlines()):
        ll = l.split()
        assert ll[0] == "Cluster"
        assert ll[1] == str(lnr + 1)
        assert ll[2] == "->"
        c = [int(lll) for lll in ll[3:]]
        clustering.append(c)

origins = []
for l in open(origin_file).readlines():
    single_pdb = True
    single_code = None
    for item in l.split("/"):
        if not item.strip():
            continue
        fields = item.split()
        assert len(fields) == 3
        code = fields[0][:4]
        if single_code is None:
            single_code = code
        else:
            if code != single_code:
                single_pdb = False
                break
    if single_pdb and single_code:
        origins.append(single_code)
    else:
        origins.append(None)
nconf = len(origins)

indices = []
for cnr, c in enumerate(clustering):
    indices += c
indices = np.array(indices, int)
indices = np.unique(indices)
indices.sort()

if indices[0] != 1:
    err("Clustering does not start at 1")
elif not np.all(indices == np.arange(len(indices)) + 1):
    err("Clustering has missing indices")

if indices[-1] != nconf:
    err(f"Clustering does not match input: {indices[-1]} vs {nconf}")

del indices

report(f"Number of clusters: {len(clustering)}")
print()
report(f"Number of fragments: {nconf}")

in_cluster = {n: [] for n in range(nconf)}
for clusnr, cluster in enumerate(clustering):
    for member in cluster:
        in_cluster[member - 1].append(clusnr)

fragments_multi_origin = {n for n in range(nconf) if origins[n] is None}
stat_trivial = report(f"with multi-PDB origin: {len(fragments_multi_origin)}")
print()
fragments_mono_origin = {n for n in range(nconf) if n not in fragments_multi_origin}
report(f"with mono-PDB origin: {len(fragments_mono_origin)}")

close_pairs = {}

cluster_hearts = []
for clusnr, cluster in enumerate(clustering):
    heart = cluster[0] - 1
    assert heart not in cluster_hearts
    assert len(in_cluster[heart]) == 1
    if heart not in fragments_mono_origin:
        continue
    cluster_hearts.append(heart)
cluster_hearts = set(cluster_hearts)

report(f"  Cluster hearts: {len(cluster_hearts)}")

cluster_singletons = []
cluster_homo_pdb_hearts = []
cluster_hetero_pdb_hearts = []
for heart in cluster_hearts:
    ori = origins[heart]
    assert ori is not None
    clus = in_cluster[heart][0]  # by definition, the only cluster
    cluster = clustering[clus]
    if len(cluster) == 1:
        cluster_singletons.append(heart)
        continue
    for mem in cluster:
        if origins[mem - 1] != ori:
            close_pairs[heart] = mem - 1
            cluster_hetero_pdb_hearts.append(heart)
            break
    else:
        cluster_homo_pdb_hearts.append(heart)

stat_singleton1 = report(f"    from singleton clusters: {len(cluster_singletons)}")
stat_singleton2 = report(f"    from homo-PDB clusters: {len(cluster_homo_pdb_hearts)}")
singletons = sorted(list(cluster_singletons) + list(cluster_homo_pdb_hearts))

stat_non_singleton1 = report(
    f"    from hetero-PDB clusters: {len(cluster_hetero_pdb_hearts)}"
)
non_cluster_hearts = [n for n in fragments_mono_origin if n not in cluster_hearts]
report(f"  Other cluster members: {len(non_cluster_hearts)}")
non_cluster_hearts_dif_origin = []
non_cluster_hearts_same_origin = []
for n in non_cluster_hearts:
    ori = origins[n]
    for clus in in_cluster[n]:
        cluster = clustering[clus]
        heart = cluster[0] - 1
        assert heart != n
        if ori != origins[heart]:
            non_cluster_hearts_dif_origin.append(n)
            close_pairs[n] = heart
            break
    else:
        non_cluster_hearts_same_origin.append(n)
stat_non_singleton2 = report(
    f"    with different origin as cluster heart: {len(non_cluster_hearts_dif_origin)}"
)
report(f"    with same origin as cluster heart: {len(non_cluster_hearts_same_origin)}")

homo_cluster = []
hetero_cluster = []
for n in non_cluster_hearts_same_origin:
    ori = origins[n]
    assert ori is not None
    for clus in in_cluster[n]:
        cluster = clustering[clus]
        for mem in cluster:
            if origins[mem - 1] != ori:
                hetero_cluster.append(n)
                break
        else:
            continue
        break
    else:
        homo_cluster.append(n)

putative_singletons = homo_cluster
stat_putative_singleton = report(
    f"      from homo-PDB clusters: {len(putative_singletons)}"
)
putative_non_singletons = hetero_cluster
stat_putative_non_singleton = report(
    f"      from hetero-PDB clusters: {len(putative_non_singletons)}"
)


print()
print(f"Trivial non-singletons: #{stat_trivial}      = {len(fragments_multi_origin)}")
print(
    f"Certain non-singletons: #{stat_non_singleton1} + #{stat_non_singleton2}   = {len(close_pairs)}"
)
print(
    f"Putative non-singletons: #{stat_putative_non_singleton}    = {len(putative_non_singletons)}"
)
print(
    f"Certain singletons: #{stat_singleton1} + #{stat_singleton2}        = {len(singletons)}"
)
print(
    f"Putative singletons: #{stat_putative_singleton}        = {len(putative_singletons)}"
)
print()

singleton_file = f"result/lib-{lib}-{motif}-{precision}.singletons.txt"
putative_singleton_file = (
    f"result/lib-{lib}-{motif}-{precision}.putative_singletons.txt"
)
close_pair_file = f"result/lib-{lib}-{motif}-{precision}.close-pairs.txt"
putative_non_singleton_file = (
    f"result/lib-{lib}-{motif}-{precision}.putative_non_singletons.txt"
)

print(
    f"""Four lists of indices (starting at 1) will now be written.

     {singleton_file}

List of conformer indices that are surely singletons, i.e. not close to any other conformer from a different PDB.

Any error in this list is a bug.

This can be verified by checking against the pre-computed closest fit (check-singletons.py {precision} --true).
Alternatively, this can be verified by brute-force superposition onto all conformers (check-singletons-brute-force.py --true).

    {putative_singleton_file}

List of PDBs that are putatively singletons. 

Note that false positives are possible.

This can be verified by checking against the pre-computed closest fit (check-singletons.py {precision} --true).
Alternatively, this can be verified by superposition (check-singletons-brute-force.py --true).


    {close_pair_file}

List of close (within {precision}) pairs of conformer indices that are from different PDBs.
For each non-trivial non-singleton, one such pair is written.

Any error in this list is a bug.

This can be verified by checking against the pre-computed closest fit (check-singletons.py {precision} --false).
Alternatively, this can be verified by superposition (verify-close-pairs-brute-force.py).


    {putative_non_singleton_file}

List of PDBs that are putatively not singletons. 
It is considered likely that a close conformer from a different PDB exists.

Note that false negatives are possible.

This can be verified by checking against the pre-computed closest fit (check-singletons.py {precision} --false).
Alternatively, this can be verified by superposition (check-singletons-brute-force.py --false).

"""
)

with open(close_pair_file, "w") as f:
    for k in sorted(list(close_pairs.keys())):
        print(k + 1, close_pairs[k] + 1, file=f)

with open(singleton_file, "w") as f:
    for ind in singletons:
        print(ind + 1, file=f)

with open(putative_singleton_file, "w") as f:
    for ind in putative_singletons:
        print(ind + 1, file=f)

with open(putative_non_singleton_file, "w") as f:
    for ind in putative_non_singletons:
        print(ind + 1, file=f)
