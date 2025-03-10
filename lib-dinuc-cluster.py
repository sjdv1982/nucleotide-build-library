import os
import sys
import json
import numpy as np
from tqdm import tqdm
import nefertiti
from nefertiti.functions.superimpose import superimpose_array

from clusterlib import write_clustering, cluster_from_pairs

USE_SEAMLESS = True

if USE_SEAMLESS:
    from clusterlib.seamless_api import peel_cluster, reassign_clustering

    import seamless

    seamless.delegate(level=3, raise_exceptions=True)

else:

    from clusterlib import peel_cluster, reassign_clustering


diseq = sys.argv[1]
rmsd_threshold = float(sys.argv[2])
strucf = f"input/lib-dinuc-nonredundant-filtered-{diseq}.npy"
struc = np.load(strucf)
print("Structures:", len(struc), file=sys.stderr)


print("Quickly peel off the (approximately) largest clusters", file=sys.stderr)

LARGEST_CLUSTERS = 50
NSAMPLES = [10, 20, 20, 50, 100, 200, 500, 1000, 1000, 2000]
if USE_SEAMLESS:

    big_clusters = peel_cluster(
        struc,
        rmsd_threshold,
        MAX_UNCLUSTERED=-LARGEST_CLUSTERS,
        NSAMPLES=NSAMPLES,
        RANDOM_SEED=0,
        SPECIAL__REPORT_PROGRESS=True,
    )
else:

    def neighbor_func(struc, struc_array):
        _, rmsd = superimpose_array(struc_array, struc)
        return rmsd < rmsd_threshold

    big_clusters = peel_cluster(
        struc,
        neighbor_func=neighbor_func,
        MAX_UNCLUSTERED=-LARGEST_CLUSTERS,
        NSAMPLES=NSAMPLES,
        RANDOM_SEED=0,
    )

json.dump(big_clusters, open(f"lib-dinuc-bigcluster-{diseq}.json", "w"))

clustering = {}

print("Assign structures to the largest clusters", file=sys.stderr)
unclustered_mask = np.ones(len(struc), bool)
for c in tqdm(big_clusters):
    _, rmsd = superimpose_array(struc, struc[c])
    rmsd[c] = np.inf
    clus = rmsd < rmsd_threshold
    clustering[c] = [c] + [int(cc) for cc in np.where(clus)[0]]
    unclustered_mask[clus] = 0
    unclustered_mask[c] = 0

print(
    "Not in the biggest cluster:", unclustered_mask.sum(), "structures", file=sys.stderr
)
unclustered = struc[unclustered_mask]
n_unclustered = len(unclustered)


def calc_closepairs(unclustered, rmsd_threshold):
    from tqdm import tqdm
    import numpy as np

    try:
        superimpose_array
    except NameError:
        from .superimpose import superimpose_array

    n_unclustered = len(unclustered)

    close_list = []
    for snr in tqdm(range(n_unclustered)):
        _, rmsd = superimpose_array(unclustered[snr:], unclustered[snr])
        curr_close = rmsd < rmsd_threshold
        close_list.append(np.where(curr_close)[0] + snr)
    p1 = np.repeat(range(n_unclustered), repeats=[len(l) for l in close_list])
    p2 = np.concatenate(close_list)
    closepairs = np.stack((p1, p2), axis=1)
    return closepairs


if USE_SEAMLESS:
    calc_closepairs = seamless.transformer(calc_closepairs)
    calc_closepairs.direct_print = True
    calc_closepairs.modules.superimpose = nefertiti.functions.superimpose
closepairs = calc_closepairs(unclustered, rmsd_threshold)

print("Final clustering")
small_ind = np.where(unclustered_mask)[0]
small_clustering = cluster_from_pairs(closepairs, len(unclustered))
for c in small_clustering:
    mc = int(small_ind[c[0]])
    cl = [mc]
    for cc in c:
        mcc = int(small_ind[cc])
        if mcc == mc:
            continue
        cl.append(mcc)
    clustering[mc] = cl

print(
    "Assign all structures to the closest cluster and to all clusters within the threshold"
)

clustering_closest, clustering_all = reassign_clustering(
    struc,
    rmsd_threshold,
    [clustering[k] for k in sorted(clustering.keys())],
)

write_clustering(f"lib-dinuc-{diseq}-{rmsd_threshold}.clust", clustering_closest)
write_clustering(f"lib-dinuc-{diseq}-{rmsd_threshold}.all.clust", clustering_all)
