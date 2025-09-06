"""Find the closest fit of all conformers, after 1A+2A clustering"""

import os
import sys
import random
import numpy as np
from clusterlib import read_clustering
import nefertiti
from nefertiti.functions.superimpose import (
    superimpose,
    superimpose_array,
)


try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda arg: arg


def err(*args):
    print(*args, file=sys.stderr)
    exit(1)


####################################################################
# Load and validate inputs
####################################################################


lib = sys.argv[1]
assert lib in ("dinuc", "trinuc")
motif = sys.argv[2]

origin_file = f"nucleotide-fragments/{lib}/origin/{motif}.txt"
coorfile = f"nucleotide-fragments/{lib}/{motif}.npy"

coors = np.load(coorfile)
assert coors.ndim == 3 and coors.shape[-1] == 3, coors.shape
coors -= coors.mean(axis=1)[:, None, :]
coors_residuals = np.einsum("ijk,ijk->i", coors, coors)


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


def get_clustering(precision):
    clusterfile = f"output/lib-{lib}-{motif}-{precision}.all.clust"
    clustering = read_clustering(clusterfile)

    indices = []
    for cnr, c in enumerate(clustering):
        indices += c
    indices = np.array(indices, int)
    indices = np.unique(indices)
    indices.sort()

    if indices[0] != 1:
        err("Clustering does not start at 1")
    elif not np.alltrue(indices == np.arange(len(indices)) + 1):
        err("Clustering has missing indices")

    if indices[-1] != nconf:
        err(f"Clustering does not match input: {indices[-1]} vs {nconf}")

    del indices

    clustering = [[cc - 1 for cc in c] for c in clustering]

    closest_clusterfile = f"output/lib-{lib}-{motif}-{precision}.clust"
    closest_clustering = read_clustering(closest_clusterfile)
    closest_clustering = [[cc - 1 for cc in c] for c in closest_clustering]

    closest_cluster = {}
    for clusnr, cluster in enumerate(closest_clustering):
        assert cluster[0] == clustering[clusnr][0]
        for member in cluster:
            closest_cluster[member] = clusnr
    return clustering, closest_cluster


result0 = {}
for conf in range(nconf):
    ori = origins[conf]
    if ori is None:
        result0[conf] = conf, 0


def get_closest_fit(
    coors,
    origins,
    clustering,
    closest_cluster,
    precision,
    *,
    done=[],
    explore_members=True,
):

    done = set(done)
    nconf = len(coors)
    SMALL_STRUC = 50
    result = {}

    closest_cluster = {int(k): v for k, v in closest_cluster.items()}

    in_cluster = {n: [] for n in range(nconf)}
    for clusnr, cluster in enumerate(clustering):
        for member in cluster:
            in_cluster[member].append(clusnr)

    confs = list(range(nconf))
    import random

    random.shuffle(confs)
    for conf in tqdm(confs):

        if conf in done:
            continue

        struc = coors[conf]
        ori = origins[conf]
        assert ori is not None

        """
        First, we try to identify one or more "bullseye" clusters:
            1A clusters where the closest fitting conformer must surely be part of.

        Candidate bullseye clusters have the following properties:
        - Not all from the same PDB as the fitted structure (just the cluster heart is ok)
        - Low RMSD
        - If possible, not too many members.

        To prove that a cluster is a bullseye cluster, we use the following triangle inequality:
        X <= Y + Z 
        where:
            X is the RMSD of the bullseye cluster heart to the closest fitting conformer
            Y is the RMSD of the bullseye cluster heart to the fitted structure
            Z is the RMSD of the fitted structure to the closest fitted structure.    
        Knowing Y and ZZ (an upper bound of Z), we try to prove that X <= cluster-precision
        """
        bullseye_candidates = []
        bullseye_rmsds = []
        closest_known_rmsd = None

        def have_bullseye():
            if closest_known_rmsd is None:
                return False
            return any([r + closest_known_rmsd < precision for r in bullseye_rmsds])

        def have_enough_bullseye():
            return have_bullseye() and any(
                [len(clustering[c]) < SMALL_STRUC for c in bullseye_candidates]
            )

        """First, consider the closest cluster"""
        cclusnr = closest_cluster[conf]
        clus = clustering[cclusnr]
        if all([origins[cc] == ori for cc in clus]):
            # The entire closest cluster is from the same PDB as the fitted structure
            pass
        else:
            cclus_heart = clus[0]
            _, closest_cluster_rmsd = superimpose(struc, coors[cclus_heart])

            bullseye_candidates.append(cclusnr)
            bullseye_rmsds.append(closest_cluster_rmsd)

            if origins[cclus_heart] != ori:
                closest_known_rmsd = closest_cluster_rmsd
            else:
                clus2 = [c for c in clus if origins[c] != ori]
                _, rmsds = superimpose_array(coors[clus2[:SMALL_STRUC]], struc)
                closest_known_rmsd = rmsds.min()

        if not have_enough_bullseye():
            for clusnr in in_cluster[conf]:
                if clusnr == closest_cluster[conf]:
                    continue
                clus = clustering[clusnr]
                if all([origins[cc] == ori for cc in clus]):
                    # The entire cluster is from the same PDB as the fitted structure
                    continue
                clus_heart = clus[0]
                _, cluster_rmsd = superimpose(struc, coors[clus_heart])

                # Now we really need a closest known RMSD
                if origins[clus_heart] != ori:
                    if closest_known_rmsd is None or cluster_rmsd < closest_known_rmsd:
                        closest_known_rmsd = cluster_rmsd
                elif closest_known_rmsd is None:
                    clus2 = [c for c in clus if origins[c] != ori]
                    _, rmsds = superimpose_array(coors[clus2[:SMALL_STRUC]], struc)
                    closest_known_rmsd = rmsds.min()

                bullseye_candidates.append(clusnr)
                bullseye_rmsds.append(cluster_rmsd)

                if have_bullseye():
                    break

        if not have_bullseye() and len(bullseye_candidates) > 0:
            # We could not find a bullseye cluster by superimposing the hearts alone.
            # Let's try superimposing the first SMALL_STRUC cluster members,
            #  to get a better closest known RMSD.
            for pos in np.argsort(bullseye_rmsds):
                clusnr = bullseye_candidates[pos]
                clus = clustering[clusnr]
                clus2 = [c for c in clus if origins[c] != ori]
                assert len(clus2)
                _, rmsds = superimpose_array(coors[clus2[:SMALL_STRUC]], struc)
                rmin = rmsds.min()
                if rmin < closest_known_rmsd:
                    closest_known_rmsd = rmin
                    if have_bullseye():
                        break

        if not have_bullseye() and len(bullseye_candidates) > 0:
            # We are unlucky. We need to consider all other bullseye candidate members,
            #  and then hope for a better closest known RMSD
            if explore_members:
                for pos in np.argsort(
                    [len(clustering[clusnr]) for clusnr in bullseye_candidates]
                ):
                    clusnr = bullseye_candidates[pos]
                    clus = clustering[clusnr]
                    clus2 = [c for c in clus if origins[c] != ori]
                    assert len(clus2)

                    success = False
                    for pos in range(SMALL_STRUC, len(clus2), SMALL_STRUC):
                        chunk = clus2[pos : pos + SMALL_STRUC]
                        _, rmsds = superimpose_array(coors[chunk], struc)
                        rmin = rmsds.min()
                        if rmin < closest_known_rmsd:
                            closest_known_rmsd = rmin
                            if have_bullseye():
                                success = True
                                break

                    if success:
                        break

        if not have_bullseye():
            # Give up
            continue

        # We have one or more bullseye clusters
        # Now we can select closest-fit candidates, they must be in *all* bullseye clusters
        candidates = None
        for clusnr, rmsd in zip(bullseye_candidates, bullseye_rmsds):
            if rmsd + closest_known_rmsd >= precision:
                continue
            members = set(clustering[clusnr])
            if candidates is None:
                candidates = members
            else:
                candidates = candidates.intersection(members)
        assert candidates is not None
        candidates = [c for c in candidates if origins[c] != ori]
        assert len(candidates)

        closest_fit_rmsd = None
        for chunkpos in range(0, len(candidates), 1000):
            chunk = candidates[chunkpos : chunkpos + 1000]
            chunk_candidate_struc = coors[chunk]
            _, rmsd = superimpose_array(chunk_candidate_struc, struc)
            chunk_best = rmsd.min()
            if closest_fit_rmsd is None or chunk_best < closest_fit_rmsd:
                closest_fit_rmsd = chunk_best
                closest_fit = chunk[rmsd.argmin()]
        result[conf] = closest_fit, closest_fit_rmsd

    return result


clustering1A, closest_cluster1A = get_clustering(1.0)
clustering2A, closest_cluster2A = get_clustering(2.0)

done = list(result0.keys())
result1A = get_closest_fit(
    coors,
    origins,
    clustering1A,
    closest_cluster1A,
    1.0,
    explore_members=False,  # True brings no benefit
    done=done,
)


done = list(result1A.keys()) + list(result0.keys())
result2A = get_closest_fit(
    coors,
    origins,
    clustering2A,
    closest_cluster2A,
    2.0,
    explore_members=False,  # True slows it down
    done=done,
)

# For the rest, we have to explicitly compare against all structures, not just a few clusters...

remaining = [
    conf
    for conf in range(len(coors))
    if conf not in result1A and conf not in result2A and conf not in result0
]

random.shuffle(remaining)
remaining = np.array(remaining, int)

result_remaining = {}

# ... However, we may be able to eliminate big clusters in bulk or in part
big_clust2A = [clus for clus in clustering2A if len(clus) > 20]
big_struc2A = coors[[clus[0] for clus in big_clust2A]]
big_conf2A = set(sum(big_clust2A, []))
big_clust2A = [np.array(l) for l in big_clust2A]

big_clust1A = [clus for clus in clustering1A if len(clus) > 20]
big_struc1A = coors[[clus[0] for clus in big_clust1A]]
big_conf1A = set(sum(big_clust1A, []))
big_clust1A = [np.array(l) for l in big_clust1A]

in_cluster = {n: [] for n in range(nconf)}
for clusnr, cluster in enumerate(clustering1A):
    for member in cluster:
        in_cluster[member].append(clusnr)

big_clust2A_rmsd = []
for n, big_clust in enumerate(tqdm(big_clust2A)):
    big_struc = big_struc2A[n]
    cluster_struc = coors[big_clust]
    _, rmsd = superimpose_array(cluster_struc, big_struc)
    big_clust2A_rmsd.append(rmsd)

all_candidates = np.ones((len(remaining), len(coors)), bool)
closest_rmsd_initial_estimates = []
for confnr, conf in enumerate(tqdm(remaining)):
    struc = coors[conf]
    ori = origins[conf]
    _, rmsd_bigclust1A = superimpose_array(big_struc1A, struc)
    _, rmsd_bigclust2A = superimpose_array(big_struc2A, struc)
    closest_rmsd_estimate = rmsd_bigclust1A.min()
    if in_cluster[conf]:
        myclusters = [clustering1A[clusnr][0] for clusnr in in_cluster[conf]]
        myclusters = [c for c in myclusters if origins[c] != ori]
        if len(myclusters):
            myclusters_struc = coors[myclusters]
            _, r = superimpose_array(myclusters_struc, struc)
            closest_rmsd_estimate = r.min()

    closest_rmsd_initial_estimates.append(closest_rmsd_estimate)

    # We have now a closest RMSD estimate ZZ
    candidates = all_candidates[confnr]
    # Eliminate-in-bulk all 1A big clusters where Y (the cluster heart RMSD) > ZZ + 1
    for clusnr in range(len(big_clust1A)):
        if rmsd_bigclust1A[clusnr] > closest_rmsd_estimate + 1:
            big_clust = big_clust1A[clusnr]
            candidates[big_clust] = 0

    # For each 2A big cluster, we have:
    #   Y, the RMSD between cluster heart and fitted conformer
    #   Q, the RMSD between cluster heart and a particular member
    # If Q < Y - ZZ, we can eliminate that member
    for clusnr in range(len(big_clust2A)):
        y = rmsd_bigclust2A[clusnr]
        elim = big_clust2A_rmsd[clusnr] < (y - closest_rmsd_estimate)
        big_clust = big_clust2A[clusnr]
        candidates[big_clust[elim]] = 0


# We will have to brute-force against the rest...
for confnr, conf in enumerate(tqdm(remaining)):
    ori = origins[conf]
    struc = coors[conf]
    closest_fit_rmsd = None
    candidates = np.nonzero(all_candidates[confnr])[0]
    candidates = [c for c in candidates if origins[c] != ori]
    for pos in range(0, len(candidates), 1000):
        chunk = candidates[pos : pos + 1000]
        chunk_candidate_struc = coors[chunk]
        _, rmsd = superimpose_array(chunk_candidate_struc, struc)
        chunk_best = rmsd.min()
        if closest_fit_rmsd is None or chunk_best < closest_fit_rmsd:
            closest_fit_rmsd = chunk_best
            closest_fit = chunk[rmsd.argmin()]

    result_remaining[conf] = closest_fit, closest_fit_rmsd

result = result0.copy()
result.update(result1A)
result.update(result2A)
result.update(result_remaining)

os.makedirs(f"output/closest-fit", exist_ok=True)
outfile = f"output/closest-fit/{lib}-{motif}.txt"
with open(outfile, "w") as f:
    for conf in range(len(coors)):
        closest_fit, closest_fit_rmsd = result[conf]
        print(closest_fit + 1, "{:.3f}".format(closest_fit_rmsd), file=f)
