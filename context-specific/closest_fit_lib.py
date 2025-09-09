import os
import sys
import random
import typing
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda arg, *, desc=None: arg


def err(*args):
    print(*args, file=sys.stderr)
    try:
        exit(1)
    except NameError:
        pass


# from clusterlib
def read_clustering(clusterfile):
    clustering = []
    if hasattr(clusterfile, "readlines"):
        f = clusterfile
    else:
        f = open(clusterfile)
    with f:
        for lnr, l in enumerate(f.readlines()):
            ll = l.split()
            assert ll[0] == "Cluster"
            assert ll[1] == str(lnr + 1)
            assert ll[2] == "->"
            c = [int(lll) for lll in ll[3:]]
            clustering.append(c)
    return clustering


# /from clusterlib

# from nefertiti

"""Functions related to Kabsch superimposition

Author: Sjoerd de Vries
License: GPLv3, https://www.gnu.org/licenses/gpl-3.0.en.html

This file is part of the Nefertiti project.
"""

import numpy as np
from numpy import sqrt
from numpy.linalg import svd, det


def superimpose(coor1, coor2):
    """Returns the rotation matrix and RMSD between coor1 and coor2

    Do coor1.dot(rotmat) to perform the superposition
    """
    assert coor1.ndim == 2 and coor1.shape[-1] == 3, coor1.shape
    assert coor2.ndim == 2 and coor2.shape[-1] == 3, coor2.shape
    assert coor2.shape[0] == coor1.shape[0]
    natoms = coor1.shape[0]

    c1 = coor1 - coor1.mean(axis=0)
    c2 = coor2 - coor2.mean(axis=0)

    residual1 = (c1 * c1).sum()
    residual2 = (c2 * c2).sum()
    covar = c1.T.dot(c2)

    v, s, wt = svd(covar)
    if det(v) * det(wt) < 0:
        s[-1] *= -1
        v[:, -1] *= -1
    rotmat = v.dot(wt)
    ss = (residual1 + residual2) - 2 * s.sum()
    if ss < 0:
        ss = 0
    rmsd = sqrt(ss / natoms)
    return rotmat, rmsd


def superimpose_array(coor1_array, coor2):
    """Returns the rotation matrix and RMSD between coor1 and coor2
    where coor1 is every element of coor1_array

    Do np.einsum("ijk,ikl->ijl", coor1_array, rotmat) to perform the superpositions
    """
    assert coor1_array.ndim == 3 and coor1_array.shape[-1] == 3, coor1_array.shape
    assert coor2.ndim == 2 and coor2.shape[-1] == 3, coor2.shape
    assert coor2.shape[0] == coor1_array.shape[1]
    natoms = coor2.shape[0]

    c1_array = coor1_array - coor1_array.mean(axis=1)[:, None, :]
    c2 = coor2 - coor2.mean(axis=0)

    residual1 = np.einsum("ijk,ijk->i", c1_array, c1_array)
    residual2 = (c2 * c2).sum()
    covar = np.einsum("ijk,jl->ikl", c1_array, c2)

    v, s, wt = svd(covar)
    reflect = det(v) * det(wt)
    s[:, -1] *= reflect
    v[:, :, -1] *= reflect[:, None]
    rotmat = np.einsum("...ij,...jk->...ik", v, wt)
    ss = (residual1 + residual2) - 2 * s.sum(axis=1)
    ss = np.maximum(ss, 0)
    rmsd = sqrt(ss / natoms)
    return rotmat, rmsd


# /from nefertiti


def closest_fit_with_context(
    original_closest_fit: list,
    coor: np.ndarray,
    *,
    ctx_any: list[bool],
    ctx_all: list[bool],
    clustering1A,
    closest_cluster1A,
    clustering2A,
    closest_cluster2A,
    allow_any2any: bool = True,
    max_sample=None,
    sample_frac=None,
) -> list[tuple[int, float]]:
    """
        Updates a previous closest fit with context information.

        Based on an abstract idea of "context" that can be positive or negative,
          it is considered that closest fit pairs cannot be between two structures
          for which the context is positive for both.
        These closest fit pairs are eliminated.

        Only structures where the context is sometimes positive (ctx_any=True) are considered.

        Current closest fits will be rejected and recomputed if the closest structure has always
         the same context (ctx_all=True). However, if allow_any2any is False, this happens also
         if the closest structure is sometimes the same.

        Recomputation is done using the multi-resolution method,

        Arguments:

        original_closest_fit, a list of (index, distance) pairs for each structure.
        "index" is a structure index counting from zero.

        coor: the structure coordinates

        clustering1A, clustering2A: a list of clusters with their member indices.
        Member indices start from zero. Every structure must be member of at least one cluster.

        closest_cluster1A, closest_cluster2A. The cluster index in clustering1A/2A indicating
        the closest cluster for each structure.

        ctx_any: a list of bools that indicate if the context may sometimes be positive.
        ctx_all: a list of bools that indicate if the context is always positive.
    .
        allow_any2any: allow closest fit pairs where both structures may sometimes (but not always) be positive.

        max_sample: refit a sample with a maximum number of structures. If None, refit all structures
        sample_frac: refit a sample with a fraction of the structures. If None, refit all structures

        Returns the refitted structures
    """
    assert len(original_closest_fit) == len(coor)
    assert len(coor) == len(ctx_all)
    assert len(ctx_all) == len(ctx_any)
    assert coor.ndim == 3 and coor.shape[-1] == 3, coor.shape

    assert max_sample is None or sample_frac is None

    ctx_all = np.array(ctx_all, bool)
    ctx_any = np.array(ctx_any, bool)

    if not allow_any2any:
        to_refit = ctx_any.copy()
        allowed_fit = ~ctx_any
    else:
        to_refit = ctx_all.copy()
        allowed_fit = ~ctx_all

    sample_mask = np.ones(len(coor), bool)
    if max_sample is not None:
        currsize = ctx_any.sum()
        if currsize > max_sample:
            sample_frac = max_sample / currsize
    if sample_frac is not None:
        sample_mask = (np.random.sample(len(coor)) <= sample_frac).astype(bool)

    to_report = ctx_any & sample_mask
    to_refit2 = to_refit & sample_mask

    # if ctx_any.sum() > to_report.sum():
    #    print("sample:", to_report.sum())
    result = closest_refit(
        original_closest_fit,
        coor,
        to_refit2,
        allowed_fit,
        clustering1A=clustering1A,
        closest_cluster1A=closest_cluster1A,
        clustering2A=clustering2A,
        closest_cluster2A=closest_cluster2A,
    )
    if allow_any2any:
        for ind in np.where(to_report & ~to_refit2)[0]:
            result[ind] = [None, 0.0]
    result = [
        result[k] for k in sorted(np.where(to_report)[0]) if result[k][1] is not None
    ]
    return result


def completeness_with_context(
    clustering,
    ctx_any: list[bool],
    ctx_all: list[bool],
    *,
    allow_any2any: bool = True,
    return_certainty: bool = False,
) -> typing.Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
        Determines the "completeness" of cluster members using direct clustering analysis

        Based on an abstract idea of "context" that can be positive or negative,
          it is considered that closest fit pairs cannot be between two structures
          for which the context is positive for both.
        These closest fit pairs are eliminated.
        If allow_any2any is False, this happens also if the closest structure is sometimes the same.
        (in case of structures that have multiple origins)

        Only structures where the context is sometimes positive (ctx_any=True) are considered.

        Arguments:

        clustering: clustering at the desired precision X.
            It is assumed that each cluster contains all members within distance X of the cluster heart,
             and that cluster hearts are at least X apart.
        ctx_any: a list of bools that indicate if the context may sometimes be positive.
        ctx_all: a list of bools that indicate if the context is always positive.
    .
        allow_any2any: allow closest fit pairs where both structures may sometimes (but not always) be positive.

        Returns a boolean numpy array, indicating completeness, for every structure where ctx_any is True.

        if return_certainty, also return a second boolean array indicating where the completeness was determined with certainty.
            For cases where certainty=0, the accuracy is still in the order of 80 %.

    """
    assert len(ctx_all) == len(ctx_any)
    ctx_all = np.array(ctx_all, bool)
    ctx_any = np.array(ctx_any, bool)

    allowed_fit_clustering = []
    for clus in clustering:
        allowed_fit = ~ctx_all if allow_any2any else ~ctx_any
        allowed_fit_clus = [allowed_fit[member] for member in clus]
        allowed_fit_clustering.append(allowed_fit_clus)

    in_cluster = {n: [] for n in range(len(ctx_any))}
    for clusnr, cluster in enumerate(clustering):
        for member in cluster:
            in_cluster[member].append(clusnr)

    hearts = {clus[0]: clusnr for clusnr, clus in enumerate(clustering)}

    result = []
    result_certain = []
    for conf in np.where(ctx_any)[0]:
        complete = None
        certain = None
        hclus = hearts.get(conf)
        if hclus is None:
            # not a cluster heart
            for clus in in_cluster[conf]:
                allowed_fit_cluster = allowed_fit_clustering[clus]
                # default: only in-ctx clusters
                #   => no known acceptable closest fit, but we aren't certain
                complete = False
                certain = False
                if allowed_fit_cluster[0]:
                    # cluster heart is an acceptable closest fit
                    complete, certain = True, True
                    break
                elif any(allowed_fit_cluster):
                    # some of the cluster is an acceptable closest fit
                    complete, certain = True, False
        else:
            # cluster heart
            assert in_cluster[conf] == [hclus]
            certain = True
            allowed_fit_cluster = allowed_fit_clustering[hclus]
            if any(allowed_fit_cluster):
                complete = True
            else:
                complete = False

        result.append(complete)
        result_certain.append(certain)

    result = np.array(result, dtype=bool)
    if not return_certainty:
        return result
    else:
        result_certain = np.array(result_certain)
        return result, result_certain


def _get_closest_fit(
    coors,
    allowed_fit,
    clustering,
    closest_cluster,
    precision,
    *,
    done=[],
    explore_members=True,
):

    done = set(done)
    nconf = len(coors)
    SMALL_STRUC = 500
    result = {}

    closest_cluster = {int(k): v for k, v in closest_cluster.items()}

    in_cluster = {n: [] for n in range(nconf)}
    for clusnr, cluster in enumerate(clustering):
        for member in cluster:
            in_cluster[member].append(clusnr)

    todo_confs = [conf for conf in range(nconf) if conf not in done]
    random.shuffle(todo_confs)
    for conf in tqdm(todo_confs, desc=f"Find {precision}A bullseye clusters..."):

        struc = coors[conf]

        """
        First, we try to identify one or more "bullseye" clusters:
            1A clusters where the closest fitting conformer must surely be part of.

        Candidate bullseye clusters have the following properties:
        - Not all with allowed_fit=False (just the cluster heart is ok)
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
        if all([not allowed_fit[cc] for cc in clus]):
            # The entire closest cluster is from the same context as the fitted structure
            pass
        else:
            cclus_heart = clus[0]
            _, closest_cluster_rmsd = superimpose(struc, coors[cclus_heart])

            bullseye_candidates.append(cclusnr)
            bullseye_rmsds.append(closest_cluster_rmsd)

            if allowed_fit[cclus_heart]:
                closest_known_rmsd = closest_cluster_rmsd
            else:
                clus2 = [c for c in clus if allowed_fit[c]]
                _, rmsds = superimpose_array(coors[clus2[:SMALL_STRUC]], struc)
                closest_known_rmsd = rmsds.min()

        if not have_enough_bullseye():
            for clusnr in in_cluster[conf]:
                if clusnr == closest_cluster[conf]:
                    continue
                clus = clustering[clusnr]
                if all([not allowed_fit[cc] for cc in clus]):
                    # The entire cluster is from the same context as the fitted structure
                    continue
                clus_heart = clus[0]
                _, cluster_rmsd = superimpose(struc, coors[clus_heart])

                # Now we really need a closest known RMSD
                if allowed_fit[clus_heart]:
                    if closest_known_rmsd is None or cluster_rmsd < closest_known_rmsd:
                        closest_known_rmsd = cluster_rmsd
                elif closest_known_rmsd is None:
                    clus2 = [c for c in clus if allowed_fit[c]]
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
                clus2 = [c for c in clus if allowed_fit[c]]
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
                    clus2 = [c for c in clus if allowed_fit[c]]
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
        candidates = [c for c in candidates if allowed_fit[c]]
        assert len(candidates)

        closest_fit_rmsd = None
        closest_fit = None
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


def closest_refit(
    original_closest_fit: list,
    coor: np.ndarray,
    to_refit: np.ndarray,
    allowed_fit: np.ndarray,
    *,
    clustering1A,
    closest_cluster1A,
    clustering2A,
    closest_cluster2A,
) -> list:
    """Updates an original closest fit of coor[to_refit] on coor.
    Coordinates are refitted using the multi-resolution method.

    Arguments:

    original_closest_fit, a list of (index, distance) pairs for each structure in coor1.
        "index" is the index of the closest fit in coor2, counting from zero.

    to_refit: Numpy array of bools (mask) indicating if a structure needs refitting

    allowed_fit: All cases where the closest fit has allowed_fit=False are rejected and recomputed

    clustering1A, clustering2A: a list of clusters with their member indices.
    Member indices start from zero. Every structure must be member of at least one cluster.

    closest_cluster1A, closest_cluster2A. The cluster index in clustering1A/2A indicating
    the closest cluster for each structure.
    """
    assert coor.ndim == 3 and coor.shape[-1] == 3, coor.shape
    assert len(to_refit) == len(coor)
    assert len(original_closest_fit) == len(coor)
    assert len(allowed_fit) == len(coor)

    coor = coor - coor.mean(axis=1)[:, None, :]

    nconf = len(coor)

    no_refit = set(np.where(~to_refit)[0])

    result0 = {}
    for conf in range(nconf):
        if conf in no_refit:
            continue
        ori = original_closest_fit[conf]
        if allowed_fit[ori[0]]:
            result0[conf] = ori

    done = list(no_refit) + list(result0.keys())

    result1A = _get_closest_fit(
        coor,
        allowed_fit,
        clustering1A,
        closest_cluster1A,
        1.0,
        explore_members=False,  # True brings no benefit
        done=done,
    )

    done = list(no_refit) + list(result1A.keys()) + list(result0.keys())

    # START: multi-resolution method
    result2A = _get_closest_fit(
        coor,
        allowed_fit,
        clustering2A,
        closest_cluster2A,
        2.0,
        explore_members=False,  # True slows it down
        done=done,
    )

    # For the rest, we have to explicitly compare against all structures, not just a few clusters...

    remaining = [
        conf
        for conf in range(len(coor))
        if conf not in no_refit
        and conf not in result1A
        and conf not in result2A
        and conf not in result0
    ]

    random.shuffle(remaining)
    remaining = np.array(remaining, int)

    result_remaining = {}

    # ... However, we may be able to eliminate big clusters in bulk or in part
    big_clust2A = [clus for clus in clustering2A if len(clus) > 20]
    big_struc2A = coor[[clus[0] for clus in big_clust2A]]
    big_conf2A = set(sum(big_clust2A, []))
    big_clust2A = [np.array(l) for l in big_clust2A]

    big_clust1A = [clus for clus in clustering1A if len(clus) > 20]
    big_struc1A = coor[[clus[0] for clus in big_clust1A]]
    big_conf1A = set(sum(big_clust1A, []))
    big_clust1A = [np.array(l) for l in big_clust1A]

    in_cluster = {n: [] for n in range(nconf)}
    for clusnr, cluster in enumerate(clustering1A):
        for member in cluster:
            in_cluster[member].append(clusnr)

    big_clust2A_rmsd = []
    for n, big_clust in enumerate(big_clust2A):
        big_struc = big_struc2A[n]
        cluster_struc = coor[big_clust]
        _, rmsd = superimpose_array(cluster_struc, big_struc)
        big_clust2A_rmsd.append(rmsd)

    all_candidates = np.ones((len(remaining), len(coor)), bool)
    closest_rmsd_initial_estimates = []
    for confnr, conf in enumerate(remaining):
        struc = coor[conf]
        _, rmsd_bigclust1A = superimpose_array(big_struc1A, struc)
        _, rmsd_bigclust2A = superimpose_array(big_struc2A, struc)
        closest_rmsd_estimate = rmsd_bigclust1A.min()
        if in_cluster[conf]:
            myclusters = [clustering1A[clusnr][0] for clusnr in in_cluster[conf]]
            myclusters = [c for c in myclusters if allowed_fit[c]]
            if len(myclusters):
                myclusters_struc = coor[myclusters]
                _, r = superimpose_array(myclusters_struc, struc)
                closest_rmsd_estimate = r.min()

        closest_rmsd_initial_estimates.append(closest_rmsd_estimate)

        # We have now a closest RMSD estimate X
        candidates = all_candidates[confnr]
        # Eliminate-in-bulk all 1A big clusters where the cluster heart RMSD > X + 1
        for clusnr in range(len(big_clust1A)):
            if rmsd_bigclust1A[clusnr] > closest_rmsd_estimate + 1:
                big_clust = big_clust1A[clusnr]
                candidates[big_clust] = 0

        # For each 2A big cluster, we have:.
        #   Y, the RMSD between cluster heart and fitted conformer
        #   Z, the RMSD between cluster heart and a particular member
        # If Z < Y - X, we can eliminate that member
        for clusnr in range(len(big_clust2A)):
            y = rmsd_bigclust2A[clusnr]
            elim = big_clust2A_rmsd[clusnr] < (y - closest_rmsd_estimate)
            big_clust = big_clust2A[clusnr]
            candidates[big_clust[elim]] = 0

    # We will have to brute-force against the rest...
    for confnr, conf in enumerate(tqdm(remaining, desc="Search remaining")):
        struc = coor[conf]
        closest_fit_rmsd = None
        closest_fit = None
        candidates = np.nonzero(all_candidates[confnr])[0]
        candidates = [c for c in candidates if allowed_fit[c]]
        for pos in range(0, len(candidates), 1000):
            chunk = candidates[pos : pos + 1000]
            chunk_candidate_struc = coor[chunk]
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

    # END: multi-resolution method

    return result
