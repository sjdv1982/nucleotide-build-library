import numpy as np


def peel_cluster(
    struc: "np.array",
    neighbor_func: callable,
    *,
    MAX_UNCLUSTERED: int = 20000,
    NSAMPLES=[10, 20, 20, 20, 50, 50, 100, 200, 500, 1000],
    REPORT_PROGRESS=True,
    RANDOM_SEED=0,
) -> list[int]:
    """'Peels off' the (approximate) largest clusters of an array of structures.

     The largest cluster is repeatedly determined, and its members are removed from the array, until
    at most MAX_UNCLUSTERED structures remain.
    A negative number for MAX_UNCLUSTERED instead calculates the -MAX_UNCLUSTERED largest clusters.

    The largest cluster is determined as follows:
        Initially, all unclustered structures are candidates. Then, for len(NSAMPLES) times:
            - A new random sample (of size NSAMPLES[n]) is selected.
            - All candidate structures have their neighbors calculated to each structure of the sample.
                The number of neighbors is counted (between 0 and NSAMPLES[n])
            - The half of the candidates with the lowest number of neighbors is eliminated
        The largest cluster is then the candidate with the highest number of neighbors to the last sample.

    Arguments:
    struc: an array of structures
    neighbor_func: a function that takes two arguments:
        - a single structure
        - an array of structures
        and must return an boolean array,
         indidating the neighborness (1 = is neighbor) between the single structure
          and each structure of the array

    Returns: a list of the indices of the largest clusters, in order.

    """
    import sys
    import numpy as np

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda iterable, **kwargs: iterable

    assert struc.ndim == 3 and struc.shape[2] == 3, struc.shape

    remain = struc.copy()
    clusters = []
    unclustered_mask = np.ones(len(struc), bool)

    np.random.seed(RANDOM_SEED)

    while 1:
        if REPORT_PROGRESS:
            print(f"Remaining structures: {len(remain)}", file=sys.stderr)
        contest = remain
        indices = np.arange((len(remain)))
        for nsample in tqdm(NSAMPLES):
            counts = np.zeros(len(contest), int)
            sample_inds = np.random.choice(len(remain), nsample, replace=True)
            sample = remain[sample_inds]

            sample2 = sample
            if REPORT_PROGRESS and len(contest) * nsample > 1000000:
                sample2 = tqdm(sample)
            for s in sample2:
                nb = neighbor_func(s, contest)
                counts += nb
            keep = np.argsort(-counts)[: int(len(contest) / 2 + 0.5)]
            keep.sort()
            contest = contest[keep]
            indices = indices[keep]
        best = indices[counts[keep].argmax()]
        # assert counts.max() == neighbor_func(remain[best], sample).sum()
        in_cluster = neighbor_func(remain[best], remain)
        unclustered = np.where(unclustered_mask)[0]
        if REPORT_PROGRESS:
            print(
                f"Structure {unclustered[best]+1}, cluster size {in_cluster.sum()}",
                file=sys.stderr,
            )
        clusters.append(int(unclustered[best]))
        unclustered_mask[unclustered_mask] = ~in_cluster
        remain = struc[unclustered_mask]

        if MAX_UNCLUSTERED > 0:
            if len(remain) <= MAX_UNCLUSTERED:
                break
        else:
            if len(clusters) == -MAX_UNCLUSTERED:
                break
    return clusters
