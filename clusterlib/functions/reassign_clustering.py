import numpy as np


def reassign_clustering(
    struc: "np.array",
    distance_threshold: float,
    initial_clustering: list[list[int]],
    distance_func: callable,
) -> tuple[list[list[int]], list[list[int]]]:
    """Reassigns structures to clusters
    Requires an existing clustering, where for each cluster:
        - The first member is the cluster center
        - The other members are guaranteed to be within distance_threshold of the center
        - Each structure is member of at least one cluster
    The format of the clustering is a list of lists as outputted by read_clustering

    Returns:
    - a new clustering with each structure assigned to the closest cluster center.
    - a clustering where each structure is member of *all* clusters under distance_threshold
    The cluster centers themselves are not changed.
    """
    import numpy as np

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda iterable, **kwargs: iterable

    ini_clustering = {int(c[0]): c for c in initial_clustering}
    cluster_indices = list(ini_clustering.keys())
    all_indices = sum(initial_clustering, [])
    cluster_indices = np.array(cluster_indices, int)
    all_indices = np.array(all_indices, int)
    all_indices.sort()

    if all_indices[0] != 0:
        raise ValueError("Clustering does not start at 0")
    elif not np.alltrue(np.unique(all_indices) == np.arange(all_indices[-1] + 1)):
        raise ValueError("Clustering has missing indices")

    del all_indices

    cluster_struc = struc[cluster_indices]

    clustering_all = {h: [] for h in cluster_indices}
    clustering_closest = {h: [] for h in cluster_indices}

    approx_threshold = 2 * distance_threshold
    for cnr, c in enumerate(tqdm(cluster_indices)):
        clus = ini_clustering[c]
        assert clus[0] == c
        distance = distance_func(struc[c], cluster_struc)
        approx_indices = np.where(distance < approx_threshold)[0]
        cpos = approx_indices.tolist().index(cnr)

        mat = np.empty((len(clus), len(approx_indices)))
        nmat = len(clus) * len(approx_indices)
        curr_struc = struc[clus]
        approx_struc = cluster_struc[approx_indices]
        if len(approx_indices) > len(clus):
            it = range(len(clus))
            if nmat > 1000000:
                it = tqdm(list(it))
            for n in it:
                s = struc[clus[n]]
                distance = distance_func(s, approx_struc)
                mat[n] = distance
        else:
            it = range(len(approx_indices))
            if nmat > 1000000:
                it = tqdm(list(it))
            for n in it:
                s = approx_struc[n]
                distance = distance_func(s, curr_struc)
                mat[:, n] = distance
        assert mat[0, cpos] < 0.1

        for ccnr, cc in enumerate(clus):
            closest = mat[ccnr].argmin()
            closest_approx_ind = approx_indices[closest]
            closest_cluster = cluster_indices[closest_approx_ind]
            clustering_closest[closest_cluster].append(cc)

            close = mat[ccnr] < distance_threshold
            if close.sum():
                close_approx_ind = approx_indices[close]
                close_clusters = cluster_indices[close_approx_ind]
                for close_cluster in close_clusters:
                    clustering_all[close_cluster].append(cc)

    list_clustering_closest = []
    list_clustering_all = []
    for c in cluster_indices:
        l = [c] + sorted(list({int(cc) for cc in clustering_closest[c] if cc != c}))
        list_clustering_closest.append(l)

        l = [c] + sorted(list({int(cc) for cc in clustering_all[c] if cc != c}))
        list_clustering_all.append(l)

    return list_clustering_closest, list_clustering_all
