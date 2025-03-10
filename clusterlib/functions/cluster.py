import numpy as np


def cluster(nb_matrix: "np.array") -> list[list[int]]:
    import numpy as np

    assert nb_matrix.dtype == bool, nb_matrix.dtype
    assert nb_matrix.ndim == 2, nb_matrix.shape
    assert nb_matrix.shape[1] == nb_matrix.shape[0], nb_matrix.shape
    nstruc = len(nb_matrix)
    _r = list(range(nstruc))
    assert np.alltrue(nb_matrix[_r, _r])  # diagonal must be 1

    m = nb_matrix.copy()
    clustering = []
    cluster_hearts = []
    while 1:
        nnb = m.sum(axis=1)
        new_heart = np.argmax(nnb)
        if nnb[new_heart] <= 1:
            singletons = set(np.where(nnb)[0])
            break
        cluster_hearts.append(new_heart)
        inds = m[new_heart].copy()
        m[inds] = 0
        m[:, inds] = 0
        inds[new_heart] = 0
        clustering.append([new_heart] + np.where(inds)[0].tolist())

    singleton_mask = np.array([n in singletons for n in range(nstruc)], bool)
    assert np.alltrue(nb_matrix[singleton_mask][:, cluster_hearts].sum(axis=1) == 0)
    assert nb_matrix[~singleton_mask][:, cluster_hearts].sum(axis=1).min() >= 1

    for singleton in sorted(singletons):
        clustering.append([singleton])
    for clus in clustering:
        clus[:] = [int(c) for c in clus]
    return clustering
