import numpy as np


def cluster_from_pairs(nb_pairs: "np.array", nr_elements: int) -> list[list[int]]:
    """Cluster a nr_elements x nr_elements neighbor matrix.
    The neighbor matrix is supplied as nb_pairs, which is a list of neighbor pairs
        (indices counting from zero).
    """
    assert nb_pairs.ndim == 2, nb_pairs.shape
    assert nb_pairs.shape[1] == 2
    assert nb_pairs.min() >= 0
    assert nr_elements > 0 and nr_elements < 1000000
    diag_mask = np.equal(nb_pairs[:, 0], nb_pairs[:, 1])
    nb_pairs = nb_pairs[~diag_mask]

    nnb = np.zeros(nr_elements, int)
    uniq, counts = np.unique(nb_pairs, axis=None, return_counts=True)
    nnb[uniq] = counts

    done = set()
    membership = {n: set() for n in uniq}
    for p1, p2 in nb_pairs:
        membership[p1].add(p2)
        membership[p2].add(p1)
    for n, mem in membership.items():
        nnb[n] = len(mem)

    clustering = []
    while 1:
        new_heart = np.argmax(nnb)
        if nnb[new_heart] == 0:
            singletons = [n for n in range(nr_elements) if n not in done]
            break
        mem0 = membership[new_heart]
        mem = sorted(list(mem0.difference(done)))
        clustering.append([new_heart] + mem)
        done.update(mem)
        done.add(new_heart)

        for n in mem:
            for nn in membership[n]:
                nnb[nn] -= 1
        nnb[new_heart] = 0
        nnb[list(mem0)] = 0

    for singleton in sorted(singletons):
        clustering.append([singleton])
    for clus in clustering:
        clus[:] = [int(c) for c in clus]
    return clustering
