import numpy as np
from nefertiti.functions import superimpose as _superimpose
from seamless import transformer

from . import functions


@transformer
def peel_cluster(
    # struc: np.array,
    struc,
    rmsd_threshold: float,
    *,
    MAX_UNCLUSTERED: int = 20000,
    NSAMPLES=[10, 20, 20, 20, 50, 50, 100, 200, 500, 1000],
    RANDOM_SEED: int = 0,
    SPECIAL__REPORT_PROGRESS=True,
):
    """Version of cluster.peel_cluster that uses Kabsch superposition from the Nefertiti project
     to determine RMSD distances.
    To be used with Seamless.
    """
    from .cluster import peel_cluster as peel_cluster_inner
    from .superimpose import superimpose_array

    def neighbor_func(struc, struc_array):
        _, rmsd = superimpose_array(struc_array, struc)
        return rmsd < rmsd_threshold

    return peel_cluster_inner(
        struc,
        neighbor_func=neighbor_func,
        MAX_UNCLUSTERED=MAX_UNCLUSTERED,
        NSAMPLES=NSAMPLES,
        RANDOM_SEED=RANDOM_SEED,
        REPORT_PROGRESS=SPECIAL__REPORT_PROGRESS,
    )


peel_cluster.modules.superimpose = _superimpose
peel_cluster.modules.cluster = functions.peel_cluster
peel_cluster.direct_print = True


@transformer
def reassign_clustering(
    # struc: np.array,
    struc,
    rmsd_threshold: float,
    initial_clustering: list[list[int]],
):
    """Version of cluster.reassign_clustering that uses Kabsch superposition from the Nefertiti project
     to determine RMSD distances.
    To be used with Seamless.
    """
    from .cluster import reassign_clustering as reassign_clustering_inner
    from .superimpose import superimpose_array

    def distance_func(struc, struc_array):
        _, rmsd = superimpose_array(struc_array, struc)
        return rmsd

    return reassign_clustering_inner(
        struc, rmsd_threshold, initial_clustering, distance_func
    )


reassign_clustering.modules.superimpose = _superimpose
reassign_clustering.modules.cluster = functions.reassign_clustering
reassign_clustering.direct_print = True
