import itertools
import sys
import numpy as np
from context_mask_fam import read_origins, context_mask_fam
from closest_fit_lib import closest_fit_with_context
from clusterlib import read_clustering

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda arg: arg


def err(*args):
    print(*args, file=sys.stderr)
    exit(1)


def _load(lib, motif):

    def get_clustering(precision):
        clusterfile = f"../output/lib-{lib}-{motif}-{precision}.all.clust"
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

        closest_clusterfile = f"../output/lib-{lib}-{motif}-{precision}.clust"
        closest_clustering = read_clustering(closest_clusterfile)
        closest_clustering = [[cc - 1 for cc in c] for c in closest_clustering]

        closest_cluster = {}
        for clusnr, cluster in enumerate(closest_clustering):
            assert cluster[0] == clustering[clusnr][0]
            for member in cluster:
                closest_cluster[member] = clusnr
        return clustering, closest_cluster

    coorfile = f"../nucleotide-fragments/{lib}/{motif}.npy"
    origin_file = f"../nucleotide-fragments/{lib}/origin/{motif}.txt"
    coors = np.load(coorfile)
    nconf = len(coors)
    origins = read_origins(origin_file)

    original_closest_fit_file = f"../output/closest-fit/{lib}-{motif}.txt"
    original_closest_fit = []
    with open(original_closest_fit_file) as f:
        for l in f:
            ll = l.split()
            ind = int(ll[0]) - 1
            rmsd = float(ll[1])
            original_closest_fit.append((ind, rmsd))
    assert len(coors) == len(origins)
    assert len(original_closest_fit) == len(coors)

    clustering1A, closest_cluster1A = get_clustering(1.0)
    clustering2A, closest_cluster2A = get_clustering(2.0)
    return (
        coors,
        origins,
        original_closest_fit,
        clustering1A,
        closest_cluster1A,
        clustering2A,
        closest_cluster2A,
    )


def closest_fit(lib, motif, database, fam_id, max_sample, sample_frac):
    (
        coors,
        origins,
        original_closest_fit,
        clustering1A,
        closest_cluster1A,
        clustering2A,
        closest_cluster2A,
    ) = _load(lib, motif)
    ctx_any, ctx_all = context_mask_fam(origins, database, fam_id)
    if sum(ctx_any) == 0:
        err(f"{database} entry {fam_id} does not exist in the dataset")
    print(f"context: any={sum(ctx_any)}, all={sum(ctx_all)}, total={len(coors)}")
    result = closest_fit_with_context(
        original_closest_fit,
        coors,
        ctx_any=ctx_any,
        ctx_all=ctx_all,
        clustering1A=clustering1A,
        closest_cluster1A=closest_cluster1A,
        clustering2A=clustering2A,
        closest_cluster2A=closest_cluster2A,
        max_sample=max_sample,
        sample_frac=sample_frac,
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("lib", help="library (dinuc or trinuc)")
    parser.add_argument("database", help="database (pfam or rfam)")
    parser.add_argument("fam_id", help="database family name, e.g RF00005")
    parser.add_argument("outfile", help="output file")
    parser.add_argument(
        "--sample",
        help="""Do not fit all fragments. 
Instead for each sequence motif, obtain a sample of approximately this size.
If there are less fragments than this, fit all fragments""",
        type=float,
    )
    args = parser.parse_args()

    lib = args.lib
    assert lib in ("dinuc", "trinuc"), lib
    database = args.database
    assert database in ("pfam", "rfam"), database
    fam_id = args.fam_id
    outfile = args.outfile

    n_nuc = 2 if lib == "dinuc" else 3
    motifs = ["".join(c) for c in itertools.product("AC", repeat=n_nuc)]
    result = []
    sample_frac = None
    max_sample = args.sample
    for motif in tqdm(motifs):
        motif_result, motif_n_result = closest_fit(
            lib, motif, database, fam_id, max_sample=max_sample, sample_frac=sample_frac
        )
        if sample_frac is None and max_sample is not None:
            if motif_n_result == len(motif_result):
                sample_frac = None
            else:
                sample_frac = len(motif_result) / motif_n_result
            max_sample = None
        result += motif_result
    with open(outfile, "w") as f:
        for closest_fit, closest_fit_rmsd in result:
            ind = -1
            r = 0
            if closest_fit is not None:
                ind = closest_fit + 1
                r = closest_fit_rmsd
            print(ind, "{:.3f}".format(r), file=f)
