import itertools
import random
import sys
import numpy as np
from context_mask_fam import read_origins, context_mask_fam, get_database_mapping
from closest_fit_lib import (
    closest_fit_with_context,
    completeness_with_context,
    read_clustering,
)

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
        elif not np.all(indices == np.arange(len(indices)) + 1):
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

    clustering05A, closest_cluster05A = get_clustering(0.5)
    clustering1A, closest_cluster1A = get_clustering(1.0)
    clustering2A, closest_cluster2A = get_clustering(2.0)
    return (
        coors,
        origins,
        original_closest_fit,
        clustering05A,
        closest_cluster05A,
        clustering1A,
        closest_cluster1A,
        clustering2A,
        closest_cluster2A,
    )


class ClosestFit:
    def __init__(self, lib):
        assert lib in ("dinuc", "trinuc")
        n_nuc = 2 if lib == "dinuc" else 3
        self.motifs = ["".join(c) for c in itertools.product("AC", repeat=n_nuc)]
        self._loaded = {}
        self._database_mappings = {}
        for motif in self.motifs:
            self._loaded[motif] = _load(lib, motif)

    def get_database_mapping(self, database):
        if database not in self._database_mappings:
            self._database_mappings[database] = get_database_mapping(database)
        return self._database_mappings[database]

    def get_completeness(self, database, fam_id):
        result = {}

        for motif in self.motifs:
            loaded = self._loaded[motif]
            origins = loaded[1]

            ctx_any, ctx_all = context_mask_fam(
                origins,
                database,
                fam_id,
                database_mapping=self.get_database_mapping(database),
            )
            if sum(ctx_any) == 0:
                err(f"{database} entry {fam_id} does not exist in the dataset")

            clusterings = loaded[3::2]
            for nclustering, precision in enumerate((0.5, 1.0, 2.0)):
                clustering = clusterings[nclustering]
                completeness, certainty = completeness_with_context(
                    clustering, ctx_any, ctx_all, return_certainty=True
                )
                result[motif, precision] = completeness, certainty
        return result

    def get_closest_fit(self, database, fam_id, max_sample=None):
        """Obtain the closest fit from multi-resolution analysis, re-using the earlier result.

            Do this for all motifs of a library.

            lib: library (dinuc or trinuc)
            database: pfam or rfam
            fam_id: database family name, e.g RF00005
            max_sample: optional.

                    Do not fit all fragments.
        Instead for each sequence motif, obtain a sample of approximately this size.
        If there are less fragments than this, fit all fragments

        Returns:
        - result dict: the results for all motifs
        """

        if max_sample is not None and max_sample <= 0:
            max_sample = None
        sample_frac = None
        result = {}
        for motif in tqdm(self.motifs, desc="Iterate over sequence motifs..."):

            (
                coors,
                origins,
                original_closest_fit,
                _,
                _,
                clustering1A,
                closest_cluster1A,
                clustering2A,
                closest_cluster2A,
            ) = self._loaded[motif]
            ctx_any, ctx_all = context_mask_fam(
                origins,
                database,
                fam_id,
                database_mapping=self.get_database_mapping(database),
            )
            if sum(ctx_any) == 0:
                err(f"{database} entry {fam_id} does not exist in the dataset")
            motif_result0 = closest_fit_with_context(
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
            motif_result = np.array([v[1] for v in motif_result0 if v[1] is not None])

            motif_n_context = sum(ctx_any)

            if sample_frac is None and max_sample is not None:
                if motif_n_context == len(motif_result):
                    sample_frac = None  # sample all structures
                else:
                    sample_frac = len(motif_result) / motif_n_context
                max_sample = None
            result[motif] = motif_result
        return result

    def get_closest_fit_baseline(self, database, fam_id):
        """Obtain the baseline (eliminate fragments from same PDB) closest fit.
        Requires that the baseline analysis is present in ../output/closest-fit

            Do this for all motifs of a library.

            lib: library (dinuc or trinuc)
            database: pfam or rfam
            fam_id: database family name, e.g RF00005
            max_sample: optional.

        Returns:
        - result dict: the results for all motifs
        """

        result = {}
        for motif in self.motifs:

            (
                origins,
                original_closest_fit,
            ) = self._loaded[
                motif
            ][1:3]
            original_rmsd = np.array([v[1] for v in original_closest_fit])
            ctx_any, ctx_all = context_mask_fam(
                origins,
                database,
                fam_id,
                database_mapping=self.get_database_mapping(database),
            )
            result[motif] = original_rmsd[ctx_any]
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
Instead for each sequence motif, obtain a random sample of approximately this size.
If there are less fragments than this, fit all fragments""",
        type=float,
    )
    parser.add_argument(
        "--seed",
        help="""Random generator seed""",
        default=0,
        type=int,
    )
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    lib = args.lib
    assert lib in ("dinuc", "trinuc"), lib
    database = args.database
    assert database in ("pfam", "rfam"), database

    obj = ClosestFit(lib)
    completeness = obj.get_completeness(args.database, args.fam_id)

    print("Completeness at 0.5A, direct cluster analysis")
    print(
        [
            (k[0], "{0:.3f}".format(v[0].mean()))
            for k, v in completeness.items()
            if k[1] == 0.5
        ]
    )
    print("uncertainty")
    print(
        [
            (k[0], "{0:.3f}".format(1 - v[1].mean()))
            for k, v in completeness.items()
            if k[1] == 0.5
        ],
    )
    print("Completeness at 1.0A, direct cluster analysis")
    print(
        [
            (k[0], "{0:.3f}".format(v[0].mean()))
            for k, v in completeness.items()
            if k[1] == 1.0
        ]
    )
    print("uncertainty")
    print(
        [
            (k[0], "{0:.3f}".format(1 - v[1].mean()))
            for k, v in completeness.items()
            if k[1] == 1.0
        ],
    )
    result = obj.get_closest_fit(args.database, args.fam_id, max_sample=args.sample)
    print("Completeness at 0.5A, explicit closest fit calculation")
    print(
        [
            (k, "{0:.3f}".format((np.array([v for v in result[k]]) < 0.5).mean()))
            for k in result
        ]
    )
    print("Completeness at 1.0A, explicit closest fit calculation")
    print(
        [
            (k, "{0:.3f}".format((np.array([v for v in result[k]]) < 1.0).mean()))
            for k in result
        ]
    )

    result = np.concatenate(list(result.values()))

    with open(args.outfile, "w") as f:
        for r in result:
            print("{:.3f}".format(r), file=f)
