import sys
import numpy as np
import itertools

from clusterlib import read_clustering
from nefertiti.functions.superimpose import superimpose_array

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda arg: arg

lib = sys.argv[1]
assert lib in ("dinuc", "trinuc")
precision = sys.argv[2]

n_nuc = 2 if lib == "dinuc" else 3
motifs = ["".join(c) for c in itertools.product("AC", repeat=n_nuc)]

singleton_fittings = []
singleton_fittings_with_extension = []

for motif in motifs:
    with open(f"build-library-{lib}-{motif}-{precision}-singleton-fit.txt") as f:
        for l in f.readlines():
            if not len(l.strip()) or l.strip().startswith("#"):
                continue
            ll = l.split()
            rmsd1 = float(ll[4])
            singleton_fittings.append(rmsd1)

            rmsd2 = float(ll[7])
            rmsd = min(rmsd1, rmsd2) if rmsd2 != -1 else rmsd1
            singleton_fittings_with_extension.append(rmsd)

with open(f"library-fit-{lib}-{precision}-singleton-primary.txt", "w") as f:
    for rmsd in singleton_fittings:
        print("{:.3f}".format(rmsd), file=f)

with open(f"library-fit-{lib}-{precision}-singleton-with-extension.txt", "w") as f:
    for rmsd in singleton_fittings_with_extension:
        print("{:.3f}".format(rmsd), file=f)


def read_origins(origin_file):
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
    return origins


all_intra_cluster_fittings = []
for motif in tqdm(motifs):
    ####################################################################
    # Load and validate inputs
    ####################################################################

    origin_file = f"lib-{lib}-nonredundant-filtered-{motif}-origin.txt"
    clusterfile = f"lib-{lib}-{motif}-{precision}.all.clust"
    coorfile = f"lib-{lib}-nonredundant-filtered-{motif}.npy"

    coors = np.load(coorfile)

    clustering = read_clustering(clusterfile)
    clustering = [[cc - 1 for cc in c] for c in clustering]

    origins = read_origins(origin_file)
    nconf = len(origins)
    assert nconf == len(coors)

    primary_indices = []
    with open(f"build-library-{lib}-{motif}-{precision}-primary-indices.txt") as f:
        for l in f.readlines():
            l = l.strip()
            if not len(l):
                continue
            primary_indices.append(int(l) - 1)

    replacement_indices = []
    with open(f"build-library-{lib}-{motif}-{precision}-replacement-indices.txt") as f:
        for l in f.readlines():
            l = l.strip()
            if not len(l):
                continue
            pos = len(replacement_indices)
            ind = primary_indices[pos]
            if l != "----":
                ind = int(l) - 1
            replacement_indices.append(ind)
    assert len(replacement_indices) == len(primary_indices)

    ####################################################################
    # Fit to primary index or replacement index, for each primary cluster
    ####################################################################

    all_clustered = set()
    closest_rmsd = np.full(nconf, np.inf)

    for clusnr, clus in enumerate(clustering):
        clus = np.array(clus)
        clus_origins = [origins[c] for c in clus]
        if clus_origins[0] is not None:
            replacement_mask = np.array(
                [c == clus_origins[0] for c in clus_origins], bool
            )
            if all(replacement_mask):
                # singleton
                continue
            assert len(replacement_mask) == len(clus)

        for ind in clus:
            all_clustered.add(ind)

        primary_index = clus[0]
        pos = primary_indices.index(primary_index)
        replacement_index = replacement_indices[pos]
        primary_struc, replacement_struc = (
            coors[primary_index],
            coors[replacement_index],
        )
        if clus_origins[0] is not None:
            assert replacement_index in clus, (replacement_index, clus)
            fit_to_primary_ind = clus[~replacement_mask]
            fit_to_replacement_ind = clus[replacement_mask]

            fit_to_primary_struc = coors[fit_to_primary_ind]
            _, rmsd1 = superimpose_array(fit_to_primary_struc, primary_struc)
            assert rmsd1.max() < 2 * float(precision) + 0.1, rmsd1.max()
            fit_to_replacement_struc = coors[fit_to_replacement_ind]
            _, rmsd2 = superimpose_array(fit_to_replacement_struc, replacement_struc)
            assert rmsd2.max() < 2 * float(precision) + 0.1, rmsd2.max()

            rmsd = np.empty(len(clus))
            rmsd[~replacement_mask] = rmsd1
            rmsd[replacement_mask] = rmsd2
        else:
            cluster_struc = coors[clus]
            _, rmsd = superimpose_array(cluster_struc, primary_struc)
            assert rmsd.max() < float(precision) + 0.1, rmsd.max()

        curr_clus_closest = closest_rmsd[clus]
        closest_rmsd[clus] = np.minimum(curr_clus_closest, rmsd)

    intra_cluster_fittings = closest_rmsd[list(all_clustered)]
    all_intra_cluster_fittings.append(intra_cluster_fittings)

all_intra_cluster_fittings = np.concatenate(all_intra_cluster_fittings)

with open(f"library-fit-{lib}-{precision}-intra-cluster.txt", "w") as f:
    for rmsd in all_intra_cluster_fittings:
        print("{:.3f}".format(rmsd), file=f)
