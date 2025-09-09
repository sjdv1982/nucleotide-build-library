import sys
import numpy as np
from clusterlib import read_clustering
from nefertiti.functions.superimpose import (
    superimpose,
    superimpose_array,
    superimpose_array_from_covar,
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
precision = sys.argv[3]


origin_file = f"input/lib-{lib}-nonredundant-filtered-{motif}-origin.txt"
clusterfile = f"output/lib-{lib}-{motif}-{precision}.all.clust"
coorfile = f"input/lib-{lib}-nonredundant-filtered-{motif}.npy"

coors = np.load(coorfile)
assert coors.ndim == 3 and coors.shape[-1] == 3, coors.shape
coors -= coors.mean(axis=1)[:, None, :]
coors_residuals = np.einsum("ijk,ijk->i", coors, coors)

clustering = read_clustering(clusterfile)

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

####################################################################
# Determine primary library
####################################################################

# Sort clustering by cluster length; use cluster heart index to resolve ties
clustering = sorted(clustering, key=lambda clus: clus[0])
clustering = sorted(clustering, key=lambda clus: len(clus), reverse=True)
# Subtract 1 from each index
clustering = [[cc - 1 for cc in c] for c in clustering]

in_cluster = {n: [] for n in range(nconf)}
for clusnr, cluster in enumerate(clustering):
    for member in cluster:
        in_cluster[member].append(clusnr)


# Sort into primary clusters and singletons
primary_clusters = []
singletons = []

for clus in clustering:
    heart = clus[0]
    is_singleton = False
    if origins[heart] is not None:
        is_singleton = True
        for clusnr2 in in_cluster[heart]:
            clus2 = clustering[clusnr2]
            if not all([origins[c] == origins[heart] for c in clus2]):
                is_singleton = False
                break
    if is_singleton:
        singletons.append(heart)
    else:
        primary_clusters.append(clus)

with open(
    f"output/build-library-{lib}-{motif}-{precision}-primary-indices.txt", "w"
) as f:
    for clus in primary_clusters:
        print(clus[0] + 1, file=f)

print(
    "Primary library: identify replacements and calculate replacement RMSD",
    file=sys.stderr,
)

primary_coors = coors[[clus[0] for clus in primary_clusters]]
replacement = []
replacement_rmsd = []

for clus in tqdm(primary_clusters):
    heart = clus[0]
    if origins[heart] is None:
        rep = None
        r = None
    else:
        others = [cc for cc in clus if origins[cc] != origins[heart]]
        struc = coors[heart]
        other_struc = coors[others]
        _, rmsd = superimpose_array(other_struc, struc)
        ind = rmsd.argmin()
        r = rmsd[ind]
        rep = others[ind]
    replacement_rmsd.append(r)
    replacement.append(rep)

with open(
    f"output/build-library-{lib}-{motif}-{precision}-replacement-indices.txt", "w"
) as f:
    for rep in replacement:
        rep2 = rep + 1 if rep is not None else "----"
        print(rep2, file=f)

replacement_coors0 = [coors[rep] if rep is not None else None for rep in replacement]
replacement_coors = [
    c1 if c1 is not None else c2 for c1, c2 in zip(replacement_coors0, primary_coors)
]
replacement_coors = np.array(replacement_coors)

print("Singletons: calculate RMSD towards primary library", file=sys.stderr)
primary_origins = {origins[clus[0]] for clus in primary_clusters}
singleton_closest_primary_cluster = []
singleton_closest_primary_rmsd = []
for ind in tqdm(singletons):
    struc = coors[ind]
    _, rmsd = superimpose_array(primary_coors, struc)
    ori = origins[ind]
    if ori in primary_origins:
        repmask = [origins[clus[0]] == ori for clus in primary_clusters]
        repstruc = replacement_coors[repmask]
        _, rmsd2 = superimpose_array(repstruc, struc)
        rmsd[np.where(repmask)[0]] = rmsd2
    singleton_closest_primary_cluster.append(rmsd.argmin())
    singleton_closest_primary_rmsd.append(rmsd.min())


print("Singletons: calculate RMSD towards other singletons", file=sys.stderr)
singleton_origins = [origins[ind] for ind in singletons]
singleton_coors = coors[singletons]
singleton_residuals = coors_residuals[singletons]
singleton_closest_singleton = []
singleton_closest_singleton_rmsd = []
for ind in tqdm(singletons):
    struc = coors[ind]
    covar = np.einsum("ijk,jl->ikl", singleton_coors, struc)
    residuals = singleton_residuals + coors_residuals[ind]
    sd = superimpose_array_from_covar(covar, residuals, len(struc), return_sd=True)
    rmsd = np.sqrt(sd / len(struc))

    ori = origins[ind]
    ori_mask = [(o == ori) for o in singleton_origins]
    rmsd[ori_mask] = np.inf

    singleton_closest_singleton.append(rmsd.argmin())
    singleton_closest_singleton_rmsd.append(rmsd.min())

print("Singletons: extract extension library", file=sys.stderr)

singleton_closest_singleton = np.array(singleton_closest_singleton)
singleton_closest_singleton_rmsd = np.array(singleton_closest_singleton_rmsd)
singleton_closest_primary_cluster = np.array(singleton_closest_primary_cluster)
singleton_closest_primary_rmsd = np.array(singleton_closest_primary_rmsd)

extension = []
extension_rev = {}
for indnr, ind in enumerate(singletons):
    rmsd1 = singleton_closest_primary_rmsd[indnr]
    rmsd2 = singleton_closest_singleton_rmsd[indnr]
    # The extension library must be at least 0.1 A better than the primary library
    if rmsd1 - rmsd2 < 0.1:
        continue
    ind_extension = singletons[singleton_closest_singleton[indnr]]
    try:
        pos = extension_rev[ind_extension]
    except KeyError:
        pos = len(extension)
        extension.append(ind_extension)
        extension_rev[ind_extension] = pos

print(f"{len(extension)} singletons in the extension library")
extension_coors = coors[extension]


print("Write library files", file=sys.stderr)
base = f"library/{lib}-{motif}-{precision}"
np.save(f"{base}.npy", primary_coors)
np.save(f"{base}-replacement.npy", replacement_coors)
with open(f"{base}-replacement.txt", "w") as f:
    print("#origin #RMSD #repl-origin", file=f)
    for n, clus in enumerate(primary_clusters):
        origin = origins[clus[0]]
        if origin is None:
            o1, r, o2 = "----", 0, "----"
        else:
            o1 = origin
            r = replacement_rmsd[n]
            o2 = origins[replacement[n]]
            if o2 is None:
                o2 = "----"
        print(o1, "{:.3f}".format(r), o2, file=f)

np.save(f"{base}-extension.npy", extension_coors)
with open(f"{base}-extension.origin.txt", "w") as f:
    for ind in extension:
        print(origins[ind], file=f)

with open(
    f"output/build-library-{lib}-{motif}-{precision}-singleton-fit.txt", "w"
) as f:
    print(
        "#origin #prim-ind #is-repl #prim-origin #prim-RMSD #ext-ind #ext-origin #ext-RMSD",
        file=f,
    )
    for indnr, ind in enumerate(tqdm(singletons)):
        clus = singleton_closest_primary_cluster[indnr]
        ind_clus = primary_clusters[clus][0]

        rmsd1 = singleton_closest_primary_rmsd[indnr]
        is_replacement = origins[ind] == origins[ind_clus]
        # validate
        prim_struc = primary_coors[clus]
        prim_struc2 = coors[ind_clus]
        if is_replacement:
            prim_struc = replacement_coors[clus]
            prim_struc2 = coors[replacement[clus]]
        _, r = superimpose(singleton_coors[indnr], prim_struc)
        assert np.isclose(r, rmsd1)
        _, r = superimpose(coors[ind], prim_struc)
        assert np.isclose(r, rmsd1)
        _, r = superimpose(coors[ind], prim_struc2)
        assert np.isclose(r, rmsd1)
        # /validate

        rmsd2 = singleton_closest_singleton_rmsd[indnr]
        ind_singleton = singleton_closest_singleton[indnr]

        # validate
        _, r = superimpose(coors[ind], singleton_coors[ind_singleton])
        assert np.isclose(r, rmsd2)
        # /validate

        o1 = origins[ind_clus]
        if is_replacement:
            o1 = origins[replacement[clus]]

        ind_extension0 = singletons[singleton_closest_singleton[indnr]]
        try:
            ind_extension = extension_rev[ind_extension0]
            if rmsd2 > rmsd1:
                raise KeyError
        except KeyError:
            ind_extension = -1
            rmsd2 = "-1"
            o2 = "----"
        else:
            o2 = origins[ind_extension0]
            # validate
            _, r = superimpose(coors[ind], coors[ind_extension0])
            assert np.isclose(r, rmsd2)
            _, r = superimpose(coors[ind], extension_coors[ind_extension])
            assert np.isclose(r, rmsd2)
            # /validate
            rmsd2 = "{:.3f}".format(rmsd2)

        if o1 is None:
            o1 = "----"
        if o2 is None:
            o2 = "----"

        print(
            origins[ind],
            "{:4d}".format(clus + 1),
            int(is_replacement),
            o1,
            "{:.3f}".format(rmsd1),
            "{:4d}".format(ind_extension + 1),
            o2,
            rmsd2,
            file=f,
        )
