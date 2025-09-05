import sys
import numpy as np
from numpy.linalg import svd, det

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda arg: arg

singleton_file = sys.argv[1]
lib = sys.argv[2]
assert lib in ("dinuc", "trinuc")
motif = sys.argv[3]
precision = sys.argv[4]
true_false = sys.argv[5]
assert true_false in (("--true", "--false"))
true_false = true_false == "--true"


def err(*args):
    print(*args, file=sys.stderr)
    exit(1)


def superimpose_from_covar(covar, residuals1, residuals2):
    v, s, wt = svd(covar)
    reflect = det(v) * det(wt)
    s[:, -1] *= reflect
    sd = (residuals1 + residuals2) - 2 * s.sum(axis=1)
    sd = np.maximum(sd, 0)
    return sd


origin_file = f"../nucleotide-fragments/{lib}/origin/{motif}.txt"
coorfile = f"../nucleotide-fragments/{lib}/{motif}.npy"

coors = np.load(coorfile)
assert coors.ndim == 3 and coors.shape[-1] == 3, coors.shape
coors -= coors.mean(axis=1)[:, None, :]
coors_residuals = np.einsum("ijk,ijk->i", coors, coors)

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

singletons = []
for l in open(singleton_file).readlines():
    if not len(l.strip()):
        continue
    ll = l.split()
    assert len(ll) == 1
    ind = int(ll[0]) - 1
    assert ind >= 0 and ind < nconf
    singletons.append(ind)

precision = float(precision)
sd_precision = (precision**2) * coors.shape[1]
for ind in tqdm(singletons):
    ori = origins[ind]
    assert ori is not None
    curr_coor = coors[ind]
    curr_residuals = coors_residuals[ind]
    mask = np.array([(ori2 != ori) for ori2 in origins], bool)
    if not mask.sum():
        continue
    other_coors = coors[mask]
    other_residuals = coors_residuals[mask]
    covar = np.einsum("ijk,jl->ikl", other_coors, curr_coor)
    sd = superimpose_from_covar(covar, curr_residuals, other_residuals)
    is_singleton = sd.min() > sd_precision
    if is_singleton != true_false:
        rmsd = np.sqrt(sd.min() / coors.shape[1])
        if is_singleton:
            if rmsd - 0.001 >= precision:
                print("False negative", ind, "%.3f" % rmsd)
        else:
            if rmsd + 0.001 <= precision:
                print(
                    "False positive",
                    ind,
                    "%.3f" % rmsd,
                    np.where(mask)[0][sd.argmin()] + 1,
                )
