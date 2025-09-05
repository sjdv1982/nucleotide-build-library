import sys
import numpy as np
from numpy.linalg import svd, det

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda arg: arg

lib = sys.argv[1]
assert lib in ("dinuc", "trinuc")
motif = sys.argv[2]
precision = sys.argv[3]


def err(*args):
    print(*args, file=sys.stderr)
    exit(1)


def superimpose_from_covar(covar, residuals1, residuals2):
    v, s, wt = svd(covar)
    reflect = det(v) * det(wt)
    s[-1] *= reflect
    sd = (residuals1 + residuals2) - 2 * s.sum()
    return sd


origin_file = f"../nucleotide-fragments/{lib}/origin/{motif}.txt"
coorfile = f"../nucleotide-fragments/{lib}/{motif}.npy"
close_pair_file = f"result/lib-{lib}-{motif}-{precision}.close-pairs.txt"

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

close_pairs = []
for l in open(close_pair_file).readlines():
    if not len(l.strip()):
        continue
    ll = l.split()
    assert len(ll) == 2
    p1, p2 = int(ll[0]) - 1, int(ll[1]) - 1
    close_pairs.append((p1, p2))

sd_precision = (float(precision) ** 2) * coors.shape[1]
for p1, p2 in tqdm(close_pairs):
    coor1, coor2 = coors[p1], coors[p2]
    residuals1, residuals2 = coors_residuals[p1], coors_residuals[p2]
    covar = np.einsum("jk,jl->kl", coor1, coor2)
    sd = superimpose_from_covar(covar, residuals1, residuals2)
    if sd > sd_precision:
        rmsd = np.sqrt(sd / coors.shape[1])
        print("False positive", p1, p2, "{:3f}".format(rmsd))
