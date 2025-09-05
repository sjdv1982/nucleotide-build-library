import sys
import numpy as np
from numpy.linalg import svd, det

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


closest_fit_file = f"../output/closest-fit/{lib}-{motif}.txt"

closest_fit = []
for l in open(closest_fit_file).readlines():
    ll = l.split()
    closest_conf = int(ll[0])
    closest_rmsd = float(ll[1])
    closest_fit.append((closest_conf, closest_rmsd))

nconf = len(closest_fit)

singletons = []
for l in open(singleton_file).readlines():
    if not len(l.strip()):
        continue
    ll = l.split()
    ind = int(ll[0]) - 1
    assert ind >= 0 and ind < nconf
    singletons.append(ind)

precision = float(precision)
for ind in singletons:
    closest_conf, rmsd = closest_fit[ind]
    is_singleton = rmsd > precision
    if is_singleton != true_false:
        if is_singleton:
            if rmsd - 0.001 >= precision:
                print("False negative", ind, "%.3f" % rmsd)
        else:
            if rmsd + 0.001 <= precision:
                print("False positive", ind, "%.3f" % rmsd, closest_conf)
