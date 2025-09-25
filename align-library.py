import sys
import numpy as np

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
sublib = sys.argv[4]
assert sublib in ("primary", "extension"), sublib

base = f"{lib}-{motif}-{precision}"
if sublib == "primary":
    coorfile = f"output/{base}.npy"
    outfile = f"library/{base}.npy"
else:
    coorfile = f"output/{base}-{sublib}.npy"
    outfile = f"library/{base}-{sublib}.npy"

coors = np.load(coorfile)
assert coors.ndim == 3 and coors.shape[-1] == 3, coors.shape
coors -= coors.mean(axis=1)[:, None, :]

####################################################################
# Align each conformer on its principal components
####################################################################

def get_structure_tensor(conf):
    curr_tensor = np.eye(3)
    niter = 0
    while 1:
        conft = conf.dot(curr_tensor)
        
        conf0 = conft - conft.mean(axis=0)
        v, s, wt = np.linalg.svd(conf0) 
        scalevec = s/np.sqrt(len(conf))
        tensor = wt.T
        if np.linalg.det(tensor) < 0:
            tensor[2] *= -1
        assert np.linalg.det(tensor) > 0.999

        curr_tensor = curr_tensor.dot(tensor)
        assert np.linalg.det(curr_tensor) > 0.999

        if np.abs(tensor - np.eye(3)).sum() < 0.01:
            break
        niter += 1
        if niter > 1000:
            if (np.abs(tensor) - np.eye(3)).sum() < 0.01:
                break

        if niter > 10000:
            print(niter, np.abs(tensor - np.eye(3)).sum(), tensor, curr_tensor)
        if niter > 10010:
            exit(1)

    return curr_tensor, scalevec

for confnr, conf in enumerate(tqdm(coors)):
    tensor, _ = get_structure_tensor(conf)
    conf = conf.dot(tensor)
    coors[confnr] = conf

np.save(outfile, coors)