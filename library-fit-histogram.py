import sys
import numpy as np
import pandas as pd
import itertools
import seaborn as sns
from matplotlib import pyplot as plt

lib = sys.argv[1]
assert lib in ("dinuc", "trinuc")


fig, ax = plt.subplots()

data = pd.DataFrame()
bins = [0] + np.arange(0.2, 3.1, 0.05).tolist()
data.index = bins


def dat(d1, d2):
    d = d1
    if d2 is not None:
        d = np.concatenate([d1, d2])
    result = []
    for bin in bins:
        fail = (d >= bin).sum() / len(d)
        result.append(fail)
    return result


n_nuc = 2 if lib == "dinuc" else 3
motifs = ["".join(c) for c in itertools.product("AC", repeat=n_nuc)]

baseline = []
for motif in motifs:
    motif_baseline = np.loadtxt(f"lib-{lib}-{motif}-closest-fit.txt")[:, 1]
    baseline.append(motif_baseline)
baseline = np.concatenate(baseline)
baseline = dat(baseline, None)
data["baseline"] = baseline

precision = "1.0"
ic = np.loadtxt(f"library-fit-{lib}-{precision}-intra-cluster.txt")
sing1 = np.loadtxt(f"library-fit-{lib}-{precision}-singleton-primary.txt")
sing2 = np.loadtxt(f"library-fit-{lib}-{precision}-singleton-with-extension.txt")

data["1A"] = dat(ic, sing1)
data["1A with extension"] = dat(ic, sing2)

precision = "0.5"
ic = np.loadtxt(f"library-fit-{lib}-{precision}-intra-cluster.txt")
sing1 = np.loadtxt(f"library-fit-{lib}-{precision}-singleton-primary.txt")
sing2 = np.loadtxt(f"library-fit-{lib}-{precision}-singleton-with-extension.txt")
maximum = np.max(np.concatenate([ic, sing1, sing2]))


data["0.5A"] = dat(ic, sing1)
data["0.5A with extension"] = dat(ic, sing2)

sns.lineplot(data, ax=ax)

ax.set_ybound(0, 1)
fig.savefig(f"library-fit-{lib}-plot.png")

ax.set_yscale("log")
ax.set_ybound(0.00001, 1)
fig.savefig(f"library-fit-{lib}-plot-log.png")
