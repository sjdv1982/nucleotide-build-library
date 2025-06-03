"""
Analyzes primary dinucleotide pairings in the primary trinucleotide library

Creates:

1/ dinuc-trinuc-pairs/{trinuc_seq}-pairing.txt
For each trinucleotide, all corresponding dinucleotide pairings:
    - Format is ((conf1, conf2), cRMSD) , where conf1 and conf2 are indices STARTING FROM ZERO
    - conf1 or conf2 must fit within 0.5A.
        If there is no such conf, -1 is written for the conf and for the cRMSD
    - The pair (with conf1 and conf2 under 0.5A) with the lowest cRMSD is always written
    - All such pairs with cRMSD<0.5A are written

2/ dinuc-trinuc-pairs/{trinuc_seq}-pairlist.txt
Contains a list of all pairs with INDICES STARTING FROM ONE
- Each pair has conf1 and conf2 fit within 0.5A on at least one trinucleotide
- Each pair has cRMSD < 0.5A

"""

import itertools
import json
import os
import sys

import numpy as np
from nefertiti.functions.parse_mmcif import atomic_dtype
from nefertiti.functions.superimpose import superimpose, superimpose_array
from tqdm import trange

trinuc_seq = sys.argv[1]


def err(msg):
    print(msg, file=sys.stderr)
    exit(1)


bases = ("A", "C")
trinuc_sequences = ["".join(s) for s in itertools.product(bases, repeat=3)]
assert trinuc_seq in trinuc_sequences, (trinuc_seq, trinuc_sequences)
dinuc_sequences = ["".join(s) for s in itertools.product(bases, repeat=2)]

template_pdbs = {}
templates_res_sizes = {}
for seq in trinuc_sequences + dinuc_sequences:
    filename = f"templates/{seq}-template-ppdb.npy"
    template = np.load(filename)
    if template.dtype != atomic_dtype:
        err(f"Template '{filename}' does not contain a parsed PDB")
    template_pdbs[seq] = template
    pos = 0
    res_sizes = []
    for n in range(len(seq)):
        mask = np.nonzero(template["resid"] == n + 1)[0]
        size = len(mask)
        assert size, (seq, n)
        assert mask[0] == pos and mask[-1] == size + pos - 1, (seq, n, mask)
        res_sizes.append(size)
        pos += size
    templates_res_sizes[seq] = res_sizes

dinuc_lib = {}  # read only 0.5A primary library
for seq in dinuc_sequences:
    filename = f"library/dinuc-{seq}-0.5.npy"
    lib = np.load(filename)
    dinuc_lib[seq] = lib

seq = trinuc_seq
res_sizes = templates_res_sizes[seq]
diseq1, diseq2 = seq[:2], seq[1:]
assert templates_res_sizes[diseq1] == res_sizes[:2]
assert templates_res_sizes[diseq2] == res_sizes[1:]

filename = f"library/trinuc-{seq}-0.5.npy"  # read only 0.5A primary library
lib = np.load(filename)
prelib = lib[:, : sum(res_sizes[:2])]
postlib = lib[:, -sum(res_sizes[1:]) :]
dilib1 = dinuc_lib[diseq1]
dilib2 = dinuc_lib[diseq2]
middle_size = res_sizes[1]
middle_dilib1 = dilib1[:, -middle_size:]
middle_dilib2 = dilib2[:, :middle_size]

pairs = []
for confnr in trange(0, len(lib)):
    candidates = []
    for trilib, dilib in ((prelib, dilib1), (postlib, dilib2)):
        rmsds = superimpose_array(dilib, trilib[confnr])[1]
        cands = np.where(rmsds < 0.5)[0].tolist()
        cands.sort(key=lambda c: rmsds[c])
        candidates.append(cands)
    cand1, cand2 = candidates
    if len(cand1) == 0 and len(cand2) == 0:
        pairs.append([((-1, -1), -1)])
    elif len(cand1) == 0:
        pairs.append([((-1, cand2[0]), -1)])
    elif len(cand2) == 0:
        pairs.append([((cand1[0], -1), -1)])
    if len(cand1) == 0 or len(cand2) == 0:
        continue
    min_crmsd = None
    min_pair = None
    curr_pairs = []
    for cnr1, c1 in enumerate(cand1):
        coor1 = middle_dilib1[c1]
        for cnr2, c2 in enumerate(cand2):
            coor2 = middle_dilib2[c2]
            crmsd = superimpose(coor1, coor2)[1]
            if min_crmsd is None or crmsd < min_crmsd:
                if min_pair is not None and min_crmsd < 0.5:
                    cnr1, cnr2 = min_pair
                    curr_pairs.append(((cand1[cnr1], cand2[cnr2]), min_crmsd))
                min_crmsd = crmsd
                min_pair = cnr1, cnr2
            elif crmsd < 0.5:
                curr_pairs.append(((cand1[cnr1], cand2[cnr2]), crmsd))
    cnr1, cnr2 = min_pair
    curr_pairs.append(((cand1[cnr1], cand2[cnr2]), min_crmsd))
    pairs.append(curr_pairs)

assert len(pairs) == len(lib)


pair_set = set()
n_fitting_pairs = 0
n_good_crmsd = 0
for p in pairs:
    if len(p) == 1 and p[0][1] == -1:
        continue
    n_fitting_pairs += 1
    ok = 0
    for pp, crmsd in p:
        if crmsd < 0.5:
            ok = 1
            pair_set.add(pp)
    if ok:
        n_good_crmsd += 1
print(
    "Primary trinucleotides with a pair of <0.5A dinucleotides: {:.1f} %".format(
        n_fitting_pairs / len(lib) * 100
    )
)
print("... and an cRMSD <0.5A: {:.1f} %".format(n_good_crmsd / len(lib) * 100))

os.makedirs("dinuc-trinuc-pairs", exist_ok=True)
with open(f"dinuc-trinuc-pairs/{trinuc_seq}-pairing.txt", "w") as f:
    json.dump(pairs, f)
pair_list = list(pair_set)
pair_list.sort(key=lambda p: p[1])
pair_list.sort(key=lambda p: p[0])
with open(f"dinuc-trinuc-pairs/{trinuc_seq}-pairlist.txt", "w") as f:
    for p in pair_list:
        print(p[0] + 1, p[1] + 1, file=f)
