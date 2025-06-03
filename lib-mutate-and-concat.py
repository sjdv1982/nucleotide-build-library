"""
- Mutates the library from A/C to A/C/G/U
- Concatenates the
Output dir: library-concat/
"""

import os
import numpy as np
from mutate import mutate
import itertools

os.makedirs("library-concat", exist_ok=True)

bases = ("A", "C", "G", "U")
dinuc_sequences = ["".join(s) for s in itertools.product(bases, repeat=2)]
trinuc_sequences = ["".join(s) for s in itertools.product(bases, repeat=3)]

mut_coor = {}
mut_ori = {}

for lib, seqs in (("trinuc", trinuc_sequences), ("dinuc", dinuc_sequences)):
    seqs_map = {}
    for seq in seqs:
        seq0 = seq.replace("G", "A").replace("U", "C")
        if seq0 == seq:
            continue
        if seq0 not in seqs_map:
            seqs_map[seq0] = []
        seqs_map[seq0].append(seq)
    for seq0 in seqs_map:
        for precision in (0.5, 1.0):
            exts = ("", "-extension")
            origins = []
            for ext in exts:
                pattern = "library/{lib}-{seq}-{precision}{ext}.origin.txt"
                with open(pattern) as f:
                    origins.append(f.read().rstrip("\n"))
            origins = "\n".join(origins)

            pattern = "library/{lib}-{seq}-{precision}{ext}.npy"
            coors = []
            for ext in exts:
                coorf = pattern.format(lib=lib, seq=seq0, ext=ext, precision=precision)
                coor = np.load(coorf)
                coors.append(coor)

            pattern = "library-concat/{lib}-{seq}-{precision}{tail}"
            coor = np.concatenate(coors)
            for seq in seqs_map[seq0]:
                mut_coor = mutate(coor, seq0, seq)
                outfile = pattern.format(
                    lib=lib, seq=seq, precision=precision, tail=".npy"
                )
                print(outfile)
                np.save(outfile, mut_coor)
