import numpy as np
from mutate import mutate
import itertools


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
            for ext in ("", "-extension"):
                pattern = "library/{lib}-{seq}-{precision}{ext}.npy"
                coorf = pattern.format(lib=lib, seq=seq0, ext=ext, precision=precision)
                coor = np.load(coorf)
                for seq in seqs_map[seq0]:
                    mut_coor = mutate(coor, seq0, seq)
                    outfile = pattern.format(
                        lib=lib, seq=seq, ext=ext, precision=precision
                    )
                    print(coorf, outfile)
                    np.save(outfile, mut_coor)
