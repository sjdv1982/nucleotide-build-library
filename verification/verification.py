txt0 = """The {libname} library consists of {nfrag} non-redundant fragments. 
Of these, {trivial_pct:.1f} % are trivially complete: they come from 0.2 A clusters originating
from multiple PDBs.

Here, we will focus on the remaining non-trivial fragments instead.
"""
txt1 = """
Of the non-trivial fragments, {single_pct:.1f} % are alone in their cluster.
Another {homo_heart_pct:.2f} % are the hearts of clusters that come from a single PDB.
Together, these two categories are the certain singletons. 
"""

txt2 = """
Of the non-trivial fragments, {member_dif_ori_pct:.1f} % come from a different PDB than the heart of their cluster.
Another {hetero_heart_pct:.2f} % are the hearts of clusters that come from multiple PDBs.
Together, these two categories are the certain non-singletons. 
"""

txt3 = """
All certain fragments together make up {certain_pct:.2f} % of the non-trivial fragments.
Verification showed that {error_certain} certain fragments were assigned in error.
"""

txt4 = """
The remaining {putative_pct:.2f} % of the fragments come from the same PDB as their cluster heart, 
but are not a cluster heart themselves. 
(Note that a non-heart fragment can belong to multiple clusters; 
to be precise, all those cluster hearts come from the same PDB as the fragment).
If all members of the cluster (or all clusters) come from that PDB too, the fragment is designated
as a putative singleton; otherwise, as a putative non-singleton. 

Among the putative fragments, {error_putative} ({putative_relative_error_pct:.1f} %) are assigned incorrectly. 
However, since putative fragments are so rare, these errors are only {putative_absolute_error_pct:.3f} % of all non-trivial fragments.

"""


def run(lib, libname):
    result = f"# {libname[0].upper() + libname[1:]} library analysis\n\n"
    assert lib in ("dinuc", "trinuc"), lib
    motifs = (
        ["AA", "AC", "CA", "CC"]
        if lib == "dinuc"
        else ["AAA", "AAC", "ACA", "ACC", "CAA", "CAC", "CCA", "CCC"]
    )
    precisions = ["0.5", "1.0"]

    classif = {}
    for motif in motifs:
        for precision in precisions:
            classifile = f"result/classify-fragments-{lib}-{motif}-{precision}.out"
            with open(classifile) as f:
                for l in f:
                    l = l.strip()
                    if not l:
                        continue
                    if l[0] == "#":
                        ll = l[1:].split()
                        field = int(ll[0])
                        value = int(ll[-1])
                        classif[motif, precision, field] = value
    nfrag = 0
    ntrivial = 0
    p = precisions[0]
    for motif in motifs:
        nfrag += classif[motif, p, 2]
        ntrivial += classif[motif, p, 3]
    trivial_pct = ntrivial / nfrag * 100
    tot = nfrag - ntrivial
    result += txt0.format(**locals())
    for precision in precisions:
        result += f"\n## {precision}A precision analysis\n"

        single = 0
        homo_heart = 0
        error_singletons = 0
        for motif in motifs:
            single += classif[motif, precision, 6]
            homo_heart += classif[motif, precision, 7]
            check_singleton_file = (
                f"result/check-{lib}-{motif}-{precision}.singletons.out"
            )
            with open(check_singleton_file) as f:
                error_singletons += len(f.readlines())
        single_pct = single / tot * 100
        homo_heart_pct = homo_heart / tot * 100
        result += txt1.format(**locals())

        hetero_heart = 0
        member_dif_ori = 0
        error_non_singletons = 0
        for motif in motifs:
            hetero_heart += classif[motif, precision, 8]
            member_dif_ori += classif[motif, precision, 10]
            check_non_singleton_file = (
                f"result/check-{lib}-{motif}-{precision}.close-pairs.out"
            )
            with open(check_non_singleton_file) as f:
                error_non_singletons += len(f.readlines())
        member_dif_ori_pct = member_dif_ori / tot * 100
        hetero_heart_pct = hetero_heart / tot * 100

        result += txt2.format(**locals())

        certain = single + member_dif_ori + homo_heart + hetero_heart
        certain_pct = certain / tot * 100
        error_certain = error_non_singletons + error_singletons
        result += txt3.format(**locals())

        putative = tot - certain
        putative_pct = putative / tot * 100

        error_putative = 0
        for motif in motifs:
            hetero_heart += classif[motif, precision, 8]
            member_dif_ori += classif[motif, precision, 10]
            putative_singleton_file = (
                f"result/check-{lib}-{motif}-{precision}.putative_singletons.out"
            )
            with open(putative_singleton_file) as f:
                error_putative += len(f.readlines())
            putative_non_singleton_file = (
                f"result/check-{lib}-{motif}-{precision}.putative_non_singletons.out"
            )
            with open(putative_non_singleton_file) as f:
                error_putative += len(f.readlines())

        putative_relative_error_pct = error_putative / putative * 100
        putative_absolute_error_pct = error_putative / tot * 100

        result += txt4.format(**locals())

    return result


result = run("dinuc", "dinucleotide")
print(result)
result = run("trinuc", "trinucleotide")
print(result)

raise Exception("TODO: notebook")
