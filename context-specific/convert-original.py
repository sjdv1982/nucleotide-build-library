pfams = []
with open("original/pdb_chain_pfam.lst") as f:
    for l in f.readlines()[2:]:
        ll = l.split()
        pdb = ll[0].lower() + ll[1]
        pfam = ll[3]
        pfams.append((pdb, pfam))
with open("pdb2pfam.txt", "w") as f:
    for pdb, pfam in pfams:
        print(pdb, pfam, file=f)


rfams = []
with open("original/Rfam-pdb.txt") as f:
    for l in f.readlines()[2:]:
        ll = l.split()
        rfam = ll[0]
        pdb = ll[1].lower() + ll[2]
        rfams.append((pdb, rfam))
with open("pdb2rfam.txt", "w") as f:
    for pdb, rfam in rfams:
        print(pdb, rfam, file=f)
