def write_clustering(clusterfile, clustering):
    if hasattr(clusterfile, "write"):
        f = clusterfile
    else:
        f = open(clusterfile, "w")
    with f:
        for clust_nr, cluster in enumerate(clustering):
            print(f"Cluster {clust_nr+1} ->", end=" ", file=f)
            for ind in cluster:
                print(ind + 1, end=" ", file=f)
            print("", file=f)


def read_clustering(clusterfile):
    clustering = []
    if hasattr(clusterfile, "readlines"):
        f = clusterfile
    else:
        f = open(clusterfile)
    with f:
        for lnr, l in enumerate(f.readlines()):
            ll = l.split()
            assert ll[0] == "Cluster"
            assert ll[1] == str(lnr + 1)
            assert ll[2] == "->"
            c = [int(lll) for lll in ll[3:]]
            clustering.append(c)
    return clustering
