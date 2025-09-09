import os
from typing import Literal, Optional

currdir = os.path.abspath(os.path.dirname(__file__))


def read_origins(origin_file: str) -> list[list[str]]:
    origins = []
    for l in open(origin_file).readlines():
        ori = []
        for item in l.split("/"):
            if not item.strip():
                continue
            fields = item.split()
            assert len(fields) == 3
            code = fields[0][:4]
            ori.append(code)
        origins.append(ori)
    return origins


def get_database_mapping(database: Literal["pfam", "rfam"]) -> dict[str, set[str]]:
    mappingfile = os.path.join(currdir, f"pdb2{database}.txt")
    mapping = {}
    with open(mappingfile) as f:
        for l in f:
            pdb, fam = l.split()

            # mapping at the whole PDB code level, not the chain level.
            # In theory, Rfam could be at the chain level, but the chains
            #  in the original file don't make any sense to me.
            pdb = pdb[:4]

            if pdb not in mapping:
                mapping[pdb] = set()
            mapping[pdb].add(fam)
    return mapping


def context_mask_fam(
    origins: list[list[str]],
    database: Literal["pfam", "rfam"],
    fam_id: str,
    database_mapping: Optional[dict] = None,
) -> tuple[list[bool], list[bool]]:
    """Returns two masks, "any" and "all".
    A mask contains for each item in origin,
      if any/all of its PDB codes are mapped to fam_id in the 'pdb2<database> mapping file.

    Rfam is mapped at the level of PDB chains.
    Pfam is mapped at the level of whole PDB codes.
    """
    if database_mapping is None:
        database_mapping = get_database_mapping(database)

    any_mask, all_mask = [], []
    for pdb_codes in origins:
        curr_any = False
        curr_all = True
        for pdb in pdb_codes:

            # see get_database_mapping
            pdb = pdb[:4]

            if fam_id in database_mapping.get(pdb, []):
                curr_any = True
            else:
                curr_all = False
        if not curr_any:
            curr_all = False
        any_mask.append(curr_any)
        all_mask.append(curr_all)
    return any_mask, all_mask
