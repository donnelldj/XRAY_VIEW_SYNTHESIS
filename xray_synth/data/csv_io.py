from __future__ import annotations

import csv
from typing import List


def load_npz_paths_from_csv(csv_path: str) -> List[str]:
    """
    Robust CSV reader:
    - Accepts either a single-column CSV of paths
    - OR a header row containing 'npz_path'
    """
    paths: List[str] = []

    with open(str(csv_path), "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return paths

    header = [c.strip() for c in rows[0]]
    if any(h.lower() == "npz_path" for h in header):
        idx = [i for i, h in enumerate(header) if h.lower() == "npz_path"][0]
        for r in rows[1:]:
            if not r:
                continue
            v = r[idx].strip()
            if v:
                paths.append(v)
        return paths

    # No header: assume first column is a path
    for r in rows:
        if not r:
            continue
        p = r[0].strip()
        if p and p.endswith(".npz"):
            paths.append(p)

    return paths
