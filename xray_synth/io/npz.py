from __future__ import annotations

from pathlib import Path
from typing import Union


def resolve_path(project_root: Path, p: str) -> Path:
    """
    Resolve file paths robustly across:
    - Windows CSV entries using backslashes
    - absolute paths
    - repo-relative paths
    """
    p2 = p.replace("\\", "/")
    cand = Path(p2)
    if cand.exists():
        return cand

    cand2 = project_root / p2
    if cand2.exists():
        return cand2

    return cand  # best-effort (caller may raise if missing)
