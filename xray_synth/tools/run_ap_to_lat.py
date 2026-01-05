from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root (contains xray_synth/)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xray_synth.cli.ap_to_lat import main  # noqa: E402

if __name__ == "__main__":
    main()
