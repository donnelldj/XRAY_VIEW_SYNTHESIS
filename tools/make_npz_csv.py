import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", required=True)
    p.add_argument("--npz_dir", required=True, help="Folder containing .npz files")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--id_col", default="case_id", help="Column in input csv to match npz names (optional)")
    args = p.parse_args()

    in_csv = Path(args.in_csv)
    npz_dir = Path(args.npz_dir)
    out_csv = Path(args.out_csv)

    df = pd.read_csv(in_csv)

    # list all npz files
    npzs = sorted(npz_dir.glob("*.npz"))
    if not npzs:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")

    # If the input CSV has a usable id column AND npz files follow that naming,
    # we build exact paths. Otherwise we just enumerate the npzs (common in prepared datasets).
    if args.id_col in df.columns:
        # map: basename(without ext) -> full path
        m = {p.stem: str(p.as_posix()) for p in npzs}
        npz_paths = []
        missing = 0
        for cid in df[args.id_col].astype(str).tolist():
            if cid in m:
                npz_paths.append(m[cid])
            else:
                missing += 1
                npz_paths.append("")
        df["npz_path"] = npz_paths
        if missing > 0:
            print(f"[WARN] {missing} rows had no matching npz by {args.id_col}.")
            print("       If your npz files are not named by case_id, use the fallback mode below.")
    else:
        # fallback: just take first N npz files
        N = min(len(df), len(npzs))
        df = df.iloc[:N].copy()
        df["npz_path"] = [str(p.as_posix()) for p in npzs[:N]]
        print(f"[OK] Fallback mode: paired first {N} rows with first {N} npz files.")

    # drop empty npz_path rows
    if (df["npz_path"] == "").any():
        df = df[df["npz_path"] != ""].copy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote: {out_csv} (rows={len(df)})")
    print(df[["npz_path"]].head(3).to_string(index=False))

if __name__ == "__main__":
    main()
