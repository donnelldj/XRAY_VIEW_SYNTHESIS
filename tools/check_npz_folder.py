import os, glob
import numpy as np

NPZ_DIR = r"C:\Users\catal\XRAY_VIEW_SYNTHESIS\runs\runs_final\data\drr_pairs_fixed\npz"

paths = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
print("npz count:", len(paths))
p = paths[0]
d = np.load(p)
print("sample:", p)
print("keys:", d.files)
for k in d.files:
    v = d[k]
    if hasattr(v, "shape"):
        print(f"{k:12s} shape={v.shape} dtype={v.dtype}")
