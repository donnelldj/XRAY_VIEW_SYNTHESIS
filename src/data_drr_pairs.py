import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class DRRPairsDataset(Dataset):
    """
    Loads the generated DRR pairs (ap, lat_gt) and also pulls ct_zyx from the original source npz.
    Each DRR npz includes 'src' which points to the original data/drp_pairs/npz file.
    """
    def __init__(self, drr_npz_dir=r"data/drr_pairs/npz"):
        self.files = sorted(glob.glob(drr_npz_dir + "/*.npz"))
        if len(self.files) == 0:
            raise RuntimeError(f"No DRR pair files found in {drr_npz_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        d = np.load(p)

        ap = d["ap"].astype(np.float32)       # (256,256)
        lat = d["lat"].astype(np.float32)     # (256,256)

        # Original CT source path
        src = str(d["src"][0])
        dsrc = np.load(src)
        ct = dsrc["ct_zyx"].astype(np.float32)  # (96,256,256)

        # Back-project AP into volume by repeating along Z
        # BP shape: (96,256,256)
        bp = np.repeat(ap[None, :, :], ct.shape[0], axis=0)

        # torch tensors
        # 3D conv expects (C,D,H,W). We'll use C=1.
        bp_t = torch.from_numpy(bp)[None, ...]   # (1,96,256,256)
        ct_t = torch.from_numpy(ct)[None, ...]   # (1,96,256,256)

        ap_t = torch.from_numpy(ap)[None, ...]   # (1,256,256)
        lat_t = torch.from_numpy(lat)[None, ...] # (1,256,256)

        return {
            "bp": bp_t,
            "ct": ct_t,
            "ap": ap_t,
            "lat": lat_t,
            "pair_path": p,
            "src_path": src,
        }
