import os, glob, random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.vis.drr import drr_lat  # uses your forward projection

NPZ_DIR = r"C:\Users\catal\XRAY_VIEW_SYNTHESIS\runs\runs_final\data\drr_pairs_fixed\npz"
OUT_DIR = r"C:\Users\catal\XRAY_VIEW_SYNTHESIS\runs\runs_final\examples10_out"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_png01(x01: np.ndarray, path: str):
    x = np.clip(x01, 0, 1)
    img = (x * 255.0).astype(np.uint8)
    Image.fromarray(img).save(path)

def backproject_ap_to_volume(ap_zx: np.ndarray, y: int) -> np.ndarray:
    # (Z,X) -> (Z,Y,X)
    return np.repeat(ap_zx[:, None, :], repeats=y, axis=1).astype(np.float32)

class NPZ10(Dataset):
    def __init__(self, npz_paths):
        self.paths = npz_paths

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        d = np.load(p)

        ap = d["ap"].astype(np.float32)   # expected (Z,X)
        lat = d["lat"].astype(np.float32) # expected (Z,Y)

        # enforce float32 and shapes
        if ap.ndim != 2 or lat.ndim != 2:
            raise ValueError(f"bad dims ap={ap.shape} lat={lat.shape} in {p}")

        Z, X = ap.shape
        Z2, Y = lat.shape
        assert Z == Z2, f"Z mismatch ap {ap.shape} lat {lat.shape} in {p}"

        vol_in = backproject_ap_to_volume(ap, y=Y)  # (Z,Y,X)


        return {
            "path": p,
            "ap": ap,
            "lat": lat,
            "vol_in": vol_in,
        }


# ---- Minimal 3D UNet-ish (small, fast) --------------------------------------
class SmallUNet3D(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv3d(1, c, 3, padding=1), nn.ReLU(), nn.Conv3d(c, c, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(nn.Conv3d(c, 2*c, 3, padding=1), nn.ReLU(), nn.Conv3d(2*c, 2*c, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool3d(2)

        self.mid = nn.Sequential(nn.Conv3d(2*c, 4*c, 3, padding=1), nn.ReLU(), nn.Conv3d(4*c, 2*c, 3, padding=1), nn.ReLU())

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec2 = nn.Sequential(nn.Conv3d(4*c, 2*c, 3, padding=1), nn.ReLU(), nn.Conv3d(2*c, c, 3, padding=1), nn.ReLU())
        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec1 = nn.Sequential(nn.Conv3d(2*c, c, 3, padding=1), nn.ReLU(), nn.Conv3d(c, c, 3, padding=1), nn.ReLU())

        self.out = nn.Conv3d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        m = self.mid(p2)

        u2 = self.up2(m)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return torch.sigmoid(self.out(d1))  # [0,1]

# ---- Train on 10 examples (image-space LAT loss; optional CT loss) -----------
def forward_project_lat_torch(ct_zyx: torch.Tensor) -> torch.Tensor:
    # ct_zyx: (B,1,Z,Y,X)
    # lat: sum along X -> (B,1,Z,Y)
    return ct_zyx.sum(dim=4)

def main():
    paths = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
    assert len(paths) >= 10, "not enough npz files"
    random.seed(0)
    pick = paths[:10]  # deterministic; swap to random.sample(paths, 10) if you want
    ds = NPZ10(pick)

    # Build a loader (batch=1 to keep it simple and stable)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    # Infer Y from first sample lat shape
    sample = ds[0]
    Z, X = sample["ap"].shape
    Z2, Y = sample["lat"].shape
    print("Using shapes:", "ap(Z,X)=", (Z,X), "lat(Z,Y)=", (Z2,Y), "vol(Z,Y,X)=", sample["vol_in"].shape)

    model = SmallUNet3D(c=12).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    # quick train loop (keep it short for deadline)
    EPOCHS = 1

    for ep in range(1, EPOCHS+1):
        model.train()
        total = 0.0
        for batch in loader:
            vol_in = batch["vol_in"].numpy()[0]  # (Z,Y,X)
            lat_gt = batch["lat"].numpy()[0]     # (Z,Y)

            x = torch.from_numpy(vol_in)[None,None].to(DEVICE)   # (1,1,Z,Y,X)
            lat_gt_t = torch.from_numpy(lat_gt)[None,None].to(DEVICE)

            ct_pred = model(x)  # (1,1,Z,Y,X)

            # LAT forward projection loss
            lat_pred = forward_project_lat_torch(ct_pred)
            lat_pred = (lat_pred - lat_pred.min()) / (lat_pred.max() - lat_pred.min() + 1e-8)
            loss_lat = F.l1_loss(lat_pred, lat_gt_t)

            loss = loss_lat

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss_lat.detach().cpu().item())

        print(f"epoch {ep:03d} | lat_l1={total/len(loader):.5f}")

    # Export 1 examples
    model.eval()
    for i in range(len(ds)):
        s = ds[i]
        ap = s["ap"]     # (Z,X)
        lat_gt = s["lat"]# (Z,Y)
        vol_in = s["vol_in"]

        x = torch.from_numpy(vol_in)[None,None].to(DEVICE)
        with torch.no_grad():
            ct_pred = model(x)[0,0].detach().cpu().numpy()  # (Z,Y,X)

        lat_pred = drr_lat(ct_pred, invert=True)  # uses your drr_lat (sum axis=2 -> (Z,Y), then normalize)
        # Save
        base = os.path.splitext(os.path.basename(s["path"]))[0]
        save_png01(ap, os.path.join(OUT_DIR, f"{base}__ap.png"))
        save_png01(lat_gt, os.path.join(OUT_DIR, f"{base}__lat_gt.png"))
        save_png01(lat_pred, os.path.join(OUT_DIR, f"{base}__lat_pred.png"))

    torch.save({"model": model.state_dict()}, os.path.join(OUT_DIR, "ap2ct_smallunet3d.pt"))
    print("DONE ->", OUT_DIR)

if __name__ == "__main__":
    main()
