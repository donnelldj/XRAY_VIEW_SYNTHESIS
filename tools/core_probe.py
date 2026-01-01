import numpy as np
import torch
from pathlib import Path
from tools.train_ap2lat import DRRPairDataset, SmallUNet2D

data_dir = Path(r"runs\runs_final\data\drr_pairs_fixed")
ckpt_path = r"runs\runs_final\ap2lat_baseline_e40_b8\best.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ds = DRRPairDataset(data_dir / "val.csv")
ckpt = torch.load(ckpt_path, map_location=device)
m = SmallUNet2D().to(device)
m.load_state_dict(ckpt["model"])
m.eval()

def corr(a,b):
    a=a.reshape(-1); b=b.reshape(-1)
    return float(np.corrcoef(a,b)[0,1])

def mm(x):
    x=x.astype(np.float32)
    return (x-x.min())/(x.max()-x.min()+1e-8)

for idx in [0, len(ds)//3, 2*len(ds)//3, len(ds)-1]:
    ap, lat = ds[idx]
    ap_np  = ap.numpy()[0]
    lat_np = lat.numpy()[0]
    with torch.no_grad():
        pred = m(ap.to(device)[None,...]).cpu().numpy()[0,0]

    # compare in normalized view-space
    ap_n, lat_n, pred_n = mm(ap_np), mm(lat_np), mm(pred)

    print(f"\nidx={idx}")
    print("corr(pred, ap) =", corr(pred_n, ap_n))
    print("corr(pred, lat)=", corr(pred_n, lat_n))
    print("mae(pred, lat) =", float(np.mean(np.abs(pred_n-lat_n))))