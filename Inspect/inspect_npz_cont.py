# import glob, numpy as np, torch
# import SimpleITK as sitk
# from src.projection_simple import forward_project_lat_from_ct

# p = glob.glob("data/drp_pairs_debug/npz/*.npz")[0]
# d = np.load(p, allow_pickle=True)

# src = d["src"]
# src = src.item() if hasattr(src, "item") else str(src)

# print("NPZ:", p)
# print("CT src:", src)

# img = sitk.ReadImage(str(src))
# ct_zyx = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X)
# Z = ct_zyx.shape[0]

# # center-crop to 96 slices in Z
# z0 = max(0, (Z - 96)//2)
# ct_zyx = ct_zyx[z0:z0+96]

# # min-max normalize (matches your ap/lat range convention)
# ct_zyx = (ct_zyx - ct_zyx.min()) / (ct_zyx.max() - ct_zyx.min() + 1e-8)

# ct_t = torch.from_numpy(ct_zyx)[None,None]  # (1,1,96,Y,X)
# lat_from_ct = forward_project_lat_from_ct(ct_t).detach().cpu().numpy()[0,0]
# lat_gt = d["lat"]

# print("ct_zyx:", ct_zyx.shape, "lat_from_ct:", lat_from_ct.shape, "lat_gt:", lat_gt.shape)
# print("MSE:", float(np.mean((lat_from_ct - lat_gt)**2)))
# print("lat_from_ct min/max:", float(lat_from_ct.min()), float(lat_from_ct.max()))
# print("lat_gt      min/max:", float(lat_gt.min()), float(lat_gt.max()))
