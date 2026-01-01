# import os, glob, argparse
# import numpy as np
# from pathlib import Path
# from PIL import Image

# def normalize01(x: np.ndarray) -> np.ndarray:
#     x = x.astype(np.float32)
#     x = x - x.min()
#     x = x / (x.max() + 1e-8)
#     return x

# def drr_from_ct(ct_zyx: np.ndarray, view: str) -> np.ndarray:
#     """
#     Simple DRR: sum along an axis of ct_zyx (Z,Y,X).
#     - 'ap'  : integrate along Z -> (Y,X)
#     - 'lat' : integrate along X -> (Z,Y)
#     Then resize/crop to (256,256) downstream if needed.
#     """
#     ct = ct_zyx.astype(np.float32)

#     if view == "ap":
#         img = ct.sum(axis=0)          # (Y, X)
#     elif view == "lat":
#         img = ct.sum(axis=2)          # (Z, Y)
#     else:
#         raise ValueError("view must be 'ap' or 'lat'")

#     img = normalize01(img)
#     return img

# def to_uint8(img01: np.ndarray) -> np.ndarray:
#     img = np.clip(img01, 0.0, 1.0)
#     return (img * 255.0).round().astype(np.uint8)

# def save_png(img01: np.ndarray, out_path: Path):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     Image.fromarray(to_uint8(img01)).save(out_path)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--npz_dir", default=r"data/drp_pairs/npz")
#     ap.add_argument("--out_dir", default=r"data/drr_pairs")
#     ap.add_argument("--n", type=int, default=30, help="how many cases to export")
#     ap.add_argument("--seed", type=int, default=0)
#     args = ap.parse_args()

#     np.random.seed(args.seed)

#     files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
#     if not files:
#         raise SystemExit(f"No .npz files found in {args.npz_dir}")

#     # choose subset
#     if args.n < len(files):
#         idx = np.random.choice(len(files), size=args.n, replace=False)
#         files = [files[i] for i in sorted(idx)]

#     out_dir = Path(args.out_dir)
#     (out_dir / "npz").mkdir(parents=True, exist_ok=True)
#     (out_dir / "png").mkdir(parents=True, exist_ok=True)

#     exported = 0

#     for p in files:
#         d = np.load(p)
#         if "ct_zyx" not in d.files:
#             print("Skipping (no ct_zyx):", p)
#             continue

#         ct = d["ct_zyx"]  # (Z,Y,X), already 0..1 per your logs

#         ap_img = drr_from_ct(ct, "ap")       # (Y,X)
#         lat_img = drr_from_ct(ct, "lat")     # (Z,Y)

#         # Make both 256x256 for consistency by center-cropping/padding
#         # AP should already be (256,256) if ct is (96,256,256)
#         # LAT will be (96,256), so we pad to (256,256) by padding along first dim.
#         if ap_img.shape != (256, 256):
#             # center crop/pad to 256x256
#             ap_img = ap_img[:256, :256]

#         if lat_img.shape[1] != 256:
#             lat_img = lat_img[:, :256]

#         # pad lat from (Z,Y)=(96,256) -> (256,256) by padding zeros on Z
#         if lat_img.shape[0] < 256:
#             pad = 256 - lat_img.shape[0]
#             lat_img = np.pad(lat_img, ((pad//2, pad - pad//2), (0, 0)), mode="constant", constant_values=0.0)
#         elif lat_img.shape[0] > 256:
#             start = (lat_img.shape[0] - 256)//2
#             lat_img = lat_img[start:start+256, :]

#         case_id = Path(p).stem

#         # Save pair as npz
#         out_npz = out_dir / "npz" / f"{case_id}.npz"
#         np.savez_compressed(out_npz, ap=ap_img.astype(np.float16), lat=lat_img.astype(np.float16), src=np.array([str(p)]))

#         # Save quicklook PNGs
#         out_png_dir = out_dir / "png" / case_id
#         save_png(ap_img, out_png_dir / "ap.png")
#         save_png(lat_img, out_png_dir / "lat_gt.png")

#         exported += 1

#     print(f"Exported {exported} DRR pairs to: {out_dir}")

# if __name__ == "__main__":
#     main()
