# scripts/make_html_report.py
#
# Build a simple HTML report:
# - Summary metrics (PSNR / SSIM)
# - Table of AP / Lat_pred / Lat_gt triplets from examples_dir

import os
import json
from pathlib import Path
from glob import glob


def find_triplets(examples_dir: Path):
    triplets = []
    pattern = str(examples_dir / "*_ap.png")
    for ap_path in sorted(glob(pattern)):
        ap_path = Path(ap_path)
        stem = ap_path.stem  # e.g., "000_ap"
        root = stem[:-3]     # "000_"
        if stem.endswith("_ap"):
            root = stem[:-3]  # "000"
        else:
            # fallback if naming changes
            root = stem.split("_ap")[0]

        pred_path = ap_path.with_name(root + "_lat_pred.png")
        gt_path = ap_path.with_name(root + "_lat_gt.png")

        if pred_path.exists() and gt_path.exists():
            triplets.append((ap_path, pred_path, gt_path))

    return triplets


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--examples_dir", required=True)
    ap.add_argument("--metrics_json", required=True)
    ap.add_argument("--out_html", required=True)
    args = ap.parse_args()

    examples_dir = Path(args.examples_dir).resolve()
    metrics_path = Path(args.metrics_json).resolve()
    out_html = Path(args.out_html).resolve()

    if not examples_dir.exists():
        raise FileNotFoundError(examples_dir)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    metrics = json.load(open(metrics_path, "r"))
    triplets = find_triplets(examples_dir)

    if not triplets:
        print(f"No AP / Lat triplets found in {examples_dir}")
        return

    out_html.parent.mkdir(parents=True, exist_ok=True)

    # Paths in HTML should be relative to the HTML file location
    def rel(p: Path) -> str:
        return os.path.relpath(p, start=out_html.parent).replace("\\", "/")

    psnr_mean = metrics.get("psnr_mean", None)
    ssim_mean = metrics.get("ssim_mean", None)
    n_test = metrics.get("n_test", len(triplets))

    rows = []
    for i, (ap, pred, gt) in enumerate(triplets):
        rows.append(f"""
        <tr>
          <td style="text-align:center;">{i}</td>
          <td><img src="{rel(ap)}" style="max-width:256px;"></td>
          <td><img src="{rel(pred)}" style="max-width:256px;"></td>
          <td><img src="{rel(gt)}" style="max-width:256px;"></td>
        </tr>
        """)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>AP→Lat View Synthesis Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#111; color:#eee; }}
    h1, h2 {{ text-align:center; }}
    table {{ border-collapse: collapse; margin: 0 auto; }}
    th, td {{ border: 1px solid #444; padding: 8px; }}
    th {{ background:#222; }}
  </style>
</head>
<body>
  <h1>New X-ray View Synthesis – AP→Lat Baseline</h1>
  <h2>Metrics</h2>
  <p style="text-align:center;">
    n_test = {n_test} &nbsp; | &nbsp;
    PSNR_mean = {psnr_mean:.3f} dB &nbsp; | &nbsp;
    SSIM_mean = {ssim_mean:.3f}
  </p>
  <h2>Example Triplets</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>AP input</th>
        <th>Predicted LAT</th>
        <th>Ground truth LAT</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print("Wrote HTML report to:", out_html)


if __name__ == "__main__":
    main()
