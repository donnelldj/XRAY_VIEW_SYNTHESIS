# X-Ray View Synthesis
**AP → backprojection volume → 3D U-Net → CT_pred → forward-project → LAT_pred**

This repository implements an end-to-end baseline for synthesizing a lateral (LAT, 90°) X-ray view from a single anteroposterior (AP, 0°) view.

The pipeline is designed to mirror the data generation logic used by the included Streamlit exporter and to serve as a clear, reproducible reference implementation for the “New X-ray View Synthesis” task described in Section III-D of the paper.

---

## Problem Setting

Given a single AP projection of a CT volume, the goal is to predict the corresponding LAT projection.

- Reconstruct a coarse 3D proxy volume from the AP view
- Predict a CT-like latent representation using supervised learning
- Forward-project the predicted volume to obtain the LAT view

Both quantitative metrics and qualitative visualizations are produced.

---

## Pipeline Overview

1. **Input**  
   A single AP DRR image with shape `(Z, X)`, normalized to `[0, 1]`.

2. **Backprojection**  
   The AP image is expanded into a 3D volume by replication along the Y axis, producing a tensor of shape `(Z, Y, X)`.

3. **Latent CT Prediction**  
   A compact 3D U-Net operates on a downsampled (latent) version of the backprojected volume to predict a CT-like latent representation. Supervision is provided by CT data.

4. **Forward Projection**  
   The predicted CT volume is projected into a LAT view by averaging along the X axis, followed by a fixed orientation transform.

5. **Evaluation and Visualization**  
   - **Quantitative:** PSNR and SSIM  
   - **Qualitative:** triplets `[ AP | Predicted LAT | Ground-Truth LAT ]`

---

## Ground-Truth Definition (Exporter Contract)

Ground-truth projections are defined by the Streamlit exporter using the following conventions:

### AP projection
'''python
ap = flipud(mean(ct_norm, axis=Y))  # shape (Z, X)
'''

### LAT projection
'''python
lat = flipud(mean(ct_norm, axis=X))  # shape (Z, Y)
'''

To ensure consistency, the forward projection used during training and evaluation applies the same operations in the same order. In particular, the flip along the Z axis is mandatory for matching the exported ground truth.

Projection/orientation logic lives in:

'''xray_synth/physics/projectors.py
'''

---

## Repository Layout (high-level)

- **Streamlit dataset generation & auditing:** `xray_synth/ui/`, `xray_synth/export/`, `xray_synth/app.py`
- **Training / evaluation entrypoint:** `xray_synth/tools/run_ap_to_lat.py`
- **Core pipeline modules:** `xray_synth/models/`, `xray_synth/train/`, `xray_synth/physics/`, `xray_synth/data/`, `xray_synth/io/`, `xray_synth/metrics/`, `xray_synth/vis/`

---

## NPZ File Format

Each case is stored as a `.npz` file with the following fields.

### Required
- `case_id` : string  
- `mhd_path` : string  
- `ap` : float32 array `(Z, X)`  
- `lat` : float32 array `(Z, Y)`  
- `spacing_zyx` : float32 array `(3,)` representing `(sz, sy, sx)`

### Optional
- `ct_zyx_norm` : float16 or float32 array `(Z, Y, X)` normalized to `[0, 1]`

If `ct_zyx_norm` is present, it is used directly for supervision.  
If not present, the CT volume is loaded from `mhd_path`, resampled to a fixed grid while preserving physical extent, clipped to the specified HU range, and normalized.

---

## Quickstart (Windows / PowerShell)

### 1) Clone
'''powershell
git clone https://github.com/donnelldj/XRAY_VIEW_SYNTHESIS.git
cd XRAY_VIEW_SYNTHESIS
'''


### 2) Create + activate virtual environment
'''powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
'''

### 3) Install dependencies
'''powershell
pip install -r requirements.txt
'''

### 4) Set PYTHONPATH
'''powershell
$env:PYTHONPATH="."
'''

---

## Download LUNA16 (Kaggle)

This repo does not redistribute LUNA16.

### kagglehub (recommended)
powershell
pip install kagglehub[pandas-datasets]


To download and place LUNA16 automatically run:
```bash
python download_luna16_and_place.py


or 


python
import kagglehub
dataset_path = kagglehub.dataset_download("avc0706/luna16")
print("Downloaded to:", dataset_path)

&

Ensure the dataset is under:

xray_synth/data/luna16/


---

## Streamlit Exporter (Generate Training Triplets)

Run:
powershell
streamlit run xray_synth\app.py


Use the sidebar to configure HU range, export size, and click **Execute Randomized Export**.

Outputs:

training_triplets/
  npz_registry/
  audit_samples/
  train.csv
  val.csv


---

## Train + Evaluate

powershell
python xray_synth/tools/run_ap_to_lat.py `
  --train_csv training_triplets/train.csv `
  --val_csv   training_triplets/val.csv `
  --out_dir   xray_synth/examples `
  --epochs 16 --batch 1 --lr 2e-4 `
  --latent_down 4 `
  --device cuda --amp `
  --num_examples 10


---

## Outputs

- `checkpoint.pt`
- `metrics.json`
- `metrics.txt`
- `examples/` (PNG triplets)

---

'''

## Scope

This implementation is intended as a clear and minimal baseline aligned with the provided data generation pipeline. The emphasis is on correctness, consistency with ground-truth definitions, and ease of review.
