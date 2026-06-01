# X-Ray-Triage-In-LMIC

Reproducible code for a calibrated, lightweight chest X-ray triage classifier
for pulmonary tuberculosis (TB), with internal evaluation on Qatar v1
TB-CXR and external evaluation on TBX11K (China) and the NIAID TB Portals
LMIC cohort.

This repository accompanies the manuscript *"Calibrated Lightweight Chest
X-ray Triage for Tuberculosis in Resource-Constrained Settings"*. The
model itself is a small ResNet-18 fine-tuned
with focal loss and post-hoc temperature scaling; the contribution of the
work is not a new architecture but the integration of (i) operating-point
reporting, (ii) calibration, (iii) external evaluation on twelve LMIC
cohorts, and (iv) a CPU-only deployment.

---

## Headline results

| Setting                              | n     | Acc.   | AUROC  | Sens@90 | ECE    |
|---|---:|---:|---:|---:|---:|
| Qatar/Dhaka v1 (internal test)       |   170 | 0.988  | 1.000  | 1.000   | 0.008  |
| TBX11K (China, external)             | 1 000 | 0.530  | 0.775  | 0.540   | 0.419  |
| TB Portals (LMIC, parts 1–3)         | 2 801 | —      | 0.985  | 0.858   | 0.154  |
| &nbsp;&nbsp; Sub-Saharan Africa subset | 43 | —      | 0.753  | 0.442   | —      |
| &nbsp;&nbsp;&nbsp;&nbsp; South Africa  | 20 | —      | —      | 0.150   | —      |
| &nbsp;&nbsp;&nbsp;&nbsp; Nigeria       | 20 | —      | —      | 0.750   | —      |

Inference latency on a commodity Intel i5 CPU is `30 ± 3 ms` per image.
The temperature constant fitted on the Qatar validation split is `T = 0.58`.

![Reliability diagram](images/reliability_resnet18.png)
*Two-panel reliability diagram on the internal Qatar test split.
(a) Equal-mass reliability bins zoomed to the relevant range; the calibrated
markers sit on the diagonal near confidence 1.0.
(b) Predicted-confidence histogram: 95% of the 170 test samples lie above
confidence 0.95, which is why an unzoomed reliability plot would look empty.*

![External ROC](images/external_roc.png)
*ROC on the internal Qatar split (blue), TB Portals LMIC (green), TB Portals
sub-Saharan African subset (dashed orange), and TBX11K China (red). The
calibrated checkpoint produces all four curves; the gap between the green
and the orange/red is the out-of-distribution generalisation story.*

---

## Repository layout

```
.
├── _common.py                # shared helpers (datasets, backbones, focal loss, metrics)
├── 01_download_data.py       # Kaggle data downloader (Qatar + TBX11K)
├── 02_train.py               # train ResNet-18 with focal loss
├── 03_calibrate.py           # fit a single scalar T on the validation split
├── 04_evaluate.py            # internal test: Acc, AUROC, Sens@90, ECE, reliability
├── 06_gradcam.py             # Grad-CAM overlays + lung-field localisation rate
├── 08_latency.py             # CPU latency benchmark + ONNX export
├── 10_external_eval.py       # OOD eval on TBX11K (and any cohort with neg labels)
├── 12_tbportals_eval.py      # OOD eval on NIAID TB Portals (TB-only, per country)
├── 14_replot_reliability.py  # two-panel reliability with confidence histogram
├── 15_replot_external_roc.py # combined ROC across internal + external cohorts
├── requirements.txt          # Python deps (torch, onnxruntime, sklearn, pydicom, ...)
│
├── deployment/               # reference Flask app for the trained model
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── templates/index.html
│
├── results/                  # JSON metrics produced by the scripts (committed)
│   ├── eval_resnet18.json
│   ├── external_tbx11k.json
│   ├── tbportals_eval.json
│   ├── ablation.json
│   ├── calibration_resnet18.json
│   └── latency_resnet18.json
│
└── images/                   # figures used in this README
    ├── reliability_resnet18.png
    ├── external_roc.png
    ├── app_tb_positive.png
    └── app_normal.png
```

---

## Reproducing the result

### 1. Environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Unix:
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU shortens the training step; the rest of the pipeline
runs comfortably on CPU.

### 2. Data

Two of the three datasets are downloadable from Kaggle and one requires
a free Data Use Agreement.

**Qatar TB-CXR v1** (training) and **TBX11K** (external) are pulled by:

```bash
python 01_download_data.py
```

The script expects `~/.kaggle/kaggle.json` to be present (see
<https://www.kaggle.com/docs/api>). Both datasets land under `data/`.

**NIAID TB Portals** (external, LMIC, used for `12_tbportals_eval.py`)
requires a click-through DUA at
<https://tbportals.niaid.nih.gov/download-data>. The DUA is free but the
data may not be redistributed; this repository does not contain any TB
Portals image. Once approved, place the extracted DICOM files under
`data/tbportals/` and the manifest + metadata CSV at
`data/tbportals/manifest.csv` and `data/tbportals/metadata.csv` (or pass
their paths with `--manifest_csv` / `--meta_csv`).

### 3. Train, calibrate, evaluate

```bash
python 02_train.py                                        # ResNet-18, focal loss
python 03_calibrate.py --ckpt checkpoints/resnet18_best.pt
python 04_evaluate.py  --ckpt checkpoints/resnet18_best.pt
python 14_replot_reliability.py --ckpt checkpoints/resnet18_best.pt
python 08_latency.py   --ckpt checkpoints/resnet18_best.pt   # also exports ONNX
python 06_gradcam.py   --ckpt checkpoints/resnet18_best.pt --n 32
```

### 4. External validation

```bash
python 10_external_eval.py  --ckpt checkpoints/resnet18_best.pt
python 12_tbportals_eval.py --ckpt checkpoints/resnet18_best.pt
python 15_replot_external_roc.py --ckpt checkpoints/resnet18_best.pt
```

`results/*.json` and `figures/*.pdf` will be produced; the JSONs already
committed in this repo are the values reported in the manuscript.

---

## Deployment

The `deployment/` folder contains a reference Flask app that serves the
calibrated checkpoint as an ONNX model. No PyTorch dependency at runtime.

![App: TB-positive case](images/app_tb_positive.png)
*Result screen on a TB-positive radiograph: red verdict banner, the gauge
needle pinned at the far right, the model-card panel exposing the headline
metrics.*

![App: Normal case](images/app_normal.png)
*Result screen on a normal radiograph: green verdict, the needle in the
normal band well below the escalation window (0.40, 0.60).*

### Run locally

```bash
cd deployment
pip install -r requirements.txt
# Copy your ONNX export into deployment/model/resnet18.onnx and a small
# calibration.json file containing {"temperature": 0.58, "backbone": "resnet18"}
python app.py                # binds 127.0.0.1:5000
```

### Run in Docker

```bash
cd deployment
docker build -t tbtriage:latest .
docker run --rm -p 5000:5000 tbtriage:latest
```

The container is CPU-only and approximately 200 MB.

---

## Out-of-scope claims

This is a triage tool, not a diagnostic device. The classifier was trained
on a single source-pooled corpus. Validation on a sub-Saharan African
cohort, in particular, is limited (n=43 across South Africa, Nigeria, and
Senegal), and within that subset the South Africa cases show a sharply
lower sensitivity than the others. The model card committed to the
release alongside the checkpoint lists the populations, devices, and use
cases under which the system has *not* been validated; please read it
before clinical use.

---

## License

MIT, see [LICENSE](LICENSE). Datasets and the model checkpoint are governed
by their own licences (TB Portals images may not be redistributed).
