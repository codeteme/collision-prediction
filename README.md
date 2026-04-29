# Detect to Protect

Video-based collision risk prediction using deep learning on dashcam footage.

---

## Overview

Every year, tens of thousands of people are killed in vehicle collisions in the United States. Many crashes are preceded by detectable visual cues in the seconds before impact. This project builds and evaluates a pipeline that watches 1.6 seconds of dashcam video and outputs a collision probability, enabling faster emergency braking or pre-crash seat belt tensioning.

We fine-tuned [VideoMAE](https://huggingface.co/MCG-NJU/videomae-base) on the [Nexar Detect to Protect](https://www.kaggle.com/competitions/nexar-collision-prediction) dataset (1,500 labeled clips, balanced between collision and non-collision) and systematically tested three input modalities — RGB frames, depth maps (DepthAnything v2), and segmentation masks (YOLOv8) — across two clip timing offsets.

**Best result:** three-stream VideoMAE (RGB + Depth + Seg), anchor offset 0.0s — validation AUC **0.918** (95% CI: 0.884–0.945).

---

## Repository Structure

```
detect-to-protect/
├── activate.sh                       # Source this every DCC session
├── requirements.txt                  # Python dependencies
├── docs/
│   ├── final-report.md               # Full project report
│   ├── setup.md                      # DCC cluster setup guide
│   ├── project-decisions.md          # Design decisions log
│   └── lighting-analysis.md          # Night vs. day performance analysis
├── notebooks/
│   ├── preprocess.ipynb              # Frame extraction, depth, segmentation
│   └── train.ipynb                   # Interactive training exploration
├── scripts/                          # SLURM batch job scripts
│   ├── submit_train_baseline.sh
│   ├── submit_train_videomae.sh
│   ├── submit_train_videomae_depth.sh
│   ├── submit_train_videomae_seg.sh
│   ├── submit_train_videomae_full.sh
│   └── submit_predict_*.sh
├── src/
│   ├── train_baseline.py             # TinyVideoCNN trained from scratch
│   ├── train_videomae.py             # VideoMAE RGB fine-tuning
│   ├── train_videomae_depth.py       # Two-stream RGB + Depth
│   ├── train_videomae_seg.py         # Two-stream RGB + Seg
│   ├── train_videomae_full.py        # Three-stream RGB + Depth + Seg (best)
│   ├── predict_baseline.py           # Kaggle submission — baseline
│   ├── predict_videomae.py           # Kaggle submission — RGB
│   ├── predict_videomae_depth.py     # Kaggle submission — RGB + Depth
│   ├── predict_videomae_seg.py       # Kaggle submission — RGB + Seg
│   ├── predict_videomae_full.py      # Kaggle submission — RGB + Depth + Seg
│   ├── eval_save_preds.py            # Save val predictions as .npz
│   ├── compute_metrics.py            # Precision, recall, F1, bootstrap CI
│   ├── error_analysis.py             # Lighting condition and error analysis
│   └── visualize_pipeline.py         # Generate pipeline figure
└── data/                             # Not tracked in git — see Data section
    ├── train.csv
    ├── test.csv
    ├── frames/                       # RGB frames at 10 fps (.npy)
    ├── depth/                        # DepthAnything v2 depth maps (.npy)
    └── segmentation/                 # YOLOv8 segmentation masks (.npy)
```

---

## Setup

### Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key packages: `torch`, `transformers`, `ultralytics` (YOLOv8), `scikit-learn`, `pandas`, `numpy`, `tqdm`, `wandb`.

### DCC Cluster Setup

All training was run on the Duke Computing Cluster (DCC) with Tesla P100-PCIE-16GB GPUs. See [`docs/setup.md`](docs/setup.md) for the full environment setup guide.

**Each DCC session:**

```bash
ssh <netid>@dcc-login.oit.duke.edu
source /hpc/group/coursess26/ids705/team-project/detect-to-protect/activate.sh
```

This activates the shared conda environment and sets the working directory.

---

## Data

### Getting the Data

Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/nexar-collision-prediction). You will need a Kaggle account and must accept the competition rules.

```bash
# Install Kaggle CLI if needed
pip install kaggle

# Download competition data
kaggle competitions download -c nexar-collision-prediction
unzip nexar-collision-prediction.zip -d data/
```

Place the downloaded files so the structure matches:

```
data/
├── train.csv     # clip IDs and collision labels (1 = collision, 0 = normal)
├── test.csv      # clip IDs for Kaggle submission
└── videos/       # raw .mp4 clips
```

### Preprocessing

Preprocessing extracts three synchronized input streams from the raw video clips. Run `notebooks/preprocess.ipynb` to reproduce this step, or follow the steps below.

**Step 1 — Extract RGB frames** at 10 fps from each clip and save as `.npy` arrays of shape `(N, H, W, 3)`:

```bash
# Covered in notebooks/preprocess.ipynb
# Output: data/frames/train/<clip_id>.npy
#         data/frames/test/<clip_id>.npy
```

**Step 2 — Generate depth maps** using DepthAnything v2, applied frame-by-frame to each RGB clip:

```bash
# Output: data/depth/train/<clip_id>.npy
#         data/depth/test/<clip_id>.npy
```

**Step 3 — Generate segmentation masks** using YOLOv8, applied frame-by-frame to each RGB clip:

```bash
# Output: data/segmentation/train/<clip_id>.npy
#         data/segmentation/test/<clip_id>.npy
```

All preprocessing code is in `notebooks/preprocess.ipynb`. Pre-extracted `.npy` arrays are stored on the DCC shared filesystem at `/hpc/group/coursess26/ids705/team-project/detect-to-protect/data/` and are available to team members without re-running preprocessing.

---

## Reproducing Results

### Training

Submit any model as a SLURM batch job from the project root:

```bash
# Best model — three-stream RGB + Depth + Seg (anchor offset 0.0s)
sbatch scripts/submit_train_videomae_full.sh

# Anchor offset ablation variants
sbatch scripts/submit_train_videomae_full_ofs0p8.sh
sbatch scripts/submit_train_videomae_full_ofs1p0.sh

# Two-stream ablations
sbatch scripts/submit_train_videomae_depth.sh
sbatch scripts/submit_train_videomae_seg.sh

# RGB-only VideoMAE
sbatch scripts/submit_train_videomae.sh

# Baseline TinyVideoCNN (clip-length ablation)
sbatch scripts/submit_train_baseline_clip16_ofs0p0.sh
```

All runs log to [Weights & Biases](https://wandb.ai/teme/detect-to-protect). Checkpoints are saved to `outputs/best_<model_name>.pt`.

### Evaluation

Re-run validation inference on a saved checkpoint and save predictions:

```bash
PYTHON=envs/dtp/bin/python

# Best model — three-stream
$PYTHON src/eval_save_preds.py \
    --type full \
    --checkpoint outputs/best_videomae_full_ofs0p0.pt \
    --out outputs/preds_videomae_full_ofs0p0.npz

# RGB-only
$PYTHON src/eval_save_preds.py \
    --type rgb \
    --checkpoint outputs/best_videomae_clip16_ofs0p0.pt \
    --out outputs/preds_videomae_rgb_ofs0p0.npz

# Baseline
$PYTHON src/eval_save_preds.py \
    --type baseline \
    --checkpoint outputs/best_baseline_scratch_clip16_ofs0p0.pt \
    --out outputs/preds_baseline_clip16_ofs0p0.npz
```

`--type` choices: `rgb`, `depth`, `seg`, `full`, `baseline`

### Compute Metrics (AUC, Precision, Recall, F1, Bootstrap CI)

```bash
$PYTHON src/compute_metrics.py \
    --preds outputs/preds_videomae_full_ofs0p0.npz
```

This prints AUC with 95% bootstrap confidence intervals, precision, recall, and F1 at the F1-optimal threshold.

### Error and Lighting Analysis

```bash
$PYTHON src/error_analysis.py \
    --preds outputs/preds_videomae_full_ofs0p0.npz \
    --save-csv outputs/error_analysis_full.csv \
    --save-thumbnails outputs/thumbnails
```

This classifies validation clips by brightness (dark/bright proxy for night/day) and reports AUC, recall, and false alarm rate for each group. See [`docs/lighting-analysis.md`](docs/lighting-analysis.md) for the full analysis.

### Kaggle Submission

```bash
$PYTHON src/predict_videomae_full.py \
    --checkpoint-path outputs/best_videomae_full_ofs0p0.pt \
    --submission-path outputs/submission_videomae_full.csv
```

---

## Results

| Model | Modalities | Offset (s) | Val AUC | 95% CI |
|---|---|---|---|---|
| TinyVideoCNN (scratch) | RGB, clip=16 | 0.0 | 0.709 | (0.648–0.765) |
| TinyVideoCNN (scratch) | RGB, clip=16 | 0.5 | 0.645 | (0.584–0.703) |
| TinyVideoCNN (scratch) | RGB, clip=32 | 0.0 | 0.679 | (0.618–0.735) |
| TinyVideoCNN (scratch) | RGB, clip=64 | 0.0 | 0.633 | (0.571–0.693) |
| TinyVideoCNN (scratch) | RGB, clip=100 | 0.0 | 0.629 | (0.567–0.692) |
| VideoMAE fine-tuned | RGB | 0.0 | 0.769 | (0.712–0.818) |
| VideoMAE fine-tuned | RGB | 0.5 | 0.772 | (0.718–0.824) |
| VideoMAE fine-tuned | RGB + Depth | 0.0 | 0.814 | (0.766–0.861) |
| VideoMAE fine-tuned | RGB + Depth | 0.5 | 0.712 | (0.650–0.768) |
| VideoMAE fine-tuned | RGB + Seg | 0.0 | 0.682 | (0.625–0.740) |
| VideoMAE fine-tuned | RGB + Seg | 0.5 | 0.666 | (0.606–0.725) |
| **VideoMAE fine-tuned** | **RGB + Depth + Seg** | **0.0** | **0.918** | **(0.884–0.945)** |
| VideoMAE fine-tuned | RGB + Depth + Seg | 0.8 | 0.801 | (0.749–0.850) |
| VideoMAE fine-tuned | RGB + Depth + Seg | 1.0 | 0.771 | (0.716–0.823) |

95% CIs computed via bootstrap resampling (2,000 iterations) on the held-out 20% validation set.

### Classification Metrics at F1-Optimal Threshold

| Model | Modalities | Offset (s) | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| TinyVideoCNN | RGB, clip=32 | 0.0 | 0.679 | 0.559 | 0.953 | 0.704 |
| TinyVideoCNN | RGB, clip=16 | 0.0 | 0.709 | 0.619 | 0.887 | 0.729 |
| VideoMAE | RGB | 0.0 | 0.769 | 0.626 | 0.860 | 0.725 |
| VideoMAE | RGB + Depth | 0.0 | 0.814 | 0.733 | 0.787 | 0.759 |
| VideoMAE | RGB + Seg | 0.0 | 0.682 | 0.627 | 0.773 | 0.693 |
| **VideoMAE** | **RGB + Depth + Seg** | **0.0** | **0.918** | **0.781** | **0.927** | **0.848** |
| VideoMAE | RGB + Depth + Seg | 0.8 | 0.801 | 0.685 | 0.900 | 0.778 |
| VideoMAE | RGB + Depth + Seg | 1.0 | 0.771 | 0.667 | 0.840 | 0.743 |

---

## Key Findings

- **Pretraining matters.** VideoMAE fine-tuned on RGB alone (AUC 0.769) substantially outperformed a 3D CNN trained from scratch (AUC 0.679) on the same data.
- **All three modalities together are best.** The three-stream model (RGB + Depth + Seg) at offset 0.0s reached AUC 0.918. At its optimal threshold it catches 139/150 collisions (92.7% recall) with a 26% false alarm rate.
- **Segmentation alone hurts, but combined with depth it helps.** RGB+Seg scored 0.682 (below the RGB baseline), but RGB+Depth+Seg scored 0.918. Depth and segmentation carry complementary information the model can only exploit together.
- **The final 1.6 seconds are the most predictive.** Shifting the clip window back by 0.8s drops AUC from 0.918 to 0.801. Longer clip lengths also hurt: clip=16 (AUC 0.709) outperforms clip=32 (0.679), clip=64 (0.633), and clip=100 (0.629).
- **Depth is the most time-sensitive modality.** Shifting back 0.5s barely affects the RGB-only model (0.769 → 0.772) but sharply hurts the three-stream model. Proximity cues change fastest in the final half-second.
- **Night is a precision problem, not a recall problem.** The model achieves 100% recall on dark clips (0 missed nighttime collisions) but over-triggers on some dark non-collision scenes. All 11 missed collisions are bright daytime clips. This is the inverse of object-detection-based systems, which fail at night because bounding-box pipelines break in the dark. See [`docs/lighting-analysis.md`](docs/lighting-analysis.md) for the full analysis.

---

## Lighting Analysis

A brightness-based proxy analysis was run on the best model's validation predictions to assess day vs. night performance. Clips with mean last-frame brightness below 60 (out of 255) were classified as dark.

| Condition | n | AUC | Recall | False Alarm Rate |
|---|---|---|---|---|
| All clips | 300 | 0.918 | 92.7% | 26.0% |
| Bright (day proxy) | 216 | 0.915 | 89.7% | 26.6% |
| Dark (night proxy) | 84 | 0.922 | **100.0%** | 24.4% |

The model misses zero collisions at night but fires with near-certainty on some dark non-collision clips, suggesting a spurious correlation between visual darkness and collision score. See [`docs/lighting-analysis.md`](docs/lighting-analysis.md) for methodology, worst-case clip analysis, comparison with a published baseline system, and implications for future work.