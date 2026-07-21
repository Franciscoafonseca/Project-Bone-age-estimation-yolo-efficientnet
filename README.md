# Bone Age Estimation with YOLO11 and EfficientNetV2

> **Recommended repository name:** `bone-age-estimation-yolo-efficientnet`  
> **Current repository:** `Project-Bone_Detection_YOLO`  
> **Status:** Main bone-age estimation project

This repository implements a deep-learning pipeline for estimating bone age from hand and wrist radiographs.

The current workflow uses YOLO11 to locate the hand and wrist, applies a consistent fixed-ratio crop, and then uses EfficientNetV2-M to predict age by regression.

> This software is intended for research and educational purposes only. It is not a certified medical device and must not be used as a standalone clinical or forensic decision system.

---

## Pipeline

```text
Original hand/wrist radiograph
        ↓
YOLO11 hand and wrist localisation
        ↓
Fixed-ratio anatomical crop
        ↓
EfficientNetV2-M
        ↓
Bone-age prediction in months
        ↓
Evaluation on validation and test sets
```

---

## Main features

- YOLO-based hand and wrist localisation;
- fixed-ratio crop centred on the YOLO detection;
- EfficientNetV2-M regression;
- optional use of sex as auxiliary information;
- train, validation and test splits;
- Weights & Biases experiment tracking;
- MAE, RMSE and R² evaluation;
- export of predictions;
- optional Grad-CAM visualisations;
- CPU and NVIDIA GPU support.

---

## Repository structure

```text
bone-age-estimation-yolo-efficientnet/
├── configs/
│   ├── config.yaml
│   ├── default.py
│   └── sweep.yaml
├── data/
│   ├── dataset.py
│   ├── image_gen.py
│   ├── resize.py
│   ├── statistics.py
│   ├── images/                 # private, not committed
│   └── preprocessed/           # generated, not committed
├── logs/
│   ├── __init__.py
│   ├── checkpoint.py
│   └── logging.py
├── models/
│   ├── models.py
│   ├── train.py
│   └── yolo_hand/
│       ├── README.md
│       └── best.pt             # downloaded separately
├── prediction/
├── tools/
│   └── check_dataset_alignment.py
├── utils/
│   ├── gradcam.py
│   ├── image_utils.py
│   ├── preprocess.py
│   └── yolo_crop.py
├── environment.yml
├── requirements.txt
├── main.py
├── .gitignore
└── README.md
```

---

## Important corrections required before publishing

The following changes are required to make the repository internally consistent and reproducible.

### 1. Add the missing `logs/` package

`main.py` imports:

```python
from logs.logging import init_wandb, finish_wandb
```

The public repository must therefore include:

```text
logs/__init__.py
logs/logging.py
logs/checkpoint.py
```

Without this package, a new clone can fail with:

```text
ModuleNotFoundError: No module named 'logs'
```

The older baseline repository already contains this folder and can be used as the source, provided the files are compatible with the current training code.

### 2. Add the dataset alignment script or remove the command

The previous documentation refers to:

```bash
python tools/check_dataset_alignment.py
```

The public repository must include that file. If the script is not added, remove that command from the documentation.

### 3. Correct the units of the metrics

The regression target is:

```text
Idade na data da radiografia (meses)
```

Therefore:

- MAE is reported in months;
- RMSE is reported in months;
- prediction errors must not be described as years unless the code explicitly converts them.

### 4. Separate test and final configurations

Do not use the same file for a two-epoch CPU test and the final reported experiment.

Recommended files:

```text
configs/config_cpu_test.yaml
configs/config_final.yaml
```

A quick CPU test can use:

```yaml
learning_rate: 0.0001
epochs: 2
batch_size: 2
optimizer: "adam"
num_workers: 0
recreate_dataset: true
generate_heatmaps: false
```

The final experiment used in the project can be stored as:

```yaml
learning_rate: 0.01
epochs: 100
batch_size: 4
optimizer: "sgd-m"
momentum: 0.9
weight_decay: 0.0001
num_workers: 0
recreate_dataset: false
generate_heatmaps: true
seed: 42
```

Only report results produced by the frozen final configuration.

### 5. Verify age normalisation units

The current target is in months, while some configurations use:

```yaml
min_age: 5
max_age: 26
```

Verify how `normalize_age` is implemented.

- If `min_age` and `max_age` must use the same unit as the target, use the dataset range in months.
- If the code internally converts months to years first, document that conversion.
- Do not mix age values in years and targets in months.

For the current dataset, the observed target range previously used in the project was approximately:

```yaml
min_age: 72
max_age: 261
```

Use these values only if the normalisation code expects months.

### 6. Keep CPU and single-GPU execution without multiprocessing

When only one process is needed, use:

```python
train(0, 1, dataset, config, run)
```

Use `torch.multiprocessing.spawn` only when more than one GPU is actually being used. This avoids Windows and Weights & Biases multiprocessing errors.

### 7. Do not commit the trained YOLO model directly

Place the detector at:

```text
models/yolo_hand/best.pt
```

Distribute it through one of the following:

- GitHub Release;
- Git LFS;
- private institutional storage.

Keep `models/yolo_hand/README.md` in Git with placement instructions.

---

## Dataset

Private clinical data are not included in the repository.

Place the spreadsheet at:

```text
data/FCT2025_Final.xlsx
```

Place the radiographs at:

```text
data/images/
```

Expected image naming:

```text
1.jpg
2.jpg
3.jpg
...
597.jpg
```

The current mapping assumes:

```text
1.jpg   → row 1 in the spreadsheet
2.jpg   → row 2 in the spreadsheet
...
```

Do not use this row-based mapping if the spreadsheet order may change. A safer future improvement is to add an explicit image identifier column.

Expected columns include:

```text
Sexo
Data de Nascimento
Data de realização da radiografia
Idade na data da radiografia (anos)
Idade na data da radiografia (meses)
```

The regression target is:

```text
Idade na data da radiografia (meses)
```

Before training:

```bash
python tools/check_dataset_alignment.py
```

Do not train if images and spreadsheet rows are not aligned.

---

## YOLO model

The detector is trained in the separate repository:

```text
hand-wrist-yolo-detection
```

Place the selected weight file here:

```text
models/yolo_hand/best.pt
```

Recommended placeholder file:

```text
models/yolo_hand/README.md
```

with instructions explaining where the model can be obtained.

---

## Installation

Recommended Python version:

```text
Python 3.11
```

### Conda

```bash
conda env create -f environment.yml
conda activate bone_identify_age
```

Alternatively:

```bash
conda create -n bone_identify_age python=3.11 -y
conda activate bone_identify_age
python -m pip install -r requirements.txt
```

Install the correct PyTorch build for the computer being used.

Check CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Check the main dependencies:

```bash
python -c "import pandas, openpyxl, torch, torchvision, cv2, ultralytics, timm, wandb; print('OK')"
```

---

## Configuration

The main configuration is loaded from:

```text
configs/config.yaml
```

Minimum required settings:

```yaml
model_task: "regression"
model_name: "efficientnet_v2_m"

ann_file: "FCT2025_Final.xlsx"
excel_sheet: 0
id_col: null
target_col: "Idade na data da radiografia (meses)"
target_unit: "months"

use_with_treatment: false
dataset_layout: "flat"
treatment_col: null

image_processing: "yolo_crop"
input_image_dir: "data/images"
output_preprocessed_dir: "data/preprocessed"

yolo_model_path: "models/yolo_hand/best.pt"
yolo_conf: 0.30
yolo_iou: 0.45
yolo_imgsz: 640
yolo_fixed_ratio: true
yolo_crop_ratio: 0.75
yolo_crop_scale: 1.12
yolo_output_size: 384
yolo_center_shift_x: 0.0
yolo_center_shift_y: 0.05
yolo_fallback: "classical_crop"

train_split: 0.70
val_split: 0.15
test_split: 0.15
seed: 42
```

---

## First test

Use the CPU test configuration and run:

```bash
python main.py
```

The test should:

1. load the spreadsheet;
2. verify image paths;
3. load `models/yolo_hand/best.pt`;
4. create YOLO-centred crops;
5. create the dataset splits;
6. start EfficientNetV2-M training;
7. save metrics and checkpoints.

Inspect:

```text
data/preprocessed/yolo_crop/
data/preprocessed/yolo_crop_preview/
```

Confirm that:

- all fingers are visible;
- the thumb is included;
- the carpal region is included;
- the distal radius and ulna are visible;
- the hand is not cut;
- unnecessary background is limited.

When crop parameters change, regenerate the processed dataset.

---

## Final training

After validating the pipeline:

```bash
python main.py
```

Use the frozen final configuration and keep:

```yaml
recreate_dataset: false
```

after the crops have been generated and approved.

---

## Weights & Biases

Login:

```bash
wandb login
```

Run:

```bash
python main.py
```

Offline mode on Windows CMD:

```cmd
set WANDB_MODE=offline
python main.py
```

Do not commit the local `wandb/` directory.

---

## Evaluation

Final reported metrics must come from the held-out test set.

Main metrics:

- **MAE:** average absolute prediction error in months;
- **RMSE:** error in months with greater penalty for large deviations;
- **R²:** proportion of target variability explained by the model.

Recommended outputs:

```text
results/test_metrics.json
results/test_predictions.xlsx
results/predicted_vs_true.png
results/error_distribution.png
```

Validation metrics are used for model selection and hyperparameter decisions. They are not the final test results.

---

## Reproducibility checklist

Record the following for every final experiment:

- dataset version;
- train, validation and test identifiers;
- random seed;
- preprocessing method;
- YOLO weight version;
- crop parameters;
- model architecture;
- pretrained weights;
- optimizer;
- learning rate;
- batch size;
- number of epochs;
- early-stopping settings;
- software environment;
- final validation metrics;
- final test metrics.

---

## Privacy

Never commit:

- radiographs;
- DICOM files;
- patient identifiers;
- birth dates;
- radiography dates;
- private spreadsheets;
- W&B credentials;
- generated files that retain identifying metadata.

---

## Related repositories

- `hand-wrist-yolo-detection`: creates and evaluates the YOLO hand/wrist detector.
- `bone-age-estimation-legacy-baseline`: older non-YOLO baseline retained only for reproducibility.

---

## Recommended repository topics

```text
bone-age
medical-imaging
pytorch
efficientnet
yolo11
xray
regression
deep-learning
```

---

## License

Add a software license before wider distribution.

The dataset, trained weights and third-party libraries may have separate licences and access restrictions.
