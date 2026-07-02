# Bone Age Identification with YOLO Center Crop and EfficientNetV2

This project trains deep learning models for bone age estimation from hand/wrist X-ray images. The current recommended pipeline uses a YOLO detector only to locate the hand/wrist center and then applies a fixed-ratio crop before training an EfficientNetV2 model for regression.

The goal is to make the preprocessing consistent: instead of letting every YOLO bounding box define a different crop size and shape, YOLO provides the center of the useful region and the project always crops the same anatomical ratio around that center.

---

## Main Features

- YOLO-based hand/wrist localization.
- Fixed-ratio crop centered on the YOLO detection.
- Optional preview images showing the YOLO box, crop center and final fixed crop.
- EfficientNetV2-M training for bone age regression.
- Weights & Biases integration for experiment tracking.
- Automatic export of train, validation and test predictions to Excel.
- W&B prediction tables and true-vs-predicted scatter plots.
- Optional Grad-CAM heatmaps after training.
- Support for regression, binary classification, multiclass classification and stage classification.

---

## Project Structure

```text
age_bone_ia/
├── configs/
│   ├── config.yaml
│   └── default.py
│
├── data/
│   ├── images/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   ├── preprocessed/
│   ├── stages_images/
│   ├── dataset.py
│   ├── image_gen.py
│   └── resize.py
│
├── Heatmaps/
│   └── legacy heatmap notebooks and scripts
│
├── logs/
│   ├── checkpoint.py
│   └── logging.py
│
├── models/
│   ├── yolo_hand/
│   │   └── best.pt
│   ├── models.py
│   └── train.py
│
├── prediction/
│   ├── images_to_predict/
│   ├── prediction_models/
│   └── predict.py
│
├── utils/
│   ├── gradcam.py
│   ├── preprocess.py
│   └── yolo_crop.py
│
├── environment.yml
├── main.py
└── README.md
```

---

## Requirements

The project is designed to run with Conda.

Recommended:

- Python 3.10 or 3.11
- PyTorch
- Torchvision
- OpenCV
- Pandas
- Scikit-learn
- OpenPyXL
- Weights & Biases
- Ultralytics YOLO

The dependencies are listed in:

```text
environment.yml
```

---

## Environment Setup

From the project root, create the Conda environment:

```bash
conda env create -f environment.yml --prefix ./env
```

Activate it:

```bash
conda activate ./env
```

Alternatively, if you already have an environment such as `bone_identify_age`:

```bash
conda activate bone_identify_age
```

Install YOLO support if it is not already installed:

```bash
pip install ultralytics
```

---

## Dataset Placement

Place the X-ray images in:

```text
data/images/
```

Example:

```text
data/images/
├── 1.png
├── 2.png
├── 3.png
└── ...
```

Place the Excel annotation file directly inside:

```text
data/
```

Example:

```text
data/FCT2025_Final.xlsx
```

The default annotation file is configured in `configs/config.yaml`:

```yaml
ann_file: "FCT2025_Final.xlsx"
```

The image filenames must match the numeric ID used in the Excel file. For example, image `35.png` must correspond to patient/image ID `35` in the annotation spreadsheet.

---

## YOLO Model Placement

Place the trained YOLO detector here:

```text
models/yolo_hand/best.pt
```

The default path is configured in `configs/config.yaml`:

```yaml
yolo_model_path: "models/yolo_hand/best.pt"
```

---

## Recommended Configuration

The recommended configuration for bone age regression with EfficientNetV2-M is:

```yaml
model_task: "regression"
model_name: "efficientnet_v2_m"
image_processing: "yolo_crop"

ann_file: "FCT2025_Final.xlsx"

train_split: 0.7
val_split: 0.15
test_split: 0.15

batch_size: 16
epochs: 100
optimizer: "sgd-m"
learning_rate: 0.01
momentum: 0.9
weight_decay: 0.0001

recreate_dataset: True
```

For the YOLO center crop:

```yaml
yolo_model_path: "models/yolo_hand/best.pt"
yolo_conf: 0.30
yolo_iou: 0.45
yolo_imgsz: 640

yolo_fixed_ratio: True
yolo_crop_ratio: 0.75
yolo_crop_scale: 1.12
yolo_output_size: 384
yolo_center_shift_x: 0.0
yolo_center_shift_y: 0.05
yolo_pad_value: 0
yolo_save_previews: True

yolo_fallback: "classical_crop"
```

For the first run, keep:

```yaml
recreate_dataset: True
```

After the crops have been generated and verified, change it to:

```yaml
recreate_dataset: False
```

This avoids regenerating all crops every time the model is trained.

---

## How the YOLO Center Crop Works

The preprocessing pipeline is:

```text
Original X-ray image
↓
YOLO detects the approximate hand/wrist region
↓
The center of the YOLO bounding box is calculated
↓
A fixed-ratio crop is created around that center
↓
The crop is resized/padded to 384x384
↓
The processed image is used by EfficientNetV2
```

This is better than directly using the YOLO bounding box because the final training images keep a consistent anatomical framing.

Important crop parameters:

| Parameter | Meaning |
|---|---|
| `yolo_crop_ratio` | Width/height ratio of the fixed crop before final square padding. `0.75` is a vertical hand/wrist crop. `1.0` is a square crop. |
| `yolo_crop_scale` | Extra margin around the YOLO-centered crop. Increase if fingers or wrist are cut. Decrease if there is too much background. |
| `yolo_center_shift_y` | Moves the crop vertically. Positive values move the crop down toward the wrist. |
| `yolo_output_size` | Final saved image size. `384` means 384x384 pixels. |

---

## Running the Full Training Pipeline

From the project root:

```bash
python main.py
```

The script will:

1. Load `configs/config.yaml`.
2. Generate YOLO center crops if needed.
3. Create the train/validation/test split.
4. Train the selected model.
5. Log metrics to W&B.
6. Save checkpoints and best model weights in `results/`.
7. Export train, validation and test predictions to Excel.
8. Log prediction tables and scatter plots to W&B.
9. Generate Grad-CAM heatmaps if enabled.

---

## Generated Crop Folders

When using:

```yaml
image_processing: "yolo_crop"
```

processed images are saved to:

```text
data/preprocessed/yolo_crop/
```

Expected structure:

```text
data/preprocessed/yolo_crop/
├── with_treatment/
├── without_treatment/
├── unknown/
└── yolo_crop_log.csv
```

Preview images are saved to:

```text
data/preprocessed/yolo_crop_preview/
```

The preview images show:

- YOLO detection box.
- Detected center point.
- Fixed crop box.

Use this folder to confirm whether the crop is anatomically correct before training seriously.

---

## Adjusting the Crop

If fingers or wrist are being cut:

```yaml
yolo_crop_scale: 1.20
```

If there is too much background:

```yaml
yolo_crop_scale: 1.05
```

If the wrist is being cut:

```yaml
yolo_center_shift_y: 0.08
```

If the crop is too low:

```yaml
yolo_center_shift_y: 0.0
```

If a square crop is preferred:

```yaml
yolo_crop_ratio: 1.0
```

---

## Weights & Biases Logging

The project uses W&B for experiment tracking.

On first use:

```bash
wandb login
```

To run offline:

```bash
wandb offline
```

To return to online logging:

```bash
wandb online
```

During training, W&B logs:

- Training loss.
- Validation loss.
- Regression metrics such as MAE, MSE, RMSE and R².
- Final train, validation and test metrics.
- Prediction tables.
- True age vs predicted age scatter plots.
- Excel prediction files as artifacts.
- Grad-CAM heatmaps when enabled.

For regression, the most important W&B metrics are usually:

```text
val/mae
val/rmse
val/r2
final/test/mae
final/test/rmse
final/test/r2
predictions/test_scatter
```

---

## Why the Test Plot May Not Appear in W&B

The test set is only evaluated at the end of training. It is not used during each epoch.

That means there is no normal epoch-by-epoch test curve unless the code explicitly logs final test outputs. This version logs the test results at the end as:

```text
final/test/mae
final/test/mse
final/test/rmse
final/test/r2
predictions/test_table
predictions/test_scatter
predictions/test_residuals
```

The Excel file is also saved locally:

```text
results/<wandb_run_name>/test_predictions.xlsx
```

---

## Grad-CAM Heatmaps

Grad-CAM heatmaps are optional and controlled by:

```yaml
generate_heatmaps: True
heatmap_max_images: 16
```

When enabled, after training the project generates heatmaps from the test split. If no test split exists, it uses the validation split.

Saved heatmaps are placed in:

```text
results/<wandb_run_name>/heatmaps_test/
```

or:

```text
results/<wandb_run_name>/heatmaps_val/
```

They are also logged to W&B under:

```text
heatmaps/test
```

or:

```text
heatmaps/val
```

For EfficientNetV2-M, the Grad-CAM target layer is automatically selected from the model's final feature block.

---

## Results and Checkpoints

Training outputs are saved in:

```text
results/<wandb_run_name>/
```

Typical files:

```text
results/<wandb_run_name>/
├── best_regression_efficientnet_v2_m.pt
├── train_predictions.xlsx
├── val_predictions.xlsx
├── test_predictions.xlsx
└── heatmaps_test/
```

The best regression checkpoint is selected using the lowest validation MSE.

---

## Running Inference

Place trained checkpoints in:

```text
prediction/prediction_models/
```

Place images to predict in:

```text
prediction/images_to_predict/
```

Run:

```bash
python -m prediction.predict --model_name efficientnet_v2_m --model_task regression --batch_size 8
```

When prompted, choose the checkpoint to load.

---

## Common Problems

### CUDA is not available

The project will run on CPU, but EfficientNetV2-M can be slow. For CPU training, consider:

```yaml
batch_size: 4
num_workers: 0
epochs: 30
```

### Windows DataLoader error

Set:

```yaml
num_workers: 0
```

### YOLO model not found

Check that this file exists:

```text
models/yolo_hand/best.pt
```

Or update:

```yaml
yolo_model_path: "your/custom/path/best.pt"
```

### Crops are not being regenerated

Set:

```yaml
recreate_dataset: True
```

Then run:

```bash
python main.py
```

After verifying the crop previews, change it back to:

```yaml
recreate_dataset: False
```

### W&B does not show the test scatter plot

Make sure the updated code is being used and that training reaches the final export stage. The plot is logged only after training finishes, not during each epoch.

Also check that W&B is online:

```bash
wandb status
```

If it is offline, use:

```bash
wandb online
```

---

## Recommended Workflow

1. Place images in `data/images/`.
2. Place Excel annotations in `data/`.
3. Place YOLO model in `models/yolo_hand/best.pt`.
4. Set `recreate_dataset: True`.
5. Run `python main.py` once.
6. Inspect `data/preprocessed/yolo_crop_preview/`.
7. Adjust crop parameters if needed.
8. Set `recreate_dataset: False`.
9. Run the final training with `python main.py`.
10. Review metrics, scatter plots and heatmaps in W&B.

---

## Notes

This project is intended for research and educational use. Bone age estimation is a medical task and should not be used as a clinical decision tool without validation, expert supervision and appropriate regulatory approval.
