import sys
import os
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import pandas as pd
from tqdm import tqdm

from utils.preprocess import preprocess_image
from utils.yolo_crop import load_yolo_model, draw_yolo_center_preview


# --- CONFIGURATION ---
INPUT_FOLDER = "data/images"
OUTPUT_ROOT = "data/preprocessed"
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

CROP_METHODS = {
    "crop_only",
    "clahe_crop",
    "center_crop_then_crop",
    "clahe_center_crop_then_crop",
}
YOLO_METHODS = {
    "yolo_crop",
    "yolo_clahe_crop",
}
DEFAULT_CROP = "both"  # Options: 'crop_black', 'crop_symmetric', 'both'


# --- UTILITY FUNCTIONS ---

def make_dirs(path):
    os.makedirs(path, exist_ok=True)


def _is_treated(value):
    """
    Convert the treatment column into a boolean.

    Supports common Excel encodings:
        1/0, True/False, Sim/Não, Com/Sem tratamento.
    """
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return int(value) == 1

    value_str = str(value).strip().lower()

    if value_str in {"1", "true", "sim", "yes", "com", "com tratamento", "with", "with treatment"}:
        return True

    if value_str in {"0", "false", "não", "nao", "no", "sem", "sem tratamento", "without", "without treatment"}:
        return False

    # Conservative fallback: do not send unknown strings to "with_treatment".
    return False


def load_labels(excel_path):
    """Loads Excel and returns a dict: image ID string -> treatment folder."""
    df = pd.read_excel(excel_path)

    original_columns = list(df.columns)
    clean_columns = [str(c).strip().lower().replace(" ", "_") for c in original_columns]

    print(f"DEBUG: Excel columns detected: {original_columns}")

    # Default used by the original project: first column = image ID, seventh column = treatment.
    filename_col = 0
    treated_col = 6 if len(df.columns) > 6 else None

    for i, col in enumerate(clean_columns):
        if col in {"nº", "n_", "n", "id", "image_id", "filename", "file_name"} or "file" in col:
            filename_col = i

        if "tratamento" in col or "treat" in col:
            treated_col = i

    if filename_col is None:
        raise ValueError("Could not find the image ID column in the Excel sheet.")

    label_map = {}

    for _, row in df.iterrows():
        try:
            number_key = str(int(row.iloc[filename_col]))

            if treated_col is None:
                label_folder = "without_treatment"
            else:
                treated = _is_treated(row.iloc[treated_col])
                label_folder = "with_treatment" if treated else "without_treatment"

            label_map[number_key] = label_folder

        except Exception as e:
            print(f"Skipping row due to error: {e}")

    return label_map


def _numeric_filename_key(filename):
    stem = os.path.splitext(filename)[0]
    digits = ''.join(filter(str.isdigit, stem))
    return int(digits) if digits else stem.lower()


def _needs_yolo(mode):
    return mode in YOLO_METHODS


def _write_yolo_log(log_path, rows):
    if not rows:
        return

    make_dirs(os.path.dirname(log_path))

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "filename",
            "image_id",
            "label_folder",
            "status",
            "confidence",
            "yolo_x1",
            "yolo_y1",
            "yolo_x2",
            "yolo_y2",
            "center_x",
            "center_y",
            "crop_x1",
            "crop_y1",
            "crop_x2",
            "crop_y2",
            "crop_ratio",
            "crop_scale",
            "output_size",
            "save_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --- MAIN FUNCTION ---

def generate_processed_images(config):
    """
    Generate the preprocessed image folder used by main.py.

    This function keeps the original repository structure:
        data/images/
        data/preprocessed/<image_processing>/with_treatment/
        data/preprocessed/<image_processing>/without_treatment/
        data/preprocessed/<image_processing>/unknown/
    """
    mode = config.get("image_processing", "clahe_center_crop_then_crop")
    input_folder = config.get("input_image_dir", INPUT_FOLDER)
    output_root = config.get("output_preprocessed_dir", OUTPUT_ROOT)
    ann_file = os.path.join("data", config["ann_file"])

    label_map = load_labels(ann_file)

    image_files = [
        f for f in sorted(os.listdir(input_folder), key=_numeric_filename_key)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not image_files:
        raise RuntimeError(f"No image files found in {input_folder}")

    print(f"\nProcessing mode: {mode}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {os.path.join(output_root, mode)}")
    print(f"Images found: {len(image_files)}")

    crop_method = DEFAULT_CROP if mode in CROP_METHODS or mode in YOLO_METHODS else None

    yolo_model = None
    yolo_log_rows = []

    save_yolo_previews = bool(config.get("yolo_save_previews", True))
    preview_root = os.path.join(output_root, mode + "_preview")

    if _needs_yolo(mode):
        yolo_model_path = config.get("yolo_model_path", "models/yolo_hand/best.pt")
        print(f"Loading YOLO model: {yolo_model_path}")
        yolo_model = load_yolo_model(yolo_model_path)
        if save_yolo_previews:
            make_dirs(preview_root)

    for fname in tqdm(image_files):
        image_path = os.path.join(input_folder, fname)
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            print(f"⚠️ Skipping unreadable image: {fname}")
            continue

        first_number = ''.join(filter(str.isdigit, os.path.splitext(fname)[0]))
        if first_number == "":
            first_number = os.path.splitext(fname)[0]

        # Convert "0001" to "1", matching the Excel ID and dataset filtering logic.
        try:
            image_id = str(int(first_number))
        except ValueError:
            image_id = first_number

        label_folder = label_map.get(image_id, "unknown")
        output_folder = os.path.join(output_root, mode, label_folder)
        make_dirs(output_folder)
        save_path = os.path.join(output_folder, image_id + ".png")

        try:
            if _needs_yolo(mode):
                processed, metadata = preprocess_image(
                    img_gray,
                    mode=mode,
                    crop_method=crop_method,
                    yolo_model=yolo_model,
                    yolo_config=config,
                    return_metadata=True,
                )

                yolo_log_rows.append({
                    "filename": fname,
                    "image_id": image_id,
                    "label_folder": label_folder,
                    "status": metadata.get("status", ""),
                    "confidence": metadata.get("confidence", ""),
                    "yolo_x1": metadata.get("yolo_x1", ""),
                    "yolo_y1": metadata.get("yolo_y1", ""),
                    "yolo_x2": metadata.get("yolo_x2", ""),
                    "yolo_y2": metadata.get("yolo_y2", ""),
                    "center_x": metadata.get("center_x", ""),
                    "center_y": metadata.get("center_y", ""),
                    "crop_x1": metadata.get("crop_x1", ""),
                    "crop_y1": metadata.get("crop_y1", ""),
                    "crop_x2": metadata.get("crop_x2", ""),
                    "crop_y2": metadata.get("crop_y2", ""),
                    "crop_ratio": metadata.get("crop_ratio", ""),
                    "crop_scale": metadata.get("crop_scale", ""),
                    "output_size": metadata.get("output_size", ""),
                    "save_path": save_path,
                })

                if save_yolo_previews:
                    preview = draw_yolo_center_preview(img_gray, metadata)
                    preview_folder = os.path.join(preview_root, label_folder)
                    make_dirs(preview_folder)
                    preview_path = os.path.join(preview_folder, image_id + ".png")
                    cv2.imwrite(preview_path, preview)
            else:
                processed = preprocess_image(
                    img_gray,
                    mode=mode,
                    crop_method=crop_method,
                )

        except Exception as e:
            print(f"❌ Failed processing {fname}: {e}")
            continue

        cv2.imwrite(save_path, processed)

    if _needs_yolo(mode):
        log_path = os.path.join(output_root, mode, "yolo_crop_log.csv")
        _write_yolo_log(log_path, yolo_log_rows)
        print(f"YOLO crop log saved in: {log_path}")
        if save_yolo_previews:
            print(f"YOLO previews saved in: {preview_root}")

    print("\n✅ Processed images generated successfully.")


if __name__ == "__main__":
    config = {
        "ann_file": "FCT2025_Final.xlsx",
        "image_processing": "yolo_crop",
        "yolo_model_path": "models/yolo_hand/best.pt",
        "yolo_conf": 0.30,
        "yolo_iou": 0.45,
        "yolo_imgsz": 640,
        "yolo_padding_x": 0.04,
        "yolo_padding_y": 0.04,
        "yolo_fixed_ratio": True,
        "yolo_crop_ratio": 0.75,
        "yolo_crop_scale": 1.12,
        "yolo_output_size": 384,
        "yolo_center_shift_x": 0.0,
        "yolo_center_shift_y": 0.05,
        "yolo_pad_value": 0,
        "yolo_save_previews": True,
        "yolo_fallback": "classical_crop",
        "yolo_device": "",
    }
    generate_processed_images(config)
