import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    """Return a 2D grayscale image, accepting grayscale, BGR, RGB, or 1-channel arrays."""
    if image is None:
        return image
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Return a 3-channel BGR image, accepting grayscale, BGR, BGRA, or 1-channel arrays."""
    if image is None:
        return image
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported image shape for BGR conversion: {image.shape}")


def load_yolo_model(model_path: str):
    """
    Load an Ultralytics YOLO model only when YOLO preprocessing is enabled.

    Keeping the import inside this function allows the rest of the project to run
    without ultralytics installed when image_processing is not a YOLO mode.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}. "
            "Place your trained hand/wrist detector at this path or update "
            "'yolo_model_path' in configs/config.yaml."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "The package 'ultralytics' is required for YOLO preprocessing. "
            "Install it with: python -m pip install ultralytics"
        ) from exc

    return YOLO(model_path)


def _empty_metadata(status: str) -> Dict[str, Any]:
    return {
        "status": status,
        "confidence": "",
        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",
        "yolo_x1": "",
        "yolo_y1": "",
        "yolo_x2": "",
        "yolo_y2": "",
        "center_x": "",
        "center_y": "",
        "crop_x1": "",
        "crop_y1": "",
        "crop_x2": "",
        "crop_y2": "",
        "crop_ratio": "",
        "crop_scale": "",
        "output_size": "",
    }


def _clip(value: float, min_value: int, max_value: int) -> int:
    return max(min_value, min(int(round(value)), max_value))


def _prediction_kwargs(conf: float, iou: float, imgsz: int, device: Optional[str]) -> Dict[str, Any]:
    kwargs = {
        "conf": conf,
        "iou": iou,
        "max_det": 1,
        "imgsz": imgsz,
        "verbose": False,
    }

    # Ultralytics accepts device="cpu", device=0, etc. If empty/None, it chooses automatically.
    if device not in (None, ""):
        kwargs["device"] = device

    return kwargs


def _crop_with_padding_gray(
    img_gray: np.ndarray,
    cx: float,
    cy: float,
    crop_w: float,
    crop_h: float,
    pad_value: int = 0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop a fixed-size window centered on (cx, cy).

    If the crop goes outside the original image, it is padded instead of being
    clipped. This keeps the anatomical crop ratio stable across all images.
    """
    img_h, img_w = img_gray.shape[:2]

    crop_w = max(1, int(round(crop_w)))
    crop_h = max(1, int(round(crop_h)))

    x1 = int(round(cx - crop_w / 2))
    y1 = int(round(cy - crop_h / 2))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    output = np.full((crop_h, crop_w), pad_value, dtype=img_gray.dtype)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(img_w, x2)
    src_y2 = min(img_h, y2)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        output[dst_y1:dst_y2, dst_x1:dst_x2] = img_gray[src_y1:src_y2, src_x1:src_x2]

    return output, (x1, y1, x2, y2)


def _letterbox_gray_to_square(img_gray: np.ndarray, output_size: int, pad_value: int = 0) -> np.ndarray:
    """
    Convert a rectangular crop to output_size x output_size without distortion.

    The crop is resized proportionally and padded. This is safer for radiographs
    than stretching width/height independently.
    """
    output_size = int(output_size)
    h, w = img_gray.shape[:2]

    if h <= 0 or w <= 0:
        return np.full((output_size, output_size), pad_value, dtype=img_gray.dtype)

    scale = min(output_size / w, output_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    output = np.full((output_size, output_size), pad_value, dtype=img_gray.dtype)

    x_offset = (output_size - new_w) // 2
    y_offset = (output_size - new_h) // 2
    output[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return output


def draw_yolo_center_preview(img_gray: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
    """
    Create a BGR preview showing:
      blue  = original YOLO box
      green = fixed-ratio crop window used before letterboxing
      red   = center point used for the crop
    """
    preview = _ensure_bgr(img_gray).copy()

    def has_number(key: str) -> bool:
        return metadata.get(key, "") not in ("", None)

    if all(has_number(k) for k in ["yolo_x1", "yolo_y1", "yolo_x2", "yolo_y2"]):
        yx1 = int(round(float(metadata["yolo_x1"])))
        yy1 = int(round(float(metadata["yolo_y1"])))
        yx2 = int(round(float(metadata["yolo_x2"])))
        yy2 = int(round(float(metadata["yolo_y2"])))
        cv2.rectangle(preview, (yx1, yy1), (yx2, yy2), (255, 0, 0), 2)
        cv2.putText(preview, "YOLO box", (max(5, yx1), max(20, yy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if all(has_number(k) for k in ["crop_x1", "crop_y1", "crop_x2", "crop_y2"]):
        h, w = preview.shape[:2]
        cx1 = _clip(float(metadata["crop_x1"]), 0, w - 1)
        cy1 = _clip(float(metadata["crop_y1"]), 0, h - 1)
        cx2 = _clip(float(metadata["crop_x2"]), 0, w - 1)
        cy2 = _clip(float(metadata["crop_y2"]), 0, h - 1)
        cv2.rectangle(preview, (cx1, cy1), (cx2, cy2), (0, 255, 0), 3)
        cv2.putText(preview, "fixed crop", (max(5, cx1), min(h - 10, cy1 + 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if all(has_number(k) for k in ["center_x", "center_y"]):
        px = int(round(float(metadata["center_x"])))
        py = int(round(float(metadata["center_y"])))
        cv2.circle(preview, (px, py), 6, (0, 0, 255), -1)

    return preview


def crop_with_yolo(
    img_gray: np.ndarray,
    yolo_model,
    conf: float = 0.30,
    iou: float = 0.45,
    imgsz: int = 640,
    padding_x: float = 0.04,
    padding_y: float = 0.04,
    device: Optional[str] = None,
    fixed_ratio: bool = True,
    crop_ratio: float = 0.75,
    crop_scale: float = 1.12,
    output_size: int = 384,
    center_shift_x: float = 0.0,
    center_shift_y: float = 0.0,
    pad_value: int = 0,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Detect the hand/wrist region with YOLO and return a grayscale crop.

    New behaviour for EfficientNetV2:
      1. YOLO detects an approximate hand/wrist box.
      2. The center of that detection is calculated.
      3. A fixed-ratio crop is created around that center.
      4. The crop is letterboxed to output_size x output_size.

    This means YOLO no longer needs to produce a perfect anatomical crop. It only
    needs to find a stable center for the hand/wrist region.

    Set fixed_ratio=False to recover the old behaviour: crop directly from the
    YOLO box plus padding_x/padding_y.
    """
    if img_gray is None:
        return None, _empty_metadata("read_error")

    # Radiographs may be loaded as 2D grayscale, 3D one-channel, BGR, or BGRA.
    # Normalize explicitly to avoid OpenCV errors such as "Invalid number of channels ... scn is 1".
    img_gray = _ensure_gray(img_gray)
    img_bgr = _ensure_bgr(img_gray)

    img_h, img_w = img_gray.shape[:2]

    results = yolo_model.predict(
        source=img_bgr,
        **_prediction_kwargs(conf=conf, iou=iou, imgsz=imgsz, device=device),
    )

    if not results:
        return None, _empty_metadata("no_detection")

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return None, _empty_metadata("no_detection")

    # max_det=1 should already return a single box, but this keeps the function safe.
    best_idx = int(boxes.conf.argmax().cpu().item())
    yolo_x1, yolo_y1, yolo_x2, yolo_y2 = boxes.xyxy[best_idx].cpu().numpy().tolist()
    confidence = float(boxes.conf[best_idx].cpu().item())

    box_w = max(1.0, yolo_x2 - yolo_x1)
    box_h = max(1.0, yolo_y2 - yolo_y1)

    cx = (yolo_x1 + yolo_x2) / 2.0
    cy = (yolo_y1 + yolo_y2) / 2.0

    # Optional center correction. Positive y moves the crop down toward the wrist.
    cx += box_w * float(center_shift_x)
    cy += box_h * float(center_shift_y)

    metadata = {
        "status": "cropped_fixed_ratio" if fixed_ratio else "cropped_yolo_box",
        "confidence": confidence,
        "yolo_x1": yolo_x1,
        "yolo_y1": yolo_y1,
        "yolo_x2": yolo_x2,
        "yolo_y2": yolo_y2,
        "center_x": cx,
        "center_y": cy,
        "crop_ratio": crop_ratio,
        "crop_scale": crop_scale,
        "output_size": output_size,
    }

    if fixed_ratio:
        crop_ratio = float(crop_ratio)
        crop_scale = float(crop_scale)

        if crop_ratio <= 0:
            raise ValueError("yolo_crop_ratio must be > 0. Example: 0.75 or 1.0")
        if crop_scale <= 0:
            raise ValueError("yolo_crop_scale must be > 0. Example: 1.12")

        # crop_ratio = width / height.
        # crop_ratio < 1 gives a vertical crop, usually better for hand+wrist X-rays.
        crop_h = max(box_h, box_w / crop_ratio) * crop_scale
        crop_w = crop_h * crop_ratio

        crop, (crop_x1, crop_y1, crop_x2, crop_y2) = _crop_with_padding_gray(
            img_gray=img_gray,
            cx=cx,
            cy=cy,
            crop_w=crop_w,
            crop_h=crop_h,
            pad_value=pad_value,
        )

        metadata.update({
            "x1": crop_x1,
            "y1": crop_y1,
            "x2": crop_x2,
            "y2": crop_y2,
            "crop_x1": crop_x1,
            "crop_y1": crop_y1,
            "crop_x2": crop_x2,
            "crop_y2": crop_y2,
        })

        if crop.size == 0:
            metadata["status"] = "empty_crop"
            return None, metadata

        crop = _letterbox_gray_to_square(crop, int(output_size), pad_value=pad_value)
        return crop, metadata

    # Old behaviour: crop directly from YOLO box plus padding.
    x1 = yolo_x1 - box_w * padding_x
    x2 = yolo_x2 + box_w * padding_x
    y1 = yolo_y1 - box_h * padding_y
    y2 = yolo_y2 + box_h * padding_y

    x1 = _clip(x1, 0, img_w - 1)
    y1 = _clip(y1, 0, img_h - 1)
    x2 = _clip(x2, 1, img_w)
    y2 = _clip(y2, 1, img_h)

    metadata.update({
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_x2": x2,
        "crop_y2": y2,
    })

    if x2 <= x1 or y2 <= y1:
        metadata["status"] = "invalid_box"
        return None, metadata

    crop = img_gray[y1:y2, x1:x2]

    if crop.size == 0:
        metadata["status"] = "empty_crop"
        return None, metadata

    return crop, metadata
