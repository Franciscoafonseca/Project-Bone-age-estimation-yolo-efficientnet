import cv2
import numpy as np
from utils.yolo_crop import crop_with_yolo

def apply_clahe(img_gray):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)

def center_crop_with_shift(img_gray, crop_percent=0.80, vertical_shift_ratio=0.10):
    """Center crop the image with a vertical shift."""
    height, width = img_gray.shape
    crop_width = int(width * crop_percent)
    crop_height = int(height * crop_percent)
    left = (width - crop_width) // 2
    top = int((height - crop_height) // 2 + crop_height * vertical_shift_ratio)
    right = left + crop_width
    bottom = min(top + crop_height, height)
    return img_gray[top:bottom, left:right]

def crop_black_margins(img_array, threshold=100, black_value=10):
    height, width = img_array.shape
    THRESH_VAL = 252
    AREA_MIN, AREA_MAX = 300, 10000
    LEFT_REGION_FRAC = 0.1
    RIGHT_REGION_FRAC = 0.9
    BOTTOM_REGION_FRAC = 0.20

    left_crop_needed = 0
    right_crop_needed = 0

    _, mask = cv2.threshold(img_array, THRESH_VAL, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, tw, th = cv2.boundingRect(cnt)
        area = tw * th
        if area < AREA_MIN or area > AREA_MAX:
            continue
        if y < BOTTOM_REGION_FRAC * height:
            continue
        if x < LEFT_REGION_FRAC * width:
            left_crop_needed = max(left_crop_needed, x + tw)
        if (x + tw) > RIGHT_REGION_FRAC * width:
            right_crop_needed = max(right_crop_needed, width - x)

    crop_width = max(left_crop_needed, right_crop_needed)
    if crop_width * 2 >= width or crop_width == 0:
        cropped = img_array.copy()
    else:
        cropped = img_array[:, crop_width: width - crop_width]

    height, width = cropped.shape
    height, width = cropped.shape
    row_thresh = min(threshold, height - 1)
    col_thresh = min(threshold, width - 1)

    top = 0
    for i in range(row_thresh):
        if np.mean(cropped[i, :]) < black_value:
            top = i

    bot = height - 1
    for bottom in range(height - 1, height - row_thresh - 1, -1):
        if np.mean(cropped[bottom, :]) < black_value:
            bot = bottom

    left_side = 0
    for left in range(col_thresh):
        if np.mean(cropped[:, left]) < black_value:
            left_side = left

    right_side = width - 1
    for right in range(width - 1, width - col_thresh, -1):
        if np.mean(cropped[:, right]) < black_value:
            right_side = right

    # Safety check: ensure cropped area is valid
    #if top >= bot or left_side >= right_side:
    #    print("⚠️ Invalid crop detected, returning original image")
    #    return img_array.copy()
    
       # Final safety check before return
    if cropped.shape[1] == 0 or cropped.shape[0] == 0:
        print("⚠️ crop_black_margins: Cropped image is empty — returning original")
        return img_array.copy()

    cropped = cropped[top:bot, left_side:right_side]

    MIN_WIDTH_FRAC  = 0.60   # keep at least 70% width
    MIN_HEIGHT_FRAC = 0.60   # keep at least 70% height

    final_h, final_w = cropped.shape
    orig_h, orig_w   = img_array.shape

    if final_w < MIN_WIDTH_FRAC * orig_w or final_h < MIN_HEIGHT_FRAC * orig_h:
        return img_array.copy()

    return cropped

def symmetric_crop_by_corners(img_gray,
                              thr=250,
                              bot_frac=0.85,
                              win_frac=0.20,
                              pad=10,
                              max_pct=0.10):
    h, w = img_gray.shape
    top = int(bot_frac * h)
    leftW = int(win_frac * w)

    left_strip = img_gray[top:, :leftW]
    xs_left = np.where(left_strip >= thr)[1]
    l_need = (xs_left.max() + pad) if xs_left.size else 0

    right_strip = img_gray[top:, w - leftW:]
    xs_right = np.where(right_strip >= thr)[1]
    r_need = (leftW - 1 - xs_right.min() + pad) if xs_right.size else 0

    crop_w = max(l_need, r_need)
    if crop_w == 0 or crop_w > int(max_pct * w):
        return img_gray, 0

    left_bound = crop_w
    right_bound = w - crop_w

    if left_bound >= right_bound:
        print("⚠️ Invalid symmetric crop — returning original image")
        return img_gray.copy(), 0

    cropped = img_gray[:, crop_w: w - crop_w]
    return cropped, crop_w

def crop_symmetric(img_gray):
    """Wrapper for symmetric_crop_by_corners to return just the cropped image."""
    cropped, _ = symmetric_crop_by_corners(img_gray)
    return cropped

def _fallback_crop(img, crop_method):
    """Classical fallback crop used when YOLO does not detect a valid hand/wrist box."""
    if crop_method == 'crop_black':
        return crop_black_margins(img)
    if crop_method == 'crop_symmetric':
        return crop_symmetric(img)
    if crop_method == 'both':
        cropped = crop_black_margins(img)
        return crop_symmetric(cropped)
    return img


def _should_return_metadata(return_metadata, processed, metadata):
    if return_metadata:
        return processed, metadata
    return processed


def preprocess_image(
    img,
    mode,
    crop_method=None,
    yolo_model=None,
    yolo_config=None,
    return_metadata=False,
):
    """
    Preprocess image according to mode and crop_method.

    mode options:
        'original'                     : no processing
        'center_crop'                  : center crop with vertical shift
        'clahe'                        : CLAHE only
        'crop_only'                    : crop only (requires crop_method)
        'clahe_crop'                   : CLAHE + crop (requires crop_method)
        'center_crop_then_crop'        : center crop + crop (requires crop_method)
        'clahe_center_crop_then_crop'  : CLAHE + center crop + crop (requires crop_method)
        'yolo_crop'                    : YOLO hand/wrist crop only
        'yolo_clahe_crop'              : YOLO hand/wrist crop + CLAHE

    crop_method options for classical modes/fallback:
        'crop_black'                   : crop_black_margins
        'crop_symmetric'               : symmetric_crop_by_corners
        'both'                         : crop_black then symmetric_crop
        None                           : no crop

    YOLO options are read from yolo_config:
        yolo_conf, yolo_iou, yolo_imgsz, yolo_padding_x, yolo_padding_y,
        yolo_fixed_ratio, yolo_crop_ratio, yolo_crop_scale, yolo_output_size,
        yolo_center_shift_x, yolo_center_shift_y, yolo_pad_value,
        yolo_device, yolo_fallback.
    """
    metadata = {
        "status": "not_yolo",
        "confidence": "",
        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",
    }

    if mode == 'original':
        return _should_return_metadata(return_metadata, img, metadata)

    if mode == 'center_crop':
        return _should_return_metadata(return_metadata, center_crop_with_shift(img), metadata)

    if mode == 'clahe':
        return _should_return_metadata(return_metadata, apply_clahe(img), metadata)

    if mode == 'crop_only':
        if crop_method in ('crop_black', 'crop_symmetric', 'both'):
            return _should_return_metadata(return_metadata, _fallback_crop(img, crop_method), metadata)
        raise ValueError("Invalid crop_method for crop_only mode")

    if mode == 'clahe_crop':
        img_clahe = apply_clahe(img)
        if crop_method in ('crop_black', 'crop_symmetric', 'both'):
            return _should_return_metadata(return_metadata, _fallback_crop(img_clahe, crop_method), metadata)
        raise ValueError("Invalid crop_method for clahe_crop mode")

    if mode == 'center_crop_then_crop':
        img_cropped_center = center_crop_with_shift(img)
        if crop_method in ('crop_black', 'crop_symmetric', 'both'):
            return _should_return_metadata(return_metadata, _fallback_crop(img_cropped_center, crop_method), metadata)
        raise ValueError("Invalid crop_method for center_crop_then_crop mode")

    if mode == 'clahe_center_crop_then_crop':
        img_clahe = apply_clahe(img)
        img_cropped_center = center_crop_with_shift(img_clahe)
        if crop_method in ('crop_black', 'crop_symmetric', 'both'):
            return _should_return_metadata(return_metadata, _fallback_crop(img_cropped_center, crop_method), metadata)
        raise ValueError("Invalid crop_method for clahe_center_crop_then_crop mode")

    if mode in ('yolo_crop', 'yolo_clahe_crop'):
        if yolo_model is None:
            raise ValueError(
                f"Mode '{mode}' requires a loaded YOLO model. "
                "Load it with utils.yolo_crop.load_yolo_model()."
            )

        yolo_config = yolo_config or {}
        crop, metadata = crop_with_yolo(
            img_gray=img,
            yolo_model=yolo_model,
            conf=float(yolo_config.get("yolo_conf", 0.30)),
            iou=float(yolo_config.get("yolo_iou", 0.45)),
            imgsz=int(yolo_config.get("yolo_imgsz", 640)),
            padding_x=float(yolo_config.get("yolo_padding_x", 0.04)),
            padding_y=float(yolo_config.get("yolo_padding_y", 0.04)),
            device=yolo_config.get("yolo_device", None),
            fixed_ratio=bool(yolo_config.get("yolo_fixed_ratio", True)),
            crop_ratio=float(yolo_config.get("yolo_crop_ratio", 0.75)),
            crop_scale=float(yolo_config.get("yolo_crop_scale", 1.12)),
            output_size=int(yolo_config.get("yolo_output_size", 384)),
            center_shift_x=float(yolo_config.get("yolo_center_shift_x", 0.0)),
            center_shift_y=float(yolo_config.get("yolo_center_shift_y", 0.0)),
            pad_value=int(yolo_config.get("yolo_pad_value", 0)),
        )

        if crop is None:
            fallback = yolo_config.get("yolo_fallback", "original")
            metadata["status"] = f'{metadata["status"]}_fallback_{fallback}'

            if fallback == "original":
                crop = img.copy()
            elif fallback == "classical_crop":
                crop = _fallback_crop(img, crop_method or 'both')
            elif fallback == "skip":
                raise RuntimeError(f"YOLO failed and yolo_fallback='skip': {metadata}")
            else:
                raise ValueError("Invalid yolo_fallback. Use 'original', 'classical_crop', or 'skip'.")

        if mode == 'yolo_clahe_crop':
            crop = apply_clahe(crop)

        return _should_return_metadata(return_metadata, crop, metadata)

    raise ValueError(f"Unknown mode: {mode}")

