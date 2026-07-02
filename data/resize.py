import os
import cv2
from tqdm import tqdm

# --- MODEL INPUT SIZES ---
def model_resize_size(model_name):
    """Returns the input size for the given model."""
    sizes = {
        "resnet18": (224, 224),
        "resnet50": (224, 224),
        "resnet152": (224, 224),
        "vgg16": (224, 224),
        "vgg19": (224, 224),
        "efficientnet_b5": (456, 456),
        "efficientnet_v2_m": (384, 384),
        "densenet121": (224, 224),
        "densenet169": (224, 224),
        "xception": (299, 299)
    }
    return sizes.get(model_name, (224, 224))