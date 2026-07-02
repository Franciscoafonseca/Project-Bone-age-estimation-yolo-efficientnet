"""Grad-CAM utilities for visualizing what the model uses in hand/wrist X-rays.

The implementation is intentionally lightweight and works with the torchvision
models used in this project, including EfficientNetV2, ResNet, DenseNet and VGG.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

try:
    import wandb
except Exception:  # pragma: no cover - wandb is optional when running locally
    wandb = None


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the real model if it is wrapped with DistributedDataParallel."""
    return model.module if hasattr(model, "module") else model


def get_default_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Pick a sensible final convolutional layer for common architectures."""
    model = unwrap_model(model)

    # EfficientNet, VGG, DenseNet-like torchvision models
    if hasattr(model, "features"):
        return model.features[-1]

    # ResNet-like torchvision models
    if hasattr(model, "layer4"):
        return model.layer4[-1]

    raise ValueError(
        "Could not infer Grad-CAM target layer for this model. "
        "Add a custom target layer in utils/gradcam.py."
    )


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_module, _inputs, output):
            self.activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, input_tensor: torch.Tensor, task: str = "regression"):
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)

        if output.ndim == 2 and output.size(1) == 1:
            output_scalar = output.squeeze(1)
        else:
            output_scalar = output

        if task == "regression":
            score = output_scalar.mean()
            pred_value = output_scalar.detach().view(-1)[0].item()
        else:
            class_idx = int(torch.argmax(output_scalar, dim=1).item())
            score = output_scalar[:, class_idx].sum()
            pred_value = class_idx

        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activations = self.activations[0]
        gradients = self.gradients[0]

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_value


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization and convert CxHxW tensor to uint8 RGB image."""
    tensor = tensor.detach().cpu()
    tensor = tensor * IMAGENET_STD + IMAGENET_MEAN
    tensor = tensor.clamp(0, 1)
    image = tensor.permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    """Overlay a Grad-CAM heatmap over an RGB image."""
    h, w = image_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay


def generate_gradcam_heatmaps(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    config: dict,
    run=None,
    split: str = "test",
    output_dir: str | os.PathLike = "results/heatmaps",
    max_images: int = 16,
) -> List[str]:
    """Generate and optionally log Grad-CAM heatmaps for a loader split."""
    if max_images <= 0:
        return []

    model.eval()
    model_for_hooks = unwrap_model(model)
    target_layer = get_default_target_layer(model_for_hooks)
    gradcam = GradCAM(model_for_hooks, target_layer)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    task = config.get("model_task", "regression")

    try:
        for batch in data_loader:
            inputs, labels, img_ids, sex_batch, treatment_batch = batch

            for i in range(inputs.size(0)):
                if len(saved_paths) >= max_images:
                    break

                input_tensor = inputs[i:i + 1].to(device)
                cam, pred_value = gradcam(input_tensor, task=task)

                image_rgb = tensor_to_uint8_image(inputs[i])
                overlay = overlay_cam(image_rgb, cam)

                img_id = int(img_ids[i].item()) if hasattr(img_ids[i], "item") else img_ids[i]
                label_value = labels[i].item() if hasattr(labels[i], "item") else labels[i]
                treatment_value = treatment_batch[i].item() if hasattr(treatment_batch[i], "item") else treatment_batch[i]

                filename = f"{split}_gradcam_{len(saved_paths) + 1:03d}_img_{img_id}.png"
                save_path = output_dir / filename
                cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                saved_paths.append(str(save_path))

                if run is not None and wandb is not None:
                    caption = (
                        f"{split} | image={img_id} | label={label_value:.3f} | "
                        f"prediction={pred_value:.3f} | treatment={treatment_value}"
                    )
                    run.log({f"heatmaps/{split}": wandb.Image(str(save_path), caption=caption)})

            if len(saved_paths) >= max_images:
                break
    finally:
        gradcam.remove_hooks()

    return saved_paths
