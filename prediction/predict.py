import torch
import argparse
import os
import cv2 
import PIL

from data.resize import model_resize_size
from models.models import get_model
from utils.preprocess import preprocess_image
from utils.yolo_crop import load_yolo_model

from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, imgs, image_files, transform):
        self.imgs = imgs
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        image = self.imgs[idx]
        image = self.transform(image)
        return image, self.image_files[idx]


def predict(config):

    model = get_model(config)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(config["model_path"], map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)

    image_files = [
        f for f in sorted(os.listdir("prediction/images_to_predict"))
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ]
    processed_imgs = []

    image_processing = config.get("image_processing", "yolo_crop")
    crop_method = config.get("crop_method", "both")
    yolo_model = None

    if image_processing in ("yolo_crop", "yolo_clahe_crop"):
        yolo_model = load_yolo_model(config.get("yolo_model_path", "models/yolo_hand/best.pt"))

    for fname in image_files:
        image_path = os.path.join("prediction/images_to_predict", fname)
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        processed_img = preprocess_image(
            img_gray,
            mode=image_processing,
            crop_method=crop_method,
            yolo_model=yolo_model,
            yolo_config=config,
        )

        processed_image = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        processed_img = PIL.Image.fromarray(processed_image)
        processed_imgs.append(processed_img)

    img_transform = v2.Compose([
        v2.Resize(model_resize_size(config["model_name"])),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    dataset = ImageDataset(imgs=processed_imgs, image_files=image_files, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    model.eval()

    all_preds = []
    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(device)
            outputs = model(images)
            if config['model_task'] == "binary":
                if outputs.ndim == 2 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                probs = torch.sigmoid(outputs)
                preds = (probs>=0.5).long()
            elif config['model_task'] == "multiclass":
                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)
            elif config['model_task'] == "regression":
                outputs = model(images)
                if outputs.ndim == 2 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                preds = outputs.detach()

            for file, pred in zip(filenames, preds):
                print(f"{file}: {pred.item()}")
                all_preds.append((file, pred.item()))

    return all_preds

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True, help="Name of the model architecture.", choices=["resnet18","resnet50","resnet152","vgg16","vgg19","efficientnet_b5","efficientnet_v2_m","densenet121","densenet169","xception"])
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for prediction.")
    parser.add_argument("-t", "--model_task", type=str, default="binary", help="Type of prediction task.", choices=["binary", "multiclass", "regression"])
    parser.add_argument("-a", "--age_threshold", type=int, default=16, help="Age threshold for prediction.", choices=[10,12,14,16,18,21])
    parser.add_argument("--image_processing", type=str, default="yolo_crop", choices=["original", "center_crop", "clahe", "crop_only", "clahe_crop", "center_crop_then_crop", "clahe_center_crop_then_crop", "yolo_crop", "yolo_clahe_crop"])
    parser.add_argument("--yolo_model_path", type=str, default="models/yolo_hand/best.pt")
    parser.add_argument("--yolo_conf", type=float, default=0.30)
    parser.add_argument("--yolo_iou", type=float, default=0.45)
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument("--yolo_padding_x", type=float, default=0.04)
    parser.add_argument("--yolo_padding_y", type=float, default=0.04)
    parser.add_argument("--yolo_fallback", type=str, default="classical_crop", choices=["original", "classical_crop", "skip"])
    parser.add_argument("--yolo_device", type=str, default="")
    args = parser.parse_args()

    model_dir = "prediction/prediction_models/"
    available_models = [
        model_file for model_file in os.listdir(model_dir)
        if model_file != ".model_placeholder"
    ]
    print("Available models for prediction:")
    for i, model_file in enumerate(available_models):
        print(f"{i+1}. - {model_file}")

    model_index = int(input("Select the model number to use for prediction: ")) - 1

    if model_index < 0 or model_index >= len(available_models):
        raise ValueError(f"Number must be between 1 and {len(available_models)}. You entered {model_index + 1}.")
    
    selected_model_file = available_models[model_index]

    config = vars(args)
    config["model_path"] = os.path.join(model_dir.rstrip("/"), selected_model_file)
    config["pretrained"] = False
    config["use_regression_sigmoid"] = False
    predict(config)