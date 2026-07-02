import cv2
import os
import pandas as pd
import PIL
import torch
import argparse

from data.resize import model_resize_size
from models.models import get_model
from utils.preprocess import preprocess_image
from utils.yolo_crop import load_yolo_model


from data.dataset import date_diff_years, age_to_class
from models.train import export_predictions_to_excel

from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torch import nn

class ExternalDataset(Dataset):
    def __init__(self, ann_file="data/FCT2025_amostra_VF.xlsx", img_dir="data/preprocessed/clahe_center_crop_then_crop/", 
            id_col= " Nº ", sex_col="Sexo", birth_date_col="Data de Nascimento", opg_date_col="Data Rx Panorâmico",
            treatment_col="Com/Sem tratamento", transform=None, target_transform=None, config=None):

        self.model_task = config.get("model_task", "binary")  # "binary", "multiclass", or "regression"
        self.age_threshold = config.get("age_threshold", 16) # used specifically for binary classification
        self.config = config
        self.image_processing = config.get("image_processing", "yolo_crop")
        self.crop_method = config.get("crop_method", "both")
        self.yolo_model = None

        if self.image_processing in ("yolo_crop", "yolo_clahe_crop"):
            self.yolo_model = load_yolo_model(config.get("yolo_model_path", "models/yolo_hand/best.pt"))

        self.id_col = id_col
        self.sex_col = sex_col
        self.birth_date_col = birth_date_col
        self.opg_date_col = opg_date_col
        self.treatment_col = treatment_col

        self.img_dir = img_dir
        self.supported_file_types = ['.png', '.jpg', '.PNG', 'jpeg', '.JPEG', '.JPG']
        self.image_files = [f for f in sorted(os.listdir(self.img_dir)) if any(f.endswith(ext) for ext in self.supported_file_types)]

        self.img_labels = pd.read_excel(ann_file)

        # make sure the ID column is numeric and use it as index
        self.img_labels[self.id_col] = self.img_labels[self.id_col].astype(int)
        self.img_labels = self.img_labels.set_index(self.id_col)

        valid_ids = set(self.img_labels.index.astype(int))
        self.image_files = [f for f in self.image_files if int(f.split('.')[0]) in valid_ids]

        print(valid_ids.difference(set(int(f.split('.')[0]) for f in self.image_files)))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_number = int(self.image_files[idx].split('.')[0])

        image = cv2.imread(os.path.join(self.img_dir, self.image_files[idx]), cv2.IMREAD_GRAYSCALE)
        processed_img = preprocess_image(
            image,
            mode=self.image_processing,
            crop_method=self.crop_method,
            yolo_model=self.yolo_model,
            yolo_config=self.config,
        )
        image = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        image = PIL.Image.fromarray(image)

        row = self.img_labels.loc[img_number]

        try:
            if self.model_task in ["binary", "multiclass"]:
                age = date_diff_years(row[self.birth_date_col], row[self.opg_date_col], time_format="%d/%m/%Y", normalize=False)
            elif self.model_task == "regression":
                age = date_diff_years(row[self.birth_date_col], row[self.opg_date_col], time_format="%d/%m/%Y", normalize=True)
        except:
            print(f"Error parsing dates for ID {img_number}. Check date format.")
            return self.transform(image), -1, img_number, "M", 0
        if self.model_task == "binary":
            label = (age >= self.age_threshold)
        elif self.model_task == "regression":
            label = age
        elif self.model_task == "multiclass":
            #classes 10/12/14/16/18/21
            label = age_to_class(age)

        sex = row[self.sex_col]
        if isinstance(sex, str):
            sex = sex.strip()

        if self.treatment_col is not None:
            treatment = row[self.treatment_col]
        else:
            treatment = 0  # default value if no treatment column

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, img_number, sex, treatment
    
def test(config):
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

    img_dir="prediction/images_to_predict/"
    
    img_transform = v2.Compose([
        v2.Resize(model_resize_size(config["model_name"])),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dataset = ExternalDataset(
        ann_file = "prediction/annotations.xlsx",
        img_dir = img_dir,
        id_col = "ID",
        sex_col = "sex",
        birth_date_col = "DOB",
        opg_date_col = "DOA",
        treatment_col = None,
        transform=img_transform,
        config=config
    )
    
    if config['model_task'] == "binary":
        criterion = nn.BCEWithLogitsLoss()
    elif config['model_task'] in ["multiclass", "stages"]:
        criterion = nn.CrossEntropyLoss()
    elif config['model_task'] == "regression":
        criterion = nn.MSELoss()

    dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    export_predictions_to_excel(model, dataloader, criterion, device, config, rank=0, world_size=1, output_path=f"prediction/results/{config['model_name']}", split=f"external_test_{config['model_name']}_{config['age_threshold']}")

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
    test(config)