import cv2
import os
import pandas as pd
import PIL

from torch.utils.data import Dataset
from datetime import datetime

HAAVIKKO_STAGES = [
    "O",
    "Ci",
    "Cco",
    "Cr 1/2",
    "Cr 3/4",
    "Crc",
    "Ri",
    "R 1/4",
    "R 1/2",
    "R 3/4",
    "Rc",
    "Ac",
]

KULLMAN_STAGES = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
]

DEMIRJIAN_STAGES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
]

STAGE_ENCODERS = {
    "H": {stage: i for i, stage in enumerate(HAAVIKKO_STAGES)},
    "K": {stage: i for i, stage in enumerate(KULLMAN_STAGES)},
    "D": {stage: i for i, stage in enumerate(DEMIRJIAN_STAGES)},
}

STAGE_DECODERS = {k: {v: s for s, v in enc.items()} for k, enc in STAGE_ENCODERS.items()}

class TeethDataset(Dataset):
    def __init__(self, ann_file="data/FCT2025_amostra_VF.xlsx", img_dir="data/preprocessed/clahe_center_crop_then_crop/", 
            id_col= " Nº ", sex_col="Sexo", birth_date_col="Data de Nascimento", opg_date_col="Data Rx Panorâmico",
            age_col="Idade (anos) ao Rx", treatment_col="Com/Sem tratamento", transform=None, target_transform=None, 
            config=None):

        self.use_with_treatment = config.get("use_with_treatment", True)
        self.model_task = config.get("model_task", "binary")  # "binary", "multiclass", or "regression"
        self.age_threshold = config.get("age_threshold", 16) # used specifically for binary classification

        self.id_col = id_col
        self.sex_col = sex_col
        self.birth_date_col = birth_date_col
        self.opg_date_col = opg_date_col
        self.age_col = age_col
        self.treatment_col = treatment_col

        self.without_treatment_img_dir = os.path.join(img_dir, "without_treatment")
        self.supported_file_types = ['.png', '.jpg', '.PNG', 'jpeg', '.JPEG', '.JPG']
        self.image_files = [(f,0) for f in sorted(os.listdir(self.without_treatment_img_dir)) if any(f.endswith(ext) for ext in self.supported_file_types)]

        if self.use_with_treatment:
            self.with_treatment_img_dir = os.path.join(img_dir, "with_treatment")
            self.with_treatment_files= [(f,1) for f in sorted(os.listdir(self.with_treatment_img_dir)) if any(f.endswith(ext) for ext in self.supported_file_types)]
            self.image_files.extend(self.with_treatment_files)

        self.img_labels = pd.read_excel(ann_file)

        # make sure the ID column is numeric and use it as index
        self.img_labels[self.id_col] = self.img_labels[self.id_col].astype(int)
        self.img_labels = self.img_labels.set_index(self.id_col)

        valid_ids = set(self.img_labels.index.astype(int))
        self.image_files = [(f, t) for (f, t) in self.image_files if int(f.split('.')[0]) in valid_ids]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_number = int(self.image_files[idx][0].split('.')[0])

        if self.image_files[idx][1]==0:
            img_dir=self.without_treatment_img_dir
        else:
            img_dir=self.with_treatment_img_dir

        image = cv2.imread(os.path.join(img_dir,self.image_files[idx][0]), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = PIL.Image.fromarray(image)

        row = self.img_labels.loc[img_number]

        if self.model_task == "binary":
            label = (row[self.age_col] >= self.age_threshold)
        elif self.model_task == "regression":
            label = date_diff_years(row[self.birth_date_col], row[self.opg_date_col])
        elif self.model_task == "multiclass":
            #classes 10/12/14/16/18/21
            age  = row[self.age_col]
            label = age_to_class(age)

        sex = row[self.sex_col]
        if isinstance(sex, str):
            sex = sex.strip()

        treatment = row[self.treatment_col]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, img_number, sex, treatment
    
class StagesDataset(Dataset):
    def __init__(self, ann_file="data/FCT2025_amostra_VF.xlsx", img_dir="data/stages_images/", 
            id_col= " Nº ", sex_col="Sexo", birth_date_col="Data de Nascimento", opg_date_col="Data Rx Panorâmico",
            treatment_col="Com/Sem tratamento", transform=None, target_transform=None, config=None):

        self.method = config.get("stage_method", "H")  # "H", "K", or "D"

        self.id_col = id_col
        self.sex_col = sex_col
        self.birth_date_col = birth_date_col
        self.opg_date_col = opg_date_col
        self.treatment_col = treatment_col
        self.img_dir = img_dir
        self.stage_jaw_selection = config.get("stage_jaw_selection", "both")  # "upper", "lower", or "both"

        self.supported_file_types = ['.png', '.jpg', '.PNG', 'jpeg', '.JPEG', '.JPG']

        if self.stage_jaw_selection == "upper":
            self.image_files = [f for f in sorted(os.listdir(self.img_dir)) if any(f.endswith(ext) for ext in self.supported_file_types) and int(f.split('.')[1]) in ['18','28']]
        elif self.stage_jaw_selection == "lower":
            self.image_files = [f for f in sorted(os.listdir(self.img_dir)) if any(f.endswith(ext) for ext in self.supported_file_types) and int(f.split('.')[1]) in ['38','48']]
        else:  # both
            self.image_files = [f for f in sorted(os.listdir(self.img_dir)) if any(f.endswith(ext) for ext in self.supported_file_types)]

        self.img_labels = pd.read_excel(ann_file, sheet_name=0)
        self.teeth_stages = pd.read_excel(ann_file, sheet_name=1)

        # make sure the ID column is numeric and use it as index
        self.img_labels[self.id_col] = self.img_labels[self.id_col].astype(int)
        self.img_labels = self.img_labels.set_index(self.id_col)
        self.teeth_stages[self.id_col] = self.teeth_stages[self.id_col].astype(int)
        self.teeth_stages = self.teeth_stages.set_index(self.id_col)

        valid_ids = set(self.img_labels.index.astype(int))
        self.image_files = [f for f in self.image_files if int(f.split('.')[0]) in valid_ids and self.exists_tooth(int(f.split('.')[0]), f.split('.')[1])]

        self.transform = transform
        self.target_transform = target_transform

    def get_age(self, img_number):
        row = self.img_labels.loc[img_number]
        return date_diff_years(row[self.birth_date_col], row[self.opg_date_col])
    
    def exists_tooth(self, img_number, tooth_number):
        val = self.teeth_stages.loc[img_number, self.method + '_' + tooth_number]
        return not pd.isna(val) and val != '-'

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_number = int(self.image_files[idx].split('.')[0])
        tooth_number = self.image_files[idx].split('.')[1]

        image = cv2.imread(os.path.join(self.img_dir, self.image_files[idx]), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = PIL.Image.fromarray(image)

        if self.stage_jaw_selection == "both" and tooth_number in ['18','28']:
            image = image.rotate(180, expand=True)

        row = self.img_labels.loc[img_number]

        raw_stage = self.teeth_stages.loc[img_number, self.method + '_' + tooth_number]

        try:
            stage = STAGE_ENCODERS[self.method][raw_stage]
        except KeyError:
            print(f"Unknown stage '{raw_stage}' for method '{self.method}' (img {img_number}, tooth {tooth_number})")      

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
            stage = self.target_transform(stage)

        return image, stage, img_number, sex, treatment

def _clean_date(d):
    d = str(d).strip()
    d = d.rstrip("/")        
    d = d.split()[0]         
    return d
    
def date_diff_years(date1, date2, min_age=5, max_age=26, time_format="%Y/%m/%d", normalize=True) -> float:
    if isinstance(date1, str):
        date1 = datetime.strptime(_clean_date(date1), time_format)
    if isinstance(date2, str):
        date2 = datetime.strptime(_clean_date(date2), time_format)
    
    # bounds the age difference to a maximum of max_age years
    days = min(abs((date2 - date1).days), max_age * 365.25)
    years = days / 365.25
    if normalize:
        return (years-min_age) / (max_age-min_age)  # normalize to [0, 1]
    else:
        return years

def transform_to_real_age(pred, min_age=5, max_age=26):
    return pred*(max_age-min_age)+min_age

def age_to_class(age, bins=[10, 12, 14, 16, 18, 21]):
    for i, b in enumerate(bins):
        if age < b and age >= (bins[i-1] if i > 0 else 0):  #less than upper bound and greater than or equal to lower bound
            return i
    return len(bins)
