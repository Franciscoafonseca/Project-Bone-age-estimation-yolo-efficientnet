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

STAGE_DECODERS = {
    k: {v: s for s, v in enc.items()} for k, enc in STAGE_ENCODERS.items()
}


class TeethDataset(Dataset):
    def __init__(
        self,
        ann_file="data/FCT2025_Final.xlsx",
        img_dir="data/preprocessed/yolo_crop/",
        id_col=None,
        sex_col="Sexo",
        birth_date_col="Data de Nascimento",
        opg_date_col="Data de realização da radiografia",
        age_col="Idade na data da radiografia (anos)",
        treatment_col=None,
        transform=None,
        target_transform=None,
        config=None,
    ):
        self.config = config or {}

        self.model_task = self.config.get("model_task", "regression")
        self.age_threshold = self.config.get("age_threshold", 16)

        self.id_col = self.config.get("id_col", id_col)
        self.sex_col = self.config.get("sex_col", sex_col)
        self.birth_date_col = self.config.get("birth_date_col", birth_date_col)
        self.opg_date_col = self.config.get("opg_date_col", opg_date_col)
        self.age_col = self.config.get("age_col", age_col)
        self.target_col = self.config.get("target_col", None)
        self.target_unit = self.config.get("target_unit", "months")
        self.treatment_col = self.config.get("treatment_col", treatment_col)

        self.min_age = float(self.config.get("min_age", 5))
        self.max_age = float(self.config.get("max_age", 26))
        self.normalize_age = bool(self.config.get("normalize_age", True))

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.supported_file_types = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".PNG",
            ".JPG",
            ".JPEG",
            ".BMP",
            ".TIF",
            ".TIFF",
        )

        self.image_files = self._find_images_recursive(self.img_dir)

        if not self.image_files:
            raise RuntimeError(f"No image files found in {self.img_dir}")

        self.img_labels = pd.read_excel(
            ann_file, sheet_name=self.config.get("excel_sheet", 0)
        )

        if self.id_col is not None and self.id_col in self.img_labels.columns:
            self.img_labels[self.id_col] = self.img_labels[self.id_col].astype(int)
            self.img_labels = self.img_labels.set_index(self.id_col)
        else:
            # Row-order mapping:
            # 1.jpg or 1.png -> first Excel row
            # 2.jpg or 2.png -> second Excel row
            # ...
            self.img_labels = self.img_labels.copy()
            self.img_labels["__image_id"] = range(1, len(self.img_labels) + 1)
            self.img_labels = self.img_labels.set_index("__image_id")

        valid_ids = set(self.img_labels.index.astype(int))

        filtered = []
        missing_in_excel = []

        for path in self.image_files:
            img_id = self._image_id_from_path(path)
            if img_id in valid_ids:
                filtered.append(path)
            else:
                missing_in_excel.append(path)

        self.image_files = filtered

        if not self.image_files:
            raise RuntimeError(
                "No image files matched the Excel rows. "
                "Check if images are named 1.jpg, 2.jpg, ... "
                "and if the Excel has the same row order."
            )

        print(f"Dataset loaded: {len(self.image_files)} images matched with Excel.")
        print(f"Image folder: {self.img_dir}")
        print(f"Target column: {self.target_col}")

    def _find_images_recursive(self, folder):
        files = []

        for root, _, filenames in os.walk(folder):
            for fname in filenames:
                if fname.endswith(self.supported_file_types):
                    files.append(os.path.join(root, fname))

        def sort_key(path):
            try:
                return self._image_id_from_path(path)
            except Exception:
                return os.path.basename(path)

        return sorted(files, key=sort_key)

    def _image_id_from_path(self, path):
        stem = os.path.splitext(os.path.basename(path))[0]
        digits = "".join(filter(str.isdigit, stem))

        if digits == "":
            raise ValueError(f"Could not extract numeric ID from filename: {path}")

        return int(digits)

    def __len__(self):
        return len(self.image_files)

    def _normalize_years(self, years):
        years = float(years)

        if self.normalize_age:
            years = min(max(years, self.min_age), self.max_age)
            return (years - self.min_age) / (self.max_age - self.min_age)

        return years

    def _target_from_row(self, row):
        if self.target_col is not None and self.target_col in row.index:
            value = row[self.target_col]

            if pd.isna(value):
                return float("nan")

            value = float(value)

            if self.target_unit == "months":
                years = value / 12.0
            else:
                years = value

            return self._normalize_years(years)

        if self.birth_date_col in row.index and self.opg_date_col in row.index:
            return date_diff_years(
                row[self.birth_date_col],
                row[self.opg_date_col],
                min_age=self.min_age,
                max_age=self.max_age,
                normalize=self.normalize_age,
            )

        if self.age_col in row.index:
            years = float(row[self.age_col])
            return self._normalize_years(years)

        raise ValueError(
            "Could not find target age. "
            "Check target_col, date columns, or age_col in config.yaml."
        )

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        img_number = self._image_id_from_path(image_path)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = PIL.Image.fromarray(image)

        row = self.img_labels.loc[img_number]

        target_age = self._target_from_row(row)

        if self.model_task == "regression":
            label = target_age

        elif self.model_task == "binary":
            if self.normalize_age:
                real_age = target_age * (self.max_age - self.min_age) + self.min_age
            else:
                real_age = target_age

            label = real_age >= self.age_threshold

        elif self.model_task == "multiclass":
            if self.normalize_age:
                real_age = target_age * (self.max_age - self.min_age) + self.min_age
            else:
                real_age = target_age

            label = age_to_class(real_age)

        else:
            label = target_age

        if self.sex_col in row.index:
            sex = row[self.sex_col]
            if isinstance(sex, str):
                sex = sex.strip()
        else:
            sex = "unknown"

        if self.treatment_col is not None and self.treatment_col in row.index:
            treatment = row[self.treatment_col]
            try:
                treatment = int(treatment)
            except Exception:
                treatment = 0
        else:
            treatment = 0

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label, img_number, sex, treatment


class StagesDataset(Dataset):
    def __init__(
        self,
        ann_file="data/FCT2025_amostra_VF.xlsx",
        img_dir="data/stages_images/",
        id_col=" Nº ",
        sex_col="Sexo",
        birth_date_col="Data de Nascimento",
        opg_date_col="Data Rx Panorâmico",
        treatment_col="Com/Sem tratamento",
        transform=None,
        target_transform=None,
        config=None,
    ):

        self.method = config.get("stage_method", "H")  # "H", "K", or "D"

        self.id_col = id_col
        self.sex_col = sex_col
        self.birth_date_col = birth_date_col
        self.opg_date_col = opg_date_col
        self.treatment_col = treatment_col
        self.img_dir = img_dir
        self.stage_jaw_selection = config.get(
            "stage_jaw_selection", "both"
        )  # "upper", "lower", or "both"

        self.supported_file_types = [".png", ".jpg", ".PNG", "jpeg", ".JPEG", ".JPG"]

        if self.stage_jaw_selection == "upper":
            self.image_files = [
                f
                for f in sorted(os.listdir(self.img_dir))
                if any(f.endswith(ext) for ext in self.supported_file_types)
                and int(f.split(".")[1]) in ["18", "28"]
            ]
        elif self.stage_jaw_selection == "lower":
            self.image_files = [
                f
                for f in sorted(os.listdir(self.img_dir))
                if any(f.endswith(ext) for ext in self.supported_file_types)
                and int(f.split(".")[1]) in ["38", "48"]
            ]
        else:  # both
            self.image_files = [
                f
                for f in sorted(os.listdir(self.img_dir))
                if any(f.endswith(ext) for ext in self.supported_file_types)
            ]

        self.img_labels = pd.read_excel(ann_file, sheet_name=0)
        self.teeth_stages = pd.read_excel(ann_file, sheet_name=1)

        # make sure the ID column is numeric and use it as index
        self.img_labels[self.id_col] = self.img_labels[self.id_col].astype(int)
        self.img_labels = self.img_labels.set_index(self.id_col)
        self.teeth_stages[self.id_col] = self.teeth_stages[self.id_col].astype(int)
        self.teeth_stages = self.teeth_stages.set_index(self.id_col)

        valid_ids = set(self.img_labels.index.astype(int))
        self.image_files = [
            f
            for f in self.image_files
            if int(f.split(".")[0]) in valid_ids
            and self.exists_tooth(int(f.split(".")[0]), f.split(".")[1])
        ]

        self.transform = transform
        self.target_transform = target_transform

    def get_age(self, img_number):
        row = self.img_labels.loc[img_number]
        return date_diff_years(row[self.birth_date_col], row[self.opg_date_col])

    def exists_tooth(self, img_number, tooth_number):
        val = self.teeth_stages.loc[img_number, self.method + "_" + tooth_number]
        return not pd.isna(val) and val != "-"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_number = int(self.image_files[idx].split(".")[0])
        tooth_number = self.image_files[idx].split(".")[1]

        image = cv2.imread(
            os.path.join(self.img_dir, self.image_files[idx]), cv2.IMREAD_GRAYSCALE
        )
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = PIL.Image.fromarray(image)

        if self.stage_jaw_selection == "both" and tooth_number in ["18", "28"]:
            image = image.rotate(180, expand=True)

        row = self.img_labels.loc[img_number]

        raw_stage = self.teeth_stages.loc[img_number, self.method + "_" + tooth_number]

        try:
            stage = STAGE_ENCODERS[self.method][raw_stage]
        except KeyError:
            print(
                f"Unknown stage '{raw_stage}' for method '{self.method}' (img {img_number}, tooth {tooth_number})"
            )

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


def date_diff_years(
    date1, date2, min_age=5, max_age=26, time_format="%Y/%m/%d", normalize=True
) -> float:
    if isinstance(date1, str):
        date1 = datetime.strptime(_clean_date(date1), time_format)
    if isinstance(date2, str):
        date2 = datetime.strptime(_clean_date(date2), time_format)

    # bounds the age difference to a maximum of max_age years
    days = min(abs((date2 - date1).days), max_age * 365.25)
    years = days / 365.25
    if normalize:
        return (years - min_age) / (max_age - min_age)  # normalize to [0, 1]
    else:
        return years


def transform_to_real_age(pred, min_age=5, max_age=26):
    return pred * (max_age - min_age) + min_age


def age_to_class(age, bins=[10, 12, 14, 16, 18, 21]):
    for i, b in enumerate(bins):
        if age < b and age >= (
            bins[i - 1] if i > 0 else 0
        ):  # less than upper bound and greater than or equal to lower bound
            return i
    return len(bins)
