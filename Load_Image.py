import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

SUBFOLDERS = ["HAM10000_images_part_1", "HAM10000_images_part_2", ""]

class SkinDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self._find_image(row.image_id)
        img      = Image.open(img_path).convert("RGB")
        label    = 1 if row.dx == "mel" else 0
        if self.transform:
            img = self.transform(img)
        return img, label

    def _find_image(self, image_id):
        for sf in SUBFOLDERS:
            p = (os.path.join(self.img_dir, sf, f"{image_id}.jpg") if sf
                 else os.path.join(self.img_dir, f"{image_id}.jpg"))
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Image not found: {image_id}")
