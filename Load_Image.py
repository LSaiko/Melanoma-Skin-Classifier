import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class SkinDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path) #load the labels spreadsheet 
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df) # how many images total

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row.image_id}.jpg"
        img = Image.open(img_path)
        label = 1 if row.dx == "mel" else 0 # melanoma = 1, non-melanoma = 0
        
        if self.transform:
            img = self.transform(img)
        
        return img, label