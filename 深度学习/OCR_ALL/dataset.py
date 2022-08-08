import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class MyDataset(Dataset):

    def __init__(self, path, mode="test", transformer=ToTensor()):
        info_path = f"{path}/{mode}_info.txt"
        path = f"{path}/{mode}"
        info = pd.read_csv(info_path, sep=" ", header=None)
        labels = np.array(info.iloc[:-1, 0])
        img_paths = []
        for i, label in enumerate(labels):
            for file_name in os.listdir(f"{path}/{label}"):
                img_paths.append((i, f"{path}/{label}/{file_name}"))
        self.img_path = img_paths
        self.transformer = transformer

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        i, path = self.img_path[idx]
        image = Image.open(path)
        return self.transformer(image), i
