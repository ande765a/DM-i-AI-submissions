import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class WaldoDataset(Dataset):
  def __init__(self, data_path="data", test=False, transform=None):
    self.test = test
    self.root_path = os.path.join(data_path, "test" if test else "train")
    self.df = pd.read_csv(os.path.join(self.root_path, "data.csv"))
    self.transform = transform
    
  def __len__(self):
    return len(self.df) if self.test else 10000

  def __getitem__(self, index):
    image_filename, x, y = self.df.iloc[index % len(self.df)]
    image_path = os.path.join(self.root_path, "images", image_filename)
    #mask_path = os.path.join(self.root_path, "masks", image_filename)
    image = Image.open(image_path)
    #mask = Image.open(mask_path)
    mask = image.copy()

    coord = (x, y)
    visible = 1
    if self.transform is not None:
      image, mask, coord, visible = self.transform(image, mask, coord, visible)

    return image, mask, coord, visible