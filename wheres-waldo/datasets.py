import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image


class WaldoDataset(Dataset):
  def __init__(self, data_path="data", test=False, transform=None):
    self.root_path = os.path.join(data_path, "test" if test else "train")
    self.df = pd.read_csv(os.path.join(self.root_path, "data.csv"))
    self.transform = transform
    
  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    image_filename, x, y = self.df.iloc[index]
    image_path = os.path.join(self.root_path, "images", image_filename)
    image = Image.open(image_path)

    w, h = image.size

    mask = torch.zeros(h, w)
    mask[y, x] = 1

    mask = to_pil_image(mask)

    if self.transform is not None:
      image, mask = self.transform(image, mask)

    return image, mask