import os
import pandas as pd
from torch.utils.data import Dataset

class MovieReviewDataset(Dataset):
  def __init__(self, data_path="data", transform=None, test=False):
    self.transform = transform
    self.df = pd.read_table(os.path.join(data_path, "test.tsv" if test else "data.tsv"), sep='\t')
    
  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    url, title_id, rating, text = self.df.iloc[index]
    if self.transform is not None:
      text = self.transform(text)

    return text, float(rating)