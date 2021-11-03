import sys
import torch
from math import sqrt
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from models import UNet
from datasets import WaldoDataset
from transforms import ToTensor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3).to(device)
model.load_state_dict(torch.load("model.torch", map_location=device))

test_dataset = WaldoDataset(test=True, transform=ToTensor())
test_data = DataLoader(test_dataset, batch_size=1, shuffle=False)

if __name__ == "__main__":
  model.eval()
  error = 0
  for image, _, (x, y) in test_data:
    image = image.to(device) # Add 
    logits = model(image)

    preds = torch.argmax(logits.reshape(logits.shape[0], -1), dim=1)

    px = (preds % image.shape[3]).item()
    py = (preds / image.shape[2]).item()

    error += sqrt((x - px) ** 2 + (y - py) ** 2) / len(test_dataset)

  print("Mean euclidean distance is {}".format(error))