import sys
import torch
from PIL import Image
from models import UNet
from torchvision.transforms.functional import to_grayscale, to_pil_image, to_tensor
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3).to(device)
model.load_state_dict(torch.load("model.torch", map_location=device))


if __name__ == "__main__":
  image_path = sys.argv[1]
  image = Image.open(image_path)

  image = to_tensor(image)
  image = image[None, :].to(device)

  mask = model(image)
  mask = torch.sigmoid(mask)

  preds = torch.argmax(mask.reshape(mask.shape[0], -1), dim=1)

  px = (preds % image.shape[3]).item()
  py = (preds / image.shape[2]).item()

  print(f"Found at ({px}, {py})")

  mask = to_pil_image(mask[0])

  mask.save("mask.jpg")





  