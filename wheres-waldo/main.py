import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from datasets import WaldoDataset
from models import BaselineCNN
from torch.utils.data import DataLoader
from tqdm import tqdm

from transforms import Compose, RandomCrop, ToTensor, RandomScale

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = WaldoDataset(transform=Compose([
  RandomScale((0.5, 1.2)),
  RandomCrop(size=(300, 300)),
  ToTensor()
]))
test_dataset = WaldoDataset(test=True, transform=ToTensor())

train_data = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)

model = BaselineCNN(in_channels=3).to(device)

num_epochs = 10
learning_rate = 0.001
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(5):
  print(f"Training epoch {epoch + 1}")
  
  tqdm_train_data = tqdm(train_data)
  for image, mask in tqdm_train_data:
    image, mask = image.to(device), mask.to(device)

    optim.zero_grad()
    output = model(image)
    loss = criterion(output.reshape(output.shape[0], -1), mask.reshape(mask.shape[0], -1))
    loss.backward()

    tqdm_train_data.set_description(f"Loss: {loss.item():.4f}")
    optim.step()


  print(f"Testing epoch {epoch + 1}")
  tqdm_test_data = tqdm(test_dataset)
  for image, mask in tqdm_test_data:
    image, mask = image[None, :], mask[None, :] # Add batch dimension
    image, mask = image.to(device), mask.to(device) # Move to GPU

    output = model(image)
    print(output)