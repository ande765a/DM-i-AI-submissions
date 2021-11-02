from math import sqrt
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from datasets import WaldoDataset
from models import BaselineCNN
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from transforms import Compose, RandomCrop, ToTensor, RandomScale

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WIDTH, HEIGHT = 300, 300

train_dataset = WaldoDataset(transform=Compose([
  RandomScale((0.5, 1.2)),
  RandomCrop(size=(WIDTH, HEIGHT)),
  ToTensor()
]))
test_dataset = WaldoDataset(test=True, transform=ToTensor())

train_data = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)

model = BaselineCNN(in_channels=3).to(device)

num_epochs = 10
learning_rate = 0.001
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

loss_history = []


wandb.init(project="wheres-waldo", entity="andersthuesen")

wandb.config = {
  'num_epochs': num_epochs,
  'learning_rate': learning_rate
}


for epoch in range(5):
  print(f"Training epoch {epoch + 1}")
  
  tqdm_train_data = tqdm(train_data)
  for image, mask, _ in tqdm_train_data:
    image, mask = image.to(device), mask.to(device)

    optim.zero_grad()
    logits = model(image)
    logits = logits.reshape(logits.shape[0], -1) # Flatten
    output = torch.softmax(logits, dim=1)
    mask = mask.reshape(mask.shape[0], -1) # Flatten

    loss = criterion(logits, mask)
    loss.backward()

    loss_history.append(loss.item())
    wandb.log({'loss': loss.item()})
    wandb.watch(model)

    tqdm_train_data.set_description(f"Loss: {loss.item():.4f}")
    optim.step()


  print(f"Testing epoch {epoch + 1}")
  error = 0
  tqdm_test_data = tqdm(test_dataset)
  for image, mask, (x, y) in tqdm_test_data:
    image, mask = image[None, :], mask[None, :] # Add batch dimension
    image, mask = image.to(device), mask.to(device) # Move to GPU

    logits = model(image)
    logits = logits.reshape(logits.shape[0], -1) # Flatten

    preds = torch.argmax(logits, dim=1)
  
    px = (preds % image.shape[3]).item()
    py = (preds / image.shape[2]).item()

    error += sqrt((x - px) ** 2 + (y - py) ** 2) / len(tqdm_test_data)
  
  print("Mean euclidian distance is {}".format(error))
