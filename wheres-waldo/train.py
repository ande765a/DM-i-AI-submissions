from math import sqrt
import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataset import ConcatDataset
from torchvision.transforms.functional import to_pil_image
from datasets import WaldoDataset
from models import BaselineCNN, UNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from transforms import Compose, RandomCrop, ToTensor, RandomScale

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WIDTH, HEIGHT = 256, 256

original_train_dataset = WaldoDataset(transform=ToTensor())
train_dataset = WaldoDataset(transform=Compose([
  RandomScale((0.5, 1.2)),
  RandomCrop(size=(WIDTH, HEIGHT)),
  ToTensor()
]))

batch_size = 64

test_dataset = WaldoDataset(test=True, transform=ToTensor())
test_data = DataLoader(test_dataset, batch_size=1, shuffle=False)
train_data = DataLoader(ConcatDataset([original_train_dataset, train_dataset]), batch_size=batch_size, shuffle=True, num_workers=24)

model = BaselineCNN(in_channels=3).to(device) #BaselineCNN(in_channels=3).to(device)
#model.load_state_dict(torch.load("model.torch", map_location=device)) # Load model

num_epochs = 20
learning_rate = 1e-1
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
#pos_weight = torch.full((HEIGHT, WIDTH), 50).to(device) # Weight positive examples more.
criterion = torch.nn.BCEWithLogitsLoss() 

loss_history = []

wandb.init(project="wheres-waldo", entity="andersthuesen")

wandb.config = {
  'num_epochs': num_epochs,
  'learning_rate': learning_rate
}


try:
  
  for epoch in range(num_epochs):
    print(f"Training epoch {epoch + 1}")

    model.train()
    tqdm_train_data = tqdm(train_data)
    for image, mask, _ in tqdm_train_data:
      image, mask = image.to(device), mask.to(device)

      optim.zero_grad()
      logits = model(image)

      loss = criterion(logits, mask)
      loss.backward()

      loss_history.append(loss.item())
      wandb.log({'loss': loss.item()})

      tqdm_train_data.set_description(f"Loss: {loss.item():.4f}")
      optim.step()


    print(f"Testing epoch {epoch + 1}")
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
    wandb.log({ 'mean-euclidean': error })

except Exception as e:
  print(e)
  pass


print("Saving model")
torch.save(model.state_dict(), "model.torch")