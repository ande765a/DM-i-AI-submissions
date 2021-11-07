import torch
from transformers import BertTokenizer
from models import BertForSentiment
from datasets import MovieReviewDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from utils import collate


if __name__ == "__main__":
  dataset = MovieReviewDataset()
  train_data = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  learning_rate = 5e-5
  num_epochs = 50


  model = BertForSentiment().to(device)
  model.load_state_dict(torch.load("saved-models/model.torch", map_location=device))
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  criterion = torch.nn.MSELoss()

  wandb.init(project="movie-reviews")
  wandb.config = {
    "learning_rate": learning_rate,
    "num_epochs": num_epochs
  }

  try:
    for epoch in range(num_epochs):
      train_data_tqdm = tqdm(train_data)
      for texts, input_ids, attention_masks, ratings in train_data_tqdm:
        input_ids, attention_masks, ratings = input_ids.to(device), attention_masks.to(device), ratings.to(device)

        optimizer.zero_grad()

        # ratings_cat = torch.zeros((ratings.shape[0], 10))
        # for i in range(1, 10):
        #   ratings_cat[ratings == (i / 2), i - 1] = 1

        # ratings_cat = ratings_cat.to(device)
      
        pred = model(input_ids, attention_masks).view(-1)
        loss = criterion(pred, ratings)
        loss.backward()

        wandb.log({"loss": loss.item()})

        train_data_tqdm.set_description(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}")
        optimizer.step()

  except Exception as e:
    print(e)


  finally:
    print("Saving model")
    torch.save(model.state_dict(), "model.torch")