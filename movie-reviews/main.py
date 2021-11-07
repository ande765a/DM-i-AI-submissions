import torch
from transformers import BertTokenizer
import re
from models import BertForSentiment
from datasets import MovieReviewDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

def text_preprocessing(text):
  """
  - Remove entity mentions (eg. "@united")
  - Correct errors (eg. "&amp;" to "&")
  @param    text (str): a string to be processed.
  @return   text (Str): the processed string.
  """
  # Remove "@name"
  text = re.sub(r"(@.*?)[\s]", " ", text)

  # Replace "&amp;" with "&"
  text = re.sub(r"&amp;", "&", text)

  # Remove trailing whitespace
  text = re.sub(r"\s+", " ", text).strip()

  return text

MAX_LEN = 64
def preprocessing_for_bert(data):
  """Perform required preprocessing steps for pretrained BERT.
  @param    data (np.array): Array of texts to be processed.
  @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
  @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                tokens should be attended to by the model.
  """
  # Create empty lists to store outputs
  input_ids = []
  attention_masks = []

  # For every sentence...
  for sent in data:
    # `encode_plus` will:
    #    (1) Tokenize the sentence
    #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
    #    (3) Truncate/Pad sentence to max length
    #    (4) Map tokens to their IDs
    #    (5) Create attention mask
    #    (6) Return a dictionary of outputs
    encoded_sent = tokenizer.encode_plus(
      text=text_preprocessing(sent),  # Preprocess sentence
      add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
      max_length=MAX_LEN,             # Max length to truncate/pad
      padding="max_length",           # Pad sentence to max length
      return_attention_mask=True,     # Return attention mask,
      return_tensors="pt",            # Return pytorch tensors,    
      truncation=True
    )
    
    # Add the outputs to the lists
    input_ids.append(encoded_sent.get("input_ids"))
    attention_masks.append(encoded_sent.get("attention_mask"))


  return input_ids, attention_masks


def collate(inputs):
  texts, ratings = zip(*inputs)
  input_ids, attention_masks = preprocessing_for_bert(texts)
  return texts, torch.concat(input_ids, dim=0), torch.concat(attention_masks, dim=0), torch.tensor(ratings)



if __name__ == "__main__":



  dataset = MovieReviewDataset()
  train_data = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

  input_ids, attention_masks = preprocessing_for_bert(["Hello world", "This is super cool!"])
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  learning_rate = 1e-3
  num_epochs = 50


  model = BertForSentiment().to(device)
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

        pred_ratings = model(input_ids, attention_masks).view(-1)
        loss = criterion(pred_ratings, ratings)
        loss.backward()

        wandb.log({"loss": loss.item()})

        train_data_tqdm.set_description(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}")
        optimizer.step()

        #break
      break
  finally:
    print("Saving model")
    torch.save(model.state_dict(), "model.torch")