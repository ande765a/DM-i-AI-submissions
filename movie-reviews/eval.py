import torch
from models import BertForSentiment
from datasets import MovieReviewDataset
from utils import preprocessing_for_bert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSentiment().to(device)
model.load_state_dict(torch.load("model.torch", map_location=device))

test_dataset = MovieReviewDataset(test=True)

if __name__ == "__main__":
  for text, rating in test_dataset:
    
    input_ids, attention_masks = preprocessing_for_bert([text])

    input_ids = torch.concat(input_ids, dim=0).to(device)
    attention_masks = torch.concat(attention_masks, dim=0).to(device)

    pred = model(input_ids, attention_masks)


    
    print(f"{text[0:100]} \t {pred.item()} {rating}")
