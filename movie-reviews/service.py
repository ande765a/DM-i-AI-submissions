import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
from models import BertForSentiment
from utils import preprocessing_for_bert

app = FastAPI()

class PredictRequest(BaseModel):
  reviews: List[str]

class PredictResponse(BaseModel):
  ratings: List[float]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSentiment().to(device)
model.load_state_dict(torch.load("model.torch", map_location=device))

@app.get("/api")
def health_check():
  return { "status": "ok" }

@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:

  ratings = []
  for review in request.reviews:
    input_ids, attention_masks = preprocessing_for_bert([review])
    input_ids = torch.concat(input_ids, dim=0).to(device)
    attention_masks = torch.concat(attention_masks, dim=0).to(device)

    pred = model(input_ids, attention_masks)

    ratings.append(pred.item())

  return PredictResponse(ratings=ratings)