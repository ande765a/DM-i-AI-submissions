from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os

app = FastAPI()

class PredictRequest(BaseModel):
  reviews: List[str]

class PredictResponse(BaseModel):
  ratings: List[float]


@app.get("/api")
def health_check():
  return { "status": "ok" }

@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
  print(request.reviews)
  
  
  with open("reviews.csv", "w") as f:
    f.writelines(["%s\n" % review for review in request.reviews])

  ratings = [
    round(((1 - i / len(request.reviews)) * 4.5) * 2) / 2 + 0.5 for i, review in enumerate(request.reviews)
  ]

  return PredictResponse(ratings=ratings)