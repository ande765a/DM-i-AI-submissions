import numpy as np
import itertools
from fastapi import FastAPI
from PIL import Image
from binary_transforms import make_transforms
from unary_transforms import all_unary_transforms
from pydantic import BaseModel
from typing import List
from io import BytesIO
import base64

def get_tile(image, col, row):
    return image.crop((col * 2 * 110, row * 110, (col * 2 + 1) * 110, (row + 1) * 110))


app = FastAPI()

@app.get("/api")
async def health_check():
    return { "ok": True }

class PredictRequest(BaseModel):
    image_base64: str
    image_choices_base64: List[str]


@app.post("/api/predict")
async def predict(predict_request: PredictRequest):
  main_image = Image.open(BytesIO(base64.b64decode(predict_request.image_base64)))
  image_choices = [Image.open(BytesIO(base64.b64decode(choice_base64))) for choice_base64 in predict_request.image_choices_base64]

  min_error = np.inf
  best_transform = None

  for row in range(0, 2):
    im1 = get_tile(main_image, 0, row)
    im2 = get_tile(main_image, 1, row)
    im3 = get_tile(main_image, 2, row)

    for make_transform in make_transforms:
      binary_transform = make_transform(im1, im2, im3)

      for unary_transforms in itertools.permutations(all_unary_transforms, r=2):
        error = 0
        for row in range(0, 2):
          im1 = get_tile(main_image, 0, row)
          im2 = get_tile(main_image, 1, row)
          im3 = get_tile(main_image, 2, row)

          pred = binary_transform(im1, im2)
          for unary_transform in unary_transforms:
            pred = unary_transform(pred)
          error += np.sum(np.square(np.array(pred) - np.array(im3)))

        if error < min_error:
          min_error = error
          best_transform = (binary_transform, unary_transforms)
        

  binary_transform, unary_transforms = best_transform

  pred = binary_transform(
    get_tile(main_image, 0, 3),
    get_tile(main_image, 1, 3)
  )

  for unary_transform in unary_transforms:
    pred = unary_transform(pred)
  
  min_error = np.inf
  best_choice = 0
  for i, image_choice in enumerate(image_choices):
    error = np.sum(np.square(np.array(pred) - np.array(image_choice)))
    if error < min_error:
      min_error = error
      best_choice = i

  print(best_choice)
  return { "next_image_index": best_choice }