import io
import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from pydantic import BaseModel
import typing
from torchvision.transforms.functional import to_tensor
import time
from models import TransferModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TransferModel().to(device)
model.load_state_dict(torch.load("saved-models/transfer-100.torch", map_location=device))
model.eval()

class PredictRequest(UploadFile):
  def __init__(self, filename: str, file: typing.IO = None, content_type: str = "") -> None:
    super().__init__(filename, file=file, content_type=content_type)

        
class PredictResponse(BaseModel):
  x: int
  y: int


app = FastAPI()

@app.get("/api")
async def health_check():
    return { "ok": True }


@app.post("/api/predict")
async def predict(request: PredictRequest = File(...)) -> PredictResponse:

  image = Image.open(io.BytesIO(await request.read())).convert("RGB")
  name = f"test/{time.time()}.jpg"
  #image.save(name)

  image = to_tensor(image)
  image = image[None, :].to(device)
  mask = model(image)
  #mask = torch.sigmoid(mask)

  preds = mask.reshape(mask.shape[0], -1).argmax(dim=1)

  x = int(((preds % mask.shape[3]).item() + 2.5) * 8)
  y = int(((preds / mask.shape[2]).item() + 2.5) * 8)

  print(f"Found at {x},{y}")

  return PredictResponse(x=x, y=y)

