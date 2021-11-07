import numpy as np
from fastapi import FastAPI
from fastapi.logger import logger
from pydantic import BaseModel
from typing import Optional
from typing import Optional
from enum import Enum
from fastapi.middleware.cors import CORSMiddleware
import pygame

from pygame import K_LEFT, K_RIGHT, K_UP, K_DOWN


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"])

class Velocity(BaseModel):
    x: int
    y: int


class SensorReadings(BaseModel):
    left_side: Optional[int]
    left_front: Optional[int]
    front: Optional[int]
    right_front: Optional[int]
    right_side: Optional[int]
    right_back: Optional[int]
    back: Optional[int]
    left_back: Optional[int]


class PredictRequest(BaseModel):
  elapsed_time_ms: int
  distance: int
  velocity: Velocity
  sensors: SensorReadings
  did_crash: bool


class ActionType(str, Enum):
  ACCELERATE = 'ACCELERATE'
  DECELERATE = 'DECELERATE'
  STEER_RIGHT = 'STEER_RIGHT'
  STEER_LEFT = 'STEER_LEFT'
  NOTHING = 'NOTHING'


class PredictResponse(BaseModel):
  action: ActionType


def sigmoid(z):
  return 1 / (1 + np.exp(-z))


def softmax(z):
  return np.exp(z) / np.sum(np.exp(z), axis=0)

num_candidates = 5
candidates = []

for i in range(num_candidates):
  candidates.append((0, np.random.randn(5, 10), np.random.randn(5)))



@app.get("/api")
def health_check():
    return {"ok": True}

app.candidate_index = 0

pygame.init()
screen = pygame.display.set_mode([500, 500])
screen.fill((255, 255, 255))
pygame.display.flip()

@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:

  if (request.elapsed_time_ms > 8000):
    return PredictResponse(action=ActionType.STEER_LEFT)
  
  if (request.elapsed_time_ms > 1000 and request.distance == 0): # Crash car if no progress
    return PredictResponse(action=ActionType.STEER_RIGHT)

  if request.did_crash:
    _, A, B = candidates[app.candidate_index]
    candidates[app.candidate_index] = request.distance, A, B

    print(f"Candidate {app.candidate_index} crashed.")

    if app.candidate_index >= num_candidates - 1:
      # Mutate stuff here
      score, A, B = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
      logger.warn(f"Winner candidate got score of {score}")
      app.candidate_index = 0
    else:
      app.candidate_index += 1
      logger.warn(f"Candidate {app.candidate_index}")

    return PredictResponse(action=ActionType.NOTHING)
    # Mutate stuff here.

  X = np.array([
    request.sensors.left_side if request.sensors.left_side is not None else np.inf,
    request.sensors.left_front if request.sensors.left_front is not None else np.inf,
    request.sensors.front if request.sensors.front is not None else np.inf,
    request.sensors.right_front if request.sensors.right_front is not None else np.inf,
    request.sensors.right_side if request.sensors.right_side is not None else np.inf,
    request.sensors.right_back if request.sensors.right_back is not None else np.inf,
    request.sensors.back if request.sensors.back is not None else np.inf,
    request.sensors.left_back if request.sensors.left_back is not None else np.inf,
    request.velocity.x,
    request.velocity.y,
  ])

  score, A, B= candidates[app.candidate_index]
  action_num = softmax(A @ X + B).argmax()

  action = [ActionType.ACCELERATE, ActionType.DECELERATE, ActionType.STEER_RIGHT, ActionType.STEER_LEFT, ActionType.NOTHING][action_num]

  return PredictResponse(action=action)