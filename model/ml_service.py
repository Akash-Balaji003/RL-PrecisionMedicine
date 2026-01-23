from fastapi import FastAPI
from model.RL_model import QNetwork, TabularTransformer, recommend_treatment
from pydantic import BaseModel
import torch
import pickle
import numpy as np

app = FastAPI()

# Load encoders first
with open("model/saved_models/label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

STATE_DIM = 6
NUM_ACTIONS = len(encoders["TreatmentPlan"].classes_)

# Load models
encoder = TabularTransformer(STATE_DIM)
model = QNetwork(embed_dim=32, action_dim=NUM_ACTIONS)

encoder.load_state_dict(torch.load("model/saved_models/cf_encoder.pth", map_location="cpu"))
model.load_state_dict(torch.load("model/saved_models/cf_qnetwork.pth", map_location="cpu"))

encoder.eval()
model.eval()


class PatientInput(BaseModel):
    age: int
    gender: str
    diagnosis: str
    heart_rate: int
    respiratory_rate: int
    oxygen_saturation: int

@app.post("/recommend")
def recommend(patient: PatientInput):
    print("Received patient data:", patient)
    result = recommend_treatment(
        age=patient.age,
        gender=patient.gender,
        diagnosis=patient.diagnosis,
        heart_rate=patient.heart_rate,
        respiratory_rate=patient.respiratory_rate,
        oxygen_saturation=patient.oxygen_saturation,
        encoder=encoder,
        model=model,
        encoders=encoders
    )
    return result

@app.get("/test")
def test():
    return {"Test": "Working"}