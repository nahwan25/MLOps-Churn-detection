import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model
model = joblib.load("model/model.joblib")

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Model API is running 🚀"}

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}
