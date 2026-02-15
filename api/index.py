from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model/model.joblib")

@app.post("/predict")
def predict(data: dict):
    prediction = model.predict([list(data.values())])
    return {"prediction": prediction.tolist()}
