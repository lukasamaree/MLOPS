# lab8app/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import mlflow.sklearn

# Connect to MLflow server
mlflow.set_tracking_uri("http://localhost:5001")

# Load the latest registered model
model_name = "BestModel_registered"
model_uri = f"models:/{model_name}/latest"
model = mlflow.sklearn.load_model(model_uri)

# Initialize FastAPI app
app = FastAPI()

# Define the input data format
class PredictRequest(BaseModel):
    features: list  # list of features

@app.get("/")
async def root():
    return {
        "message": "Welcome to the ML Prediction API",
        "endpoints": {
            "/predict": "POST endpoint for making predictions"
        }
    }

# Define a route for predictions
@app.post("/predict")
async def predict(request: PredictRequest):
    # Prepare input for prediction
    input_data = np.array(request.features).reshape(1, -1)
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}

