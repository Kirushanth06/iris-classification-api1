from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan function (startup + shutdown)
async def lifespan(app: FastAPI):
    """Manage the application lifecycle including startup and shutdown"""
    try:
        app.state.model = joblib.load("model.pkl")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        app.state.model = None

    yield   # Keep app alive until shutdown

    logger.info("Shutting down application")
    app.state.model = None


# FastAPI app instance
app = FastAPI(
    title="Iris Classification API",
    version="1.0.0",
    lifespan=lifespan
)


# Pydantic models
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionResult(BaseModel):
    species: str
    species_id: int


# Root endpoint (for quick check)
@app.get("/")
async def root():
    return {"message": "Iris API is running"}


# Model info endpoint
@app.get("/model-info", response_model=Dict[str, object], tags=["Model Info"])
async def model_info():
    """Get information about the model"""
    return {
        "model_type": "RandomForestClassifier",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "target_names": ["setosa", "versicolor", "virginica"],
        "model_loaded": app.state.model is not None
    }


# Prediction endpoint
@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(features: IrisFeatures):
    """Make a prediction using the loaded model"""
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available - service unavailable"
        )

    try:
        input_data = np.array([
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]).reshape(1, -1)

        prediction = app.state.model.predict(input_data)[0]
        species = ["setosa", "versicolor", "virginica"][prediction]

        return PredictionResult(
            species=species,
            species_id=int(prediction)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )
