from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from model.singleton_model_loader import SingletonModelLoader
# from model.predict import SingletonModelLoader
import numpy as np
router = APIRouter()

# Initialize the SingletonModelLoader for word prediction
word_predictor = SingletonModelLoader("models/word_prediction_model.pkl")

class PredictionRequest(BaseModel):
    input_text: str


@router.post("/predict-word/")
def predict_word(request: PredictionRequest):
    try:
        input_text = request.input_text.lower()
        prediction = word_predictor.predict(input_text)
        
        # Ensure the prediction is JSON-serializable
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()  # Convert NumPy array to list
        
        return {"status": "success", "prediction": prediction}
    except Exception as e:
        print(f"Error during prediction: {e}")  # Debugging log
        raise HTTPException(status_code=500, detail=str(e))


# @router.post("/predict-word/")
# def predict_word(request: PredictionRequest):
#     try:
#         prediction = word_predictor.predict(request)
#         return {"status": "status", "prediction": prediction}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
