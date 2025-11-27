from fastapi import FastAPI,HTTPException 
from fastapi.responses import JSONResponse
from typing import Literal, Annotated
from pydantic import BaseModel , Field
from predict import InsurancePredictor
class InsuranceInput(BaseModel):
    age: Annotated[int, Field(ge=18, le=65)]
    sex: Literal["male", "female"]
    bmi: Annotated[float, Field(ge=15.0, le=50.0)]
    children: Annotated[int, Field(ge=0, le=5)]
    smoker: Literal["yes", "no"]

app=FastAPI(title="Insurance Prediction API",
    description="Predicts annual medical charges using a trained ML pipeline.")

try:
    predictor_service = InsurancePredictor(model_path='insaurance.joblib')
except RuntimeError as e:
    print(f"FATAL ERROR: {e}")
    predictor_service = None

@app.post("/predict_insurance_charge")
async def predict_charge(data: InsuranceInput):
    if predictor_service is None:
        raise HTTPException(
            status_code=503,
            detail="Model service is currently unavailable. Check model file path."
        )

    input_dict = data.model_dump()

    try:
        result = predictor_service.predict_and_classify(input_dict)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed due to internal error: {e}")
