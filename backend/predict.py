import joblib
import pandas as pd
import numpy as np

LOW_RISK_THRESHOLD = 12000
MEDIUM_RISK_THRESHOLD = 30000

class InsurancePredictor:
    def __init__(self, model_path='insaurance.joblib'):
        try:
            self.pipeline = joblib.load(model_path)
            print(f"INFO: Pipeline loaded successfully from {model_path}")
        except FileNotFoundError:
            self.pipeline = None
            raise RuntimeError(f"Model file not found at: {model_path}. Please train and save the pipeline first.")

    def _classify_risk(self, charge):
        if charge < LOW_RISK_THRESHOLD:
            return "Standard (Tier 1)"
        elif charge < MEDIUM_RISK_THRESHOLD:
            return "Elevated Risk (Tier 2)"
        else:
            return "High-Risk/Complex Case (Tier 3)"

    def predict_and_classify(self, input_data: dict) -> dict:
        if self.pipeline is None:
            raise RuntimeError("Prediction failed: Model pipeline is not loaded.")

        input_df = pd.DataFrame([input_data])

        predicted_charge = self.pipeline.predict(input_df)[0]
        predicted_charge = float(np.round(predicted_charge, 2))

        risk_category = self._classify_risk(predicted_charge)

        return {
            "predicted_charge": predicted_charge,
            "risk_category": risk_category
        }