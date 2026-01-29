import numpy as np
import tensorflow as tf
import joblib
import logging
import shap
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TrustLens API", description="Real-time Fraud Detection with XAI")

app.add_middleware(
    CORSMiddleware,
    # Allow common local dev origins used by React (create-react-app) and Vite
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(filename='trustlens_audit.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model = None
scaler = None
explainer = None

@app.on_event("startup")
def load_assets():
    global model, scaler, explainer
    try:
        model = tf.keras.models.load_model('trustlens_model.h5')
        scaler = joblib.load('rob_scaler.pkl')
        # Use a more representative background for SHAP if possible, but zeros work for speed
        background_data = np.zeros((10, 30)) 
        explainer = shap.DeepExplainer(model, background_data)
        logging.info("System Assets Loaded Successfully.")
    except Exception as e:
        logging.error(f"Initialization Error: {e}")

class Transaction(BaseModel):
    Time: float
    V_features: list[float] = Field(..., min_items=28, max_items=28)
    Amount: float

def process_xai(processed_data):
    """
    Extracts numerical SHAP values for the frontend chart.
    """
    try:
        shap_values = explainer.shap_values(processed_data)
        # DeepExplainer returns a list of arrays for each output neuron
        vals = np.abs(shap_values[0]).flatten() if isinstance(shap_values, list) else np.abs(shap_values).flatten()
        
        feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        
        # Get top 4 most impactful features for the chart
        top_indices = np.argsort(vals)[-4:]
        
        chart_data = []
        for i in top_indices:
            chart_data.append({
                "name": feature_names[i],
                "impact": float(vals[i]) # Must be standard float for JSON
            })
            
        # Generate the text explanation based on top 2
        top_2 = [feature_names[i] for i in top_indices[-2:]]
        text_reason = f"Anomalous patterns identified in {', '.join(top_2)}. These features show significant deviation from legitimate transaction clusters."
        
        return text_reason, chart_data
    except Exception as e:
        logging.error(f"SHAP Error: {e}")
        return "Pattern detected in latent features.", []



@app.post("/predict")
async def predict(data: Transaction, background_tasks: BackgroundTasks):
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    try:
        # Scale only the amount (matching your training preprocessing)
        scaled_amount = scaler.transform(np.array([[data.Amount]]))[0][0]
        processed_data = np.array([[data.Time] + data.V_features + [scaled_amount]])
        
        prediction = model.predict(processed_data)
        fraud_prob = float(prediction[0][0])
        is_fraud = fraud_prob > 0.2 # Lowered threshold for higher recall in prototype
        
        explanation_text, xai_chart_data = process_xai(processed_data)
        
        if is_fraud:
            background_tasks.add_task(logging.info, f"Fraud Alert: {fraud_prob}")
            
        return {
            "is_fraud": is_fraud,
            "risk_score": round(fraud_prob * 100, 2),
            "status": "BLOCKED" if is_fraud else "APPROVED",
            "explanation": explanation_text if is_fraud else "Transaction follows established legitimate behavioral patterns.",
            "xai_data": xai_chart_data
        }
    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal analysis failure")