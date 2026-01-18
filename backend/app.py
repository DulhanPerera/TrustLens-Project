import numpy as np
import tensorflow as tf
import joblib
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# --- 1. App Configuration ---
app = FastAPI(title="TrustLens API", description="Real-time Fraud Detection System")

# Allow your React Frontend to talk to this Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Secure Audit Logging
logging.basicConfig(
    filename='trustlens_audit.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# --- 2. Load the Brain ---
model = None
scaler = None

@app.on_event("startup")
def load_brain():
    global model, scaler
    try:
        model = tf.keras.models.load_model('trustlens_model.h5')
        scaler = joblib.load('rob_scaler.pkl')
        print("✅ TrustLens Intelligence Loaded Successfully!")
    except Exception as e:
        print(f"❌ Error loading model/scaler: {e}")

# --- 3. Define Input Data Format ---
class Transaction(BaseModel):
    Time: float
    Amount: float
    V_features: list[float]  # Expecting [V1, V2, ... V28]

# --- 4. Helper: Simulate SMS Alert ---
def send_sms_alert(prob: float):
    # In a real app, you'd use Twilio here.
    print(f"📲 [SMS SENT] Suspicious Transaction Detected! Risk Score: {prob*100:.2f}%")

# --- 5. The Prediction Endpoint ---
@app.post("/predict")
async def predict(data: Transaction, background_tasks: BackgroundTasks):
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare Features: [Time, V1..V28, Amount]
        features = [data.Time] + data.V_features + [data.Amount]
        input_array = np.array([features])
        
        # Scale Data
        processed_data = scaler.transform(input_array)
        
        # Predict
        prediction = model.predict(processed_data)
        fraud_prob = float(prediction[0][0])
        is_fraud = fraud_prob > 0.5
        
        # Log Result
        log_msg = f"Analysis: Risk={fraud_prob:.4f}, Fraud={is_fraud}"
        logging.info(log_msg)
        
        # Trigger SMS if Fraud
        if is_fraud:
            background_tasks.add_task(send_sms_alert, fraud_prob)
            
        # Return Response
        return {
            "is_fraud": is_fraud,
            "risk_score": round(fraud_prob * 100, 2),
            "status": "BLOCKED" if is_fraud else "APPROVED",
            "explanation": "High deviation in V14 & V17" if is_fraud else "Normal Pattern"
        }

    except Exception as e:
        return {"error": str(e)}