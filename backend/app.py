import numpy as np
import tensorflow as tf
import joblib
import logging
import shap
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# --- 1. App Configuration ---
app = FastAPI(title="TrustLens API", description="Real-time Fraud Detection with XAI")

# Security: CORS configuration for React [cite: 214]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Restrict to React dev port
    allow_methods=["*"],
    allow_headers=["*"],
)

# Audit Logging setup per NF06 
logging.basicConfig(
    filename='trustlens_audit.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- 2. Load Models & XAI Explainer ---
model = None
scaler = None
explainer = None

@app.on_event("startup")
def load_assets():
    global model, scaler, explainer
    try:
        # Load the Deep Learning model and Scaler [cite: 563, 569]
        model = tf.keras.models.load_model('trustlens_model.h5')
        scaler = joblib.load('rob_scaler.pkl')
        
        # Initialize SHAP explainer for Deep Learning (DeepExplainer) [cite: 312, 570]
        # Note: We use a small background dataset for the explainer to remain "computationally efficient" (RG2)
        background_data = np.zeros((10, 30)) 
        explainer = shap.DeepExplainer(model, background_data)
        
        logging.info("System Assets Loaded Successfully.")
        print("✅ TrustLens Intelligence & XAI explainer Ready!")
    except Exception as e:
        logging.error(f"Failed to load assets: {e}")
        print(f"❌ Initialization Error: {e}")

# --- 3. Data Schema ---
class Transaction(BaseModel):
    # Mapping to Kaggle dataset features [cite: 331, 335]
    Time: float
    V_features: list[float] = Field(..., min_items=28, max_items=28)
    Amount: float

# --- 4. Logic Functions ---
def send_realtime_notification(prob: float):
    # Implementation of UC-14 [cite: 1594]
    # In production, integrate SMS/Email gateway here
    logging.info(f"Notification Sent: High Risk Alert ({prob*100:.2f}%)")

def generate_xai_reason(processed_data):
    """
    Identifies top contributory features using SHAP[cite: 1117].
    Fulfills NF04 (Transparency).
    """
    shap_values = explainer.shap_values(processed_data)
    # Convert to flat array and find indices of max impact
    vals = np.abs(shap_values).flatten()
    indices = np.argsort(vals)[-2:] # Top 2 features
    
    # Feature names map to [Time, V1..V28, Amount]
    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    reasons = [feature_names[i] for i in indices]
    return f"Abnormal patterns detected in {', '.join(reasons)}"

# --- 5. Predict Endpoint (FR01, FR03) ---
@app.post("/predict")
async def predict(data: Transaction, background_tasks: BackgroundTasks):
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model not ready")
    
    try:
        # 1. Preprocessing (FR09) [cite: 190, 561]
        # Scale only the Amount feature (scaler was trained on Amount only)
        scaled_amount = scaler.transform(np.array([[data.Amount]]))[0][0]
        
        # Build final feature array: [Time, V1..V28, Scaled_Amount]
        processed_data = np.array([[data.Time] + data.V_features + [scaled_amount]])
        
        # 2. Prediction (FR01) [cite: 1160]
        prediction = model.predict(processed_data)
        fraud_prob = float(prediction[0][0])
        is_fraud = fraud_prob > 0.5 # Threshold per UC-11 [cite: 1104]
        
        # 3. Explain (UC-12) [cite: 1117]
        explanation = generate_xai_reason(processed_data) if is_fraud else "Normal Pattern"
        
        # 4. Background Notification (UC-14) [cite: 1594]
        if is_fraud:
            background_tasks.add_task(send_realtime_notification, fraud_prob)
            
        # 5. Secure Audit Logging (NF06) 
        logging.info(f"Transaction Processed - Risk: {fraud_prob:.4f}, Action: {'BLOCKED' if is_fraud else 'APPROVED'}")
        
        return {
            "is_fraud": is_fraud,
            "risk_score": round(fraud_prob * 100, 2),
            "status": "BLOCKED" if is_fraud else "APPROVED",
            "explanation": explanation
        }

    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during analysis")