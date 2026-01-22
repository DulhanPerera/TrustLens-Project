import React, { useState } from 'react';
import { ShieldAlert, Activity, CheckCircle } from 'lucide-react';
import { getFraudPrediction } from './api';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const testTransaction = async () => {
    setLoading(true);
    // Sample fraud data format matching your backend schema
    const sampleData = {
      Time: 406.0,
      V_features: Array(28).fill(0).map(() => Math.random() * -2), // Simulated anomaly
      Amount: 0.00
    };

    try {
      const result = await getFraudPrediction(sampleData);
      setPrediction(result);
    } catch (err) {
      alert("Ensure your FastAPI backend is running on port 8000");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '40px', fontFamily: 'sans-serif', backgroundColor: '#f9fafb', minHeight: '100vh' }}>
      <header style={{ marginBottom: '30px', display: 'flex', alignItems: 'center', gap: '10px' }}>
        <ShieldAlert size={32} color="#2563eb" />
        <h1 style={{ fontSize: '24px', fontWeight: 'bold' }}>TrustLens Admin Panel</h1>
      </header>

      <main style={{ maxWidth: '800px' }}>
        <button 
          onClick={testTransaction}
          disabled={loading}
          style={{ padding: '12px 24px', backgroundColor: '#2563eb', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer' }}
        >
          {loading ? "Analyzing..." : "Analyze New Transaction"}
        </button>

        {prediction && (
          <div style={{ marginTop: '30px', padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <h2 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {prediction.is_fraud ? <ShieldAlert color="#dc2626" /> : <CheckCircle color="#16a34a" />}
              {prediction.status}
            </h2>
            <div style={{ marginTop: '10px' }}>
              <p><strong>Risk Score:</strong> {prediction.risk_score}%</p>
              <p><strong>Reasoning (XAI):</strong> {prediction.explanation}</p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;