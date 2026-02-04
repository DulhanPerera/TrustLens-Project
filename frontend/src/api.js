/**
 * =============================================================================
 * TrustLens Frontend - API Service Layer
 * =============================================================================
 * 
 * This module handles all HTTP communication with the TrustLens FastAPI backend.
 * It provides a centralized API client using Axios for making HTTP requests.
 * 
 * Backend endpoints:
 * - POST /predict       - Single transaction fraud prediction
 * - POST /predict_batch - Batch transaction analysis (50-500 transactions)
 * - POST /report        - Detailed fraud analysis report (TrustLens Card)
 * - GET  /health        - API health check
 * 
 * File: api.js
 * =============================================================================
 */

import axios from 'axios';

// =============================================================================
// API Configuration
// =============================================================================
// The base URL where the FastAPI backend is running.
// Update this for production deployment (e.g., https://api.trustlens.com)
// =============================================================================
const API_BASE_URL = "https://trustlens-project-production.up.railway.app/";

/**
 * Axios instance configured with the backend base URL.
 * This instance can be extended with interceptors for:
 * - Authentication tokens
 * - Request/response logging
 * - Error handling
 */
const api = axios.create({
    baseURL: API_BASE_URL,
});

// =============================================================================
// API Functions
// =============================================================================

/**
 * Send a transaction to the backend for fraud prediction analysis.
 * 
 * This function calls the /predict endpoint which:
 * 1. Scales the input features using the trained scaler
 * 2. Runs MLP model for fraud probability
 * 3. Runs Autoencoder for anomaly detection
 * 4. Combines scores for final decision
 * 5. Generates SHAP explanations (if enabled)
 * 
 * @param {Object} transactionData - Transaction features to analyze
 * @param {number} transactionData.Time - Timestamp (seconds from dataset start)
 * @param {number[]} transactionData.V_features - 28 PCA-transformed features (V1-V28)
 * @param {number} transactionData.Amount - Transaction amount
 * 
 * @returns {Promise<Object>} Prediction response containing:
 *   - is_fraud: boolean - Whether transaction is flagged as fraud
 *   - status: string - "BLOCKED" or "APPROVED"
 *   - risk_score: number - Combined risk score (0-100%)
 *   - explanation: string - Human-readable explanation
 *   - xai_data: Array - SHAP feature attributions
 * 
 * @throws {Error} Network or server errors are re-thrown after logging
 */
export const getFraudPrediction = async (transactionData) => {
    try {
        const response = await api.post('/predict', transactionData);
        return response.data;
    } catch (error) {
        // Log error details for debugging
        console.error("API Error:", error);
        // Re-throw to allow calling code to handle the error
        throw error;
    }
};

// Export the configured axios instance for direct use if needed
export default api;