import axios from 'axios';

// The URL where your FastAPI backend is running
const API_BASE_URL = "http://127.0.0.1:8000";

const api = axios.create({
    baseURL: API_BASE_URL,
});

export const getFraudPrediction = async (transactionData) => {
    try {
        const response = await api.post('/predict', transactionData);
        return response.data;
    } catch (error) {
        console.error("API Error:", error);
        throw error;
    }
};

export default api;