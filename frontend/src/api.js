import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const getFraudPrediction = async (transactionData) => {
  try {
    const response = await api.post('/predict', transactionData);
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

export const updateUserRole = async (userId, role) => {
  try {
    const response = await api.patch(`/users/${userId}/role`, { role });
    return response.data;
  } catch (error) {
    console.error('Update Role Error:', error);
    throw error;
  }
};

export const markTransactionLegitimate = async (mongoId) => {
  try {
    const response = await api.patch(`/transactions/${mongoId}/mark-legitimate`);
    return response.data;
  } catch (error) {
    console.error('Mark Legitimate Error:', error);
    throw error;
  }
};

export const saveAnalystNote = async (mongoId, note) => {
  try {
    const response = await api.post(`/transactions/${mongoId}/note`, { note });
    return response.data;
  } catch (error) {
    console.error('Save Analyst Note Error:', error);
    throw error;
  }
};

export const getSystemSettings = async () => {
  try {
    const response = await api.get('/settings');
    return response.data;
  } catch (error) {
    console.error('Get Settings Error:', error);
    throw error;
  }
};

export const updateSystemSettings = async (settings) => {
  try {
    const response = await api.post('/settings', settings);
    return response.data;
  } catch (error) {
    console.error('Update Settings Error:', error);
    throw error;
  }
};

export const getActivityLogs = async (limit = 50) => {
  try {
    const response = await api.get(`/activity-logs?limit=${limit}`);
    return response.data;
  } catch (error) {
    console.error('Get Activity Logs Error:', error);
    throw error;
  }
};

export default api;