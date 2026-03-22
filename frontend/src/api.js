import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// For Vite:
// add this to your frontend .env file:
// VITE_ADMIN_SETUP_KEY=your_admin_setup_key_here

const ADMIN_SETUP_KEY = import.meta.env.VITE_ADMIN_SETUP_KEY || '';

const api = axios.create({
  baseURL: API_BASE_URL,
});

// -----------------------------
// Header helpers
// -----------------------------
const getAdminHeaders = () => ({
  'X-Admin-Setup-Key': ADMIN_SETUP_KEY,
});

// -----------------------------
// User management
// -----------------------------
export const updateUserRole = async (userId, role) => {
  try {
    const response = await api.patch(`/users/${userId}/role`, { role });
    return response.data;
  } catch (error) {
    console.error('Update Role Error:', error);
    throw error;
  }
};

// -----------------------------
// Transaction / analyst actions
// -----------------------------
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

// -----------------------------
// System settings
// -----------------------------
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

// -----------------------------
// Activity logs
// -----------------------------
export const getActivityLogs = async (limit = 50) => {
  try {
    const response = await api.get(`/activity-logs?limit=${limit}`);
    return response.data;
  } catch (error) {
    console.error('Get Activity Logs Error:', error);
    throw error;
  }
};

// -----------------------------
// API key management
// -----------------------------
export const createApiKey = async (clientName) => {
  try {
    const response = await api.post(
      '/api-keys',
      { client_name: clientName },
      {
        headers: getAdminHeaders(),
      }
    );
    return response.data;
  } catch (error) {
    console.error('Create API Key Error:', error);
    throw error;
  }
};

export const getApiKeys = async (limit = 50) => {
  try {
    const response = await api.get(`/api-keys?limit=${limit}`, {
      headers: getAdminHeaders(),
    });
    return response.data;
  } catch (error) {
    console.error('Get API Keys Error:', error);
    throw error;
  }
};

export const getApiKeyById = async (apiKeyId) => {
  try {
    const response = await api.get(`/api-keys/${apiKeyId}`, {
      headers: getAdminHeaders(),
    });
    return response.data;
  } catch (error) {
    console.error('Get API Key By ID Error:', error);
    throw error;
  }
};

export const updateApiKeyStatus = async (apiKeyId, status) => {
  try {
    const response = await api.patch(
      `/api-keys/${apiKeyId}/status`,
      { status },
      {
        headers: getAdminHeaders(),
      }
    );
    return response.data;
  } catch (error) {
    console.error('Update API Key Status Error:', error);
    throw error;
  }
};

export const getRequestLogs = async (limit = 50) => {
  try {
    const response = await api.get(`/request-logs?limit=${limit}`, {
      headers: getAdminHeaders(),
    });
    return response.data;
  } catch (error) {
    console.error('Get Request Logs Error:', error);
    throw error;
  }
};

// -----------------------------
// Health
// -----------------------------
export const getHealthStatus = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health Check Error:', error);
    throw error;
  }
};

export default api;