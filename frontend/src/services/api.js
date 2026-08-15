import axios from 'axios';

// Resolve and sanitize API base URL from environment or local development fallback
const getApiBaseUrl = () => {
  const envUrl = import.meta.env.VITE_API_URL;
  if (envUrl && typeof envUrl === 'string' && envUrl.trim() !== '') {
    return envUrl.trim().replace(/\/+$/, '');
  }
  return 'http://127.0.0.1:8000';
};

const API_BASE_URL = getApiBaseUrl();

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 45000, // 45s timeout to gracefully accommodate Render cold starts
});

/**
 * Verifies backend API and model status
 */
export async function checkHealth() {
  const response = await apiClient.get('/health');
  return response.data;
}

/**
 * Retrieves the supported ASL alphabet class list
 */
export async function getClasses() {
  const response = await apiClient.get('/classes');
  return response.data;
}

/**
 * Sends an image file or webcam frame blob to the prediction endpoint
 * @param {Blob | File} imageFile 
 */
export async function predictImage(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile, 'capture.jpg');

  const response = await apiClient.post('/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

/**
 * Returns the URL for a canonical class reference image
 * @param {string} className 
 */
export function getReferenceImageUrl(className) {
  if (!className || className.startsWith('Uncertain')) return null;
  return `${API_BASE_URL}/reference/${encodeURIComponent(className)}`;
}

export { API_BASE_URL };

export default {
  checkHealth,
  getClasses,
  predictImage,
  getReferenceImageUrl,
  API_BASE_URL,
};

