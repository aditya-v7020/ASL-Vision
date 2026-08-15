import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
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

export default {
  checkHealth,
  getClasses,
  predictImage,
  getReferenceImageUrl,
};

