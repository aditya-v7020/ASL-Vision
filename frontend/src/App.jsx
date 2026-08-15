import React, { useState, useEffect } from 'react';
import { Camera, UploadCloud, Info, AlertTriangle, X } from 'lucide-react';
import Header from './components/Header';
import WebcamPredictor from './components/WebcamPredictor';
import ImageUploadPredictor from './components/ImageUploadPredictor';
import PredictionResult from './components/PredictionResult';
import { checkHealth, getClasses } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('webcam'); // 'webcam' | 'upload'
  const [predictionResult, setPredictionResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [backendHealth, setBackendHealth] = useState(null);
  const [loadingHealth, setLoadingHealth] = useState(true);
  const [availableClasses, setAvailableClasses] = useState([]);
  const [errorMessage, setErrorMessage] = useState(null);

  // Check backend health on mount and periodically
  useEffect(() => {
    const fetchHealthAndClasses = async () => {
      try {
        setLoadingHealth(true);
        const health = await checkHealth();
        setBackendHealth(health);
        
        const classesData = await getClasses();
        setAvailableClasses(classesData.classes || []);
      } catch (err) {
        console.warn('Backend currently unreachable:', err.message);
        setBackendHealth({ status: 'offline', model_loaded: false });
      } finally {
        setLoadingHealth(false);
      }
    };

    fetchHealthAndClasses();
    const interval = setInterval(fetchHealthAndClasses, 20000);
    return () => clearInterval(interval);
  }, []);

  const handlePrediction = (result) => {
    setPredictionResult(result);
    setErrorMessage(null);
  };

  const handleError = (msg) => {
    setErrorMessage(msg);
  };

  return (
    <div className="app-container">
      {/* Top Header */}
      <Header backendHealth={backendHealth} loadingHealth={loadingHealth} />

      {/* Global Error Banner */}
      {errorMessage && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.15)',
          border: '1px solid rgba(239, 68, 68, 0.35)',
          borderRadius: 'var(--radius-md)',
          padding: '12px 18px',
          color: '#fca5a5',
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <AlertTriangle size={18} />
            <span style={{ fontSize: '0.9rem' }}>{errorMessage}</span>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            style={{ background: 'transparent', border: 'none', color: '#fca5a5', cursor: 'pointer' }}
          >
            <X size={16} />
          </button>
        </div>
      )}

      {/* Tab Selectors */}
      <div className="tabs-container">
        <button
          id="tab-webcam"
          className={`tab-btn ${activeTab === 'webcam' ? 'active' : ''}`}
          onClick={() => setActiveTab('webcam')}
        >
          <Camera size={18} />
          Webcam Live Recognition
        </button>
        <button
          id="tab-upload"
          className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          <UploadCloud size={18} />
          Image File Upload
        </button>
      </div>

      {/* Main Grid */}
      <div className="main-grid">
        {/* Left Column: Input Source */}
        <div>
          {activeTab === 'webcam' ? (
            <WebcamPredictor
              onPrediction={handlePrediction}
              onError={handleError}
              isAnalyzing={isAnalyzing}
              setIsAnalyzing={setIsAnalyzing}
            />
          ) : (
            <ImageUploadPredictor
              onPrediction={handlePrediction}
              onError={handleError}
              isAnalyzing={isAnalyzing}
              setIsAnalyzing={setIsAnalyzing}
            />
          )}
        </div>

        {/* Right Column: Prediction Results */}
        <div>
          <PredictionResult result={predictionResult} loading={isAnalyzing} />
        </div>
      </div>

      {/* ASL Alphabet Supported Classes Grid / Cheatsheet */}
      {availableClasses.length > 0 && (
        <div className="glass-card" style={{ marginTop: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
            <Info size={18} color="var(--accent-primary)" />
            <h3 style={{ fontSize: '1.05rem' }}>Supported ASL Sign Classes ({availableClasses.length})</h3>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(60px, 1fr))',
            gap: '8px',
            textAlign: 'center'
          }}>
            {availableClasses.map((cls, idx) => (
              <div
                key={idx}
                style={{
                  background: predictionResult?.prediction === cls ? 'var(--accent-gradient)' : 'rgba(255, 255, 255, 0.04)',
                  color: predictionResult?.prediction === cls ? '#ffffff' : 'var(--text-secondary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '8px 4px',
                  fontWeight: 700,
                  fontSize: '0.9rem',
                  transition: 'var(--transition)'
                }}
              >
                {cls}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
