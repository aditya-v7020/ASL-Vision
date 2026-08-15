import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, Sparkles, X, AlertCircle } from 'lucide-react';
import { predictImage } from '../services/api';

export default function ImageUploadPredictor({ onPrediction, onError, isAnalyzing, setIsAnalyzing }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      onError('Please select a valid image file (JPEG, PNG).');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      onError('Image size exceeds 10MB limit.');
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile || isAnalyzing) return;

    try {
      setIsAnalyzing(true);
      const result = await predictImage(selectedFile);
      onPrediction(result);
    } catch (err) {
      console.error('Upload prediction error:', err);
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to predict uploaded image.';
      onError(errorMsg);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const clearSelectedImage = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="glass-card">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h2 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <UploadCloud size={20} color="var(--accent-primary)" />
          Manual Image Upload
        </h2>
        {selectedFile && (
          <button
            onClick={clearSelectedImage}
            className="badge badge-danger"
            style={{ cursor: 'pointer', background: 'transparent' }}
          >
            <X size={12} />
            Clear
          </button>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        id="image-file-input"
        style={{ display: 'none' }}
        onChange={(e) => e.target.files && handleFile(e.target.files[0])}
      />

      {previewUrl ? (
        <div className="viewport-wrapper">
          <img src={previewUrl} alt="Sign Preview" className="viewport-image" />
          <div className="target-box" style={{ width: '80%', height: '80%' }}>
            <span className="target-box-label">{selectedFile?.name}</span>
          </div>
        </div>
      ) : (
        <div
          className={`dropzone ${isDragOver ? 'dragover' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
        >
          <div style={{
            width: '54px',
            height: '54px',
            borderRadius: '50%',
            background: 'rgba(99, 102, 241, 0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 12px',
            color: 'var(--accent-primary)'
          }}>
            <ImageIcon size={28} />
          </div>
          <h3 style={{ fontSize: '1.1rem', marginBottom: '6px' }}>Drag & Drop sign image here</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem' }}>or click to browse files</p>
          <span style={{ display: 'inline-block', marginTop: '12px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Supports JPG, JPEG, PNG (Max 10MB)
          </span>
        </div>
      )}

      <div style={{ marginTop: '18px', display: 'flex', gap: '12px' }}>
        <button
          className="btn btn-secondary"
          onClick={() => fileInputRef.current?.click()}
        >
          <ImageIcon size={18} />
          {previewUrl ? 'Change Image' : 'Select Image'}
        </button>

        <button
          id="analyze-image-btn"
          className="btn btn-primary"
          onClick={handleAnalyze}
          disabled={!selectedFile || isAnalyzing}
        >
          <Sparkles size={18} />
          {isAnalyzing ? 'Analyzing...' : 'Analyze Image'}
        </button>
      </div>
    </div>
  );
}
