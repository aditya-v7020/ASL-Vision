import React, { useRef, useState, useEffect, useCallback } from 'react';
import {
  Camera,
  CameraOff,
  Sparkles,
  AlertCircle,
  FlipHorizontal,
  Eye,
  Sliders,
  ShieldCheck,
  Activity,
  CheckCircle2,
  HelpCircle,
  Sun,
  Focus,
  Maximize2
} from 'lucide-react';
import { predictImage, getReferenceImageUrl } from '../services/api';

export default function WebcamPredictor({ onPrediction, onError, isAnalyzing, setIsAnalyzing }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [autoContinuous, setAutoContinuous] = useState(false);
  const [mirrorCapture, setMirrorCapture] = useState(true); // Default true: aligns right-hand webcam view with right-hand dataset
  const [debugMode, setDebugMode] = useState(false);
  
  // Last capture & debug inspection states
  const [lastCropPreview, setLastCropPreview] = useState(null);
  const [lastRawPreview, setLastRawPreview] = useState(null);
  const [lastPredictionData, setLastPredictionData] = useState(null);
  const [frameStats, setFrameStats] = useState(null);

  // Temporal smoothing history buffer (rolling window)
  const historyQueueRef = useRef([]); // Stores last N raw results
  const consecutiveAgreeCountRef = useRef(0);
  const lastStableClassRef = useRef(null);

  // Sync stream with video element whenever cameraActive or stream changes
  useEffect(() => {
    if (cameraActive && streamRef.current && videoRef.current) {
      const video = videoRef.current;
      if (video.srcObject !== streamRef.current) {
        video.srcObject = streamRef.current;
      }
      video.play().catch((err) => {
        console.warn('Video auto-play interrupted or failed:', err);
      });
    }
  }, [cameraActive]);

  // Start Webcam Stream
  const startCamera = async () => {
    setCameraError(null);
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Webcam API (navigator.mediaDevices.getUserMedia) is not supported in this browser.');
      }

      // Stop any existing stream first
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch((err) => {
          console.warn('Video play error on start:', err);
        });
      }

      setCameraActive(true);
    } catch (err) {
      console.error('Camera access error:', err);
      let msg = 'Could not access camera.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = 'Camera permission denied. Please allow camera access in your browser settings.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        msg = 'No camera device detected on this system.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        msg = 'Camera is already in use by another application or tab.';
      } else {
        msg = err.message || msg;
      }
      setCameraError(msg);
      onError(msg);
      setCameraActive(false);
    }
  };

  // Stop Webcam Stream
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setAutoContinuous(false);
    historyQueueRef.current = [];
    consecutiveAgreeCountRef.current = 0;
    lastStableClassRef.current = null;
    setLastPredictionData(null);
  }, []);

  // Clean up stream on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  // Robust Multi-Frame Temporal Prediction Stabilization & Hand Presence Gating
  const applyTemporalSmoothing = useCallback((rawResult) => {
    if (!rawResult) return rawResult;

    const hasHand = rawResult.hand_presence?.has_hand ?? !rawResult.is_uncertain;

    // If no hand is detected or frame is uncertain, pass through immediately and reset stability
    if (!hasHand || rawResult.is_uncertain) {
      historyQueueRef.current.push(rawResult);
      if (historyQueueRef.current.length > 4) historyQueueRef.current.shift();
      consecutiveAgreeCountRef.current = 0;
      lastStableClassRef.current = null;
      return rawResult;
    }

    const queue = historyQueueRef.current;
    queue.push(rawResult);
    if (queue.length > 4) queue.shift();

    // Aggregate probabilities over valid recent frames where hand is confirmed present
    const validFrames = queue.filter(f => !f.is_uncertain && f.top_predictions && (f.hand_presence?.has_hand !== false));
    if (validFrames.length === 0) return rawResult;

    const classScores = {};
    const weights = [0.15, 0.25, 0.30, 0.30].slice(4 - validFrames.length);
    const weightSum = weights.reduce((a, b) => a + b, 0);

    validFrames.forEach((frame, idx) => {
      const w = weights[idx] / weightSum;
      frame.top_predictions.forEach(item => {
        classScores[item.class] = (classScores[item.class] || 0) + (item.confidence * w);
      });
    });

    const sortedSmoothed = Object.entries(classScores)
      .map(([cls, score]) => ({ class: cls, confidence: Math.round(score * 10) / 10 }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 5);

    const smoothedTopClass = sortedSmoothed[0]?.class || rawResult.prediction;

    // Stability verification counter (requires 2 consecutive agreements)
    if (smoothedTopClass === lastStableClassRef.current) {
      consecutiveAgreeCountRef.current += 1;
    } else {
      consecutiveAgreeCountRef.current = 1;
      lastStableClassRef.current = smoothedTopClass;
    }

    const isStable = consecutiveAgreeCountRef.current >= 2;

    return {
      ...rawResult,
      prediction: smoothedTopClass,
      confidence: sortedSmoothed[0]?.confidence || rawResult.confidence,
      top_predictions: sortedSmoothed.length > 0 ? sortedSmoothed : rawResult.top_predictions,
      raw_prediction: rawResult.prediction,
      is_stable: isStable,
      stability_count: consecutiveAgreeCountRef.current,
    };
  }, []);

  // Capture Frame matching the hand guide box and send to FastAPI backend
  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || !cameraActive || isAnalyzing) return;

    try {
      setIsAnalyzing(true);
      const video = videoRef.current;

      const videoWidth = video.videoWidth || 640;
      const videoHeight = video.videoHeight || 480;

      // Central square crop matching the dataset aspect ratio and calibrated scale
      const minDimension = Math.min(videoWidth, videoHeight);
      const cropSize = minDimension * 0.72; // ~72% dimension, perfectly matching dataset hand framing
      const startX = (videoWidth - cropSize) / 2;
      const startY = (videoHeight - cropSize) / 2;

      // 200x200 canvas matching the dataset 200x200 resolution
      const canvas = document.createElement('canvas');
      canvas.width = 200;
      canvas.height = 200;

      const ctx = canvas.getContext('2d');
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';

      // Mirror capture if active (standard for user right hand)
      if (mirrorCapture) {
        ctx.save();
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(
          video,
          startX,
          startY,
          cropSize,
          cropSize,
          0,
          0,
          canvas.width,
          canvas.height
        );
        ctx.restore();
      } else {
        ctx.drawImage(
          video,
          startX,
          startY,
          cropSize,
          cropSize,
          0,
          0,
          canvas.width,
          canvas.height
        );
      }

      // Generate visual thumbnail previews for debugging
      const cropDataUrl = canvas.toDataURL('image/jpeg', 0.95);
      setLastCropPreview(cropDataUrl);

      if (debugMode) {
        const rawCanvas = document.createElement('canvas');
        rawCanvas.width = 320;
        rawCanvas.height = 240;
        const rawCtx = rawCanvas.getContext('2d');
        rawCtx.drawImage(video, 0, 0, 320, 240);
        setLastRawPreview(rawCanvas.toDataURL('image/jpeg', 0.8));
      }

      setFrameStats({
        rawResolution: `${videoWidth}x${videoHeight}`,
        cropArea: `${Math.round(cropSize)}x${Math.round(cropSize)} (at X:${Math.round(startX)}, Y:${Math.round(startY)})`,
        targetCanvas: '200x200 RGB',
        modelInput: '128x128x3 Float32',
        orientation: mirrorCapture ? 'Right-Hand Canonical (Mirrored)' : 'Left-Hand Direct',
      });

      // Convert canvas to JPEG Blob for multipart upload
      canvas.toBlob(async (blob) => {
        if (!blob) {
          setIsAnalyzing(false);
          return;
        }

        try {
          const rawResult = await predictImage(blob);
          const finalResult = autoContinuous ? applyTemporalSmoothing(rawResult) : rawResult;
          setLastPredictionData(finalResult);
          onPrediction(finalResult);
        } catch (err) {
          console.error('Prediction API Error:', err);
          const errorMsg = err.response?.data?.detail || err.message || 'Failed to analyze sign.';
          onError(errorMsg);
        } finally {
          setIsAnalyzing(false);
        }
      }, 'image/jpeg', 0.95);

    } catch (err) {
      console.error('Capture error:', err);
      setIsAnalyzing(false);
      onError('Failed to capture frame from webcam.');
    }
  }, [cameraActive, isAnalyzing, mirrorCapture, debugMode, autoContinuous, applyTemporalSmoothing, onPrediction, onError, setIsAnalyzing]);

  // Auto-Continuous prediction loop
  useEffect(() => {
    let intervalId = null;
    if (cameraActive && autoContinuous && !isAnalyzing) {
      intervalId = setInterval(() => {
        captureAndAnalyze();
      }, 1000);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [cameraActive, autoContinuous, isAnalyzing, captureAndAnalyze]);

  const activePredClass = lastPredictionData && !lastPredictionData.is_uncertain ? lastPredictionData.prediction : null;
  const referenceImgUrl = activePredClass ? getReferenceImageUrl(activePredClass) : null;
  const qMetrics = lastPredictionData?.quality_metrics;

  return (
    <div className="glass-card">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h2 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Camera size={20} color="var(--accent-primary)" />
          Live Webcam Feed
        </h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {cameraActive && (
            <span className="badge badge-success">
              <span className="live-indicator" />
              Camera Active
            </span>
          )}
          <button
            type="button"
            onClick={() => setDebugMode(!debugMode)}
            className={`btn ${debugMode ? 'btn-primary' : 'btn-secondary'}`}
            style={{ fontSize: '0.75rem', padding: '4px 8px', height: '26px', gap: '4px' }}
            title="Toggle Preprocessing & Dataset Reference Inspector"
          >
            <Sliders size={13} />
            Debug Mode
          </button>
        </div>
      </div>

      {cameraError && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.12)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: 'var(--radius-md)',
          padding: '12px 16px',
          color: '#f87171',
          fontSize: '0.9rem',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          marginBottom: '16px',
        }}>
          <AlertCircle size={18} />
          <span>{cameraError}</span>
        </div>
      )}

      {/* Video Viewport with Calibrated Target Guide Box */}
      <div className="viewport-wrapper">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          id="webcam-video-element"
          className="viewport-video"
          style={{
            display: cameraActive ? 'block' : 'none',
            transform: 'scaleX(-1)', // Mirror preview for natural user experience
            zIndex: 1,
          }}
          onLoadedMetadata={() => {
            if (videoRef.current) {
              videoRef.current.play().catch((err) => console.warn('LoadedMetadata play:', err));
            }
          }}
        />

        {cameraActive && (
          <div className="target-box" style={{ zIndex: 10 }}>
            <div className="target-box-corner tl" />
            <div className="target-box-corner tr" />
            <div className="target-box-corner bl" />
            <div className="target-box-corner br" />
            <div style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: '40px',
              height: '40px',
              border: '1px dashed rgba(255, 255, 255, 0.25)',
              borderRadius: '50%',
              pointerEvents: 'none'
            }} />
            <span className="target-box-label">Position Hand Inside Guide (~70% Frame)</span>
          </div>
        )}

        {!cameraActive && (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '20px', zIndex: 2 }}>
            <CameraOff size={48} style={{ opacity: 0.4, marginBottom: '12px' }} />
            <p style={{ fontSize: '0.95rem' }}>Camera is currently turned off.</p>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              Click "Start Camera" below to begin live recognition.
            </p>
          </div>
        )}
      </div>

      {/* Real-time Quality & Orientation Status Strip */}
      {cameraActive && lastPredictionData && (
        <div style={{
          marginTop: '12px',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '8px',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 12px',
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-sm)',
          fontSize: '0.78rem'
        }}>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
            <span style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <ShieldCheck size={13} color={lastPredictionData.hand_presence?.has_hand ? '#34d399' : '#f87171'} />
              Hand: <strong>{lastPredictionData.hand_presence?.has_hand ? 'Detected' : lastPredictionData.hand_presence?.status === 'NO_HAND' ? 'None' : 'Edge/Small'}</strong>
            </span>
            <span style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Sun size={13} color={qMetrics && (qMetrics.brightness < 35 || qMetrics.brightness > 230) ? '#f87171' : '#34d399'} />
              Light: <strong>{qMetrics ? `${qMetrics.brightness}` : 'OK'}</strong>
            </span>
            <span style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Focus size={13} color={qMetrics && qMetrics.sharpness < 25 ? '#f87171' : '#34d399'} />
              Sharpness: <strong>{qMetrics ? (qMetrics.sharpness >= 25 ? 'Sharp' : 'Blurry') : 'OK'}</strong>
            </span>
            {lastPredictionData.prototype_similarity && (
              <span style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <CheckCircle2 size={13} color={lastPredictionData.prototype_similarity >= 0.817 ? '#34d399' : '#fbbf24'} />
                Ref Match: <strong>{(lastPredictionData.prototype_similarity * 100).toFixed(1)}%</strong>
              </span>
            )}
          </div>
          {lastPredictionData.is_stable && (
            <span className="badge badge-success" style={{ fontSize: '0.7rem', padding: '2px 6px' }}>
              Stable
            </span>
          )}
        </div>
      )}

      {/* Primary Controls */}
      <div style={{ marginTop: '14px', display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', alignItems: 'center' }}>
          {!cameraActive ? (
            <button id="start-camera-btn" className="btn btn-primary" onClick={startCamera}>
              <Camera size={18} />
              Start Camera
            </button>
          ) : (
            <>
              <button id="stop-camera-btn" className="btn btn-danger" onClick={stopCamera}>
                <CameraOff size={18} />
                Stop Camera
              </button>
              <button
                id="analyze-webcam-btn"
                className="btn btn-primary"
                onClick={captureAndAnalyze}
                disabled={isAnalyzing}
              >
                <Sparkles size={18} />
                {isAnalyzing ? 'Analyzing...' : 'Capture & Analyze'}
              </button>
            </>
          )}
        </div>

        {cameraActive && (
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
              <input
                type="checkbox"
                checked={autoContinuous}
                onChange={(e) => setAutoContinuous(e.target.checked)}
                style={{ accentColor: 'var(--accent-primary)', width: '15px', height: '15px' }}
              />
              Auto-Stream Scan
            </label>

            <button
              type="button"
              className="btn btn-secondary"
              style={{ fontSize: '0.8rem', padding: '6px 10px', gap: '5px' }}
              onClick={() => {
                setMirrorCapture(!mirrorCapture);
                historyQueueRef.current = [];
              }}
              title="Toggle horizontal orientation to match right vs left hand"
            >
              <FlipHorizontal size={14} />
              {mirrorCapture ? 'Right Hand (Mirrored)' : 'Left Hand (Direct)'}
            </button>
          </div>
        )}
      </div>

      {/* Visual Reference & Crop Comparison Card */}
      {cameraActive && lastCropPreview && (
        <div style={{
          marginTop: '14px',
          padding: '12px 14px',
          background: 'rgba(255, 255, 255, 0.03)',
          border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-md)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Eye size={14} color="var(--accent-primary)" />
              Model Preprocessed Input vs Dataset Reference
            </div>
            <span className="badge" style={{ fontSize: '0.72rem', background: 'rgba(99, 102, 241, 0.15)', color: '#818cf8', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
              {mirrorCapture ? 'Right-Hand Canonical' : 'Left-Hand Direct'}
            </span>
          </div>

          <div style={{ display: 'flex', gap: '14px', alignItems: 'center' }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
              <img
                src={lastCropPreview}
                alt="Webcam Model Crop"
                style={{ width: '56px', height: '56px', borderRadius: '6px', border: '1px solid var(--border-color)', objectFit: 'cover' }}
              />
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Webcam Crop</span>
            </div>

            {referenceImgUrl && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
                <img
                  src={referenceImgUrl}
                  alt={`Dataset Reference for ${activePredClass}`}
                  onError={(e) => { e.target.style.display = 'none'; }}
                  style={{ width: '56px', height: '56px', borderRadius: '6px', border: '1px solid rgba(99, 102, 241, 0.4)', objectFit: 'cover' }}
                />
                <span style={{ fontSize: '0.7rem', color: 'var(--accent-primary)', fontWeight: 600 }}>
                  Dataset Ref ({activePredClass})
                </span>
              </div>
            )}

            <div style={{ flex: 1, fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              <div>
                <strong>Pipeline:</strong> 200×200 crop $\rightarrow$ Bilinear 128×128 tensor.
              </div>
              <div style={{ marginTop: '2px', color: 'var(--text-muted)' }}>
                Comparing hand silhouette with 87,000 reference training prototypes.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Comprehensive Debug Inspector Panel */}
      {debugMode && cameraActive && frameStats && (
        <div style={{
          marginTop: '16px',
          padding: '14px',
          background: 'rgba(15, 23, 42, 0.95)',
          border: '1px solid rgba(99, 102, 241, 0.3)',
          borderRadius: 'var(--radius-md)',
          fontSize: '0.8rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--accent-primary)', fontWeight: 700, marginBottom: '10px' }}>
            <ShieldCheck size={16} />
            PREPROCESSING & DATASET REFERENCE INSPECTOR
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '12px' }}>
            {lastRawPreview && (
              <div>
                <div style={{ color: 'var(--text-muted)', marginBottom: '4px' }}>Raw Camera Stream:</div>
                <img src={lastRawPreview} alt="Raw Stream" style={{ width: '100%', borderRadius: '4px', border: '1px solid var(--border-color)' }} />
              </div>
            )}
            {lastCropPreview && (
              <div>
                <div style={{ color: 'var(--text-muted)', marginBottom: '4px' }}>Model Input (debug_webcam_input.jpg):</div>
                <img src={lastCropPreview} alt="Cropped Input" style={{ width: '100%', borderRadius: '4px', border: '1px solid var(--border-color)' }} />
              </div>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '8px' }}>
            <div style={{ background: 'rgba(255, 255, 255, 0.04)', padding: '6px 10px', borderRadius: '4px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Raw Resolution: </span>
              <strong>{frameStats.rawResolution}</strong>
            </div>
            <div style={{ background: 'rgba(255, 255, 255, 0.04)', padding: '6px 10px', borderRadius: '4px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Crop Box Area: </span>
              <strong>{frameStats.cropArea}</strong>
            </div>
            <div style={{ background: 'rgba(255, 255, 255, 0.04)', padding: '6px 10px', borderRadius: '4px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Model Input: </span>
              <strong>{frameStats.modelInput}</strong>
            </div>
            <div style={{ background: 'rgba(255, 255, 255, 0.04)', padding: '6px 10px', borderRadius: '4px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Orientation: </span>
              <strong>{frameStats.orientation}</strong>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

