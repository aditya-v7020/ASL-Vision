import React from 'react';
import { Award, BarChart3, CheckCircle2, AlertTriangle, Info, HelpCircle, Sun, Focus, ShieldCheck } from 'lucide-react';

export default function PredictionResult({ result, loading }) {
  if (loading) {
    return (
      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '380px', textAlign: 'center' }}>
        <div style={{
          width: '56px',
          height: '56px',
          border: '4px solid rgba(99, 102, 241, 0.2)',
          borderTopColor: 'var(--accent-primary)',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          marginBottom: '16px'
        }} />
        <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
        <h3 style={{ fontSize: '1.2rem', marginBottom: '6px' }}>Analyzing Sign...</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Running TensorFlow CNN inference</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '380px', textAlign: 'center' }}>
        <div style={{
          width: '64px',
          height: '64px',
          borderRadius: '50%',
          background: 'rgba(255, 255, 255, 0.04)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: '16px',
          color: 'var(--text-muted)'
        }}>
          <BarChart3 size={32} />
        </div>
        <h3 style={{ fontSize: '1.25rem', marginBottom: '8px', color: 'var(--text-primary)' }}>Awaiting Sign Input</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', maxWidth: '300px' }}>
          Capture a frame from your webcam or upload a sign image to see the model prediction.
        </p>
      </div>
    );
  }

  const isUncertain = result.is_uncertain === true || result.prediction?.startsWith('Uncertain');
  const confidence = result.confidence || 0;
  const isHighConfidence = confidence >= 70;
  const isModerateConfidence = confidence >= 45 && confidence < 70;
  const isLowConfidence = confidence < 45;

  return (
    <div className="glass-card">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '1.15rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Award size={18} color="var(--accent-primary)" />
          Prediction Result
        </h3>
        {isUncertain ? (
          <span className="badge" style={{ background: 'rgba(245, 158, 11, 0.15)', color: '#fbbf24', border: '1px solid rgba(245, 158, 11, 0.3)' }}>
            <AlertTriangle size={12} />
            Uncertain Frame
          </span>
        ) : isHighConfidence ? (
          <span className="badge badge-success">
            <CheckCircle2 size={12} />
            High Confidence
          </span>
        ) : isModerateConfidence ? (
          <span className="badge" style={{ background: 'rgba(245, 158, 11, 0.15)', color: '#fbbf24', border: '1px solid rgba(245, 158, 11, 0.3)' }}>
            <Info size={12} />
            Moderate Confidence
          </span>
        ) : (
          <span className="badge badge-danger">
            <AlertTriangle size={12} />
            Low Confidence
          </span>
        )}
      </div>

      {/* Main Prediction Display */}
      {isUncertain ? (
        <div style={{
          padding: '24px 20px',
          background: result.hand_presence && !result.hand_presence.has_hand && result.hand_presence.status === 'NO_HAND'
            ? 'rgba(99, 102, 241, 0.08)'
            : 'rgba(245, 158, 11, 0.08)',
          border: result.hand_presence && !result.hand_presence.has_hand && result.hand_presence.status === 'NO_HAND'
            ? '1px solid rgba(99, 102, 241, 0.3)'
            : '1px solid rgba(245, 158, 11, 0.3)',
          borderRadius: 'var(--radius-lg)',
          textAlign: 'center',
          marginBottom: '16px'
        }}>
          <div style={{
            width: '52px',
            height: '52px',
            borderRadius: '50%',
            background: result.hand_presence && !result.hand_presence.has_hand && result.hand_presence.status === 'NO_HAND'
              ? 'rgba(99, 102, 241, 0.18)'
              : 'rgba(245, 158, 11, 0.18)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 12px auto',
            color: result.hand_presence && !result.hand_presence.has_hand && result.hand_presence.status === 'NO_HAND'
              ? 'var(--accent-primary)'
              : '#fbbf24'
          }}>
            {result.hand_presence && !result.hand_presence.has_hand && result.hand_presence.status === 'NO_HAND' ? (
              <Focus size={28} />
            ) : (
              <HelpCircle size={28} />
            )}
          </div>
          <h4 style={{
            fontSize: '1.2rem',
            color: result.hand_presence && !result.hand_presence.has_hand && result.hand_presence.status === 'NO_HAND'
              ? 'var(--accent-primary)'
              : '#fbbf24',
            fontWeight: 700,
            marginBottom: '6px'
          }}>
            {result.hand_presence && !result.hand_presence.has_hand && result.hand_presence.status === 'NO_HAND'
              ? 'No hand detected — place your hand inside the guide.'
              : result.hand_presence && result.hand_presence.status === 'PARTIAL_HAND_EDGE'
              ? 'Move your hand into the guide'
              : result.quality_metrics && result.quality_metrics.sharpness < 25
              ? 'Hold your hand steady'
              : result.quality_metrics && result.quality_metrics.brightness < 30
              ? 'Improve lighting'
              : 'Uncertain — adjust your hand position'}
          </h4>
          {result.uncertainty_reason && (
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '14px' }}>
              {result.uncertainty_reason}
            </p>
          )}

          <div style={{
            background: 'rgba(0, 0, 0, 0.25)',
            borderRadius: 'var(--radius-sm)',
            padding: '10px 14px',
            textAlign: 'left',
            fontSize: '0.8rem',
            color: 'var(--text-secondary)',
            display: 'flex',
            flexDirection: 'column',
            gap: '6px'
          }}>
            <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '2px' }}>
              Guidance Tips:
            </div>
            <div>• <strong>Placement:</strong> Position hand inside the guide box (~70% frame fill).</div>
            <div>• <strong>Stability:</strong> Hold steady for a sharp capture without motion blur.</div>
            <div>• <strong>Illumination:</strong> Ensure bright, even lighting on palm and fingers.</div>
            <div>• <strong>Orientation:</strong> Use Right Hand canonical mode (or toggle for left hand).</div>
          </div>
        </div>
      ) : (
        <div className="sign-display-card">
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Detected ASL Sign
          </p>
          <div className="sign-letter" id="detected-sign-result">
            {result.prediction}
          </div>
          <div style={{ marginTop: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.88rem', marginBottom: '6px' }}>
              <span style={{ color: 'var(--text-secondary)' }}>CNN Confidence</span>
              <span style={{ fontWeight: 700, color: isHighConfidence ? 'var(--text-primary)' : isModerateConfidence ? '#fbbf24' : '#f87171' }}>
                {result.confidence}%
              </span>
            </div>
            <div className="progress-bar-container">
              <div
                className="progress-bar-fill"
                style={{
                  width: `${Math.min(result.confidence, 100)}%`,
                  background: isHighConfidence ? 'var(--accent-gradient)' : isModerateConfidence ? 'linear-gradient(90deg, #f59e0b, #d97706)' : 'linear-gradient(90deg, #ef4444, #b91c1c)'
                }}
              />
            </div>
          </div>

          {result.prototype_similarity && (
            <div style={{
              marginTop: '12px',
              padding: '6px 10px',
              background: 'rgba(255, 255, 255, 0.03)',
              borderRadius: 'var(--radius-sm)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '0.78rem'
            }}>
              <span style={{ color: 'var(--text-muted)' }}>Dataset Prototype Alignment:</span>
              <span style={{ fontWeight: 600, color: result.prototype_similarity >= 0.817 ? '#34d399' : '#fbbf24' }}>
                {(result.prototype_similarity * 100).toFixed(1)}% (Ref Class: {result.prototype_match})
              </span>
            </div>
          )}
        </div>
      )}

      {/* Top Predictions Breakdown */}
      {result.top_predictions && result.top_predictions.length > 0 && (
        <div style={{ marginTop: '20px' }}>
          <h4 style={{ fontSize: '0.92rem', color: 'var(--text-secondary)', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Top Predictions
          </h4>
          <div>
            {result.top_predictions.map((item, idx) => (
              <div key={idx} className="top-prediction-item">
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{
                    width: '24px',
                    height: '24px',
                    borderRadius: '50%',
                    background: idx === 0 ? 'var(--accent-primary)' : 'rgba(255, 255, 255, 0.08)',
                    color: '#ffffff',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '0.75rem',
                    fontWeight: 700
                  }}>
                    {idx + 1}
                  </span>
                  <span style={{ fontWeight: 600, fontSize: '1rem' }}>{item.class}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', width: '50%' }}>
                  <div className="progress-bar-container" style={{ height: '6px' }}>
                    <div
                      className="progress-bar-fill"
                      style={{
                        width: `${Math.min(item.confidence, 100)}%`,
                        background: idx === 0 ? 'var(--accent-gradient)' : 'rgba(255, 255, 255, 0.2)'
                      }}
                    />
                  </div>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', minWidth: '46px', textAlign: 'right' }}>
                    {item.confidence}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

