import React from 'react';
import { Activity, Layers, Sparkles } from 'lucide-react';

export default function Header({ backendHealth, loadingHealth }) {
  const isHealthy = backendHealth?.status === 'healthy';

  return (
    <header className="glass-card" style={{ marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px' }}>
            <div style={{
              background: 'var(--accent-gradient)',
              borderRadius: '10px',
              padding: '6px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: 'var(--accent-glow)'
            }}>
              <Sparkles size={22} color="#ffffff" />
            </div>
            <h1 style={{ fontSize: '1.75rem', fontWeight: 800 }}>SignSense AI</h1>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem' }}>
            Deep Learning ASL Alphabet Recognition (TensorFlow + Keras CNN & FastAPI)
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className={`badge ${isHealthy ? 'badge-success' : 'badge-danger'}`}>
            <span className={isHealthy ? 'live-indicator' : ''} style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: isHealthy ? 'var(--success)' : 'var(--danger)',
              display: 'inline-block'
            }} />
            <span>
              {loadingHealth ? 'Connecting...' : isHealthy ? 'Backend Connected' : 'Backend Offline / Pending'}
            </span>
          </div>

          {backendHealth?.classes_count && (
            <div className="badge" style={{ background: 'rgba(255, 255, 255, 0.05)', color: 'var(--text-secondary)', border: '1px solid var(--border-color)' }}>
              <Layers size={14} />
              <span>{backendHealth.classes_count} Signs</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
