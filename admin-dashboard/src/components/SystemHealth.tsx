import React from 'react';

export default function SystemHealth({ health }: { health: any }) {
  if (!health) return null;

  const items = [
    { label: 'API', ok: health.api === 'ok' },
    { label: 'Database', ok: health.database === 'ok' },
    { label: 'ML Model', ok: health.ml_model_loaded },
    { label: 'Rule Engine', ok: health.rule_engine_loaded },
    { label: 'Explainer', ok: health.explainer_loaded },
  ];

  return (
    <div className="card">
      <h3>System Health</h3>
      <div style={{ display: 'flex', gap: 24, marginTop: 8 }}>
        {items.map(i => (
          <div key={i.label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ width: 12, height: 12, borderRadius: '50%', background: i.ok ? '#10b981' : '#ef4444' }} />
            <span>{i.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
