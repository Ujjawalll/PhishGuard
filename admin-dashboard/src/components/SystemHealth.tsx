import React, { useEffect, useState } from 'react';

export default function SystemHealth({ health }: { health: any }) {
  const [modelInfo, setModelInfo] = useState<any>(null);

  useEffect(() => {
    const fetchModel = async () => {
      try {
        const base = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const res = await fetch(`${base}/model`);
        if (res.ok) setModelInfo(await res.json());
      } catch (e) {
        console.error("Failed to fetch model info", e);
      }
    };
    fetchModel();
  }, []);

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
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginTop: '16px' }}>
        <div>
          <h4 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#6b7280' }}>Component Status</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {items.map(i => (
              <div key={i.label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: i.ok ? '#10b981' : '#ef4444' }} />
                <span style={{ fontSize: '14px' }}>{i.label}</span>
              </div>
            ))}
          </div>
        </div>
        
        {modelInfo && (
          <div>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#6b7280' }}>Engine Details</h4>
            <div style={{ fontSize: '13px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <div><strong>Model:</strong> {modelInfo.model_name}</div>
              <div><strong>Version:</strong> {modelInfo.model_version}</div>
              <div><strong>Strategy:</strong> {modelInfo.fusion_strategy}</div>
              <div><strong>Rules:</strong> v{modelInfo.rule_config_version}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
