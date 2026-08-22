import React from 'react';

export default function AlertsTable({ alerts }: { alerts: any[] }) {
  const riskClass = (level: string) => level === 'HIGH_RISK' ? 'high-risk' : 'suspicious';

  return (
    <div className="card">
      <h3>⚠️ Recent Alerts</h3>
      <div style={{ maxHeight: 400, overflowY: 'auto' }}>
        <table>
          <thead><tr><th>Time</th><th>URL</th><th>Risk</th><th>Score</th><th>Rules Triggered</th></tr></thead>
          <tbody>
            {alerts.map(a => (
              <tr key={a.scan_id}>
                <td>{a.timestamp ? new Date(a.timestamp).toLocaleString() : '—'}</td>
                <td style={{ maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis' }}>{a.url}</td>
                <td><span className={`badge ${riskClass(a.risk_level)}`}>{a.risk_level}</span></td>
                <td>{a.fused_score?.toFixed(2)}</td>
                <td>{a.triggered_rules?.length || 0}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
