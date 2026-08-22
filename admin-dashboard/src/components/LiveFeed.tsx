import React from 'react';

export default function LiveFeed({ events }: { events: any[] }) {
  const riskClass = (level: string) => level === 'HIGH_RISK' ? 'high-risk' : level === 'SUSPICIOUS' ? 'suspicious' : 'safe';

  return (
    <div className="card">
      <h3><span className="live-dot" />Live Scan Feed</h3>
      <div style={{ maxHeight: 400, overflowY: 'auto' }}>
        <table>
          <thead>
            <tr><th>Time</th><th>Domain</th><th>Risk</th><th>Score</th><th>Stage</th></tr>
          </thead>
          <tbody>
            {events.length === 0 && (
              <tr><td colSpan={5} style={{ textAlign: 'center', color: '#64748b' }}>Waiting for scans...</td></tr>
            )}
            {events.map(e => (
              <tr key={e.scan_id}>
                <td>{new Date(e.timestamp).toLocaleTimeString()}</td>
                <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>{e.domain}</td>
                <td><span className={`badge ${riskClass(e.risk_level)}`}>{e.risk_level}</span></td>
                <td>{e.fused_score?.toFixed(2)}</td>
                <td>{e.stage}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
