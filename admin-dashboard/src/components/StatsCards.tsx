import React from 'react';

export default function StatsCards({ stats }: { stats: any }) {
  if (!stats) return null;

  const cards = [
    { label: 'Total Scans', value: stats.total_scans, color: '#3b82f6' },
    { label: 'Today', value: stats.today_scans, color: '#8b5cf6' },
    { label: 'This Week', value: stats.week_scans, color: '#06b6d4' },
    { label: 'Active Users', value: stats.active_users, color: '#10b981' },
    { label: 'Avg ML Score', value: stats.avg_ml_probability?.toFixed(3), color: '#f59e0b' },
    { label: 'Avg Fused Score', value: stats.avg_fused_score?.toFixed(3), color: '#ef4444' },
  ];

  return (
    <div className="grid" style={{ gridTemplateColumns: 'repeat(6, 1fr)' }}>
      {cards.map(c => (
        <div className="card" key={c.label}>
          <h3>{c.label}</h3>
          <div className="stat-value" style={{ color: c.color }}>{c.value ?? '—'}</div>
        </div>
      ))}
    </div>
  );
}
