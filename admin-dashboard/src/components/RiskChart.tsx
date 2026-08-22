import React from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const COLORS = { SAFE: '#10b981', SUSPICIOUS: '#f59e0b', HIGH_RISK: '#ef4444' };

export default function RiskChart({ stats }: { stats: any }) {
  if (!stats?.risk_distribution) return null;

  const data = Object.entries(stats.risk_distribution).map(([name, value]) => ({ name, value }));

  return (
    <div className="card">
      <h3>Risk Distribution</h3>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label>
            {data.map((entry: any) => (
              <Cell key={entry.name} fill={COLORS[entry.name as keyof typeof COLORS] || '#64748b'} />
            ))}
          </Pie>
          <Tooltip />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
