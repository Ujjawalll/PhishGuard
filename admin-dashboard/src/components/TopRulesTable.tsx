import React from 'react';

export default function TopRulesTable({ rules }: { rules: any[] }) {
  return (
    <div className="card">
      <h3>Top Triggered Rules</h3>
      <table>
        <thead><tr><th>Rule ID</th><th>Trigger Count</th></tr></thead>
        <tbody>
          {rules.map(r => (
            <tr key={r.rule_id}><td>{r.rule_id}</td><td>{r.count}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
