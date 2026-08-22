import React, { useState, useEffect } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import LoginForm from './components/LoginForm';
import StatsCards from './components/StatsCards';
import LiveFeed from './components/LiveFeed';
import RiskChart from './components/RiskChart';
import TopRulesTable from './components/TopRulesTable';
import AlertsTable from './components/AlertsTable';
import SystemHealth from './components/SystemHealth';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const [token, setToken] = useState<string | null>(localStorage.getItem('admin_token'));
  const [stats, setStats] = useState<any>(null);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [topRules, setTopRules] = useState<any[]>([]);
  const [health, setHealth] = useState<any>(null);
  const { events, connected } = useWebSocket(token);

  const headers = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };

  const fetchAll = async () => {
    if (!token) return;
    try {
      const [statsRes, alertsRes, rulesRes, healthRes] = await Promise.all([
        fetch(`${API_URL}/admin/stats`, { headers }),
        fetch(`${API_URL}/admin/alerts?limit=50`, { headers }),
        fetch(`${API_URL}/admin/top-rules?limit=100`, { headers }),
        fetch(`${API_URL}/admin/health`, { headers })
      ]);

      if (statsRes.ok) setStats(await statsRes.json());
      if (alertsRes.ok) setAlerts(await alertsRes.json());
      if (rulesRes.ok) setTopRules(await rulesRes.json());
      if (healthRes.ok) setHealth(await healthRes.json());
    } catch (err) {
      console.error('Failed to fetch dashboard data', err);
    }
  };

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 15000); // Refresh every 15s
    return () => clearInterval(interval);
  }, [token]);

  const handleLogin = (newToken: string) => {
    setToken(newToken);
    localStorage.setItem('admin_token', newToken);
  };

  const handleLogout = () => {
    setToken(null);
    localStorage.removeItem('admin_token');
  };

  if (!token) return <LoginForm onLogin={handleLogin} apiUrl={API_URL} />;

  return (
    <div className="dashboard">
      <div className="header">
        <h1>🛡️ PhishGuard Admin</h1>
        <div>
          <span className="live-dot" />
          {connected ? 'Live' : 'Reconnecting...'}
          <button onClick={handleLogout} style={{ marginLeft: 20, padding: '6px 16px', borderRadius: 6, border: '1px solid #475569', background: 'transparent', color: '#94a3b8', cursor: 'pointer' }}>
            Logout
          </button>
        </div>
      </div>

      <StatsCards stats={stats} />

      <div className="grid" style={{ gridTemplateColumns: '2fr 1fr' }}>
        <LiveFeed events={events} />
        <RiskChart stats={stats} />
      </div>

      <div className="grid" style={{ marginTop: 20 }}>
        <AlertsTable alerts={alerts} />
        <TopRulesTable rules={topRules} />
      </div>

      <div style={{ marginTop: 20 }}>
        <SystemHealth health={health} />
      </div>
    </div>
  );
}
