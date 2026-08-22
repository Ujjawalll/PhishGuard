import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';

function App() {
  const [token, setToken] = useState<string | null>(null);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [result, setResult] = useState<any>(null);

  useEffect(() => {
    chrome.storage.local.get(['token']).then(res => {
      if (res.token) setToken(res.token);
    });
    
    chrome.tabs.query({ active: true, currentWindow: true }).then(tabs => {
      if (tabs[0]?.id) {
        chrome.storage.local.get([`result_${tabs[0].id}`]).then(res => {
          setResult(res[`result_${tabs[0].id}`]);
        });
      }
    });
  }, []);

  const login = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const { apiUrl = 'http://localhost:8000' } = await chrome.storage.local.get(['apiUrl']);
      const res = await fetch(`${apiUrl}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      if (res.ok) {
        const data = await res.json();
        setToken(data.access_token);
        chrome.storage.local.set({ token: data.access_token });
      } else {
        alert("Login failed");
      }
    } catch (err) {
      alert("Cannot reach API");
    }
  };

  const logout = () => {
    setToken(null);
    chrome.storage.local.remove('token');
  };

  if (!token) {
    return (
      <div style={{ padding: '20px' }}>
        <h2>PhishGuard Login</h2>
        <form onSubmit={login}>
          <input type="email" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} style={{ width: '100%', marginBottom: '10px', padding: '8px' }} required />
          <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} style={{ width: '100%', marginBottom: '10px', padding: '8px' }} required />
          <button type="submit" style={{ width: '100%', padding: '8px', background: '#3B82F6', color: 'white', border: 'none', borderRadius: '4px' }}>Login</button>
        </form>
      </div>
    );
  }

  return (
    <div style={{ padding: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>PhishGuard</h2>
        <button onClick={logout} style={{ background: 'transparent', border: 'none', cursor: 'pointer', textDecoration: 'underline' }}>Logout</button>
      </div>
      
      {result ? (
        <div style={{ marginTop: '20px', padding: '15px', borderRadius: '8px', background: result.risk_level === 'HIGH_RISK' ? '#FEE2E2' : result.risk_level === 'SUSPICIOUS' ? '#FEF3C7' : '#D1FAE5' }}>
          <h3 style={{ margin: '0 0 10px 0', color: result.risk_level === 'HIGH_RISK' ? '#DC2626' : result.risk_level === 'SUSPICIOUS' ? '#D97706' : '#059669' }}>
            Risk Level: {result.risk_level}
          </h3>
          <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%</p>
          <h4>Explanation:</h4>
          <ul style={{ paddingLeft: '20px', margin: '5px 0' }}>
            {result.explanation.top_reasons.map((r: string, i: number) => <li key={i}>{r}</li>)}
          </ul>
          <p style={{ fontWeight: 'bold', marginTop: '10px' }}>{result.explanation.recommendation}</p>
        </div>
      ) : (
        <p>No scan data for this tab yet. Reload the page to scan.</p>
      )}
    </div>
  );
}

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
