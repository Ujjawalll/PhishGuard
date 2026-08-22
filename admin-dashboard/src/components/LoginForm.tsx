import React, { useState } from 'react';

export default function LoginForm({ onLogin, apiUrl }: { onLogin: (token: string) => void; apiUrl: string }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await fetch(`${apiUrl}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      if (res.ok) {
        const data = await res.json();
        onLogin(data.access_token);
      } else {
        setError('Invalid credentials or not an admin');
      }
    } catch {
      setError('Cannot reach API');
    }
  };

  return (
    <div className="login-container">
      <form className="login-form" onSubmit={handleSubmit}>
        <h2 style={{ marginBottom: 24, textAlign: 'center' }}>🛡️ PhishGuard Admin</h2>
        {error && <p style={{ color: '#ef4444', marginBottom: 12 }}>{error}</p>}
        <input type="email" placeholder="Admin Email" value={email} onChange={e => setEmail(e.target.value)} required />
        <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} required />
        <button type="submit">Login</button>
      </form>
    </div>
  );
}
