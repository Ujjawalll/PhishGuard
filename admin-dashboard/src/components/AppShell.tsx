import React from 'react';

export type NavRoute = 'home' | 'models' | 'model-detail' | 'compare' | 'deep-learning' | 'live-ops';

interface AppShellProps {
  currentRoute: NavRoute;
  onRouteChange: (route: NavRoute) => void;
  connected: boolean;
  token: string | null;
  onLogout: () => void;
  onOpenLogin: () => void;
  children: React.ReactNode;
}

export default function AppShell({
  currentRoute,
  onRouteChange,
  connected,
  token,
  onLogout,
  onOpenLogin,
  children
}: AppShellProps) {
  const navItems = [
    {
      id: 'home' as NavRoute,
      label: 'Project Overview',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
          <polyline points="9 22 9 12 15 12 15 22" />
        </svg>
      )
    },
    {
      id: 'models' as NavRoute,
      label: 'Model Directory',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
          <line x1="8" y1="21" x2="16" y2="21" />
          <line x1="12" y1="17" x2="12" y2="21" />
        </svg>
      )
    },
    {
      id: 'compare' as NavRoute,
      label: 'Comparison Workspace',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="18" y1="20" x2="18" y2="10" />
          <line x1="12" y1="20" x2="12" y2="4" />
          <line x1="6" y1="20" x2="6" y2="14" />
        </svg>
      )
    },
    {
      id: 'deep-learning' as NavRoute,
      label: 'Deep Learning Lab',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="3" />
          <circle cx="4" cy="6" r="2" />
          <circle cx="20" cy="6" r="2" />
          <circle cx="4" cy="18" r="2" />
          <circle cx="20" cy="18" r="2" />
          <line x1="6" y1="7" x2="9.5" y2="10.5" />
          <line x1="18" y1="7" x2="14.5" y2="10.5" />
          <line x1="6" y1="17" x2="9.5" y2="13.5" />
          <line x1="18" y1="17" x2="14.5" y2="13.5" />
        </svg>
      )
    },
    {
      id: 'live-ops' as NavRoute,
      label: 'Live Operations & Feed',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        </svg>
      )
    }
  ];

  const getPageTitle = () => {
    switch (currentRoute) {
      case 'home': return 'PhishGuard Overview';
      case 'models': return 'Model Directory';
      case 'model-detail': return 'Model Specification';
      case 'compare': return 'Model Comparison Workspace';
      case 'deep-learning': return 'Deep Learning Lab — Character-Level CNN';
      case 'live-ops': return 'Live Threat Operations';
      default: return 'PhishGuard Analytics';
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <img src="/logo.png" alt="PhishGuard Shield" className="brand-logo" />
          <div>
            <div className="brand-title">PhishGuard</div>
            <div className="brand-subtitle">Evaluation Suite</div>
          </div>
        </div>

        <nav className="sidebar-nav">
          {navItems.map(item => {
            const isActive = currentRoute === item.id || (item.id === 'models' && currentRoute === 'model-detail');
            return (
              <button
                key={item.id}
                onClick={() => onRouteChange(item.id)}
                className={`nav-link ${isActive ? 'active' : ''}`}
              >
                {item.icon}
                <span>{item.label}</span>
                <span className="nav-indicator" />
              </button>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
            <span>Stream Status</span>
            <span className="live-indicator">
              <span className="live-dot" style={{ backgroundColor: connected ? 'var(--color-success)' : 'var(--color-warning)' }} />
              {connected ? 'Active' : 'Offline'}
            </span>
          </div>
          <div style={{ color: 'rgba(243, 248, 247, 0.45)', fontSize: 11 }}>
            Models: RF, XGBoost, LightGBM + Char-CNN
          </div>
        </div>
      </aside>

      {/* Main Workspace */}
      <div className="main-wrapper">
        <header className="top-bar">
          <div className="top-bar-title">
            <span>{getPageTitle()}</span>
          </div>

          <div className="top-bar-actions">
            <span className="status-pill lime">
              Production: XGBoost + LightGBM Ready
            </span>

            {token ? (
              <button onClick={onLogout} className="btn-secondary" style={{ fontSize: 12.5, padding: '6px 14px' }}>
                Logout
              </button>
            ) : (
              <button onClick={onOpenLogin} className="btn-lime" style={{ fontSize: 12.5, padding: '6px 14px' }}>
                Admin Login
              </button>
            )}
          </div>
        </header>

        <main className="content-body">
          {children}
        </main>
      </div>
    </div>
  );
}
