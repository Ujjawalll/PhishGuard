import { useEffect, useRef, useState, useCallback } from 'react';

interface ScanEvent {
  type: string;
  scan_id: string;
  url: string;
  domain: string;
  risk_level: string;
  fused_score: number;
  stage: string;
  timestamp: string;
}

export function useWebSocket(token: string | null) {
  const [events, setEvents] = useState<ScanEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    if (!token) return;

    const apiUrl = import.meta.env.VITE_API_URL || 'ws://localhost:8000';
    const wsUrl = apiUrl.replace('http', 'ws');
    const ws = new WebSocket(`${wsUrl}/admin/ws/live`);

    ws.onopen = () => {
      ws.send(token); // Authenticate
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'new_scan') {
          setEvents(prev => [data, ...prev].slice(0, 100)); // Keep last 100
        }
      } catch {}
    };

    ws.onclose = () => {
      setConnected(false);
      // Reconnect after 3 seconds
      setTimeout(connect, 3000);
    };

    ws.onerror = () => ws.close();

    wsRef.current = ws;
  }, [token]);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  return { events, connected };
}
