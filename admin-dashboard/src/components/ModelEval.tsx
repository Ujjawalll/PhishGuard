import React, { useEffect, useRef, useState } from 'react';

const METRICS = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'Sensitivity', 'ROC-AUC', 'MCC', 'PR-AUC'];
const PLOT_TITLES: Record<string, string> = {
  confusion_matrices: 'Confusion Matrices',
  roc_curves: 'ROC Curves',
  pr_curves: 'Precision-Recall Curves',
  feature_importances: 'Feature Importances',
};

function fmtParams(p: Record<string, any>): string {
  return Object.entries(p)
    .map(([k, v]) => `${k}=${typeof v === 'number' ? Math.round(v * 10000) / 10000 : v}`)
    .join(', ');
}

export default function ModelEval({ apiUrl, token }: { apiUrl: string; token: string }) {
  const [results, setResults] = useState<any>(null);
  const [jobs, setJobs] = useState<any[]>([]);
  const [activeJob, setActiveJob] = useState<any>(null);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [quick, setQuick] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [plotUrls, setPlotUrls] = useState<Record<string, string>>({});
  const fileRef = useRef<HTMLInputElement>(null);

  const headers = { Authorization: `Bearer ${token}` };

  const fetchResults = async (jobId: string | null) => {
    try {
      const url = jobId ? `${apiUrl}/eval/results?job_id=${jobId}` : `${apiUrl}/eval/results`;
      const res = await fetch(url, { headers });
      if (res.ok) setResults(await res.json());
      else setResults(null);
    } catch { setResults(null); }
  };

  const fetchJobs = async () => {
    try {
      const res = await fetch(`${apiUrl}/eval/jobs`, { headers });
      if (res.ok) setJobs(await res.json());
    } catch { /* ignore */ }
  };

  useEffect(() => { fetchResults(selectedJobId); }, [selectedJobId]);
  useEffect(() => { fetchJobs(); }, []);

  // poll the running job
  useEffect(() => {
    const running = jobs.find(j => j.status === 'running' || j.status === 'queued');
    if (!running) { setActiveJob(null); return; }
    const poll = async () => {
      try {
        const res = await fetch(`${apiUrl}/eval/jobs/${running.job_id}`, { headers });
        if (!res.ok) return;
        const job = await res.json();
        setActiveJob(job);
        if (job.status === 'done' || job.status === 'failed') {
          fetchJobs();
          if (job.status === 'done') {
            setSelectedJobId(job.job_id);
            setNotice(`Job ${job.job_id} finished — results loaded below.`);
          }
        }
      } catch { /* ignore */ }
    };
    poll();
    const t = setInterval(poll, 5000);
    return () => clearInterval(t);
  }, [jobs]);

  // load plot images with auth (img tags can't send headers)
  useEffect(() => {
    if (!results?.plots?.length) { setPlotUrls({}); return; }
    const urls: Record<string, string> = {};
    const created: string[] = [];
    (async () => {
      for (const name of results.plots) {
        try {
          const q = results.job_id ? `?job_id=${results.job_id}` : '';
          const res = await fetch(`${apiUrl}/eval/plots/${name}${q}`, { headers });
          if (res.ok) {
            const objUrl = URL.createObjectURL(await res.blob());
            urls[name.replace('.png', '')] = objUrl;
            created.push(objUrl);
          }
        } catch { /* skip missing plot */ }
      }
      setPlotUrls(urls);
    })();
    return () => created.forEach(u => URL.revokeObjectURL(u));
  }, [results]);

  const handleUpload = async () => {
    setError(null); setNotice(null);
    const file = fileRef.current?.files?.[0];
    if (!file) { setError('Choose a CSV file first.'); return; }
    setUploading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch(`${apiUrl}/eval/train?quick=${quick}`, { method: 'POST', headers, body: form });
      const data = await res.json();
      if (!res.ok) { setError(data.detail || 'Upload failed'); return; }
      setNotice(`Training started (job ${data.job_id}) on ${data.dataset.rows.toLocaleString()} URLs ` +
        `(${data.dataset.phishing} phishing / ${data.dataset.legitimate} legitimate). This takes a few minutes.`);
      if (fileRef.current) fileRef.current.value = '';
      fetchJobs();
    } catch (e: any) {
      setError('Upload failed: ' + e.message);
    } finally {
      setUploading(false);
    }
  };

  const rf = results?.test_metrics?.['Random Forest'];
  const xgb = results?.test_metrics?.['XGBoost'];
  const better = rf && xgb
    ? (xgb['F1-Score'] > rf['F1-Score'] ? 'XGBoost' : xgb['F1-Score'] < rf['F1-Score'] ? 'Random Forest' : 'Tie')
    : null;

  const statusColor = (s: string) =>
    s === 'done' ? '#10b981' : s === 'failed' ? '#ef4444' : s === 'running' ? '#fbbf24' : '#94a3b8';

  return (
    <div>
      {/* Upload card */}
      <div className="card">
        <h3>Train &amp; Evaluate on Your Dataset</h3>
        <p style={{ fontSize: 13, color: '#94a3b8', margin: '0 0 12px' }}>
          Upload a CSV with a <code>url</code> column and a <code>label</code> column (0 = legitimate, 1 = phishing).
          A <code>source</code> column also works (anything other than "tranco" counts as phishing). Minimum 50 rows, both classes required, file size up to 100 MB.
          Both models are re-trained and hyperparameter-tuned on your data, then evaluated on an untouched 20% test split.
        </p>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <input ref={fileRef} type="file" accept=".csv"
            style={{ color: '#e2e8f0', fontSize: 14 }} />
          <label style={{ fontSize: 13, color: '#94a3b8', display: 'flex', gap: 6, alignItems: 'center' }}>
            <input type="checkbox" checked={quick} onChange={e => setQuick(e.target.checked)} />
            Quick mode (faster tuning)
          </label>
          <button onClick={handleUpload} disabled={uploading}
            style={{ padding: '8px 20px', borderRadius: 6, border: 'none', background: '#3b82f6', color: 'white', fontWeight: 600, cursor: uploading ? 'wait' : 'pointer' }}>
            {uploading ? 'Uploading…' : 'Start Training'}
          </button>
        </div>
        {error && <div style={{ color: '#fca5a5', fontSize: 13, marginTop: 10 }}>⚠ {error}</div>}
        {notice && <div style={{ color: '#10b981', fontSize: 13, marginTop: 10 }}>{notice}</div>}
      </div>

      {/* Running job */}
      {activeJob && (activeJob.status === 'running' || activeJob.status === 'queued') && (
        <div className="card" style={{ marginTop: 20, borderColor: '#fbbf24' }}>
          <h3>Training In Progress — job {activeJob.job_id}</h3>
          <div style={{ fontSize: 13, color: '#fbbf24', marginBottom: 8 }}>
            ● {activeJob.status} — {activeJob.dataset?.rows?.toLocaleString()} rows
            {activeJob.dataset ? ` (${activeJob.dataset.phishing} phishing / ${activeJob.dataset.legitimate} legitimate)` : ''}
          </div>
          {activeJob.log_tail && (
            <pre style={{ background: '#0f172a', padding: 12, borderRadius: 8, fontSize: 11.5, overflowX: 'auto', color: '#94a3b8', maxHeight: 180 }}>
              {activeJob.log_tail}
            </pre>
          )}
        </div>
      )}

      {/* Job history */}
      {jobs.length > 0 && (
        <div className="card" style={{ marginTop: 20 }}>
          <h3>Training Jobs</h3>
          <table>
            <thead>
              <tr><th>Job</th><th>File</th><th>Rows</th><th>Status</th><th>Created</th><th></th></tr>
            </thead>
            <tbody>
              {jobs.map(j => (
                <tr key={j.job_id}>
                  <td style={{ fontSize: 12 }}>{j.job_id}</td>
                  <td>{j.filename}</td>
                  <td>{j.dataset?.rows?.toLocaleString()}</td>
                  <td><span className="badge" style={{ background: '#0f172a', color: statusColor(j.status) }}>{j.status}</span></td>
                  <td style={{ fontSize: 12, color: '#94a3b8' }}>{new Date(j.created_at).toLocaleString()}</td>
                  <td>
                    {j.status === 'done' && (
                      <button onClick={() => setSelectedJobId(j.job_id)}
                        style={{ padding: '4px 10px', borderRadius: 4, border: '1px solid #475569', background: 'transparent', color: '#94a3b8', cursor: 'pointer', fontSize: 12 }}>
                        {selectedJobId === j.job_id ? 'Viewing' : 'View results'}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {selectedJobId && (
            <button onClick={() => setSelectedJobId(null)}
              style={{ marginTop: 10, padding: '4px 10px', borderRadius: 4, border: '1px solid #475569', background: 'transparent', color: '#94a3b8', cursor: 'pointer', fontSize: 12 }}>
              ← Back to baseline results (repo dataset)
            </button>
          )}
        </div>
      )}

      {/* Results */}
      {results && rf && xgb && (
        <div style={{ marginTop: 20 }}>
          <div className="grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginTop: 0 }}>
            <div className="card"><h3>Dataset</h3>
              <div className="stat-value" style={{ color: '#3b82f6' }}>{results.dataset?.rows?.toLocaleString()}</div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>
                {results.dataset?.positive} phishing · {results.dataset?.negative} legitimate
              </div>
            </div>
            <div className="card"><h3>Train / Test</h3>
              <div className="stat-value" style={{ color: '#8b5cf6' }}>{results.dataset?.train_size?.toLocaleString()} / {results.dataset?.test_size?.toLocaleString()}</div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>80:20 stratified, seed 42</div>
            </div>
            <div className="card"><h3>Best CV F1</h3>
              <div style={{ fontSize: 15, marginTop: 6 }}>
                <div style={{ color: '#2878b5' }}>RF: {Number(results.best_cv_f1_rf).toFixed(4)}</div>
                <div style={{ color: '#ef4444' }}>XGB: {Number(results.best_cv_f1_xgb).toFixed(4)}</div>
              </div>
            </div>
            <div className="card"><h3>Winner (Test F1)</h3>
              <div className="stat-value" style={{ color: '#10b981', fontSize: 24 }}>{better}</div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>by F1 on untouched test set</div>
            </div>
          </div>

          <div className="card" style={{ marginTop: 20 }}>
            <h3>Performance Matrix — Test Set {selectedJobId ? `(job ${selectedJobId})` : '(baseline: repo dataset)'}</h3>
            <table>
              <thead><tr><th>Metric</th><th style={{ textAlign: 'right' }}>Random Forest</th><th style={{ textAlign: 'right' }}>XGBoost</th><th style={{ textAlign: 'right' }}>Better</th></tr></thead>
              <tbody>
                {METRICS.map(m => {
                  const a = Number(rf[m]), b = Number(xgb[m]);
                  return (
                    <tr key={m}>
                      <td>{m}</td>
                      <td style={{ textAlign: 'right', fontWeight: a > b ? 700 : 400, color: a > b ? '#10b981' : undefined }}>{a.toFixed(4)}</td>
                      <td style={{ textAlign: 'right', fontWeight: b > a ? 700 : 400, color: b > a ? '#10b981' : undefined }}>{b.toFixed(4)}</td>
                      <td style={{ textAlign: 'right', fontSize: 12, color: '#94a3b8' }}>{a === b ? '—' : a > b ? 'RF' : 'XGB'}</td>
                    </tr>
                  );
                })}
                <tr>
                  <td style={{ color: '#94a3b8' }}>Train F1 (overfit check)</td>
                  <td style={{ textAlign: 'right', color: '#94a3b8' }}>{Number(rf['Train-F1']).toFixed(4)}</td>
                  <td style={{ textAlign: 'right', color: '#94a3b8' }}>{Number(xgb['Train-F1']).toFixed(4)}</td>
                  <td />
                </tr>
              </tbody>
            </table>
          </div>

          <div className="grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <div className="card">
              <h3>Best Random Forest Params</h3>
              <div style={{ fontSize: 12.5, color: '#cbd5e1', wordBreak: 'break-word' }}>{fmtParams(results.best_params_rf)}</div>
            </div>
            <div className="card">
              <h3>Best XGBoost Params</h3>
              <div style={{ fontSize: 12.5, color: '#cbd5e1', wordBreak: 'break-word' }}>{fmtParams(results.best_params_xgb)}</div>
            </div>
          </div>

          {Object.keys(plotUrls).length > 0 && (
            <div className="grid" style={{ gridTemplateColumns: '1fr', marginTop: 20 }}>
              {Object.entries(plotUrls).map(([key, url]) => (
                <div className="card" key={key}>
                  <h3>{PLOT_TITLES[key] || key}</h3>
                  <img src={url} alt={key} style={{ width: '100%', borderRadius: 8, background: '#f8fafc' }} />
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {!results && (
        <div className="card" style={{ marginTop: 20 }}>
          <h3>No Results Yet</h3>
          <p style={{ fontSize: 13, color: '#94a3b8' }}>Upload a dataset above to train and evaluate both models.</p>
        </div>
      )}
    </div>
  );
}
