import React, { useEffect, useRef, useState } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';

const METRICS = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'Sensitivity', 'ROC-AUC', 'MCC', 'PR-AUC'];

const DL_COLORS: Record<string, string> = {
  'Char-CNN': '#0D596B'
};

const PLOT_TITLES: Record<string, string> = {
  confusion_matrices: 'Confusion Matrices (Test Set)',
  roc_curves: 'ROC Curves (Test Set)',
  pr_curves: 'Precision-Recall Curves (Test Set)',
  feature_importances: 'Character-Class Signal Strength',
  training_history: 'Training History (Loss / Validation F1)',
};

interface DeepLearningProps {
  apiUrl: string;
  token: string | null;
  onOpenLogin: () => void;
}

export default function DeepLearning({ apiUrl, token, onOpenLogin }: DeepLearningProps) {
  const [results, setResults] = useState<any>(null);
  const [jobs, setJobs] = useState<any[]>([]);
  const [activeJob, setActiveJob] = useState<any>(null);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [quick, setQuick] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [plotUrls, setPlotUrls] = useState<Record<string, string>>({});
  const [autoSelected, setAutoSelected] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};

  const fetchResults = async (jobId: string | null) => {
    try {
      const url = jobId ? `${apiUrl}/eval/results?job_id=${jobId}` : `${apiUrl}/eval/results`;
      const res = await fetch(url, { headers });
      if (res.ok) {
        const data = await res.json();
        // only Deep Learning results belong on this page (test_metrics keyed by CNN)
        if (data?.test_metrics && 'Char-CNN' in data.test_metrics) {
          setResults(data);
        } else {
          setResults(null);
        }
      } else {
        setResults(null);
      }
    } catch {
      setResults(null);
    }
  };

  const fetchJobs = async () => {
    if (!token) return;
    try {
      const res = await fetch(`${apiUrl}/eval/jobs`, { headers });
      if (res.ok) {
        const all = await res.json();
        const dlJobs = all.filter((j: any) => j.engine === 'dl');
        setJobs(dlJobs);
        if (!autoSelected && !selectedJobId && dlJobs.length > 0) {
          const latestDone = dlJobs.find((j: any) => j.status === 'done');
          if (latestDone) {
            setSelectedJobId(latestDone.job_id);
            fetchResults(latestDone.job_id);
          }
          setAutoSelected(true);
        }
      }
    } catch { /* ignore */ }
  };

  useEffect(() => {
    if (selectedJobId) fetchResults(selectedJobId);
  }, [selectedJobId]);
  useEffect(() => { fetchJobs(); }, [token]);

  // poll the running DL job
  useEffect(() => {
    if (!token) return;
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
            setNotice(`Deep learning job ${job.job_id} finished — results loaded below.`);
          }
        }
      } catch { /* ignore */ }
    };
    poll();
    const t = setInterval(poll, 5000);
    return () => clearInterval(t);
  }, [jobs, token]);

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
  }, [results, token]);

  const handleUpload = async () => {
    setError(null); setNotice(null);
    if (!token) { setError('Admin login required to start training jobs.'); return; }
    const file = fileRef.current?.files?.[0];
    if (!file) { setError('Choose a CSV file first.'); return; }
    setUploading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch(`${apiUrl}/eval/train?engine=dl&quick=${quick}`, { method: 'POST', headers, body: form });
      const data = await res.json();
      if (!res.ok) { setError(data.detail || 'Upload failed'); return; }
      setNotice(`Deep learning training started (job ${data.job_id}) on ${data.dataset.rows.toLocaleString()} URLs ` +
        `(${data.dataset.phishing} phishing / ${data.dataset.legitimate} legitimate). ` +
        `Char-CNN is trained on CPU — this can take several minutes.`);
      if (fileRef.current) fileRef.current.value = '';
      fetchJobs();
    } catch (e: any) {
      setError('Upload failed: ' + e.message);
    } finally {
      setUploading(false);
    }
  };

  const models: string[] = results?.test_metrics ? Object.keys(results.test_metrics) : [];
  const dataset = results?.dataset;
  const history = results?.training_history || {};
  const errorAnalysis = results?.error_analysis || {};
  const maxEpochs = Math.max(0, ...models.map(m => (history[m] || []).length));
  const historyChart = Array.from({ length: maxEpochs }, (_, i) => {
    const point: any = { epoch: i + 1 };
    models.forEach(m => {
      const entry = (history[m] || [])[i];
      point[m] = entry ? entry.val_f1 : undefined;
    });
    return point;
  });
  const winner = models.length > 1
    ? models.reduce((a, b) => Number(results.test_metrics[b]['F1-Score']) > Number(results.test_metrics[a]['F1-Score']) ? b : a)
    : null;

  const statusColor = (s: string) =>
    s === 'done' ? 'var(--color-success)' : s === 'failed' ? 'var(--color-danger)' : s === 'running' ? 'var(--color-warning)' : 'var(--color-text-muted)';

  return (
    <div>
      {/* Intro + Upload */}
      <div className="surface-card" style={{ marginBottom: 'var(--space-6)' }}>
        <h3 className="card-title">Deep Learning Lab — Character-Level CNN</h3>
        <p className="card-subtitle" style={{ marginBottom: 16, maxWidth: 720 }}>
          Train deep neural networks directly on raw URL strings: a 1D Char-CNN (learned n-gram detectors).
          Upload a CSV with a <code>url</code> column and a <code>label</code> column (0 = legitimate, 1 = phishing);
          a <code>source</code> column also works. Minimum 50 rows, both classes, up to 100 MB.
        </p>
        {token ? (
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
            <input ref={fileRef} type="file" accept=".csv" style={{ color: 'var(--color-text)', fontSize: 14 }} />
            <label style={{ fontSize: 13, color: 'var(--color-text-muted)', display: 'flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={quick} onChange={e => setQuick(e.target.checked)} />
              Quick mode (fewer epochs)
            </label>
            <button onClick={handleUpload} disabled={uploading} className="btn-lime" style={{ padding: '8px 20px', fontSize: 13 }}>
              {uploading ? 'Uploading…' : 'Train Char-CNN'}
            </button>
          </div>
        ) : (
          <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            <span style={{ fontSize: 13, color: 'var(--color-text-muted)' }}>
              Admin authentication is required to start deep learning training jobs.
            </span>
            <button onClick={onOpenLogin} className="btn-lime" style={{ padding: '6px 16px', fontSize: 12.5 }}>Admin Login</button>
          </div>
        )}
        {error && <div style={{ color: 'var(--color-danger)', fontSize: 13, marginTop: 10 }}>⚠ {error}</div>}
        {notice && <div style={{ color: 'var(--color-success)', fontSize: 13, marginTop: 10 }}>{notice}</div>}
      </div>

      {/* Running job */}
      {activeJob && (activeJob.status === 'running' || activeJob.status === 'queued') && (
        <div className="surface-card" style={{ marginBottom: 'var(--space-6)', borderColor: 'var(--color-warning)' }}>
          <h3 className="card-title">Training In Progress — job {activeJob.job_id}</h3>
          <div style={{ fontSize: 13, color: 'var(--color-warning)', marginBottom: 8 }}>
            ● {activeJob.status} — {activeJob.dataset?.rows?.toLocaleString()} rows
            {activeJob.dataset ? ` (${activeJob.dataset.phishing} phishing / ${activeJob.dataset.legitimate} legitimate)` : ''}
          </div>
          {activeJob.log_tail && (
            <pre style={{ background: 'var(--color-ink-deep)', padding: 12, borderRadius: 8, fontSize: 11.5, overflowX: 'auto', color: 'var(--color-text-muted)', maxHeight: 200 }}>
              {activeJob.log_tail}
            </pre>
          )}
        </div>
      )}

      {/* Job history */}
      {jobs.length > 0 && (
        <div className="surface-card" style={{ marginBottom: 'var(--space-6)' }}>
          <h3 className="card-title">Deep Learning Training Jobs</h3>
          <p className="card-subtitle" style={{ marginBottom: 14 }}>Every Char-CNN run on this backend, newest first</p>
          <div className="modern-table-container">
            <table className="modern-table">
              <thead>
                <tr><th>Job</th><th>File</th><th>Dataset</th><th>Status</th><th>Created</th><th></th></tr>
              </thead>
              <tbody>
                {jobs.map(j => (
                  <tr key={j.job_id}>
                    <td style={{ fontSize: 12, fontFamily: 'var(--font-mono)' }}>{j.job_id}</td>
                    <td>{j.filename}</td>
                    <td style={{ fontSize: 12.5 }}>
                      {j.dataset?.rows?.toLocaleString()} rows
                      <span style={{ color: 'var(--color-text-muted)' }}> ({j.dataset?.phishing?.toLocaleString()} phish / {j.dataset?.legitimate?.toLocaleString()} legit)</span>
                    </td>
                    <td>
                      <span className="status-pill neutral" style={{ fontSize: 11, color: statusColor(j.status) }}>{j.status}</span>
                    </td>
                    <td style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>{new Date(j.created_at).toLocaleString()}</td>
                    <td>
                      {j.status === 'done' && (
                        <button onClick={() => setSelectedJobId(j.job_id)}
                          className="btn-secondary" style={{ padding: '4px 10px', fontSize: 12 }}>
                          {selectedJobId === j.job_id ? 'Viewing' : 'View results'}
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Results */}
      {results && models.length > 0 && (
        <div>
          {/* Dataset overview */}
          <h3 className="card-title" style={{ margin: 'var(--space-6) 0 var(--space-4)' }}>
            Training Dataset {selectedJobId ? `(job ${selectedJobId})` : ''}
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
            <div className="metric-box">
              <div className="metric-label">Total URLs</div>
              <div className="metric-val teal">{dataset?.rows?.toLocaleString()}</div>
              <div className="metric-sub">{dataset?.file}</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Phishing</div>
              <div className="metric-val" style={{ color: 'var(--color-danger)' }}>{dataset?.positive?.toLocaleString()}</div>
              <div className="metric-sub">{dataset?.negative?.toLocaleString()} legitimate</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Train / Val / Test</div>
              <div className="metric-val lime-tint">
                {dataset?.train_size?.toLocaleString()} / {dataset?.val_size?.toLocaleString()} / {dataset?.test_size?.toLocaleString()}
              </div>
              <div className="metric-sub">80:10:10 stratified</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Input Encoding</div>
              <div className="metric-val teal">{dataset?.max_url_len} chars</div>
              <div className="metric-sub">vocab {dataset?.vocab_size}, {dataset?.truncated_urls} truncated</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Device</div>
              <div className="metric-val lime-tint">{results?.device || 'cpu'}</div>
              <div className="metric-sub">torch inference</div>
            </div>
          </div>

          {/* Metrics matrix */}
          <div className="surface-card" style={{ marginBottom: 'var(--space-6)' }}>
            <h3 className="card-title">Char-CNN — Performance Matrix (Test Set)</h3>
            <p className="card-subtitle" style={{ marginBottom: 14 }}>
              {winner ? `Winner: ${winner} (F1 ${Number(results.test_metrics[winner]['F1-Score']).toFixed(4)})` : ''}
            </p>
            <div className="modern-table-container">
              <table className="modern-table">
                <thead>
                  <tr>
                    <th style={{ width: '22%' }}>Metric</th>
                    {models.map(m => (
                      <th key={m} style={{ textAlign: 'right', color: DL_COLORS[m] || 'inherit' }}>{m}</th>
                    ))}
                    {models.length > 1 && <th style={{ textAlign: 'center', width: '16%' }}>Better</th>}
                  </tr>
                </thead>
                <tbody>
                  {METRICS.map(metric => {
                    const vals = models.map(m => Number(results.test_metrics[m][metric]));
                    const maxVal = Math.max(...vals);
                    return (
                      <tr key={metric}>
                        <td style={{ fontWeight: 600 }}>{metric}</td>
                        {models.map((m, i) => {
                          const isBest = models.length > 1 && vals[i] === maxVal;
                          return (
                            <td key={m} style={{
                              textAlign: 'right', fontFamily: 'var(--font-mono)',
                              fontWeight: isBest ? 700 : 400,
                              color: isBest ? 'var(--color-success)' : 'inherit',
                              backgroundColor: isBest ? 'rgba(22, 121, 75, 0.04)' : undefined
                            }}>
                              {isNaN(vals[i]) ? '—' : vals[i].toFixed(4)}
                            </td>
                          );
                        })}
                        {models.length > 1 && (
                          <td style={{ textAlign: 'center' }}>
                            <span className="status-pill lime" style={{ fontSize: 11 }}>
                              {vals[0] === vals[1] ? '—' : vals[0] > vals[1] ? models[0].split('-')[0] : models[1].split('-')[0]}
                            </span>
                          </td>
                        )}
                      </tr>
                    );
                  })}
                  <tr>
                    <td style={{ color: 'var(--color-text-muted)' }}>Train F1 (overfit check)</td>
                    {models.map(m => (
                      <td key={m} style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--color-text-muted)' }}>
                        {Number(results.test_metrics[m]['Train-F1']).toFixed(4)}
                      </td>
                    ))}
                    {models.length > 1 && <td />}
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Training history chart + table */}
          <div className="surface-card" style={{ marginBottom: 'var(--space-6)' }}>
            <h3 className="card-title">Training Progress — Validation F1 per Epoch</h3>
            <p className="card-subtitle" style={{ marginBottom: 14 }}>Early stopping restores the best-epoch weights</p>
            {historyChart.length > 0 && (
              <div style={{ height: 260, marginBottom: 18 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={historyChart} margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#D8E2DF" vertical={false} />
                    <XAxis dataKey="epoch" tick={{ fontSize: 12, fill: '#64777D' }} />
                    <YAxis tick={{ fontSize: 12, fill: '#64777D' }} domain={[0, 1]} />
                    <Tooltip contentStyle={{ background: '#FFFFFF', border: '1px solid #D8E2DF', borderRadius: 8, fontSize: 12 }} />
                    <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: 13 }} />
                    {models.map(m => (
                      <Line key={m} type="monotone" dataKey={m} stroke={DL_COLORS[m] || '#64777D'} strokeWidth={2} dot={{ r: 3 }} connectNulls />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
            <div className="modern-table-container">
              <table className="modern-table">
                <thead>
                  <tr>
                    <th>Model</th><th>Epoch</th><th>Train Loss</th><th>Val F1</th><th>Val ROC-AUC</th><th>LR</th><th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {models.flatMap(m => (history[m] || []).map((h: any) => (
                    <tr key={`${m}-${h.epoch}`}>
                      <td style={{ color: DL_COLORS[m] || 'inherit', fontWeight: 600 }}>{m}</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{h.epoch}</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{h.train_loss?.toFixed(4)}</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{h.val_f1?.toFixed(4)}</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{h.val_roc_auc?.toFixed(4)}</td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{h.lr?.toExponential(1)}</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{h.seconds}s</td>
                    </tr>
                  )))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Error analysis */}
          <div className="surface-card" style={{ marginBottom: 'var(--space-6)' }}>
            <h3 className="card-title">Error Analysis — False Positives / False Negatives</h3>
            <p className="card-subtitle" style={{ marginBottom: 14 }}>Counts on the untouched test split; examples capped at 25 per bucket</p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 'var(--space-4)' }}>
              {models.map(m => {
                const a = errorAnalysis[m] || {};
                return (
                  <div key={m} style={{ border: '1px solid var(--color-border)', borderRadius: 'var(--radius-sm)', padding: 14 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                      <strong style={{ color: DL_COLORS[m] || 'inherit' }}>{m}</strong>
                      <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                        <span style={{ color: 'var(--color-danger)' }}>FP {a.false_positives ?? '—'}</span>
                        {'  ·  '}
                        <span style={{ color: 'var(--color-warning)' }}>FN {a.false_negatives ?? '—'}</span>
                      </span>
                    </div>
                    {a.examples_false_negatives?.slice(0, 3).map((e: any, i: number) => (
                      <div key={i} style={{ fontSize: 11.5, fontFamily: 'var(--font-mono)', color: 'var(--color-text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        FN p={e.prob} · {e.url}
                      </div>
                    ))}
                    {a.examples_false_positives?.slice(0, 3).map((e: any, i: number) => (
                      <div key={i} style={{ fontSize: 11.5, fontFamily: 'var(--font-mono)', color: 'var(--color-text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        FP p={e.prob} · {e.url}
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Hyperparameters + artifacts */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
            {models.map(m => {
              const key = m === 'Char-CNN' ? 'best_params_cnn' : `best_params_${m.toLowerCase()}`;
              const params = results[key] || {};
              const bestValF1 = m === 'Char-CNN' ? results.best_val_f1_cnn : results[`best_val_f1_${m.toLowerCase()}`];
              return (
                <div key={m} className="surface-card">
                  <h3 className="card-title">{m} — Hyperparameters & Artifact</h3>
                  <p className="card-subtitle" style={{ marginBottom: 10 }}>
                    Best val F1: {bestValF1 != null ? Number(bestValF1).toFixed(4) : '—'}
                  </p>
                  <div style={{ fontSize: 12, color: 'var(--color-text)', fontFamily: 'var(--font-mono)', wordBreak: 'break-word', lineHeight: 1.8 }}>
                    {Object.entries(params).map(([k, v]) => (
                      <span key={k}>
                        <span style={{ color: 'var(--color-text-muted)' }}>{k}</span>
                        <span>={typeof v === 'number' ? Math.round(v * 10000) / 10000 : String(v)}</span>
                        {'  '}
                      </span>
                    ))}
                  </div>
                  {results.artifacts?.[m] && (
                    <div style={{ marginTop: 10, fontSize: 11.5, color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>
                      artifact: {results.artifacts[m]}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Plots */}
          {Object.keys(plotUrls).length > 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 'var(--space-4)' }}>
              {Object.entries(plotUrls).map(([key, url]) => (
                <div className="surface-card" key={key}>
                  <h3 className="card-title">{PLOT_TITLES[key] || key}</h3>
                  <img src={url} alt={key} style={{ width: '100%', borderRadius: 8, background: '#FFFFFF' }} />
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {!results && (
        <div className="surface-card">
          <h3 className="card-title">No Deep Learning Results Yet</h3>
          <p className="card-subtitle">
            Upload a URL dataset above to train the Char-CNN model. Results, training curves,
            error analysis, and plots will appear here.
          </p>
        </div>
      )}
    </div>
  );
}
