function clearWarning() {
  const overlay = document.getElementById('pg-warning-overlay');
  if (overlay) overlay.remove();
}

function normalizeUrl(url: string) {
  try {
    const u = new URL(url);
    return u.origin + u.pathname + u.search; // ignore hash for matching
  } catch {
    return url;
  }
}

function renderWarning(result: any) {
  clearWarning(); // ensure previous is cleared

  console.log(`[PhishGuard UI]
CURRENT_URL=${window.location.href}
RESULT_URL=${result.scanned_url}
RISK=${result.risk_level}
SCORE=${result.fused_score || result.normalized_score || 'N/A'}
SCAN_STAGE=${result.scan_stage}
SCAN_ID=${result.scan_id}`);

  if (result.risk_level === 'LOW_RISK') return;
  if (result.risk_level === 'ANALYSIS_UNAVAILABLE') {
    // Spec: "Show neutral technical message. Never show a red phishing warning merely because analysis failed."
    // We could show a small toast, but for now we simply don't block the page.
    return;
  }

  const overlay = document.createElement('div');
  overlay.id = 'pg-warning-overlay';
  overlay.style.position = 'fixed';
  overlay.style.top = '0';
  overlay.style.left = '0';
  overlay.style.width = '100vw';
  overlay.style.height = '100vh';
  overlay.style.backgroundColor = result.risk_level === 'HIGH_RISK' ? 'rgba(239, 68, 68, 0.95)' : 'rgba(245, 158, 11, 0.95)';
  overlay.style.color = 'white';
  overlay.style.zIndex = '2147483647'; // Max z-index
  overlay.style.display = 'flex';
  overlay.style.flexDirection = 'column';
  overlay.style.alignItems = 'center';
  overlay.style.justifyContent = 'center';
  overlay.style.fontFamily = 'sans-serif';
  
  const reasons = (result.explanation?.top_reasons || []).map((r: string) => `<li style="margin-bottom: 0.5rem;">${r}</li>`).join('');
  
  overlay.innerHTML = `
    <div style="max-width: 600px; padding: 2rem; text-align: center;">
      <h1 style="font-size: 3rem; margin-bottom: 1rem;">Phishing Warning</h1>
      <p style="font-size: 1.5rem; margin-bottom: 2rem;">PhishGuard has detected that this site is <strong>${result.risk_level.replace('_', ' ')}</strong>.</p>
      ${reasons ? `
      <div style="background: rgba(0,0,0,0.2); padding: 1.5rem; border-radius: 8px; text-align: left; margin-bottom: 2rem;">
        <h3>Reasons:</h3>
        <ul>${reasons}</ul>
      </div>` : ''}
      <p style="font-size: 1.25rem; font-weight: bold;">${result.explanation?.recommendation || 'Proceed with extreme caution.'}</p>
      <button id="pg-go-back" style="margin-top: 2rem; padding: 1rem 2rem; font-size: 1.25rem; font-weight: bold; cursor: pointer; border: none; border-radius: 4px; background: white; color: ${result.risk_level === 'HIGH_RISK' ? '#EF4444' : '#F59E0B'};">Get Me Out of Here</button>
      <button id="pg-ignore" style="margin-top: 1rem; padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; border: 1px solid white; border-radius: 4px; background: transparent; color: white;">Ignore and Proceed</button>
    </div>
  `;
  
  if (document.body) {
    document.body.appendChild(overlay);
  } else {
    document.documentElement.appendChild(overlay);
  }
  
  document.getElementById('pg-go-back')?.addEventListener('click', () => {
    window.history.back();
  });
  
  document.getElementById('pg-ignore')?.addEventListener('click', () => {
    overlay.remove();
  });
}

function processResult(result: any) {
    if (!result || !result.scanned_url) return;
    if (normalizeUrl(result.scanned_url) !== normalizeUrl(window.location.href)) {
        return; // Mismatch, ignore
    }
    
    if (result.risk_level === 'LOW_RISK' || result.risk_level === 'ANALYSIS_UNAVAILABLE') {
        clearWarning();
    } else {
        renderWarning(result);
    }
}

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'CLEAR_WARNING') {
    clearWarning();
  } else if (message.type === 'UPDATE_WARNING') {
    processResult(message.result);
  }
});

// Proactively check storage on load
chrome.runtime.sendMessage({ type: "GET_TAB_ID" }, (response) => {
    if (response && response.tabId) {
        chrome.storage.local.get([`result_${response.tabId}`], (res) => {
            const result = res[`result_${response.tabId}`];
            processResult(result);
        });
    }
});
