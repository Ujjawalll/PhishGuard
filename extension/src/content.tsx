chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'SHOW_WARNING') {
    const result = message.result;
    
    // Create interstitial
    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100vw';
    overlay.style.height = '100vh';
    overlay.style.backgroundColor = 'rgba(239, 68, 68, 0.95)';
    overlay.style.color = 'white';
    overlay.style.zIndex = '999999999';
    overlay.style.display = 'flex';
    overlay.style.flexDirection = 'column';
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';
    overlay.style.fontFamily = 'sans-serif';
    
    overlay.innerHTML = `
      <div style="max-width: 600px; padding: 2rem; text-align: center;">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">Phishing Warning</h1>
        <p style="font-size: 1.5rem; margin-bottom: 2rem;">PhishGuard has detected that this site is <strong>HIGH RISK</strong>.</p>
        <div style="background: rgba(0,0,0,0.2); padding: 1.5rem; border-radius: 8px; text-align: left; margin-bottom: 2rem;">
          <h3>Reasons:</h3>
          <ul>
            ${result.explanation.top_reasons.map((r: string) => `<li style="margin-bottom: 0.5rem;">${r}</li>`).join('')}
          </ul>
        </div>
        <p style="font-size: 1.25rem; font-weight: bold;">${result.explanation.recommendation}</p>
        <button id="pg-go-back" style="margin-top: 2rem; padding: 1rem 2rem; font-size: 1.25rem; font-weight: bold; cursor: pointer; border: none; border-radius: 4px; background: white; color: #EF4444;">Get Me Out of Here</button>
        <button id="pg-ignore" style="margin-top: 1rem; padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; border: 1px solid white; border-radius: 4px; background: transparent; color: white;">Ignore and Proceed</button>
      </div>
    `;
    
    document.body.appendChild(overlay);
    
    document.getElementById('pg-go-back')?.addEventListener('click', () => {
      window.history.back();
    });
    
    document.getElementById('pg-ignore')?.addEventListener('click', () => {
      overlay.remove();
    });
  }
});
