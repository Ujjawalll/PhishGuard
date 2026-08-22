chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  if (details.frameId !== 0) return; // Only top-level
  
  const url = details.url;
  if (!url.startsWith('http')) return;
  
  // Set badge to loading
  chrome.action.setBadgeBackgroundColor({ color: '#FCD34D', tabId: details.tabId });
  chrome.action.setBadgeText({ text: '...', tabId: details.tabId });

  try {
    const { token, apiUrl = 'http://localhost:8000' } = await chrome.storage.local.get(['token', 'apiUrl']);
    if (!token) return;

    const response = await fetch(`${apiUrl}/scan`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ url })
    });
    
    if (response.ok) {
      const result = await response.json();
      
      // Update badge
      const colors: Record<string, string> = {
        'SAFE': '#10B981',
        'SUSPICIOUS': '#F59E0B',
        'HIGH_RISK': '#EF4444'
      };
      
      chrome.action.setBadgeBackgroundColor({ color: colors[result.risk_level], tabId: details.tabId });
      chrome.action.setBadgeText({ text: '✓', tabId: details.tabId });
      
      // Tell content script if it's high risk to show interstitial
      if (result.risk_level === 'HIGH_RISK') {
        // We wait a bit for the page to load, then send message
        setTimeout(() => {
          chrome.tabs.sendMessage(details.tabId, { type: 'SHOW_WARNING', result });
        }, 1500);
      }
      
      // Save result for popup
      chrome.storage.local.set({ [`result_${details.tabId}`]: result });
      
      // Trigger deep scan if recommended
      if (result.deep_analysis_recommended) {
        chrome.action.setBadgeText({ text: '↻', tabId: details.tabId });
        fetch(`${apiUrl}/scan/deep`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({ url })
        }).then(async res => {
          if (res.ok) {
            const deepRes = await res.json();
            chrome.action.setBadgeBackgroundColor({ color: colors[deepRes.risk_level], tabId: details.tabId });
            chrome.action.setBadgeText({ text: '✓', tabId: details.tabId });
            chrome.storage.local.set({ [`result_${details.tabId}`]: deepRes });
          }
        });
      }
    }
  } catch (err) {
    chrome.action.setBadgeBackgroundColor({ color: '#9CA3AF', tabId: details.tabId });
    chrome.action.setBadgeText({ text: '!', tabId: details.tabId });
  }
});
