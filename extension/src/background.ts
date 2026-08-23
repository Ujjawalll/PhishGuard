chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  if (details.frameId !== 0) return; // Only top-level
  
  const url = details.url;
  if (!url.startsWith('http')) return;
  
  const scanId = crypto.randomUUID();

  // Step 1: Invalidate stale result for this tab
  await chrome.storage.local.remove(`result_${details.tabId}`);
  
  // Set badge to loading
  chrome.action.setBadgeBackgroundColor({ color: '#FCD34D', tabId: details.tabId });
  chrome.action.setBadgeText({ text: '...', tabId: details.tabId });
  
  // Tell content script to clear any remaining warnings
  chrome.tabs.sendMessage(details.tabId, { type: 'CLEAR_WARNING' }).catch(() => {});

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
    
    let result: any = { risk_level: 'ANALYSIS_UNAVAILABLE', scanned_url: url, scan_id: scanId, scan_stage: 'FAST' };
    
    if (response.ok) {
      result = await response.json();
    }
    
    result.scanned_url = url;
    result.scan_id = scanId;
    result.scan_stage = 'FAST';
    
    // Update badge
    const colors: Record<string, string> = {
      'LOW_RISK': '#10B981',
      'SUSPICIOUS': '#F59E0B',
      'HIGH_RISK': '#EF4444',
      'ANALYSIS_UNAVAILABLE': '#9CA3AF'
    };
    const badges: Record<string, string> = {
      'LOW_RISK': '✓',
      'SUSPICIOUS': '?',
      'HIGH_RISK': '✗',
      'ANALYSIS_UNAVAILABLE': '!'
    };
    
    chrome.action.setBadgeBackgroundColor({ color: colors[result.risk_level] || '#9CA3AF', tabId: details.tabId });
    chrome.action.setBadgeText({ text: badges[result.risk_level] || '!', tabId: details.tabId });
    
    // Save result
    await chrome.storage.local.set({ [`result_${details.tabId}`]: result });
    
    // Send to content script
    chrome.tabs.sendMessage(details.tabId, { type: 'UPDATE_WARNING', result }).catch(() => {});
    
    // Trigger deep scan if recommended
    if (result.deep_analysis_recommended && result.risk_level !== 'ANALYSIS_UNAVAILABLE') {
      chrome.action.setBadgeText({ text: '↻', tabId: details.tabId });
      
      const res = await fetch(`${apiUrl}/scan/deep`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ url })
      }).catch(() => null);
      
      if (res && res.ok) {
        const deepRes = await res.json();
        deepRes.scanned_url = url;
        deepRes.scan_id = scanId;
        deepRes.scan_stage = 'DEEP';
        
        chrome.action.setBadgeBackgroundColor({ color: colors[deepRes.risk_level] || '#9CA3AF', tabId: details.tabId });
        chrome.action.setBadgeText({ text: badges[deepRes.risk_level] || '✓', tabId: details.tabId });
        await chrome.storage.local.set({ [`result_${details.tabId}`]: deepRes });
        chrome.tabs.sendMessage(details.tabId, { type: 'UPDATE_WARNING', result: deepRes }).catch(() => {});
      }
    }
  } catch (err) {
    chrome.action.setBadgeBackgroundColor({ color: '#9CA3AF', tabId: details.tabId });
    chrome.action.setBadgeText({ text: '!', tabId: details.tabId });
    
    const errResult = { risk_level: 'ANALYSIS_UNAVAILABLE', scanned_url: url, scan_id: scanId, scan_stage: 'FAST' };
    await chrome.storage.local.set({ [`result_${details.tabId}`]: errResult });
    chrome.tabs.sendMessage(details.tabId, { type: 'UPDATE_WARNING', result: errResult }).catch(() => {});
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "GET_TAB_ID" && sender.tab) {
        sendResponse({ tabId: sender.tab.id });
    }
    return true;
});
