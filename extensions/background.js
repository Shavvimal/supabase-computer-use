chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'screenshot') {
    // Need to handle this asynchronously
    (async () => {
      try {
        const tabs = await chrome.tabs.query({active: true, currentWindow: true});
        if (!tabs[0]) {
          sendResponse({error: 'No active tab found'});
          return;
        }
        
        const dataUrl = await chrome.tabs.captureVisibleTab(
          tabs[0].windowId,
          {format: 'png', quality: 100}
        );
        
        sendResponse({imgSrc: dataUrl});
      } catch (error) {
        console.error('Screenshot error:', error);
        sendResponse({error: error.message});
      }
    })();
    return true; // Keep message channel open for async response
  }
}); 