chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'screenshot') {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      chrome.tabs.captureVisibleTab(
        tabs[0].windowId,
        {format: 'png', quality: 100},
        function(dataUrl) {
          sendResponse({imgSrc: dataUrl});
        }
      );
    });
    return true; // Required for async response
  }
}); 