// Add counter at the top level of the file
let mockResponseCounter = 0;

// Add this at the top level
let messageHistory = [];

// Add this at the top level of the file
let audioContext = null;

// Add this at the top level of the file
let isProcessingTurn = false;

// Add this at the top level of the file
let pendingToolResults = [];

// Function to simulate mouse movement
function moveMouseTo(x, y) {
  return new Promise(resolve => {
    const event = new MouseEvent('mousemove', {
      view: window,
      bubbles: true,
      cancelable: true,
      clientX: x,
      clientY: y
    });
    document.elementFromPoint(x, y)?.dispatchEvent(event);
    setTimeout(resolve, 50);
  });
}

// Function to simulate mouse clicks
function simulateClick(type) {
  return new Promise(resolve => {
    const element = document.elementFromPoint(
      window.mouseX || 0,
      window.mouseY || 0
    );
    
    if (element) {
      const event = new MouseEvent(type, {
        view: window,
        bubbles: true,
        cancelable: true,
        clientX: window.mouseX || 0,
        clientY: window.mouseY || 0
      });
      element.dispatchEvent(event);
    }
    setTimeout(resolve, 50);
  });
}

// Function to simulate keyboard input
function simulateKeyboard(text, isRawKey = false) {
  return new Promise(resolve => {
    const element = document.activeElement;
    if (element) {
      if (isRawKey) {
        const event = new KeyboardEvent('keypress', {
          key: text,
          code: `Key${text.toUpperCase()}`,
          bubbles: true
        });
        element.dispatchEvent(event);
      } else {
        element.value = (element.value || '') + text;
        element.dispatchEvent(new Event('input', { bubbles: true }));
      }
    }
    setTimeout(resolve, text.length * 12); // Same delay as backend
  });
}

// Constants and message handlers
const MESSAGE_HANDLERS = {
  mouse_move: async (instruction) => {
    await moveMouseTo(instruction.coordinate[0], instruction.coordinate[1]);
    return `Moved to (${instruction.coordinate[0]}, ${instruction.coordinate[1]})`;
  },
  
  left_click: async () => {
    await simulateClick('click');
    return 'Performed left click';
  },
  
  right_click: async () => {
    await simulateClick('contextmenu');
    return 'Performed right click';
  },
  
  double_click: async () => {
    await simulateClick('dblclick');
    return 'Performed double click';
  },
  
  type: async (instruction) => {
    await simulateKeyboard(instruction.text);
    return `Typed: ${instruction.text}`;
  },
  
  key: async (instruction) => {
    await simulateKeyboard(instruction.text, true);
    return `Pressed key: ${instruction.text}`;
  },
  
  screenshot: async () => {
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage({action: 'screenshot'}, response => {
        if (response?.imgSrc) {
          resolve(response.imgSrc);
        } else {
          reject(new Error('Failed to capture screenshot'));
        }
      });
    });
  },
  
  cursor_position: (instruction) => {
    return {
      x: window.mouseX || 0,
      y: window.mouseY || 0
    };
  },
  
  speak: async (instruction) => {
    if (instruction.text) {
      await textToSpeech(instruction.text);
      return `Spoke text: ${instruction.text}`;
    }
    return 'No text provided to speak';
  }
};

// Update the sendToAPI function
async function sendToAPI(prompt, isFollowUp = false) {
  let contentBlocks = [];
  
  // Add user message if not a follow-up
  if (!isFollowUp) {
    contentBlocks.push({
      type: "text",
      content: { text: prompt }
    });
  }
  
  // Add any pending tool results
  if (pendingToolResults.length > 0) {
    contentBlocks.push(...pendingToolResults);
    // Clear pending results after adding them
    pendingToolResults = [];
  }

  const response = await fetch('http://localhost:8000/agent', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      conversation_id: window.currentConversationId,
      content_blocks: contentBlocks,
      metadata: {
        screen: {
          width: window.screen.width,
          height: window.screen.height,
          availWidth: window.screen.availWidth,
          availHeight: window.screen.availHeight,
          colorDepth: window.screen.colorDepth,
          pixelDepth: window.screen.pixelDepth,
          orientation: window.screen.orientation?.type
        },
        window: {
          innerWidth: window.innerWidth,
          innerHeight: window.innerHeight,
          outerWidth: window.outerWidth, 
          outerHeight: window.outerHeight,
          devicePixelRatio: window.devicePixelRatio
        },
        browser: {
          userAgent: navigator.userAgent,
          platform: navigator.platform,
          language: navigator.language,
          languages: navigator.languages,
          cookieEnabled: navigator.cookieEnabled,
          doNotTrack: navigator.doNotTrack,
          vendor: navigator.vendor,
          maxTouchPoints: navigator.maxTouchPoints
        },
        connection: {
          type: navigator.connection?.type,
          effectiveType: navigator.connection?.effectiveType,
          downlink: navigator.connection?.downlink,
          rtt: navigator.connection?.rtt
        }
      }
    })
  });

  if (!response.ok) {
    throw new Error('API request failed');
  }

  const data = await response.json();
  window.currentConversationId = data.conversation_id;
  
  // Add assistant response to history
  messageHistory.push({
    role: 'assistant',
    content: data.content
  });

  return data;
}

// Add function to append message cards
function appendMessageCard(card) {
  const container = document.querySelector('.ai-messages-container');
  
  // Add card to container with animation
  card.style.opacity = '0';
  card.style.transform = 'translateY(20px)';
  
  // Insert at the bottom of the messages container
  container.appendChild(card);
  
  // Trigger animation
  requestAnimationFrame(() => {
    card.style.transition = 'transform 0.3s ease-out, opacity 0.3s ease-out';
    card.style.transform = 'translateY(0)';
    card.style.opacity = '1';
  });
  
  // Scroll to bottom
  container.scrollTop = container.scrollHeight;
}

// Add this function to initialize audio context on user interaction
function initAudioContext() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  return audioContext;
}

// Update the textToSpeech function
async function textToSpeech(text) {
  try {
    // Initialize audio context if needed
    const context = initAudioContext();
    if (context.state === 'suspended') {
      await context.resume();
    }

    const response = await fetch('http://localhost:8000/text-to-ant', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text })
    });

    if (!response.ok) {
      throw new Error('Text-to-speech request failed');
    }

    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await context.decodeAudioData(arrayBuffer);
    
    const source = context.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(context.destination);
    source.start(0);

    return new Promise((resolve, reject) => {
      source.onended = resolve;
      source.onerror = reject;
    });

  } catch (error) {
    console.error('Text-to-speech error:', error);
    throw error;
  }
}

// Update the executeInstruction function
async function executeInstruction(instruction, toolUseId) {
  const handler = MESSAGE_HANDLERS[instruction.action];
  if (handler) {
    try {
      const result = await handler(instruction);
      
      // Create tool result object
      const toolResult = {
        type: "tool_result",
        tool_use_id: toolUseId,
        content: result || "success" // Default to "success" if no result returned
      };
      
      pendingToolResults.push(toolResult);
      return toolResult;
    } catch (error) {
      // Create error result
      const errorResult = {
        type: "tool_result",
        tool_use_id: toolUseId,
        content: error.message,
        is_error: true
      };
      
      pendingToolResults.push(errorResult);
      return errorResult;
    }
  }
}

// Track mouse position
document.addEventListener('mousemove', (e) => {
  window.mouseX = e.clientX;
  window.mouseY = e.clientY;
});

// Update the createFloatingButton function to initialize audio context on click
function createFloatingButton() {
  // Create main container
  const container = document.createElement('div');
  container.className = 'ai-floating-container';
  
  // Create messages container
  const messagesContainer = document.createElement('div');
  messagesContainer.className = 'ai-messages-container';
  
  // Create input container at the bottom
  const inputContainer = document.createElement('div');
  inputContainer.className = 'ai-input-container';
  
  // Create textarea
  const textarea = document.createElement('textarea');
  textarea.className = 'ai-input-field';
  textarea.placeholder = 'Enter your prompt here...';
  
  // Create submit button
  const submitButton = document.createElement('button');
  submitButton.className = 'ai-submit-button';
  submitButton.textContent = 'Submit';
  
  inputContainer.appendChild(textarea);
  inputContainer.appendChild(submitButton);
  
  // Add components to main container
  container.appendChild(messagesContainer);
  container.appendChild(inputContainer);

  // Update click handler to just initialize audio
  container.addEventListener('click', function(e) {
    initAudioContext();
  });

  // Handle submit button click
  async function handleSubmit() {
    const prompt = textarea.value.trim();
    if (prompt) {
      try {
        // Show loading state
        textarea.disabled = true;
        submitButton.disabled = true;
        textarea.value = 'Processing...';

        // Set processing flag
        isProcessingTurn = true;

        // Initial API call with user prompt
        let response = await sendToAPI(prompt);

        // Continue processing until end_turn is true
        while (isProcessingTurn) {
          // Process text blocks
          const textBlocks = response.content.filter(block => block.type === 'text');
          const toolBlocks = response.content.filter(block => block.type === 'tool_use');
          
          // Process text blocks
          const audioPromises = textBlocks.map(block => {
            const card = createMessageCard(block);
            appendMessageCard(card);
            return textToSpeech(block.content.text);
          });

          // Process tool blocks
          const toolPromises = toolBlocks.map(async block => {
            const card = createMessageCard(block);
            appendMessageCard(card);
            
            if (block.content.name && block.content.input) {
              const instruction = {
                action: block.content.name,
                ...block.content.input
              };
              // Pass tool_use_id to executeInstruction
              await executeInstruction(instruction, block.id);
            }
          });

          // Wait for all operations to complete
          await Promise.all([...audioPromises, ...toolPromises]);

          // Break the loop if end_turn is true
          if (response.end_turn) {
            isProcessingTurn = false;
            break;
          }

          // Make another API call with any pending tool results
          response = await sendToAPI(prompt, true);
        }

        // Clear and re-enable inputs
        textarea.value = '';
        textarea.disabled = false;
        submitButton.disabled = false;
        textarea.focus();

      } catch (error) {
        console.error('Error:', error);
        textarea.value = 'Error occurred. Please try again.';
        textarea.disabled = false;
        submitButton.disabled = false;
        isProcessingTurn = false;
        setTimeout(() => {
          textarea.value = prompt;
          textarea.focus();
        }, 2000);
      }
    }
  }

  submitButton.addEventListener('click', handleSubmit);

  // Handle textarea enter key
  textarea.addEventListener('keydown', async function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  });

  // Close popup when clicking outside
  document.addEventListener('click', function(e) {
    if (!container.contains(e.target) && container.classList.contains('expanded')) {
      popupContent.classList.remove('visible');
      setTimeout(() => {
        container.classList.remove('expanded');
      }, 300);
    }
  });

  // Prevent popup from closing when clicking inside
  container.addEventListener('click', function(e) {
    e.stopPropagation();
  });

  return container;
}

// Add this function after createFloatingButton but before init()
function createActionCard(instruction) {
  const card = document.createElement('div');
  card.className = 'ai-action-card';
  
  // Create icon based on action type
  const iconMap = {
    mouse_move: '🖱️',
    left_click: '👆',
    right_click: '👆',
    double_click: '👆👆',
    type: '⌨️',
    key: '🔤',
    screenshot: '📸'
  };
  
  const icon = iconMap[instruction.action] || '❓';

  console.log(instruction);
  
  // Create card content
  let text = instruction.action;
  if (instruction.text) {
    text += `: "${instruction.text}"`;
  } else if (instruction.coordinate) {
    text += `: (${instruction.coordinate[0]}, ${instruction.coordinate[1]})`;
  }
  
  card.innerHTML = `
    <span class="ai-card-icon">${icon}</span>
    <span class="ai-card-text">${text}</span>
  `;
  
  return card;
}

// Update the injectStyles function to use external stylesheet
function injectStyles() {
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.type = 'text/css';
  link.href = chrome.runtime.getURL('styles.css');
  document.head.appendChild(link);
}

// Initialize
function init() {
  injectStyles();
  const button = createFloatingButton();
  document.body.appendChild(button);
}

// Add function to create message card
function createMessageCard(block, isUser = false) {
  const card = document.createElement('div');
  card.className = 'ai-message-card';
  if (isUser) card.classList.add('user');
  
  // Add animation class
  card.classList.add('ai-message-card-animated');
  
  let icon = isUser ? '👤' : '🤖';
  let content = '';
  
  switch (block.type) {
    case 'text':
      content = block.content.text;
      break;
      
    case 'tool_use':
      content = block.content.input.action.replace(/_/g, ' '); // Just show the action name
      icon = '🔧';
      break;
      
    case 'tool_result':
      content = block.content.is_error ? 
        `Error: ${block.content.content}` :
        block.content.content;
      icon = block.content.is_error ? '❌' : '✅';
      break;
      
    case 'image':
      const img = document.createElement('img');
      img.src = block.content.source.data;
      img.className = 'ai-message-image';
      card.appendChild(img);
      icon = '🖼️';
      break;
      
    default:
      content = JSON.stringify(block.content, null, 2);
      icon = '❓';
  }
  
  card.innerHTML = `
    <span class="ai-card-icon">${icon}</span>
    <div class="ai-card-content">
      <span class="ai-card-text">${content}</span>
    </div>
  `;
  
  return card;
}

init(); 