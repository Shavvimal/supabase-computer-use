// Add counter at the top level of the file
let mockResponseCounter = 0;

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
  },
  
  left_click: async () => {
    await simulateClick('click');
  },
  
  right_click: async () => {
    await simulateClick('contextmenu');
  },
  
  double_click: async () => {
    await simulateClick('dblclick');
  },
  
  type: async (instruction) => {
    await simulateKeyboard(instruction.text);
  },
  
  key: async (instruction) => {
    await simulateKeyboard(instruction.text, true);
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
  
  cursor_position: (instruction, sendResponse) => {
    sendResponse({
      x: window.mouseX || 0,
      y: window.mouseY || 0
    });
  }
};

const MOCK_RESPONSES = [
  {
    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
    "content": [
      {
        "action": "mouse_move",
        "text": null,
        "coordinate": [500, 300],
        "wait_ms": 50,
        "screenshot": null
      }
    ]
  },
  {
    "conversation_id": "123e4567-e89b-12d3-a456-426614174001", 
    "content": [
      {
        "action": "left_click",
        "text": null,
        "coordinate": null,
        "wait_ms": 50,
        "screenshot": null
      }
    ]
  },
  {
    "conversation_id": "123e4567-e89b-12d3-a456-426614174002",
    "content": [
      {
        "action": "type",
        "text": "Hello, I'm typing some text",
        "coordinate": null,
        "wait_ms": 276,
        "screenshot": null
      }
    ]
  },
  {
    "conversation_id": "123e4567-e89b-12d3-a456-426614174003",
    "content": [
      {
        "action": "key",
        "text": "Enter",
        "coordinate": null,
        "wait_ms": 60,
        "screenshot": null
      }
    ]
  },
  {
    "conversation_id": "123e4567-e89b-12d3-a456-426614174004",
    "content": [
      {
        "action": "mouse_move",
        "text": null,
        "coordinate": [800, 600],
        "wait_ms": 50,
        "screenshot": null
      }
    ]
  },
  {
    "conversation_id": "123e4567-e89b-12d3-a456-426614174005",
    "content": [
      {
        "action": "right_click",
        "text": null,
        "coordinate": null,
        "wait_ms": 50,
        "screenshot": null
      }
    ]
  }
] = {
    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
    "content": [
      // Sequential instructions that would be executed one after another
      {
        "action": "mouse_move",
        "text": null,
        "coordinate": [500, 300],
        "wait_ms": 50,
        "screenshot": null
      },
      {
        "action": "left_click",
        "text": null,
        "coordinate": null,
        "wait_ms": 50,
        "screenshot": null
      },
      {
        "action": "type",
        "text": "Hello, I'm typing some text",
        "coordinate": null,
        "wait_ms": 276,
        "screenshot": null
      },
      {
        "action": "key",
        "text": "Enter",
        "coordinate": null,
        "wait_ms": 60,
        "screenshot": null
      },
      {
        "action": "mouse_move",
        "text": null,
        "coordinate": [800, 600],
        "wait_ms": 50,
        "screenshot": null
      },
      {
        "action": "right_click",
        "text": null,
        "coordinate": null,
        "wait_ms": 50,
        "screenshot": null
      }
    ]
  }

// Handle floating button instruction execution
async function executeInstruction(instruction) {
  const handler = MESSAGE_HANDLERS[instruction.action];
  if (handler) {
    // Create and add action card
    const card = createActionCard(instruction);
    const container = document.querySelector('.ai-floating-button');
    document.body.appendChild(card);
    
    // Set initial position using CSS custom property
    const containerRect = container.getBoundingClientRect();
    card.style.setProperty('--card-bottom', `${containerRect.height + 20}px`);
    
    // Adjust positions of existing cards
    const existingCards = document.querySelectorAll('.ai-action-card');
    existingCards.forEach((existingCard, index) => {
      if (existingCard !== card) {
        const newBottom = containerRect.height + 20 + ((index + 1) * 60);
        existingCard.style.setProperty('--card-bottom', `${newBottom}px`);
        existingCard.classList.add('stacked');
        existingCard.style.setProperty('--stack-opacity', Math.max(0, 1 - (index * 0.2)));
      }
    });
    
    // Execute the action
    await handler(instruction);
    if (instruction.wait_ms) {
      await new Promise(resolve => setTimeout(resolve, instruction.wait_ms));
    }
    
    // Animate out and remove card after delay
    setTimeout(() => {
      card.classList.add('removing');
      setTimeout(() => card.remove(), 300);
    }, 2000);
  }
}

// Track mouse position
document.addEventListener('mousemove', (e) => {
  window.mouseX = e.clientX;
  window.mouseY = e.clientY;
});

function createFloatingButton() {
  // Create main button
  const container = document.createElement('div');
  container.className = 'ai-floating-button';
  
  // Create icon
  const icon = document.createElement('span');
  icon.className = 'ai-button-icon';
  icon.innerHTML = '📽️';
  container.appendChild(icon);

  // Create popup content
  const popupContent = document.createElement('div');
  popupContent.className = 'ai-popup-content';
  
  // Create textarea
  const textarea = document.createElement('textarea');
  textarea.className = 'ai-input-field';
  textarea.placeholder = 'Enter your prompt here...';
  
  // Create submit button
  const submitButton = document.createElement('button');
  submitButton.className = 'ai-submit-button';
  submitButton.textContent = 'Submit';
  
  popupContent.appendChild(textarea);
  popupContent.appendChild(submitButton);
  container.appendChild(popupContent);

  // Add click handler for the container
  container.addEventListener('click', function(e) {
    if (!container.classList.contains('expanded')) {
      container.classList.add('expanded');
      setTimeout(() => {
        popupContent.classList.add('visible');
        textarea.focus();
      }, 300);
    }
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

        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Get next mock response and increment counter
        const mockResponse = MOCK_RESPONSES[mockResponseCounter];
        mockResponseCounter = (mockResponseCounter + 1) % MOCK_RESPONSES.length;

        // Execute instructions from mock response
        for (const instruction of mockResponse.content) {
          await executeInstruction(instruction);
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
    key: '🔤'
  };
  
  const icon = iconMap[instruction.action] || '❓';
  
  // Create card content
  let text = instruction.action.replace('_', ' ');
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

init(); 