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

// Add these at the top level of the file
let textarea;
let submitButton;

// Add at the top level of the file
let cachedAngryAntAudio = null;

// Add at the top level of the file
let isPlayingAngryAnt = false;

// Add at the top level of the file
let isPreloadingAngryAnt = false;

// Add at the top level of the file
let mouseIndicator = null;

// Add at the top level with other variables
let tooltipElement = null;

// Add these at the top level with other variables
let threeContainer = null;
let threeRenderer = null;
let threeScene = null;
let threeCamera = null;
let threeModel = null;
let threeAnimationId = null;
let isNodding = false;
let nodParams = {
  isNodding: false,
  startTime: 0,
  duration: 1000,
  amplitude: 0.3,
  originalY: 0
};

// Function to simulate mouse movement
function moveMouseTo(x, y) {
  return new Promise(resolve => {
    createOrUpdateMouseIndicator(x, y);

    // Create or update tooltip
    if (!tooltipElement) {
      tooltipElement = document.createElement('div');
      tooltipElement.className = 'ai-tooltip';
      document.body.appendChild(tooltipElement);
    }

    // Position tooltip above the cursor
    tooltipElement.textContent = 'Click here';
    tooltipElement.style.left = `${x - tooltipElement.offsetWidth / 2}px`;
    tooltipElement.style.top = `${y - tooltipElement.offsetHeight - 25}px`;
    tooltipElement.style.opacity = '1';

    // Hide tooltip after 2 seconds
    // setTimeout(() => {
    //   if (tooltipElement) {
    //     tooltipElement.style.opacity = '0';
    //   }
    // }, 2000);

    setTimeout(resolve, 50);
  });
}

// Function to simulate mouse clicks
function simulateClick(type) {
  return new Promise(resolve => {
    // Hide tooltip immediately on click
    if (tooltipElement) {
      tooltipElement.style.opacity = '0';
    }

    const element = document.elementFromPoint(
      window.mouseX || 0,
      window.mouseY || 0
    );

    if (element && mouseIndicator) {
      mouseIndicator.style.transform = 'scale(0.8)';
      setTimeout(() => {
        mouseIndicator.style.transform = 'scale(1)';
      }, 150);

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
    // 1. Do mouse move
    await moveMouseTo(instruction.coordinate[0], instruction.coordinate[1]);

    // 3. Resolve promise and continue
    return `Moved to (${instruction.coordinate[0]}, ${instruction.coordinate[1]})`;
  },

  left_click: async () => {
    // 1. Do mouse click
    // await simulateClick('click');

    // 2. Add click event listener and wait for click
    await new Promise((resolve) => {
      const clickHandler = (event) => {
        if (tooltipElement) {
          tooltipElement.style.opacity = '0';
        }



        document.removeEventListener('click', clickHandler);
        resolve();
      };
      document.addEventListener('click', clickHandler);
    });

    // 3. Return status
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
    console.log('Taking screenshot...');
    try {
      // Store tooltip visibility state
      const tooltipWasVisible = tooltipElement && tooltipElement.style.opacity !== '0';
      
      // Hide tooltip if visible
      if (tooltipWasVisible) {
        tooltipElement.style.opacity = '0';
      }

      const response = await new Promise((resolve, reject) => {
        chrome.runtime.sendMessage({ action: 'screenshot' }, response => {
          if (chrome.runtime.lastError) {
            console.error('Chrome runtime error:', chrome.runtime.lastError);
            reject(chrome.runtime.lastError);
            return;
          }

          if (response?.error) {
            console.error('Screenshot error:', response.error);
            reject(new Error(response.error));
            return;
          }

          if (!response?.imgSrc) {
            console.error('No image data in response');
            reject(new Error('Failed to capture screenshot - no image data'));
            return;
          }

          const data = response.imgSrc.replace(/^data:image\/png;base64,/, '');

          resolve([{
            type: "image",
            source: {
              type: "base64",
              media_type: "image/png",
              data: data
            }
          }]);
        });
      });

      // Restore tooltip if it was visible before
      if (tooltipWasVisible) {
        tooltipElement.style.opacity = '1';
      }

      console.log('Screenshot captured successfully');
      return response;
    } catch (error) {
      console.error('Screenshot capture failed:', error);
      throw error;
    }
  },

  cursor_position: (instruction) => {
    return `Cursor position: x=${window.mouseX || 0}, y=${window.mouseY || 0}`;
  },

  speak: async (instruction) => {
    if (instruction.text) {
      await textToSpeech(instruction.text);
      return {
        message: `Spoke text: ${instruction.text}`
      };
    }
    return {
      message: 'No text provided to speak'
    };
  }
};

// Update the sendToAPI function
async function sendToAPI(prompt, isFollowUp = false) {
  let contentBlocks = [];

  // Add user message and screenshot if not a follow-up
  if (!isFollowUp) {
    // First add the user message block
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
  try {
    // const blockId = 'screenshot-' + Date.now();
    // // Add tool use block for screenshot
    // contentBlocks.push({
    //   type: "tool_use",
    //   id: blockId,
    //   name: "computer",
    //   input: {
    //     action: "screenshot"
    //   }
    // });



    // // Add tool result block for screenshot
    // contentBlocks.push({
    //   type: "tool_result",
    //   tool_use_id: blockId,
    //   content: [ {
    //     "type": "image",
    //     "source": {
    //       "type": "base64",
    //       "media_type": "image/png",
    //       "data": screenshotData,
    //     }
    //   }],
    //   is_error: false,
    // });

    const screenshotData = await MESSAGE_HANDLERS.screenshot();

    //  contentBlocks.push({
    //    type: "image",
    //    content: {
    //      source: {
    //        type: "base64",
    //        media_type: "image/png",
    //        data: screenshotData[0].source.data
    //      }
    //    }
    //  });

    const blockId = 'screenshot-' + Date.now();
    contentBlocks.push({
      type: "tool_use",
      content: {
        id: blockId,
        name: "computer",
        input: {
          action: "screenshot"
        }
      }
    }, {
      type: "tool_result",
      content: {
        tool_use_id: blockId,
        content: screenshotData
      }
    });
  } catch (error) {
    console.error('Failed to capture initial screenshot:', error);
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
          maxTouchPoints: navigator.maxTouchPoints,
          url: window.location.href,
          pathname: window.location.pathname,
          hostname: window.location.hostname,
          protocol: window.location.protocol,
          search: window.location.search,
          hash: window.location.hash
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

// Update the initAudioContext function to include logging
function initAudioContext() {
  try {
    if (!audioContext) {
      console.log('Creating new AudioContext...');
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      console.log('AudioContext created:', audioContext.state);
    }
    return audioContext;
  } catch (error) {
    console.error('Failed to initialize AudioContext:', error);
    return null;
  }
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
        content: {
          tool_use_id: toolUseId,
          content: result || "success" // Default to "success" if no result returned
        }
      };

      pendingToolResults.push(toolResult);
      return toolResult;
    } catch (error) {
      // Create error result
      const errorResult = {
        type: "tool_result",
        content: {
          tool_use_id: toolUseId,
          content: error.message,
          is_error: true
        }
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

// Update the preloadAngryAntAudio function to prevent parallel fetches
async function preloadAngryAntAudio() {
  // If already preloading or we have cached audio, skip
  if (isPreloadingAngryAnt || cachedAngryAntAudio) {
    console.log('Already preloading or audio cached, skipping...');
    return true;
  }

  console.log('Starting to preload angry ant audio...');
  try {
    isPreloadingAngryAnt = true;  // Set flag before starting

    const context = initAudioContext();
    if (!context) {
      console.error('No audio context available');
      isPreloadingAngryAnt = false;  // Reset flag
      return false;
    }

    console.log('Audio context state:', context.state);

    if (context.state === 'suspended') {
      try {
        console.log('Attempting to resume audio context...');
        await context.resume();
        console.log('Audio context resumed successfully');
      } catch (error) {
        console.error('Failed to resume audio context:', error);
        isPreloadingAngryAnt = false;  // Reset flag
        return false;
      }
    }

    console.log('Fetching angry ant audio...');
    const response = await fetch('http://localhost:8000/angry-ant', {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch angry ant audio: ${response.status} ${response.statusText}`);
    }

    console.log('Decoding audio data...');
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await context.decodeAudioData(arrayBuffer);
    cachedAngryAntAudio = audioBuffer;

    console.log('Successfully preloaded angry ant audio');
    isPreloadingAngryAnt = false;  // Reset flag
    return true;
  } catch (error) {
    console.error('Error preloading angry ant audio:', error);
    isPreloadingAngryAnt = false;  // Reset flag
    return false;
  }
}

// Update the playAngryAntStream function to use cached audio
async function playAngryAntStream() {
  try {
    // If already playing, don't start another one
    if (isPlayingAngryAnt) {
      console.log('Angry ant already playing, skipping...');
      return null;
    }

    const context = initAudioContext();
    if (context.state === 'suspended') {
      await context.resume();
    }

    // Use cached audio buffer if available
    const audioBuffer = cachedAngryAntAudio;
    if (!audioBuffer) {
      console.warn('No cached audio available, fetching new audio...');
      return null;
    }

    const source = context.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(context.destination);

    // Set flag before starting playback
    isPlayingAngryAnt = true;
    source.start(0);

    return new Promise((resolve) => {
      source.onended = () => {
        // Reset flag when playback ends
        isPlayingAngryAnt = false;
        resolve(source);
      };
    });
  } catch (error) {
    console.error('Error playing angry ant:', error);
    isPlayingAngryAnt = false;
    return null;
  }
}

// Update the handleSubmit function to show loading states
async function handleSubmit() {
  const prompt = textarea.value.trim();
  if (prompt) {
    try {
      const userMessageCard = createMessageCard({
        type: 'text',
        content: { text: prompt }
      }, true);
      appendMessageCard(userMessageCard);

      // Show loading state
      textarea.disabled = true;
      submitButton.disabled = true;
      textarea.value = '';

      // Add initial loading card
      const loadingCard = createLoadingCard();
      appendMessageCard(loadingCard);

      // Start API call immediately
      const responsePromise = sendToAPI(prompt);

      // Add a delay before playing the sound (500ms)
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Play the angry ant sound
      const angryAntPromise = playAngryAntStream();

      // Wait for API response
      let response = await responsePromise;

      // Remove initial loading card
      const loadingCards = document.querySelectorAll('.ai-message-card.loading');
      loadingCards.forEach(card => card.remove());

      // Set processing flag
      isProcessingTurn = true;

      // Get the audio source once it's ready
      const angryAntSound = await angryAntPromise;

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
            // Add loading card before executing instruction
            const loadingCard = createLoadingCard();
            appendMessageCard(loadingCard);

            const instruction = {
              action: block.content.name,
              ...block.content.input
            };
            
            // Execute instruction and get result
            const result = await executeInstruction(instruction, block.content.id);
            
            // Remove loading card
            loadingCard.remove();
            
            // If there's a result, show it
            if (result) {
              const resultCard = createMessageCard(result);
              appendMessageCard(resultCard);
            }
          }
        });

        // Wait for all operations to complete
        await Promise.all([...audioPromises, ...toolPromises]);

        // Break the loop if end_turn is true or if there are no tool blocks
        if (response.end_turn || toolBlocks.length === 0) {
          isProcessingTurn = false;
          break;
        }

        // Add loading card before making next API call
        const nextLoadingCard = createLoadingCard();
        appendMessageCard(nextLoadingCard);

        // Make another API call with any pending tool results
        response = await sendToAPI(prompt, true);

        // Remove loading card after getting response
        nextLoadingCard.remove();
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

// Update the init function to remove the immediate preload attempt
function init() {
  console.log('Initializing extension...');
  injectStyles();
  const button = createFloatingButton();
  document.body.appendChild(button);
}

// Make sure init runs after DOM is fully loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
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

// Add this function after the moveMouseTo function
function createOrUpdateMouseIndicator(x, y) {
  if (!mouseIndicator) {
    mouseIndicator = document.createElement('div');
    mouseIndicator.style.cssText = `
      position: fixed;
      width: 20px;
      height: 20px;
      background: rgba(75, 161, 255, 0.4);
      border: 2px solid rgba(75, 161, 255, 0.8);
      border-radius: 50%;
      pointer-events: none;
      transition: all 0.3s ease-out;
      z-index: 999999;
    `;
    document.body.appendChild(mouseIndicator);
  }

  mouseIndicator.style.transform = 'scale(1)';
  mouseIndicator.style.left = `${x - 10}px`;
  mouseIndicator.style.top = `${y - 10}px`;
}

// Add this function to create loading placeholder cards
function createLoadingCard() {
  const card = document.createElement('div');
  card.className = 'ai-message-card loading';
  
  card.innerHTML = `
    <span class="ai-card-icon">⏳</span>
    <div class="ai-card-content">
      <div class="loading-animation">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
      </div>
    </div>
  `;
  
  return card;
}

// Add the Three.js initialization function
async function initThreeJS(container) {
  try {

    const THREE = window.THREE;
    if (!THREE) {
      console.error('THREE.js failed to load');
      return;
    }

    // Create scene
    threeScene = new THREE.Scene();
    
    // Get container dimensions
    const width = 100;
    const height = 100;

    // Create camera
    threeCamera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    threeCamera.position.set(0, 1, 3);

    // Create renderer
    threeRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    threeRenderer.setSize(width, height);
    threeRenderer.setPixelRatio(window.devicePixelRatio);
    threeRenderer.setClearColor(0x000000, 0);

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 1);
    threeScene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
    directionalLight.position.set(5, 5, 5);
    threeScene.add(directionalLight);

    // Load model
    const loader = new THREE.GLTFLoader();
    const modelUrl = chrome.runtime.getURL('models/Smiling_Portrait_1123072412.glb');

    try {
      const gltf = await new Promise((resolve, reject) => {
        loader.load(
          modelUrl,
          resolve,
          (xhr) => {
            console.log(`Model loading: ${(xhr.loaded / xhr.total) * 100}% loaded`);
          },
          reject
        );
      });
      
      threeModel = gltf.scene;
      threeScene.add(threeModel);
      threeModel.position.set(0, 1, 0);
      threeModel.scale.set(1, 1, 1);
      nodParams.originalY = threeModel.position.y;
      
    } catch (modelError) {
      console.error('Error loading 3D model:', modelError);
    }


    // Start animation loop
    function animate() {
      threeAnimationId = requestAnimationFrame(animate);

      // Handle nodding
      if (nodParams.isNodding && threeModel) {
        const currentTime = performance.now();
        const elapsed = currentTime - nodParams.startTime;
        
        if (elapsed < nodParams.duration) {
          const progress = elapsed / nodParams.duration;
          const angle = progress * Math.PI;
          const yOffset = Math.sin(angle) * nodParams.amplitude;
          threeModel.position.y = nodParams.originalY + yOffset;
        } else {
          threeModel.position.y = nodParams.originalY;
          nodParams.isNodding = false;
          isNodding = false;
        }
      }

      threeRenderer.render(threeScene, threeCamera);
    }

    animate();
    
    // Append renderer to container
    container.appendChild(threeRenderer.domElement);

  } catch (error) {
    console.error('Three.js initialization error:', error);
  }
}

// Add nod function
function triggerNod() {
  if (nodParams.isNodding || !threeModel) return;
  
  nodParams.isNodding = true;
  nodParams.startTime = performance.now();
  isNodding = true;
}

// Update createFloatingButton to include Three.js container
function createFloatingButton() {
  // Create main container (existing code...)
  const container = document.createElement('div');
  container.className = 'ai-floating-container';

  // Create Three.js container
  threeContainer = document.createElement('div');
  threeContainer.className = 'ai-three-container';
  threeContainer.style.cssText = `
    width: 100px;
    height: 100px;
    position: absolute;
    bottom: 0;
    right: 100%;
    pointer-events: none;
    border-radius: 8px;
    overflow: hidden;
  `;

  // Initialize Three.js
  initThreeJS(threeContainer);

  // Add Three.js container and nod button
  container.appendChild(threeContainer);

// Update the createFloatingButton function to handle audio initialization
  console.log('Creating floating button...');

  // Create messages container
  const messagesContainer = document.createElement('div');
  messagesContainer.className = 'ai-messages-container';

  // Create input container at the bottom
  const inputContainer = document.createElement('div');
  inputContainer.className = 'ai-input-container';

  // Create textarea (now using global variable)
  textarea = document.createElement('textarea');
  textarea.className = 'ai-input-field';
  textarea.placeholder = 'Enter your prompt here...';

  // Create submit button (now using global variable)
  submitButton = document.createElement('button');
  submitButton.className = 'ai-submit-button';
  submitButton.textContent = 'Submit';

  inputContainer.appendChild(textarea);
  inputContainer.appendChild(submitButton);

  // Add components to main container
  container.appendChild(messagesContainer);
  container.appendChild(inputContainer);

  // Initialize audio on first interaction with any element
  const initAudioOnInteraction = async () => {
    console.log('User interaction detected, initializing audio...');
    const context = initAudioContext();
    if (context) {
      await context.resume();
      await preloadAngryAntAudio();
      // Remove all the interaction listeners once initialized
      container.removeEventListener('click', initAudioOnInteraction);
    }
  };

  // Add multiple opportunities to initialize audio
  container.addEventListener('click', initAudioOnInteraction);

  // Handle submit button click
  submitButton.addEventListener('click', handleSubmit);

  // Handle textarea enter key
  textarea.addEventListener('keydown', async function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  });

  return container;
}