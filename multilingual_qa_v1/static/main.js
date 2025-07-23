class ChatBox {
  constructor() {
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.stream = null;
    this.isRecording = false;
    this.isTyping = false;
    this.selectedLang = 'en';
    
    this.elements = {
      chatMessages: document.getElementById('chatMessages'),
      messageInput: document.getElementById('messageInput'),
      micButton: document.getElementById('micButton'),
      sendButton: document.getElementById('sendButton'),
      typingIndicator: document.getElementById('typingIndicator'),
      voiceIndicator: document.getElementById('voiceIndicator'),
      micIcon: document.getElementById('micIcon'),
      stopIcon: document.getElementById('stopIcon'),
      statusMessage: document.getElementById('statusMessage'),
      langSelectBox :  document.getElementById('lang-select')
    };

    this.initializeAudioRecording();
    this.bindEvents();
    this.showStatus('Ready to chat! Try asking me something.', 'success');
  }

  langSelect(e){
    console.log(e)
  }

  async sendAudioMessage (audioBlob){
    const formData = new FormData();
    formData.append("audio", audioBlob, "speech.webm");

    const translatingIndicator = document.getElementById("translatingIndicator");
    translatingIndicator.classList.remove("hidden");

    try {
        const response = await fetch("/translate", {
          method: "POST",
          body: formData,
        });
        const message = await response.json();
        const isConfirm = confirm(`Translate : ${message?.text}`)
        if (isConfirm===true){
            this.sendMessage(message?.text);
        }
    } catch (error) {
        console.log(error);
        this.addMessage("Translate problem!", 'user');
    }
    finally{
        translatingIndicator.classList.add("hidden");
    }
  }

  async initializeAudioRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this.showStatus('Audio recording not supported in this browser', 'error');
      return;
    }

    try {
      // Request microphone permission
      this.stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100
        } 
      });
      this.showStatus('Microphone ready!', 'success');
    } catch (error) {
      this.showStatus('Microphone access denied', 'error');
      console.error('Error accessing microphone:', error);
    }
  }

  startAudioRecording() {
    if (!this.stream) {
      this.showStatus('Microphone not available', 'error');
      return;
    }

    this.audioChunks = [];
    
    let mimeType = 'audio/webm;codecs=opus';
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      mimeType = 'audio/webm';
    }
    this.mediaRecorder = new MediaRecorder(this.stream, { mimeType });

    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this.audioChunks.push(event.data);
      }
    };

    this.mediaRecorder.onstop = () => {
      const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm;codecs=opus' });
      if (audioBlob.size === 0) {
        this.showStatus("No audio detected. Try again.", "error");
        return;
      }
      this.sendAudioMessage(audioBlob);
      this.stopAudioRecording();
    };

    this.mediaRecorder.onerror = (event) => {
      this.showStatus('Recording error occurred', 'error');
      console.error('MediaRecorder error:', event.error);
      this.stopRecording();
    };

    this.mediaRecorder.start(50); // Collect data every 100ms
    this.startRecording();
  }

  stopAudioRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.stop();
    } else {
      this.stopRecording();
    }
  }

  bindEvents() {
    // Send button click
    this.elements.sendButton.addEventListener('click', () => {
      this.sendMessage();
    });

    // Enter key press
    this.elements.messageInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        this.sendMessage();
      }
    });

    // Microphone button
    this.elements.micButton.addEventListener('click', () => {
      if (this.isRecording) {
        this.stopAudioRecording();
      } else {
        this.startAudioRecording();
      }
    });

    // Input focus animations
    this.elements.messageInput.addEventListener('focus', () => {
      this.elements.messageInput.parentElement.classList.add('ring-2', 'ring-blue-400');
    });

    this.elements.messageInput.addEventListener('blur', () => {
      this.elements.messageInput.parentElement.classList.remove('ring-2', 'ring-blue-400');
    });

    this.elements.langSelectBox.addEventListener("change",(e)=>{
      this.selectedLang = e.target.value;
    })
  }

  startRecording() {
    this.isRecording = true;
    this.elements.micIcon.classList.add('hidden');
    this.elements.stopIcon.classList.remove('hidden');
    this.elements.voiceIndicator.classList.remove('hidden');
    this.elements.micButton.classList.add('bg-red-500', 'hover:bg-red-600');
    this.elements.micButton.classList.remove('bg-gradient-to-br', 'from-blue-500', 'to-purple-600');
    this.showStatus('Listening... Speak now!', 'info');
  }

  stopRecording() {
    this.isRecording = false;
    this.elements.micIcon.classList.remove('hidden');
    this.elements.stopIcon.classList.add('hidden');
    this.elements.voiceIndicator.classList.add('hidden');
    this.elements.micButton.classList.remove('bg-red-500', 'hover:bg-red-600');
    this.elements.micButton.classList.add('bg-gradient-to-br', 'from-blue-500', 'to-purple-600');
    this.showStatus('Audio recorded, sending...', 'info');
  }

  async sendMessage(msg="") {
    const message = msg && msg.trim() !== "" ? msg.trim() : this.elements.messageInput.value.trim();
    if (!message || this.isTyping) return;

    // Add user message
    this.addMessage(message, 'user');
    this.elements.messageInput.value = '';

    // Show typing indicator
    this.showTypingIndicator();

    // Simulate AI response delay
    const response = await this.generateAIResponse(message);
    
    // Hide typing indicator and add AI response
    this.hideTypingIndicator();
    this.addMessage(response, 'ai');
  }

  addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'flex items-start space-x-3 animate-slide-up';

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    if (sender === 'user') {
      messageDiv.innerHTML = `
        <div class="flex-1"></div>
        <div class="flex-shrink-0 max-w-xs md:max-w-md lg:max-w-lg">
          <div class="bg-gradient-to-br from-blue-500 to-purple-600 text-white rounded-2xl rounded-tr-md p-4 shadow-lg">
            <div class="message-content">${this.escapeHtml(text)}</div>
            <span class="text-xs text-blue-100 mt-2 block">${timestamp}</span>
          </div>
        </div>
        <div class="flex-shrink-0">
          <div class="w-10 h-10 bg-gradient-to-br from-gray-400 to-gray-600 rounded-full flex items-center justify-center">
            <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
            </svg>
          </div>
        </div>
      `;
    } else {
      messageDiv.innerHTML = `
        <div class="flex-shrink-0">
          <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
            <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
            </svg>
          </div>
        </div>
        <div class="flex-1 max-w-xs md:max-w-md lg:max-w-lg">
          <div class="bg-white/40 backdrop-blur-sm rounded-2xl rounded-tl-md p-4 shadow-lg">
            <div class="message-content text-gray-800">${this.sanitizeAndRenderHtml(text)}</div>
            <span class="text-xs text-gray-500 mt-2 block">${timestamp}</span>
          </div>
        </div>
      `;
    }

    this.elements.chatMessages.appendChild(messageDiv);
    this.scrollToBottom();
  }

  showTypingIndicator() {
    this.isTyping = true;
    this.elements.typingIndicator.classList.remove('hidden');
    this.scrollToBottom();
  }

  hideTypingIndicator() {
    this.isTyping = false;
    this.elements.typingIndicator.classList.add('hidden');
  }

  async generateAIResponse(userMessage) {
    try {
        const response = await fetch(`/chat/${this.selectedLang}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: userMessage }),
        });
        if (!response.ok) {
          return "Sorry, something went wrong.";
        }

        const data = await response.json();
        return data.reply;
    } catch (error) {
        return "Sorry, something went wrong."
    }
  }

  showStatus(message, type = 'info') {
    this.elements.statusMessage.textContent = message;
    this.elements.statusMessage.className = `text-center mt-4 text-sm transition-opacity duration-300 ${
      type === 'error' ? 'text-red-600' : 
      type === 'success' ? 'text-green-600' : 
      type === 'info' ? 'text-blue-600' : 
      'text-gray-600'
    }`;
    this.elements.statusMessage.style.opacity = '1';

    // Auto-hide after 3 seconds
    setTimeout(() => {
      this.elements.statusMessage.style.opacity = '0';
    }, 3000);
  }

  scrollToBottom() {
    setTimeout(() => {
      this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    }, 100);
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  sanitizeAndRenderHtml(htmlContent) {
    // Create a temporary div to parse and sanitize HTML
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = htmlContent;
    
    // Remove potentially dangerous elements and attributes
    this.sanitizeElement(tempDiv);
    
    return tempDiv.innerHTML;
  }

  sanitizeElement(element) {
    // List of allowed tags
    const allowedTags = [
      'p', 'div', 'span', 'br', 'strong', 'b', 'em', 'i', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'ul', 'ol', 'li', 'blockquote', 'code', 'pre', 'a', 'img', 'table', 'thead', 'tbody', 'tr', 'th', 'td'
    ];
    
    // List of allowed attributes
    const allowedAttributes = ['href', 'src', 'alt', 'title', 'class'];
    
    // Remove script tags and other dangerous elements
    const dangerousTags = element.querySelectorAll('script, object, embed, iframe, form, input, button');
    dangerousTags.forEach(tag => tag.remove());
    
    // Process all elements
    const allElements = element.querySelectorAll('*');
    allElements.forEach(el => {
      // Remove elements not in allowed list
      if (!allowedTags.includes(el.tagName.toLowerCase())) {
        // Replace with span to preserve content
        const span = document.createElement('span');
        span.innerHTML = el.innerHTML;
        el.parentNode.replaceChild(span, el);
        return;
      }
      
      // Remove dangerous attributes
      Array.from(el.attributes).forEach(attr => {
        if (!allowedAttributes.includes(attr.name.toLowerCase()) && !attr.name.startsWith('data-')) {
          el.removeAttribute(attr.name);
        }
      });
      
      // Sanitize href attributes
      if (el.hasAttribute('href')) {
        const href = el.getAttribute('href');
        if (!href.startsWith('http://') && !href.startsWith('https://') && !href.startsWith('mailto:')) {
          el.removeAttribute('href');
        }
      }
    });
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Initialize the chatbox when the page loads
document.addEventListener('DOMContentLoaded', () => {
  new ChatBox();
});

// Add some additional interactive elements
document.addEventListener('DOMContentLoaded', () => {
  // Add particle animation on mouse move
  let mouseX = 0;
  let mouseY = 0;

  document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
  });

  // Create floating particles occasionally
  setInterval(() => {
    createFloatingParticle();
  }, 3000);

  function createFloatingParticle() {
    const particle = document.createElement('div');
    particle.className = 'fixed w-2 h-2 bg-blue-400 rounded-full pointer-events-none opacity-30';
    particle.style.left = Math.random() * window.innerWidth + 'px';
    particle.style.top = window.innerHeight + 'px';
    particle.style.animation = 'float-up 4s linear forwards';
    
    document.body.appendChild(particle);
    
    setTimeout(() => {
      particle.remove();
    }, 4000);
  }

  // Add CSS for floating animation
  const style = document.createElement('style');
  style.textContent = `
    @keyframes float-up {
      0% {
        transform: translateY(0) rotate(0deg);
        opacity: 0.3;
      }
      50% {
        opacity: 0.6;
      }
      100% {
        transform: translateY(-100vh) rotate(360deg);
        opacity: 0;
      }
    }
    
    /* Message content styling for HTML responses */
    .message-content h1, .message-content h2, .message-content h3,
    .message-content h4, .message-content h5, .message-content h6 {
      font-weight: bold;
      margin: 0.5em 0;
    }
    
    .message-content h1 { font-size: 1.5em; }
    .message-content h2 { font-size: 1.3em; }
    .message-content h3 { font-size: 1.1em; }
    
    .message-content p {
      margin: 0.5em 0;
      line-height: 1.5;
    }
    
    .message-content ul, .message-content ol {
      margin: 0.5em 0;
      padding-left: 1.5em;
    }
    
    .message-content li {
      margin: 0.25em 0;
    }
    
    .message-content strong, .message-content b {
      font-weight: bold;
    }
    
    .message-content em, .message-content i {
      font-style: italic;
    }
    
    .message-content code {
      background-color: rgba(0, 0, 0, 0.1);
      padding: 0.2em 0.4em;
      border-radius: 0.25em;
      font-family: 'Courier New', monospace;
      font-size: 0.9em;
    }
    
    .message-content pre {
      background-color: rgba(0, 0, 0, 0.1);
      padding: 1em;
      border-radius: 0.5em;
      overflow-x: auto;
      margin: 0.5em 0;
    }
    
    .message-content pre code {
      background: none;
      padding: 0;
    }
    
    .message-content blockquote {
      border-left: 3px solid rgba(59, 130, 246, 0.5);
      padding-left: 1em;
      margin: 0.5em 0;
      font-style: italic;
    }
    
    .message-content a {
      color: #3B82F6;
      text-decoration: underline;
    }
    
    .message-content a:hover {
      color: #1D4ED8;
    }
    
    .message-content img {
      max-width: 100%;
      height: auto;
      border-radius: 0.5em;
      margin: 0.5em 0;
    }
    
    .message-content table {
      width: 100%;
      border-collapse: collapse;
      margin: 0.5em 0;
    }
    
    .message-content th, .message-content td {
      border: 1px solid rgba(0, 0, 0, 0.2);
      padding: 0.5em;
      text-align: left;
    }
    
    .message-content th {
      background-color: rgba(0, 0, 0, 0.1);
      font-weight: bold;
    }
    
    .scrollbar-thin::-webkit-scrollbar {
      width: 6px;
    }
    
    .scrollbar-thin::-webkit-scrollbar-track {
      background: transparent;
    }
    
    .scrollbar-thin::-webkit-scrollbar-thumb {
      background: rgba(59, 130, 246, 0.3);
      border-radius: 3px;
    }
    
    .scrollbar-thin::-webkit-scrollbar-thumb:hover {
      background: rgba(59, 130, 246, 0.5);
    }
    
    /* Audio player styling */
    .audio-player {
      height: 32px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 16px;
    }
    
    .audio-player::-webkit-media-controls-panel {
      background-color: transparent;
    }
    
    .audio-player::-webkit-media-controls-play-button,
    .audio-player::-webkit-media-controls-pause-button {
      background-color: rgba(255, 255, 255, 0.8);
      border-radius: 50%;
    }
    
    .audio-player::-webkit-media-controls-timeline {
      background-color: rgba(255, 255, 255, 0.3);
      border-radius: 25px;
      margin-left: 10px;
      margin-right: 10px;
    }
    
    .audio-player::-webkit-media-controls-current-time-display,
    .audio-player::-webkit-media-controls-time-remaining-display {
      color: white;
      text-shadow: none;
      font-size: 11px;
    }
  `;
  document.head.appendChild(style);
});