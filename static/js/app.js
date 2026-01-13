// ==========================================
// Movie Critic AI - JavaScript Application
// ==========================================

class MovieCriticApp {
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.checkModelStatus();

        // Settings
        this.wordsToGenerate = 30;
        this.temperature = 1.0;
    }

    initializeElements() {
        // UI Elements
        this.statusBar = document.getElementById('statusBar');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = this.statusIndicator.querySelector('.status-text');

        this.welcomeSection = document.getElementById('welcomeSection');
        this.messagesContainer = document.getElementById('messagesContainer');

        this.seedInput = document.getElementById('seedInput');
        this.generateBtn = document.getElementById('generateBtn');

        // Settings
        this.settingsToggle = document.getElementById('settingsToggle');
        this.settingsPanel = document.getElementById('settingsPanel');
        this.wordsSlider = document.getElementById('wordsSlider');
        this.wordsValue = document.getElementById('wordsValue');
        this.tempSlider = document.getElementById('tempSlider');
        this.tempValue = document.getElementById('tempValue');
    }

    attachEventListeners() {
        // Generate button
        this.generateBtn.addEventListener('click', () => this.handleGenerate());

        // Enter key in textarea (Shift+Enter for new line)
        this.seedInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleGenerate();
            }
        });

        // Auto-resize textarea
        this.seedInput.addEventListener('input', () => {
            this.seedInput.style.height = 'auto';
            this.seedInput.style.height = Math.min(this.seedInput.scrollHeight, 120) + 'px';
        });

        // Settings toggle
        this.settingsToggle.addEventListener('click', () => {
            this.settingsPanel.classList.toggle('active');
        });

        // Sliders
        this.wordsSlider.addEventListener('input', (e) => {
            this.wordsToGenerate = parseInt(e.target.value);
            this.wordsValue.textContent = this.wordsToGenerate;
        });

        this.tempSlider.addEventListener('input', (e) => {
            this.temperature = parseFloat(e.target.value);
            this.tempValue.textContent = this.temperature.toFixed(1);
        });
    }

    async checkModelStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();

            if (data.ready) {
                this.setStatus('ready', '✓ Model Ready');
            } else {
                this.setStatus('error', '⚠ Model Not Trained');
                this.showErrorMessage('Please train the model first by running: python movie_critic_chatbot.py');
            }
        } catch (error) {
            this.setStatus('error', '⚠ Server Error');
            console.error('Status check failed:', error);
        }
    }

    setStatus(type, text) {
        this.statusIndicator.className = 'status-indicator ' + type;
        this.statusText.textContent = text;
    }

    async handleGenerate() {
        const seedText = this.seedInput.value.trim();

        if (!seedText) {
            this.showNotification('Please enter a seed phrase', 'error');
            this.seedInput.focus();
            return;
        }

        // Disable input during generation
        this.setGenerating(true);

        // Add user message
        this.addMessage('user', seedText);

        // Add loading message
        const loadingId = this.addLoadingMessage();

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    seed_text: seedText,
                    next_words: this.wordsToGenerate,
                    temperature: this.temperature
                })
            });

            const data = await response.json();

            // Remove loading message
            this.removeLoadingMessage(loadingId);

            if (data.success) {
                // Add AI response
                this.addMessage('ai', data.generated_text);

                // Clear input
                this.seedInput.value = '';
                this.seedInput.style.height = 'auto';

                // Hide welcome section if visible
                if (!this.welcomeSection.classList.contains('hidden')) {
                    this.welcomeSection.classList.add('hidden');
                    this.messagesContainer.classList.add('active');
                }
            } else {
                this.showNotification(data.error || 'Generation failed', 'error');
            }
        } catch (error) {
            this.removeLoadingMessage(loadingId);
            this.showNotification('Network error. Please try again.', 'error');
            console.error('Generation failed:', error);
        } finally {
            this.setGenerating(false);
        }
    }

    setGenerating(isGenerating) {
        this.generateBtn.disabled = isGenerating;
        this.seedInput.disabled = isGenerating;

        if (isGenerating) {
            this.generateBtn.classList.add('loading');
            this.setStatus('loading', '⚡ Generating...');
        } else {
            this.generateBtn.classList.remove('loading');
            this.setStatus('ready', '✓ Model Ready');
        }
    }

    addMessage(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;

        const label = type === 'user' ? 'Your Input' : 'AI Generated';
        const icon = type === 'user' ? '👤' : '🤖';

        messageDiv.innerHTML = `
            <div class="message-label ${type}">
                <span class="message-label-icon">${icon}</span>
                ${label}
            </div>
            <div class="message-content">${this.escapeHtml(content)}</div>
        `;

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addLoadingMessage() {
        const loadingId = 'loading-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ai';
        messageDiv.id = loadingId;

        messageDiv.innerHTML = `
            <div class="message-label ai">
                <span class="message-label-icon">🤖</span>
                AI Generated
            </div>
            <div class="message-content loading">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
                <span>Generating review...</span>
            </div>
        `;

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        return loadingId;
    }

    removeLoadingMessage(loadingId) {
        const loadingMessage = document.getElementById(loadingId);
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    showNotification(message, type = 'info') {
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = 'toast toast-' + type;
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'error' ? '#f5576c' : '#4ade80'};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            z-index: 1000;
            animation: slideInRight 0.3s ease;
            max-width: 300px;
            font-size: 0.875rem;
        `;

        document.body.appendChild(toast);

        // Auto-remove after 3 seconds
        setTimeout(() => {
            toast.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    showErrorMessage(message) {
        this.addMessage('ai', `⚠️ Error: ${message}`);
        this.welcomeSection.classList.add('hidden');
        this.messagesContainer.classList.add('active');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.app = new MovieCriticApp();
    });
} else {
    window.app = new MovieCriticApp();
}
