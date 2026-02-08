/**
 * Nokia SLM Assistant - Frontend JavaScript
 */

class NokiaSLMApp {
    constructor() {
        // DOM Elements
        this.chatContainer = document.getElementById('chat-container');
        this.queryInput = document.getElementById('query-input');
        this.sendBtn = document.getElementById('send-btn');
        this.charCount = document.getElementById('char-count');
        this.modelInfo = document.getElementById('model-info');
        this.statusIndicator = document.getElementById('status-indicator');
        this.statusText = this.statusIndicator.querySelector('.status-text');
        this.settingsBtn = document.getElementById('settings-btn');
        this.statusModal = document.getElementById('status-modal');
        this.modalClose = document.getElementById('modal-close');
        
        // State
        this.isLoading = false;
        this.welcomeVisible = true;
        
        // Initialize
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.checkStatus();
        this.autoResizeTextarea();
        
        // Periodic status check
        setInterval(() => this.checkStatus(), 30000);
    }
    
    bindEvents() {
        // Send button
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Enter key (shift+enter for new line)
        this.queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Character count
        this.queryInput.addEventListener('input', () => {
            this.updateCharCount();
            this.autoResizeTextarea();
        });
        
        // Quick actions
        document.querySelectorAll('.quick-action').forEach(btn => {
            btn.addEventListener('click', () => {
                const query = btn.dataset.query;
                this.queryInput.value = query;
                this.updateCharCount();
                this.sendMessage();
            });
        });
        
        // Settings modal
        this.settingsBtn.addEventListener('click', () => this.openModal());
        this.modalClose.addEventListener('click', () => this.closeModal());
        this.statusModal.querySelector('.modal-backdrop').addEventListener('click', () => this.closeModal());
        
        // Escape key closes modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') this.closeModal();
        });
    }
    
    updateCharCount() {
        const count = this.queryInput.value.length;
        this.charCount.textContent = `${count} / 10000`;
    }
    
    autoResizeTextarea() {
        this.queryInput.style.height = 'auto';
        this.queryInput.style.height = Math.min(this.queryInput.scrollHeight, 200) + 'px';
    }
    
    async checkStatus() {
        try {
            const response = await fetch('/api/status');
            if (response.ok) {
                const data = await response.json();
                this.updateStatus(data, true);
            } else {
                this.updateStatus(null, false);
            }
        } catch (error) {
            console.error('Status check failed:', error);
            this.updateStatus(null, false);
        }
    }
    
    updateStatus(data, connected) {
        if (connected) {
            this.statusIndicator.className = 'status-indicator connected';
            this.statusText.textContent = 'Connected';
            this.modelInfo.textContent = `Model: ${data.model}`;
            
            // Update modal stats
            document.getElementById('stat-vectors').textContent = data.vectors?.toLocaleString() || '-';
            document.getElementById('stat-chunks').textContent = data.chunks?.toLocaleString() || '-';
            document.getElementById('stat-model').textContent = data.model || '-';
            document.getElementById('stat-bm25').textContent = data.bm25_enabled ? 'Enabled' : 'Disabled';
        } else {
            this.statusIndicator.className = 'status-indicator error';
            this.statusText.textContent = 'Disconnected';
            this.modelInfo.textContent = 'Offline';
        }
    }
    
    openModal() {
        this.statusModal.classList.add('active');
        this.checkStatus(); // Refresh stats
    }
    
    closeModal() {
        this.statusModal.classList.remove('active');
    }
    
    clearWelcome() {
        if (this.welcomeVisible) {
            const welcome = this.chatContainer.querySelector('.welcome-message');
            if (welcome) {
                welcome.remove();
            }
            this.welcomeVisible = false;
        }
    }
    
    addMessage(type, content, meta = {}) {
        this.clearWelcome();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        if (meta.isRefusal) {
            messageDiv.classList.add('refusal');
        }
        
        const avatar = type === 'user' ? '👤' : '🤖';
        
        let bubbleContent = '';
        
        if (meta.isRefusal) {
            bubbleContent = `
                <div class="refusal-indicator">
                    <span>⚠️</span>
                    <span>Documentation Not Found</span>
                </div>
                <div>${this.formatContent(content)}</div>
            `;
        } else if (meta.isLoading) {
            bubbleContent = `
                <div class="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span>Thinking...</span>
            `;
            messageDiv.classList.add('loading');
        } else {
            bubbleContent = this.formatContent(content);
        }
        
        let metaHTML = '';
        if (meta.queryType) {
            metaHTML += `<span class="query-type-badge">${meta.queryType}</span>`;
        }
        if (meta.latency) {
            metaHTML += `<span>${meta.latency}ms</span>`;
        }
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-bubble">${bubbleContent}</div>
                ${metaHTML ? `<div class="message-meta">${metaHTML}</div>` : ''}
            </div>
        `;
        
        this.chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    }
    
    formatContent(content) {
        if (!content) return '';
        
        // Basic markdown-like formatting
        let formatted = content
            // Code blocks
            .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
            // Inline code
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Bold
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            // Italic
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            // Headers
            .replace(/^### (.+)$/gm, '<h4>$1</h4>')
            .replace(/^## (.+)$/gm, '<h3>$1</h3>')
            // Lists
            .replace(/^- (.+)$/gm, '• $1')
            .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
            // Paragraphs
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');
        
        return `<p>${formatted}</p>`;
    }
    
    scrollToBottom() {
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }
    
    removeLoadingMessage() {
        const loading = this.chatContainer.querySelector('.message.loading');
        if (loading) {
            loading.remove();
        }
    }
    
    async sendMessage() {
        const query = this.queryInput.value.trim();
        
        if (!query || this.isLoading) return;
        
        this.isLoading = true;
        this.sendBtn.disabled = true;
        this.queryInput.value = '';
        this.updateCharCount();
        this.autoResizeTextarea();
        
        // Add user message
        this.addMessage('user', query);
        
        // Add loading message
        this.addMessage('assistant', '', { isLoading: true });
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query })
            });
            
            this.removeLoadingMessage();
            
            if (response.ok) {
                const data = await response.json();
                
                const isRefusal = !data.grounded || 
                    data.response.includes('INSUFFICIENT DOCUMENTATION CONTEXT');
                
                this.addMessage('assistant', data.response, {
                    queryType: data.query_type,
                    latency: data.latency_ms,
                    isRefusal: isRefusal
                });
            } else {
                const error = await response.json();
                this.addMessage('assistant', `Error: ${error.error || 'Request failed'}`, {
                    isRefusal: true
                });
            }
        } catch (error) {
            this.removeLoadingMessage();
            this.addMessage('assistant', `Network error: ${error.message}`, {
                isRefusal: true
            });
        } finally {
            this.isLoading = false;
            this.sendBtn.disabled = false;
            this.queryInput.focus();
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new NokiaSLMApp();
});
