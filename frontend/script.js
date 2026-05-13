/**
 * AI Assistant - Frontend Logic
 * ChatGPT-style RAG implementation
 */

class ChatStore {
    constructor() {
        this.storageKey = 'rag_assistant_history_v2';
        this.chats = JSON.parse(localStorage.getItem(this.storageKey)) || [];
        this.currentChatId = null;
    }

    save() {
        localStorage.setItem(this.storageKey, JSON.stringify(this.chats));
    }

    createChat() {
        const newChat = {
            id: Date.now().toString(),
            title: 'New Conversation',
            messages: [],
            timestamp: new Date().toISOString()
        };
        this.chats.unshift(newChat);
        this.currentChatId = newChat.id;
        this.save();
        return newChat;
    }

    getChat(id) {
        return this.chats.find(c => c.id === id);
    }

    addMessage(chatId, role, content, sources = []) {
        const chat = this.getChat(chatId);
        if (chat) {
            chat.messages.push({ role, content, sources, timestamp: new Date().toISOString() });
            // Update title if it's the first user message
            if (role === 'user' && chat.messages.filter(m => m.role === 'user').length === 1) {
                chat.title = content.substring(0, 35) + (content.length > 35 ? '...' : '');
            }
            this.save();
        }
    }

    updateChatId(oldId, newId) {
        const chat = this.getChat(oldId);
        if (chat) {
            chat.id = newId;
            if (this.currentChatId === oldId) {
                this.currentChatId = newId;
            }
            this.save();
            return true;
        }
        return false;
    }

    deleteChat(id) {
        this.chats = this.chats.filter(c => c.id !== id);
        if (this.currentChatId === id) {
            this.currentChatId = this.chats.length > 0 ? this.chats[0].id : null;
        }
        this.save();
    }
}

class ChatUI {
    constructor(store) {
        this.store = store;
        this.selectedFile = null;
        this.isProcessing = false;
        
        this.elements = {
            sidebar: document.getElementById('sidebar'),
            historyContainer: document.getElementById('chat-history'),
            chatContainer: document.getElementById('chat-container'),
            welcomeScreen: document.getElementById('welcome-screen'),
            userInput: document.getElementById('user-input'),
            sendBtn: document.getElementById('send-btn'),
            newChatBtn: document.getElementById('new-chat-btn'),
            mobileMenuBtn: document.getElementById('mobile-menu-btn'),
            msgTemplate: document.getElementById('message-template'),
            uploadBtn: document.getElementById('upload-btn'),
            fileInput: document.getElementById('file-upload'),
            uploadStatus: document.getElementById('upload-status'),
            fileNameDisplay: document.getElementById('file-name-display'),
            removeFileBtn: document.getElementById('remove-file-btn'),
            processBtn: document.getElementById('process-btn'),
            dragOverlay: document.getElementById('drag-overlay')
        };

        this.init();
    }

    init() {
        this.bindEvents();
        this.renderHistory();
        this.loadLastChat();
    }

    bindEvents() {
        this.elements.newChatBtn.addEventListener('click', () => this.startNewChat());
        
        this.elements.userInput.addEventListener('input', () => {
            this.elements.sendBtn.disabled = !this.elements.userInput.value.trim();
            this.autoResizeTextarea();
        });

        this.elements.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSend();
            }
        });

        this.elements.sendBtn.addEventListener('click', () => this.handleSend());
        
        this.elements.mobileMenuBtn.addEventListener('click', () => {
            this.elements.sidebar.classList.toggle('open');
        });

        // Upload Events
        this.elements.uploadBtn.addEventListener('click', () => this.elements.fileInput.click());
        
        this.elements.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });

        this.elements.removeFileBtn.addEventListener('click', () => this.clearFile());
        this.elements.processBtn.addEventListener('click', () => this.handleProcess());

        // Drag and Drop
        window.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.dragOverlay.classList.add('active');
        });

        window.addEventListener('dragleave', (e) => {
            e.preventDefault();
            if (e.relatedTarget === null) {
                this.elements.dragOverlay.classList.remove('active');
            }
        });

        window.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.dragOverlay.classList.remove('active');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });

        // Suggestions
        document.querySelectorAll('.suggestion-card').forEach(card => {
            card.addEventListener('click', () => {
                this.elements.userInput.value = card.textContent;
                this.elements.userInput.focus();
                this.elements.sendBtn.disabled = false;
                this.autoResizeTextarea();
            });
        });
    }

    handleFileSelect(file) {
        // Validate file type
        const allowedTypes = ['.pdf', '.txt', '.md'];
        const extension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        
        if (!allowedTypes.includes(extension)) {
            alert('Invalid file type. Please upload PDF, TXT, or MD files.');
            return;
        }

        this.selectedFile = file;
        this.elements.fileNameDisplay.textContent = file.name;
        this.elements.uploadStatus.classList.remove('hidden');
        this.elements.processBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
            Process
        `;
        this.elements.processBtn.classList.remove('loading');
        this.elements.processBtn.disabled = false;
    }

    async handleProcess() {
        if (!this.selectedFile || this.isProcessing) return;

        this.isProcessing = true;
        this.elements.processBtn.disabled = true;
        this.elements.processBtn.classList.add('loading');
        this.elements.processBtn.innerHTML = '<span class="loader"></span> Ingesting...';

        try {
            const formData = new FormData();
            formData.append('file', this.selectedFile);

            const response = await fetch('http://localhost:8000/api/ingest/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Ingestion failed');

            const data = await response.json();
            const backendSessionId = data.session_id;

            // Adopt the backend's session_id for this conversation
            const currentId = this.store.currentChatId;
            this.store.updateChatId(currentId, backendSessionId);
            this.renderHistory();
            
            this.elements.processBtn.innerHTML = '✅ Done';
            setTimeout(() => {
                this.clearFile();
                this.appendMessageToUI('bot', `Successfully ingested "${this.selectedFile.name}". You can now ask questions about its content in this isolated session.`);
            }, 1000);

        } catch (error) {
            console.error(error);
            alert('Processing failed: ' + error.message);
            this.elements.processBtn.innerHTML = '❌ Failed';
            this.elements.processBtn.disabled = false;
        } finally {
            this.isProcessing = false;
            this.elements.processBtn.classList.remove('loading');
        }
    }

    clearFile() {
        this.selectedFile = null;
        this.elements.fileInput.value = '';
        this.elements.uploadStatus.classList.add('hidden');
    }

    autoResizeTextarea() {
        const textarea = this.elements.userInput;
        textarea.style.height = 'auto';
        textarea.style.height = (textarea.scrollHeight) + 'px';
    }

    startNewChat() {
        const chat = this.store.createChat();
        this.renderHistory();
        this.switchChat(chat.id);
        if (window.innerWidth <= 768) this.elements.sidebar.classList.remove('open');
    }

    loadLastChat() {
        if (this.store.chats.length > 0) {
            this.switchChat(this.store.chats[0].id);
        } else {
            this.startNewChat();
        }
    }

    switchChat(id) {
        if (!id) {
            this.startNewChat();
            return;
        }
        this.store.currentChatId = id;
        const chat = this.store.getChat(id);
        
        document.querySelectorAll('.history-item').forEach(item => {
            item.classList.toggle('active', item.dataset.id === id);
        });

        this.renderMessages(chat.messages);
    }

    async handleDeleteChat(id, event) {
        event.stopPropagation();
        if (!confirm('Are you sure you want to delete this conversation and its associated data?')) return;

        try {
            // 1. Delete from Backend
            const res = await fetch(`http://localhost:8000/api/session/${id}`, {
                method: 'DELETE'
            });
            
            if (!res.ok) {
                console.warn('Backend session deletion failed or not found. Removing locally anyway.');
            }

            // 2. Delete from Store
            this.store.deleteChat(id);
            
            // 3. UI Update
            this.renderHistory();
            this.switchChat(this.store.currentChatId);

        } catch (err) {
            console.error('Deletion error:', err);
            // Fallback: Delete locally anyway if backend fails
            this.store.deleteChat(id);
            this.renderHistory();
            this.switchChat(this.store.currentChatId);
        }
    }

    renderHistory() {
        this.elements.historyContainer.innerHTML = '';
        this.store.chats.forEach(chat => {
            const item = document.createElement('div');
            item.className = `history-item ${this.store.currentChatId === chat.id ? 'active' : ''}`;
            item.dataset.id = chat.id;
            item.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
                <span>${chat.title}</span>
                <button class="delete-chat-btn" title="Delete conversation">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                </button>
            `;
            item.addEventListener('click', () => this.switchChat(chat.id));
            
            const delBtn = item.querySelector('.delete-chat-btn');
            delBtn.addEventListener('click', (e) => this.handleDeleteChat(chat.id, e));
            
            this.elements.historyContainer.appendChild(item);
        });
    }

    renderMessages(messages) {
        this.elements.chatContainer.innerHTML = '';
        
        if (messages.length === 0) {
            this.elements.welcomeScreen.classList.remove('hidden');
            this.elements.chatContainer.appendChild(this.elements.welcomeScreen);
            return;
        }

        this.elements.welcomeScreen.classList.add('hidden');
        messages.forEach(msg => this.appendMessageToUI(msg.role, msg.content, msg.sources));
        this.scrollToBottom();
    }

    appendMessageToUI(role, content, sources = []) {
        const clone = this.elements.msgTemplate.content.cloneNode(true);
        const msgDiv = clone.querySelector('.message');
        msgDiv.classList.add(role);
        
        const authorSpan = clone.querySelector('.msg-author');
        authorSpan.textContent = role === 'user' ? 'You' : 'RAG Assistant';

        const textDiv = clone.querySelector('.message-text');
        textDiv.textContent = content;

        if (sources && sources.length > 0) {
            const container = clone.querySelector('.sources-container');
            container.classList.remove('hidden');
            
            const toggle = clone.querySelector('.sources-toggle');
            toggle.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m18 15-6-6-6 6"/></svg>
                Sources Used (${sources.length})
            `;
            
            const list = clone.querySelector('.sources-list');
            sources.forEach(src => {
                const item = document.createElement('div');
                item.className = 'source-item';
                item.innerHTML = `
                    <div class="source-meta">
                        <span class="source-domain">${src.domain || 'Doc'}</span>
                        <span class="source-file">${src.source}</span>
                    </div>
                    <div class="source-text">${src.content}</div>
                `;
                list.appendChild(item);
            });

            toggle.addEventListener('click', () => {
                list.classList.toggle('hidden');
                toggle.querySelector('svg').style.transform = list.classList.contains('hidden') ? '' : 'rotate(180deg)';
            });
        }

        this.elements.chatContainer.appendChild(clone);
        this.scrollToBottom();
    }

    showThinkingIndicator() {
        const div = document.createElement('div');
        div.className = 'message bot thinking-indicator';
        div.innerHTML = `
            <div class="message-content">
                <div class="message-header">
                    <div class="msg-avatar"></div>
                    <span class="msg-author">RAG Assistant</span>
                </div>
                <div class="thinking">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        `;
        this.elements.chatContainer.appendChild(div);
        this.scrollToBottom();
        return div;
    }

    scrollToBottom() {
        this.elements.chatContainer.scrollTop = this.elements.chatContainer.scrollHeight;
    }

    async handleSend() {
        const query = this.elements.userInput.value.trim();
        if (!query) return;

        if (!this.store.currentChatId) this.startNewChat();
        const chatId = this.store.currentChatId;

        // Reset UI
        this.elements.userInput.value = '';
        this.elements.userInput.style.height = 'auto';
        this.elements.sendBtn.disabled = true;
        this.elements.welcomeScreen.classList.add('hidden');

        // Add user message to UI
        this.store.addMessage(chatId, 'user', query);
        this.appendMessageToUI('user', query);
        this.renderHistory();

        const thinkingIndicator = this.showThinkingIndicator();

        try {
            const response = await fetch('http://localhost:8000/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, session_id: chatId })
            });

            if (!response.ok) throw new Error('Chat API failed');
            const data = await response.json();
            
            thinkingIndicator.remove();
            this.store.addMessage(chatId, 'bot', data.answer, data.sources);
            this.appendMessageToUI('bot', data.answer, data.sources);

        } catch (error) {
            thinkingIndicator.remove();
            this.appendMessageToUI('bot', `Error: ${error.message}. Please ensure the backend is running at http://localhost:8000`);
            console.error(error);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const store = new ChatStore();
    const ui = new ChatUI(store);
});
