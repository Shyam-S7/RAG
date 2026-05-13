/**
 * RAG Assistant - Frontend Logic
 * Production-ready vanilla JS implementation
 */

class ChatStore {
    constructor() {
        this.storageKey = 'rag_assistant_history';
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
                chat.title = content.substring(0, 30) + (content.length > 30 ? '...' : '');
            }
            this.save();
        }
    }

    deleteChat(id) {
        this.chats = this.chats.filter(c => c.id !== id);
        if (this.currentChatId === id) this.currentChatId = null;
        this.save();
    }
}

class ChatUI {
    constructor(store) {
        this.store = store;
        this.elements = {
            sidebar: document.getElementById('sidebar'),
            historyContainer: document.getElementById('chat-history'),
            chatContainer: document.getElementById('chat-container'),
            welcomeScreen: document.getElementById('welcome-screen'),
            userInput: document.getElementById('user-input'),
            sendBtn: document.getElementById('send-btn'),
            newChatBtn: document.getElementById('new-chat-btn'),
            mobileMenuBtn: document.getElementById('mobile-menu-btn'),
            msgTemplate: document.getElementById('message-template')
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
    }

    autoResizeTextarea() {
        const textarea = this.elements.userInput;
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
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
        }
    }

    switchChat(id) {
        this.store.currentChatId = id;
        const chat = this.store.getChat(id);
        
        // Update Sidebar active state
        document.querySelectorAll('.history-item').forEach(item => {
            item.classList.toggle('active', item.dataset.id === id);
        });

        this.renderMessages(chat.messages);
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
            `;
            item.addEventListener('click', () => this.switchChat(chat.id));
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
        
        const textDiv = clone.querySelector('.message-text');
        textDiv.textContent = content;

        if (sources && sources.length > 0) {
            const sourcesContainer = clone.querySelector('.sources-container');
            sourcesContainer.classList.remove('hidden');
            
            const toggle = clone.querySelector('.sources-toggle');
            toggle.textContent = `Sources (${sources.length})`;
            
            const list = clone.querySelector('.sources-list');
            sources.forEach(src => {
                const srcItem = document.createElement('div');
                srcItem.className = 'source-item';
                srcItem.innerHTML = `
                    <div class="source-meta">
                        <span class="source-domain">${src.domain}</span>
                        <span class="source-file">${src.source}</span>
                    </div>
                    <div class="source-text">${src.content.substring(0, 150)}...</div>
                `;
                list.appendChild(srcItem);
            });

            toggle.addEventListener('click', () => list.classList.toggle('hidden'));
        }

        this.elements.chatContainer.appendChild(clone);
        this.scrollToBottom();
    }

    showThinkingIndicator() {
        const div = document.createElement('div');
        div.className = 'message bot thinking-indicator';
        div.innerHTML = `
            <div class="message-content">
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

        // Ensure we have a chat started
        if (!this.store.currentChatId) {
            this.startNewChat();
        }

        const chatId = this.store.currentChatId;

        // Clear input
        this.elements.userInput.value = '';
        this.elements.userInput.style.height = 'auto';
        this.elements.sendBtn.disabled = true;
        this.elements.welcomeScreen.classList.add('hidden');

        // Add user message
        this.store.addMessage(chatId, 'user', query);
        this.appendMessageToUI('user', query);
        this.renderHistory();

        // Show thinking
        const thinkingIndicator = this.showThinkingIndicator();

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query: query,
                    session_id: chatId 
                })
            });

            if (!response.ok) throw new Error('API request failed');

            const data = await response.json();
            
            // Remove thinking
            thinkingIndicator.remove();

            // Add bot message
            this.store.addMessage(chatId, 'bot', data.answer, data.sources);
            this.appendMessageToUI('bot', data.answer, data.sources);

        } catch (error) {
            if (thinkingIndicator) thinkingIndicator.remove();
            this.appendMessageToUI('bot', 'Sorry, I encountered an error connecting to the RAG server. Please ensure the backend is running.');
            console.error('Chat Error:', error);
        }
    }
}

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    const store = new ChatStore();
    const ui = new ChatUI(store);
});
