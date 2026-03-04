/**
 * Reddit RAG Chatbot - Frontend Application
 * Modern JavaScript for API integration
 * Supports session persistence, source display, and cache indicators
 */

// API Configuration - always use relative URL (works with both dev and Docker/Nginx)
const API_BASE_URL = '/api/v1';

// State
let conversationHistory = [];
let isLoading = false;
let sessionId = null; // Managed by the backend, persists across messages

// DOM Elements
const messagesContainer = document.getElementById('messages');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const useLlmToggle = document.getElementById('use-llm');
const nSourcesSlider = document.getElementById('n-sources');
const sourcesValue = document.getElementById('sources-value');
const statusElement = document.getElementById('status');
const loadingOverlay = document.getElementById('loading');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    loadStats();

    messageInput.addEventListener('input', autoResizeTextarea);

    nSourcesSlider.addEventListener('input', () => {
        sourcesValue.textContent = nSourcesSlider.value;
    });
}

function setupEventListeners() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const tab = item.dataset.tab;
            switchTab(tab);
        });
    });
}

// ============================================
// Tab Navigation
// ============================================

function switchTab(tabName) {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.tab === tabName);
    });

    document.getElementById('chat-tab').classList.toggle('hidden', tabName !== 'chat');
    document.getElementById('search-tab').classList.toggle('hidden', tabName !== 'search');
    document.getElementById('stats-tab').classList.toggle('hidden', tabName !== 'stats');

    if (tabName === 'stats') {
        loadStats();
    }
}

// ============================================
// Chat Functions
// ============================================

async function sendMessage() {
    const message = messageInput.value.trim();

    if (!message || isLoading) return;

    messageInput.value = '';
    autoResizeTextarea();

    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }

    addMessageToUI('user', message);

    setLoading(true);
    setStatus('Thinking...');

    try {
        const requestBody = {
            message: message,
            use_llm: useLlmToggle.checked,
            n_results: parseInt(nSourcesSlider.value),
        };

        // Include session_id if we have one from a previous response
        if (sessionId) {
            requestBody.session_id = sessionId;
        }

        const response = await fetch(`${API_BASE_URL}/chat/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Persist session_id from the response for future messages
        if (data.metadata && data.metadata.session_id) {
            sessionId = data.metadata.session_id;
        }

        // Add bot response with sources and metadata
        addMessageToUI('assistant', data.message, data.sources, data.metadata);

        conversationHistory.push(
            { role: 'user', content: message },
            { role: 'assistant', content: data.message }
        );

        setStatus('');
    } catch (error) {
        console.error('Chat error:', error);
        addMessageToUI('assistant', `Sorry, an error occurred: ${error.message}`);
        setStatus('Error');
    } finally {
        setLoading(false);
    }
}

function addMessageToUI(role, content, sources, metadata) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = role === 'user' ? '&#128100;' : '&#129302;';
    const time = new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });

    let metaBadges = '';
    if (metadata) {
        const badges = [];
        if (metadata.cache_hit) {
            badges.push('<span class="meta-badge cache-hit">cache</span>');
        }
        if (metadata.reranked) {
            badges.push('<span class="meta-badge reranked">reranked</span>');
        }
        if (metadata.duration_ms) {
            badges.push(`<span class="meta-badge duration">${Math.round(metadata.duration_ms)}ms</span>`);
        }
        if (metadata.method) {
            badges.push(`<span class="meta-badge method">${metadata.method}</span>`);
        }
        metaBadges = `<div class="message-meta">${badges.join(' ')}</div>`;
    }

    let sourcesHTML = '';
    if (sources && sources.length > 0) {
        const sourceItems = sources.map((source, idx) => {
            const conv = source.conversation || source;
            const score = source.score ? (source.score * 100).toFixed(0) : '?';
            const context = conv.context || '';
            const truncated = context.length > 120 ? context.substring(0, 120) + '...' : context;
            return `<div class="source-item">
                <span class="source-rank">#${idx + 1}</span>
                <span class="source-score">${score}%</span>
                <span class="source-text">${formatMessage(truncated)}</span>
            </div>`;
        }).join('');

        sourcesHTML = `
            <details class="sources-details">
                <summary>Sources (${sources.length})</summary>
                <div class="sources-list">${sourceItems}</div>
            </details>
        `;
    }

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-bubble">
            <div class="message-content">${formatMessage(content)}</div>
            ${sourcesHTML}
            ${metaBadges}
            <div class="message-time">${time}</div>
        </div>
    `;

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function formatMessage(content) {
    return content
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\n/g, '<br>');
}

function setExample(text) {
    messageInput.value = text;
    messageInput.focus();
    autoResizeTextarea();
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + 'px';
}

// ============================================
// Search Functions
// ============================================

async function searchConversations() {
    const query = document.getElementById('search-input').value.trim();

    if (!query) return;

    const resultsContainer = document.getElementById('search-results');
    resultsContainer.innerHTML = '<p class="search-placeholder">Searching...</p>';

    try {
        const response = await fetch(`${API_BASE_URL}/chat/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: query,
                use_llm: false,
                n_results: 10
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.sources && data.sources.length > 0) {
            resultsContainer.innerHTML = data.sources.map((source, index) => {
                const conv = source.conversation || source;
                return `
                <div class="search-result-card">
                    <div class="result-header">
                        <span class="result-number">Result #${index + 1}</span>
                        <span class="result-score">${(source.score * 100).toFixed(1)}% match</span>
                        ${data.metadata && data.metadata.reranked ? '<span class="meta-badge reranked">reranked</span>' : ''}
                    </div>
                    <div class="result-context">
                        <strong>Context:</strong> ${conv.context || 'N/A'}
                    </div>
                    <div class="result-response">
                        <strong>Response:</strong> ${conv.response || data.message}
                    </div>
                </div>
            `}).join('');
        } else {
            resultsContainer.innerHTML = `
                <div class="search-result-card">
                    <div class="result-response">${data.message}</div>
                </div>
            `;
        }
    } catch (error) {
        console.error('Search error:', error);
        resultsContainer.innerHTML = `<p class="search-placeholder">Error: ${error.message}</p>`;
    }
}

// ============================================
// Stats Functions
// ============================================

async function loadStats() {
    const statsGrid = document.getElementById('stats-grid');

    try {
        const response = await fetch(`${API_BASE_URL}/chat/stats`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const stats = await response.json();

        // Build stats cards including cache and memory info
        const cards = [];

        // Main stats
        const mainKeys = ['total_conversations', 'embedding_model', 'llm_model', 'llm_available', 'reranker_enabled', 'reranker_model', 'cache_enabled'];
        for (const key of mainKeys) {
            if (stats[key] !== undefined && stats[key] !== null) {
                cards.push(buildStatCard(key, stats[key]));
            }
        }

        // Cache stats
        if (stats.cache_stats) {
            const cs = stats.cache_stats;
            cards.push(buildStatCard('cache_hit_rate', cs.hit_rate || '0.00%'));
            cards.push(buildStatCard('cache_hits', cs.hits || 0));
            cards.push(buildStatCard('cache_misses', cs.misses || 0));
        }

        // Memory stats
        if (stats.memory_stats) {
            const ms = stats.memory_stats;
            cards.push(buildStatCard('active_sessions', ms.active_sessions || 0));
            cards.push(buildStatCard('total_messages', ms.total_messages || 0));
        }

        statsGrid.innerHTML = cards.join('');

        if (stats.total_conversations) {
            document.getElementById('total-conversations').textContent =
                stats.total_conversations.toLocaleString();
        }
    } catch (error) {
        console.error('Stats error:', error);
        statsGrid.innerHTML = `<p>Error loading statistics: ${error.message}</p>`;
    }
}

function buildStatCard(key, value) {
    const icon = getStatIcon(key);
    const label = formatStatLabel(key);
    const formattedValue = formatStatValue(key, value);

    return `
        <div class="stat-card">
            <div class="stat-card-icon">${icon}</div>
            <div class="stat-card-value">${formattedValue}</div>
            <div class="stat-card-label">${label}</div>
        </div>
    `;
}

function getStatIcon(key) {
    const icons = {
        total_conversations: '&#128172;',
        embedding_model: '&#129504;',
        vector_store: '&#128230;',
        llm_provider: '&#9889;',
        llm_model: '&#129302;',
        llm_available: '&#9989;',
        reranker_enabled: '&#128269;',
        reranker_model: '&#127919;',
        cache_enabled: '&#128190;',
        cache_hit_rate: '&#127919;',
        cache_hits: '&#9989;',
        cache_misses: '&#10060;',
        active_sessions: '&#128101;',
        total_messages: '&#128172;',
        default_n_results: '&#128202;'
    };
    return icons[key] || '&#128200;';
}

function formatStatLabel(key) {
    return key
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
}

function formatStatValue(key, value) {
    if (typeof value === 'number') {
        return value.toLocaleString();
    }
    if (typeof value === 'boolean') {
        return value ? 'Yes' : 'No';
    }
    return value;
}

// ============================================
// Utility Functions
// ============================================

function setLoading(loading) {
    isLoading = loading;
    sendBtn.disabled = loading;
    loadingOverlay.classList.toggle('hidden', !loading);
}

function setStatus(status) {
    statusElement.textContent = status;
}

// ============================================
// Clear Chat
// ============================================

function clearChat() {
    conversationHistory = [];
    sessionId = null; // Reset session on clear
    messagesContainer.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon"></div>
            <h2>Welcome to Reddit RAG Chatbot!</h2>
            <p>Ask me anything in French or English. I'll find relevant conversations from Reddit to help you.</p>
            <div class="example-questions">
                <h4>Try asking:</h4>
                <div class="examples-grid">
                    <button class="example-btn" onclick="setExample('Quel t\u00e9l\u00e9phone me recommandes-tu ?')">
                        Quel t\u00e9l\u00e9phone acheter ?
                    </button>
                    <button class="example-btn" onclick="setExample('Je me sens triste aujourd\\'hui')">
                        Je me sens triste
                    </button>
                    <button class="example-btn" onclick="setExample('How do I make friends?')">
                        How to make friends?
                    </button>
                    <button class="example-btn" onclick="setExample('What are the best video games?')">
                        Best video games?
                    </button>
                </div>
            </div>
        </div>
    `;
}

// Expose functions to global scope for HTML onclick handlers
window.setExample = setExample;
window.handleKeyDown = handleKeyDown;
window.sendMessage = sendMessage;
window.searchConversations = searchConversations;
window.clearChat = clearChat;
