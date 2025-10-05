/**
 * Fractalic Chat Client - Main class that coordinates all modules
 */

import { FileBrowser } from './file-browser.js';
import { UIRenderer } from './ui-rendering.js';
import { StreamClient } from './stream-client.js';
import { formatTime } from './utils.js';

export class FractalicChatClient {
    constructor() {
        // DOM Elements
        this.messagesContainer = document.getElementById('messages');
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        this.connectionStatus = document.getElementById('connectionStatus');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.fileBrowserModal = document.getElementById('fileBrowserModal');
        this.fileList = document.getElementById('fileList');
        this.currentPathDisplay = document.getElementById('currentPath');
        this.selectedFileDisplay = document.getElementById('selectedFile');
        this.clearSelectionButton = document.getElementById('clearSelection');
        this.diffViewerModal = document.getElementById('diffViewerModal');
        this.diffContent = document.getElementById('diffContent');
        this.terminalViewerModal = document.getElementById('terminalViewerModal');
        this.terminalContent = document.getElementById('terminalContent');
        this.terminalFileName = document.getElementById('terminalFileName');
        this.terminalStatus = document.getElementById('terminalStatus');

        // Additional modal buttons
        this.selectFileButton = document.getElementById('selectFileBtn');
        this.modalCloseButton = document.getElementById('modalCloseBtn');
        this.cancelButton = document.getElementById('cancelBtn');
        this.diffModalCloseButton = document.getElementById('diffModalCloseBtn');
        this.closeDiffButton = document.getElementById('closeDiffBtn');
        this.terminalModalCloseButton = document.getElementById('terminalModalCloseBtn');
        this.closeTerminalButton = document.getElementById('closeTerminalBtn');

        // State
        this.selectedFile = null;
        this.conversation = [];
        this.executionBubbles = new Map();
        this.activeStreams = new Map();
        this.tokenStats = null;

        // Initialize modules
        this.fileBrowser = new FileBrowser(this);
        this.uiRenderer = new UIRenderer(this);
        this.streamClient = new StreamClient(this);

        // Setup event listeners
        this.initializeEventListeners();

        // Set connection status (HTTP streaming mode, no persistent WebSocket)
        this.connectionStatus.textContent = '🟢 Готово (HTTP Stream)';
        this.connectionStatus.className = 'status-connected';
    }

    initializeEventListeners() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // History toggle button
        const historyToggleBtn = document.getElementById('historyToggle');
        if (historyToggleBtn) {
            historyToggleBtn.addEventListener('click', () => {
                historyToggleBtn.classList.toggle('active');
            });
        }

        // Attach button (file selector)
        const attachBtn = document.getElementById('fileSelectBtn');
        if (attachBtn) {
            attachBtn.addEventListener('click', () => this.fileBrowser.openFileBrowser());
        }

        // File browser events
        this.selectFileButton.addEventListener('click', () => this.fileBrowser.openFileBrowser());
        this.clearSelectionButton.addEventListener('click', () => this.fileBrowser.clearFileSelection());
        this.modalCloseButton.addEventListener('click', () => this.fileBrowser.closeFileBrowser());
        this.cancelButton.addEventListener('click', () => this.fileBrowser.closeFileBrowser());
        
        // Diff viewer events
        this.diffModalCloseButton.addEventListener('click', () => this.fileBrowser.closeDiffViewer());
        this.closeDiffButton.addEventListener('click', () => this.fileBrowser.closeDiffViewer());
        
        // Terminal viewer events
        this.terminalModalCloseButton.addEventListener('click', () => this.fileBrowser.closeTerminalViewer());
        this.closeTerminalButton.addEventListener('click', () => this.fileBrowser.closeTerminalViewer());
        
        // Close modals when clicking outside
        this.fileBrowserModal.addEventListener('click', (e) => {
            if (e.target === this.fileBrowserModal) {
                this.fileBrowser.closeFileBrowser();
            }
        });
        
        this.diffViewerModal.addEventListener('click', (e) => {
            if (e.target === this.diffViewerModal) {
                this.fileBrowser.closeDiffViewer();
            }
        });
        
        this.terminalViewerModal.addEventListener('click', (e) => {
            if (e.target === this.terminalViewerModal) {
                this.fileBrowser.closeTerminalViewer();
            }
        });
    }

    async sendMessage() {
        const text = this.chatInput.value.trim();
        if (!text) return;
        if (!this.selectedFile) {
            alert('Сначала выберите markdown файл');
            return;
        }
        const timestamp = new Date().toISOString();
        // Don't store current message in history yet - it will be added after fractalic confirms
        this.uiRenderer.addMessage('user', text, formatTime(timestamp), { storeHistory: false });
        this.chatInput.value = '';
        this.uiRenderer.showTyping(true);

        // Check if history mode is enabled
        const historyToggle = document.getElementById('historyToggle');
        const useHistory = historyToggle ? historyToggle.classList.contains('active') : true;

        const history = useHistory ? this.conversation.slice(-50) : [];
        console.log('[DEBUG] Sending message with history:', {
            useHistory,
            historyLength: history.length,
            currentMessage: text
        });
        console.log('[DEBUG] Full history:', JSON.stringify(history, null, 2));

        // Format history as markdown with heading blocks
        let historyMarkdown = '';
        if (history.length > 0) {
            const markdownParts = [];
            let userMsgCount = 0;
            let assistantMsgCount = 0;

            for (let i = 0; i < history.length; i++) {
                const msg = history[i];
                if (msg.role === 'user') {
                    userMsgCount++;
                    markdownParts.push(`# User Message ${userMsgCount}\n${msg.content}`);
                } else if (msg.role === 'assistant') {
                    assistantMsgCount++;
                    // Check if assistant response already has a heading
                    const hasHeading = msg.content && msg.content.trim().startsWith('#');
                    if (hasHeading) {
                        // Use existing heading, don't add duplicate
                        markdownParts.push(msg.content);
                    } else {
                        // No heading, add one
                        markdownParts.push(`# Assistant Response ${assistantMsgCount}\n${msg.content}`);
                    }
                }
            }
            historyMarkdown = markdownParts.join('\n\n');
        }

        // Combine history with current user request
        // Count user messages to number the current one correctly
        const userMessageCount = history.filter(m => m.role === 'user').length + 1;
        const fullPrompt = historyMarkdown
            ? `${historyMarkdown}\n\n# User Message ${userMessageCount}\n${text}`
            : text;

        console.log('[DEBUG] Full prompt being sent:', fullPrompt.substring(0, 500) + '...');

        const payload = {
            message: fullPrompt,  // Send formatted markdown instead of separate history
            file_path: this.selectedFile,
        };
        this.streamClient.fetchAndStreamChat(payload);

        // Now add current user message to history for next time
        if (useHistory) {
            this.uiRenderer.appendConversationEntry('user', text);
        }
    }

    handleMessage(data) {
        const { type, message, timestamp, script_content, status, selected_file, file_path, branch_name, ctx_file, return_content, execution_id } = data;

        switch (type) {
            case 'user':
                // Dedup on reconnect (history replay)
                if (timestamp && message) {
                    const key = `user|${timestamp}|${message}`;
                    if (this.messageKeys && this.messageKeys.has(key)) return;
                }
                let displayMessage = message;
                if (selected_file) {
                    displayMessage = `📄 [${selected_file}]\n${message}`;
                }
                this.uiRenderer.addMessage('user', displayMessage, formatTime(timestamp));
                break;
                
            case 'assistant':
                if (timestamp && message) {
                    const key = `assistant|${timestamp}|${message}`;
                    if (this.messageKeys && this.messageKeys.has(key)) return;
                }
                if (script_content) {
                    // Показать сообщение с кодом
                    this.uiRenderer.addMessageWithCode('assistant', message, script_content, formatTime(timestamp));
                } else {
                    const returnVariant = data.return_variant || (return_content ? 'success' : null);
                    this.uiRenderer.addMessage('assistant', message || '', formatTime(timestamp), {
                        markdownish: true,
                        returnContent: return_content,
                        returnVariant
                    });
                }
                break;
                
            case 'execution':
                // Support both selected_file and file_path (older server may send file_path)
                const filePathForRun = selected_file || file_path;
                console.log('DEBUG: Execution message received:', { status, selected_file, file_path, execution_id, branch_name, ctx_file });
                
                if (status === 'running' && filePathForRun && execution_id) {
                    console.log('DEBUG: Calling addExecutionRunningMessage');
                    this.uiRenderer.addExecutionRunningMessage(message, formatTime(timestamp), filePathForRun, execution_id);
                } else if (status === 'completed' && execution_id) {
                    // Вместо нового сообщения – обновляем уже существующее (инлайново)
                    this.uiRenderer.updateExecutionMessage({ execution_id, return_content, message, branch_name, file_path: filePathForRun, status });
                } else {
                    this.uiRenderer.addMessage('execution', message, formatTime(timestamp));
                }
                
                this.uiRenderer.showTyping(status === 'running');
                break;

            case 'execution_update':
                this.uiRenderer.updateExecutionMessage(data);
                this.uiRenderer.showTyping(false);
                break;
                
            case 'execution_output':
                this.uiRenderer.addMessage('execution_output', message, formatTime(timestamp));
                break;
                
            case 'error':
                this.uiRenderer.addMessage('error', message, formatTime(timestamp));
                this.uiRenderer.showTyping(false);
                break;
            case 'fractalic_event': {
                const ev = data.event || {};
                const evType = ev.type || 'unknown';
                const seq = ev.seq != null ? ev.seq : '?';
                const lvl = ev.level || ev.lvl || '';
                const payloadObj = ev.payload || {};
                let category = 'generic';
                if (evType.startsWith('execution.')) category = 'execution';
                else if (evType.startsWith('return.')) category = 'return';
                else if (evType.startsWith('llm.')) category = 'llm';
                const payload = Object.keys(payloadObj).length ? JSON.stringify(payloadObj) : '';
                const display = `[${seq}] ${evType}${lvl ? ' <'+lvl+'>' : ''}${payload? '\n'+payload : ''}`;
                // Use assistant styling but prefix emoji / color hint per category
                let decorated = display;
                if (category === 'execution') decorated = '⚙️ ' + display;
                else if (category === 'return') decorated = '📤 ' + display;
                else if (category === 'llm') decorated = '🧠 ' + display;
                this.uiRenderer.addMessage('assistant', decorated, formatTime(ev.ts || timestamp || new Date().toISOString()), {
                    markdownish: true,
                    storeHistory: false
                });
                break;
            }
        }
    }
}
