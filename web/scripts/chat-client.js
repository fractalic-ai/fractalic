/**
 * Fractalic Chat Client - Main class that coordinates all modules
 */

import { FileBrowser } from './file-browser.js?v=14';
import { UIRenderer } from './ui-rendering.js?v=9';
import { StreamClient } from './stream-client.js?v=9';
import { TerminalViewer } from './terminal-viewer.js?v=9';
import { DiffViewer } from './diff-viewer.js?v=9';
import { createSVGIcon, formatTime } from './utils.js?v=9';

// Component System imports - using central module to avoid cache issues
import {
    componentRouter,
    componentRegistry,
    ImageGalleryComponent,
    MessageListComponent
} from './components.js?v=10';

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

        // Artifacts panel elements
        this.artifactsPanel = document.getElementById('artifactsPanel');
        this.artifactsContent = document.getElementById('artifactsContent');
        this.artifactsToggle = document.getElementById('artifactsToggle');

        // State
        this.selectedFile = null;
        this.conversation = [];
        this.executionBubbles = new Map();
        this.activeStreams = new Map();
        this.tokenStats = null;
        this.executionSeq = new Map();
        this.executionSeqSeen = new Map();

        // Initialize modules
        this.fileBrowser = new FileBrowser(this);
        this.uiRenderer = new UIRenderer(this);
        this.streamClient = new StreamClient(this);
        this.terminalViewer = new TerminalViewer({
            modal: this.terminalViewerModal,
            content: this.terminalContent,
            fileName: this.terminalFileName,
            status: this.terminalStatus
        });
        this.diffViewer = new DiffViewer({
            modal: this.diffViewerModal,
            content: this.diffContent
        });

        // Initialize component system
        this.initializeComponentSystem();

        // Setup event listeners
        this.initializeEventListeners();

        // Set connection status (HTTP streaming mode, no persistent WebSocket)
        this.connectionStatus.textContent = '🟢 Готово (HTTP Stream)';
        this.connectionStatus.className = 'status-connected';

        // Update sidebar connection status
        const sidebarStatus = document.getElementById('connectionStatusSidebar');
        if (sidebarStatus) {
            sidebarStatus.innerHTML = `
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                    <circle cx="12" cy="12" r="10"/>
                </svg>
                Готово
            `;
            sidebarStatus.className = 'status-connected';
        }
    }

    initializeComponentSystem() {
        // Register components with their manifests
        componentRegistry.register(
            'image-gallery',
            ImageGalleryComponent,
            ImageGalleryComponent.getManifest()
        );

        componentRegistry.register(
            'message-list',
            MessageListComponent,
            MessageListComponent.getManifest()
        );

        // Initialize router with mount points
        componentRouter.initialize({
            chat: this.messagesContainer,
            artifacts: this.artifactsContent
        });

        console.log('[ComponentSystem] Initialized:', {
            components: componentRegistry.getAllComponents().length,
            mountPoints: ['chat', 'artifacts']
        });
    }

    initializeEventListeners() {
        if (this.sendButton) {
            this.sendButton.addEventListener('click', () => this.sendMessage());
        }
        if (this.chatInput) {
            this.chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        // History toggle buttons (sidebar and inline)
        const historyToggleBtn = document.getElementById('historyToggle');
        const historyToggleInline = document.getElementById('historyToggleInline');

        const toggleHistoryMode = () => {
            const isActive = historyToggleBtn ? historyToggleBtn.classList.contains('active') : false;
            const newState = !isActive;

            if (historyToggleBtn) {
                historyToggleBtn.classList.toggle('active', newState);
            }
            if (historyToggleInline) {
                historyToggleInline.classList.toggle('active', newState);
            }
        };

        if (historyToggleBtn) {
            historyToggleBtn.addEventListener('click', toggleHistoryMode);
        }
        if (historyToggleInline) {
            historyToggleInline.addEventListener('click', toggleHistoryMode);
        }

        // Attach button (file selector) - main area
        const attachBtn = document.getElementById('fileSelectBtn');
        if (attachBtn) {
            attachBtn.addEventListener('click', () => this.fileBrowser.openFileBrowser());
        }

        // File selector button in sidebar
        const sidebarFileSelectBtn = document.getElementById('selectFileBtnSidebar');
        if (sidebarFileSelectBtn) {
            sidebarFileSelectBtn.addEventListener('click', () => this.fileBrowser.openFileBrowser());
        }

        // File browser events
        if (this.selectFileButton) {
            this.selectFileButton.addEventListener('click', () => this.fileBrowser.openFileBrowser());
        }
        if (this.clearSelectionButton) {
            this.clearSelectionButton.addEventListener('click', () => this.fileBrowser.clearFileSelection());
        }
        if (this.modalCloseButton) {
            this.modalCloseButton.addEventListener('click', () => this.fileBrowser.closeFileBrowser());
        }
        if (this.cancelButton) {
            this.cancelButton.addEventListener('click', () => this.fileBrowser.closeFileBrowser());
        }
        
        // Diff viewer events
        if (this.diffModalCloseButton) {
            this.diffModalCloseButton.addEventListener('click', () => this.diffViewer.close());
        }
        if (this.closeDiffButton) {
            this.closeDiffButton.addEventListener('click', () => this.diffViewer.close());
        }

        // Terminal viewer events
        if (this.terminalModalCloseButton) {
            this.terminalModalCloseButton.addEventListener('click', () => this.terminalViewer.close());
        }
        if (this.closeTerminalButton) {
            this.closeTerminalButton.addEventListener('click', () => this.terminalViewer.close());
        }

        // Close modals when clicking outside
        if (this.fileBrowserModal) {
            this.fileBrowserModal.addEventListener('click', (e) => {
                if (e.target === this.fileBrowserModal) {
                    this.fileBrowser.closeFileBrowser();
                }
            });
        }

        if (this.diffViewerModal) {
            this.diffViewerModal.addEventListener('click', (e) => {
                if (e.target === this.diffViewerModal) {
                    this.diffViewer.close();
                }
            });
        }

        if (this.terminalViewerModal) {
            this.terminalViewerModal.addEventListener('click', (e) => {
                if (e.target === this.terminalViewerModal) {
                    this.terminalViewer.close();
                }
            });
        }

        // Artifacts panel toggle - button in header
        const artifactsToggleBtn = document.getElementById('artifactsToggleBtn');
        if (artifactsToggleBtn && this.artifactsPanel) {
            artifactsToggleBtn.addEventListener('click', () => {
                this.artifactsPanel.classList.toggle('hidden');
                const isHidden = this.artifactsPanel.classList.contains('hidden');
                artifactsToggleBtn.title = isHidden ? 'Показать артефакты' : 'Скрыть артефакты';
            });
        }

        // Artifacts panel close button (inside panel)
        if (this.artifactsToggle && this.artifactsPanel) {
            this.artifactsToggle.addEventListener('click', () => {
                this.artifactsPanel.classList.add('hidden');
            });
        }
    }

    async sendMessage() {
        const text = this.chatInput.value.trim();
        if (!text) return;
        if (!this.selectedFile) {
            alert('Сначала выберите markdown файл');
            return;
        }

        // Clean up previous execution bubbles and tracking Maps before starting new request
        console.log('[CLEANUP] Clearing execution bubbles from previous run');
        this.executionBubbles.clear();
        this.executionSeq.clear();
        this.executionSeqSeen.clear();

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

    handleStreamEvent(data, receivedAtIso) {
        const receivedIso = receivedAtIso || new Date().toISOString();
        const now = formatTime(receivedIso);
        const sessionId = data.session_id || data.execution_id;
        const eventSeqRaw = data.seq;
        const seq = Number.isFinite(eventSeqRaw) ? eventSeqRaw : (typeof eventSeqRaw === 'string' && !Number.isNaN(Number(eventSeqRaw)) ? Number(eventSeqRaw) : null);

        const bumpSeq = (executionId) => {
            if (!executionId || seq == null) {
                return true;
            }

            if (!this.executionSeqSeen.has(executionId)) {
                this.executionSeqSeen.set(executionId, new Set());
            }

            const seenSet = this.executionSeqSeen.get(executionId);
            if (seenSet.has(seq)) {
                console.warn(`[SEQ] Duplicate event for ${executionId}: seq=${seq}`);
                return false;
            }
            seenSet.add(seq);

            const lastSeq = this.executionSeq.get(executionId) || 0;
            if (seq < lastSeq) {
                console.warn(`[SEQ] Out-of-order event for ${executionId}: seq=${seq} last=${lastSeq}`);
            } else if (seq > lastSeq) {
                this.executionSeq.set(executionId, seq);
            }

            if (this.executionBubbles.has(executionId)) {
                const bubbleRefs = this.executionBubbles.get(executionId);
                const prevSeq = bubbleRefs.lastSeq || 0;
                if (seq > prevSeq) {
                    bubbleRefs.lastSeq = seq;
                }
            }
            return true;
        };

        console.log('📥 Event:', data.type,
            data.role ? `(role: ${data.role})` : '',
            data.return_content ? '(has return_content)' : ''
        );

        switch (data.type) {
            case 'chat_message': {
                const isUserMessage = data.role === 'user';
                const isAssistantMessage = data.role === 'assistant';
                const execId = data.execution_id;

                const isSystemMessage = data.content && (
                    data.content.includes('Фракталик запущен') ||
                    data.content.includes('Фракталик завершил') ||
                    data.content.startsWith('⚙️') ||
                    data.content.startsWith('✅') ||
                    data.content.startsWith('❌')
                );

                const shouldStoreInHistory = (isUserMessage || isAssistantMessage) && !isSystemMessage;

                if (isAssistantMessage) {
                    console.log('🔍 Assistant message:',
                        '\n  execId:', execId,
                        '\n  hasExecBubble:', execId && this.executionBubbles.has(execId),
                        '\n  isSystemMessage:', isSystemMessage,
                        '\n  eventId:', data.event_id,
                        '\n  contentPreview:', data.content ? data.content.substring(0, 100) : 'no content'
                    );
                }

                if (isAssistantMessage && execId && this.executionBubbles.has(execId)) {
                    const bubbleRefs = this.executionBubbles.get(execId);
                    if (!bumpSeq(execId)) break;
                    const placeholder = bubbleRefs.responseContent.querySelector('[data-placeholder="true"]');
                    if (placeholder) {
                        placeholder.remove();
                    }

                    if (!data.return_content) {
                        const rendered = this.uiRenderer.renderMarkdownish(data.content);
                        const messageDiv = document.createElement('div');
                        messageDiv.className = 'message-content system-message';
                        messageDiv.style.cssText = 'margin-bottom: 12px;';
                        messageDiv.innerHTML = rendered;
                        bubbleRefs.responseContent.appendChild(messageDiv);
                    }

                    if (shouldStoreInHistory) {
                        this.conversation.push({ role: 'assistant', content: data.content });
                    }
                } else {
                    this.uiRenderer.addMessage(isUserMessage ? 'user' : 'assistant', data.content, now, {
                        session_id: sessionId,
                        storeHistory: shouldStoreInHistory,
                        markdownish: isAssistantMessage
                    });
                }
                break;
            }

            case 'workflow_start': {
                const nestedExecId = data.nested_execution_id || data.execution_id;
                const parentExecId = data.parent_execution_id;
                const blockId = data.block_id;
                const filePath = data.target || '';

                if (!this.executionBubbles.has(nestedExecId)) {
                    const bubbleRefs = this.uiRenderer.createExecutionBubble(
                        nestedExecId,
                        filePath,
                        now,
                        parentExecId,
                        blockId
                    );
                    this.executionBubbles.set(nestedExecId, bubbleRefs);
                    bubbleRefs.isCompleted = false;
                    bumpSeq(nestedExecId);

                    if (parentExecId && this.executionBubbles.has(parentExecId)) {
                        const parentBubble = this.executionBubbles.get(parentExecId);
                        // FIXED: Use nestedExecId for terminal stream instead of parent's rootExecutionId
                        // Each nested module has its own terminal stream keyed by its nestedExecutionId
                        const streamExecutionId = nestedExecId;
                        const childMeta = {
                            executionId: nestedExecId,
                            blockId: blockId || null,
                            currentLocation: 'response',
                            savedParent: bubbleRefs.bubble.parentNode || parentBubble.responseContent,
                            savedNextSibling: bubbleRefs.bubble.nextSibling || null,
                            placeholderEl: null,
                            streamExecutionId
                        };
                        parentBubble.childBubbles.push(childMeta);

                        bubbleRefs.terminalStreamId = streamExecutionId;
                        bubbleRefs.isCompleted = false;

                        if (parentBubble.inspectPanel && parentBubble.inspectPanel.classList.contains('active')) {
                            this.uiRenderer.moveChildBubblesToInspect(parentBubble, nestedExecId);
                        }
                    }
                } else {
                    bumpSeq(nestedExecId);
                }
                break;
            }

            case 'workflow_complete': {
                const nestedExecId = data.nested_execution_id || data.execution_id;

                if (nestedExecId && this.executionBubbles.has(nestedExecId)) {
                    if (!bumpSeq(nestedExecId)) break;
                    const bubbleRefs = this.executionBubbles.get(nestedExecId);
                    const filePath = data.target || '';
                    const fileName = filePath.split('/').pop() || filePath;

                    const checkIcon = createSVGIcon('checkCircle', 16, '#83d69d');
                    bubbleRefs.title.innerHTML = `
                        ${checkIcon}
                        <span>✓ Completed: ${fileName}</span>
                        <span style="font-size: 11px; color: #83d69d; margin-left: 8px;">Done</span>
                    `;
                    bubbleRefs.bubble.style.borderColor = '#83d69d';
                    bubbleRefs.isCompleted = true;

                    if (data.return_content) {
                        const placeholder = bubbleRefs.responseContent.querySelector('[data-placeholder="true"]');
                        if (placeholder) {
                            placeholder.remove();
                        }
                        const finalDiv = document.createElement('div');
                        finalDiv.className = 'message-content final-result';
                        finalDiv.style.cssText = 'margin-bottom: 12px;';
                        finalDiv.innerHTML = this.uiRenderer.renderMarkdownish(data.return_content);
                        bubbleRefs.responseContent.appendChild(finalDiv);
                    }
                }

                // Mark terminal as complete (but don't delete from Maps - will be cleaned on next sendMessage)
                if (nestedExecId) {
                    this.terminalViewer.markComplete(nestedExecId);
                }

                break;
            }

            case 'workflow_error': {
                const nestedExecId = data.nested_execution_id || data.execution_id;

                if (nestedExecId && this.executionBubbles.has(nestedExecId)) {
                    if (!bumpSeq(nestedExecId)) break;
                    const bubbleRefs = this.executionBubbles.get(nestedExecId);
                    const filePath = data.target || '';
                    const fileName = filePath.split('/').pop() || filePath;
                    const errorMessage = data.error_message || 'Unknown error';

                    const errorIcon = createSVGIcon('x', 16, '#d33f3f');
                    bubbleRefs.title.innerHTML = `
                        ${errorIcon}
                        <span>✗ Error: ${fileName}</span>
                        <span style="font-size: 11px; color: #d33f3f; margin-left: 8px;">Failed</span>
                    `;
                    bubbleRefs.bubble.style.borderColor = '#d33f3f';

                    const errorBlock = document.createElement('div');
                    errorBlock.className = 'workflow-error-content';
                    errorBlock.style.cssText = `
                        margin: 8px 0;
                        padding: 12px;
                        background: rgba(211, 63, 63, 0.08);
                        border: 1px solid rgba(211, 63, 63, 0.24);
                        border-radius: 10px;
                        font-size: 13px;
                        color: #f28b8b;
                    `;
                    errorBlock.textContent = `Error: ${errorMessage}`;
                    bubbleRefs.responseContent.appendChild(errorBlock);
                    bubbleRefs.isCompleted = true;
                }

                // Mark terminal as complete (but don't delete from Maps - will be cleaned on next sendMessage)
                if (nestedExecId) {
                    this.terminalViewer.markComplete(nestedExecId);
                }

                break;
            }

            case 'execution_start': {
                const execId = data.execution_id;

                if (!this.executionBubbles.has(execId)) {
                    const filePath = data.target || data.file_path || (this.selectedFile || '');
                    const bubbleRefs = this.uiRenderer.createExecutionBubble(execId, filePath, now);
                    this.executionBubbles.set(execId, bubbleRefs);
                    bubbleRefs.isCompleted = false;
                    bubbleRefs.terminalStreamId = bubbleRefs.rootExecutionId || execId;
                    bumpSeq(execId);
                } else {
                    if (!bumpSeq(execId)) break;
                    const bubbleRefs = this.executionBubbles.get(execId);
                    bubbleRefs.isCompleted = false;
                    bubbleRefs.terminalStreamId = bubbleRefs.rootExecutionId || execId;
                }

                break;
            }

            case 'execution_complete': {
                const execId = data.execution_id;
                if (execId && this.executionBubbles.has(execId)) {
                    if (!bumpSeq(execId)) break;
                    const bubbleRefs = this.executionBubbles.get(execId);
                    const checkIcon = createSVGIcon('checkCircle', 16, '#83d69d');
                    const filePath = data.target || data.file_path || '';
                    bubbleRefs.title.innerHTML = `
                        ${checkIcon}
                        <span>Completed: ${filePath}</span>
                        <span style="font-size: 11px; color: #83d69d; margin-left: 8px;">✓ Done</span>
                    `;
                    bubbleRefs.bubble.style.borderColor = '#83d69d';
                    bubbleRefs.isCompleted = true;
                }

                // Mark terminal as complete (but don't delete from Maps - will be cleaned on next sendMessage)
                if (execId) {
                    this.terminalViewer.markComplete(execId);
                }

                break;
            }

            case 'execution_error': {
                const execId = data.execution_id;
                if (execId && this.executionBubbles.has(execId)) {
                    if (!bumpSeq(execId)) break;
                    const bubbleRefs = this.executionBubbles.get(execId);
                    const errorIcon = createSVGIcon('error', 16, '#d33f3f');
                    const filePath = data.target || data.file_path || '';
                    bubbleRefs.title.innerHTML = `
                        ${errorIcon}
                        <span>Error: ${filePath}</span>
                        <span style="font-size: 11px; color: #d33f3f; margin-left: 8px;">✗ Failed</span>
                    `;
                    bubbleRefs.bubble.style.borderColor = '#d33f3f';
                    bubbleRefs.isCompleted = true;
                }

                // Mark terminal as complete (but don't delete from Maps - will be cleaned on next sendMessage)
                if (execId) {
                    this.terminalViewer.markComplete(execId);
                }

                break;
            }

            case 'error':
                this.uiRenderer.addMessage('error', data.message || 'Ошибка', now, { session_id: sessionId });
                break;

            case 'termination':
            case 'stream_end':
                this.uiRenderer.showTyping(false);
                break;

            case 'ast_update': {
                const execId = data.execution_id;
                if (execId && this.executionBubbles.has(execId)) {
                    if (!bumpSeq(execId)) break;
                    const bubbleRefs = this.executionBubbles.get(execId);
                    this.uiRenderer.renderASTBlocks(data.blocks || [], data.operation, bubbleRefs.astContainer, bubbleRefs);
                }
                break;
            }

            case 'tool_call': {
                const execId = data.execution_id;
                if (execId && this.executionBubbles.has(execId)) {
                    if (!bumpSeq(execId)) break;
                    const toolIcon = createSVGIcon('tool', 14, '#4A54F5');
                    const eventHtml = `
                        <div style="font-weight: 500; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                            ${toolIcon}
                            <span>Tool: <strong style="color: #afcaf5;">${data.tool_name}</strong></span>
                        </div>
                        <div style="font-size: 11px; color: #a0a8b2; margin-bottom: 4px; font-family: monospace;">ID: ${data.tool_call_id}</div>
                        <div style="font-family: 'Monaco', 'Menlo', monospace; font-size: 11px; white-space: pre-wrap; max-height: 150px; overflow-y: auto; background: #3c424a; padding: 8px; border-radius: 8px;">${data.arguments || '{}'}</div>
                    `;
                    const timestamp = data.timestamp ? data.timestamp * 1000 : Date.now();
                    this.uiRenderer.addPendingBlockAfterActive(execId, eventHtml, timestamp);
                }
                break;
            }

            case 'tool_result': {
                const execId = data.execution_id;
                if (execId && this.executionBubbles.has(execId)) {
                    if (!bumpSeq(execId)) break;
                    const resultPreview = data.result && data.result.length > 200
                        ? data.result.substring(0, 200) + '...'
                        : data.result || '(empty)';
                    const checkIcon = createSVGIcon('checkCircle', 14, '#83d69d');
                    const eventHtml = `
                        <div style="font-weight: 500; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                            ${checkIcon}
                            <span>Result: <strong style="color: #b8eec9;">${data.tool_name}</strong></span>
                        </div>
                        <div style="font-size: 11px; color: #a0a8b2; margin-bottom: 4px; font-family: monospace;">ID: ${data.tool_call_id}</div>
                        <div style="font-family: 'Monaco', 'Menlo', monospace; font-size: 11px; white-space: pre-wrap; max-height: 150px; overflow-y: auto; background: #3c424a; padding: 8px; border-radius: 8px;">${resultPreview}</div>
                    `;
                    const timestamp = data.timestamp ? data.timestamp * 1000 : Date.now();
                    this.uiRenderer.addPendingBlockAfterActive(execId, eventHtml, timestamp);
                }
                break;
            }

            case 'token_usage':           // Legacy support - treat as call
            case 'token_usage_call':      // Individual LLM call token usage
            case 'token_usage_summary': { // Aggregated file/operation summary
                const execId = data.execution_id;
                const blockId = data.block_id;
                // Determine type: check event type first, fallback to is_summary flag for legacy
                const isSummary = data.type === 'token_usage_summary' || data.is_summary === true;

                if (execId && this.executionBubbles.has(execId)) {
                    if (!bumpSeq(execId)) break;
                    const bubbleRefs = this.executionBubbles.get(execId);
                    const inputTokens = data.input_tokens || 0;
                    const outputTokens = data.output_tokens || 0;

                    const responseCost = data.response_cost || 0.0;
                    const inputCost = data.input_cost || 0.0;
                    const outputCost = data.output_cost || 0.0;
                    const toolUsageCost = data.tool_usage_cost || 0.0;

                    if (!bubbleRefs.totalTokens.cost) {
                        bubbleRefs.totalTokens.cost = 0.0;
                    }

                    if (isSummary) {
                        bubbleRefs.totalTokens.input = inputTokens;
                        bubbleRefs.totalTokens.output = outputTokens;
                        bubbleRefs.totalTokens.cost = responseCost;
                    } else {
                        bubbleRefs.totalTokens.input += inputTokens;
                        bubbleRefs.totalTokens.output += outputTokens;
                        bubbleRefs.totalTokens.cost += responseCost;
                    }

                    // For summary events, always create a display block even without blockId
                    // For call events, only create display if blockId is present
                    const shouldCreateDisplay = isSummary || blockId;

                    if (blockId) {
                        bubbleRefs.blockTokens.set(blockId, {
                            input: inputTokens,
                            output: outputTokens,
                            inputCost: inputCost,
                            outputCost: outputCost,
                            toolUsageCost: toolUsageCost,
                            totalCost: responseCost
                        });

                        if (!isSummary) {
                            this.uiRenderer.refreshBlockTokenBadge(execId, blockId);
                        }
                    }

                    if (shouldCreateDisplay) {
                        const chartIcon = createSVGIcon('chart', 14, isSummary ? '#6a9955' : '#d7a558');
                        const inputStr = inputTokens.toLocaleString();
                        const outputStr = outputTokens.toLocaleString();
                        const totalStr = (inputTokens + outputTokens).toLocaleString();
                        const modelName = data.model || 'unknown';
                        const sourceFile = data.source_file || '';

                        let costDisplay = '';
                        if (responseCost > 0) {
                            if (isSummary) {
                                costDisplay = ` | Cost: $${responseCost.toFixed(6)}`;
                            } else {
                                const inputCostStr = inputCost > 0 ? `$${inputCost.toFixed(6)}` : '$0';
                                const outputCostStr = outputCost > 0 ? `$${outputCost.toFixed(6)}` : '$0';
                                costDisplay = ` | Cost: ${inputCostStr} / ${outputCostStr}`;
                                if (toolUsageCost > 0) {
                                    costDisplay += ` [tool: $${toolUsageCost.toFixed(6)}]`;
                                }
                                costDisplay += ` = $${responseCost.toFixed(6)}`;
                            }
                        }

                        const tokenEventHtml = `
                            <div style="font-weight: 500; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
                                ${chartIcon}
                                <span style="color: ${isSummary ? '#6a9955' : '#d7a558'};">${isSummary ? 'File Summary' : 'Tokens'}</span>
                            </div>
                            <div style="font-size: 11px; color: #a0a8b2; font-family: monospace;">
                                ${isSummary && sourceFile ? `File: ${sourceFile}<br/>` : ''}
                                ${isSummary ? 'Total ' : ''}${!isSummary ? `Model: ${modelName} | ` : ''}Input: ${inputStr} | ${isSummary ? 'Total ' : ''}Output: ${outputStr} | Total: ${totalStr}${costDisplay}
                            </div>
                        `;

                        const timestamp = data.timestamp ? data.timestamp * 1000 : Date.now();
                        this.uiRenderer.addPendingBlockAfterActive(execId, tokenEventHtml, timestamp, isSummary);
                    }

                    this.uiRenderer.updateTokenCounter(bubbleRefs);

                    if (bubbleRefs.parentExecutionId && this.executionBubbles.has(bubbleRefs.parentExecutionId)) {
                        let currentParent = bubbleRefs.parentExecutionId;
                        while (currentParent) {
                            const parentBubble = this.executionBubbles.get(currentParent);
                            if (parentBubble) {
                                this.uiRenderer.updateTokenCounter(parentBubble);
                                currentParent = parentBubble.parentExecutionId;
                            } else {
                                break;
                            }
                        }
                    }
                } else {
                    this.uiRenderer.addTokenUsageMessage(data, now);
                }
                break;
            }

            case 'block_processing': {
                const execId = data.execution_id;
                if (execId && this.executionBubbles.has(execId)) {
                    if (!bumpSeq(execId)) break;
                    const bubbleRefs = this.executionBubbles.get(execId);
                    bubbleRefs.activeBlockId = data.block_id;
                    this.uiRenderer.highlightActiveBlock(data.block_id, execId);
                }
                break;
            }

            case 'terminal_output': {
                // Forward terminal output to terminal viewer for buffering
                this.terminalViewer.handleTerminalOutput(data);
                break;
            }

            case 'keepalive':
                break;

            default:
                // Check if any component handles this custom event
                const executionId = data.execution_id || data.nested_execution_id || 'default';
                if (componentRegistry.hasHandlerForEvent(data.type)) {
                    console.log(`[ComponentRouter] Routing custom event: ${data.type} to components`);
                    componentRouter.routeEvent(data.type, data, executionId);
                } else {
                    console.log('Unknown event type:', data.type, data);
                }
                break;
        }

        if (data.return_content && data.type !== 'workflow_complete' && data.type !== 'workflow_error') {
            console.log('[DEBUG] Received event with return_content field (unexpected!)');
            const execId = data.execution_id;
            if (execId && this.executionBubbles.has(execId)) {
                const bubbleRefs = this.executionBubbles.get(execId);
                const placeholder = bubbleRefs.responseContent.querySelector('[data-placeholder="true"]');
                if (placeholder) {
                    placeholder.remove();
                }
                const rendered = this.uiRenderer.renderMarkdownish(data.return_content);
                const resultDiv = document.createElement('div');
                resultDiv.className = 'message-content final-result';
                resultDiv.style.cssText = 'margin-bottom: 12px;';
                resultDiv.innerHTML = rendered;
                bubbleRefs.responseContent.appendChild(resultDiv);
                if (data.role === 'assistant') {
                    this.uiRenderer.appendConversationEntry('assistant', data.return_content);
                }
            } else {
                this.uiRenderer.addMessage('assistant', 'Ответ Fractalic:', now, {
                    returnContent: data.return_content,
                    returnVariant: data.return_variant,
                    session_id: sessionId,
                    storeHistory: false
                });
            }
        }
    }

    onStreamError(message) {
        const timestamp = formatTime(new Date().toISOString());
        this.uiRenderer.addMessage('error', message || 'Ошибка запуска потока', timestamp);
        this.uiRenderer.showTyping(false);
    }

    onStreamException(error) {
        const timestamp = formatTime(new Date().toISOString());
        const errorMessage = error?.message ? `Ошибка потока: ${error.message}` : 'Ошибка потока';
        this.uiRenderer.addMessage('error', errorMessage, timestamp);
        this.uiRenderer.showTyping(false);
    }

    onStreamFinished() {
        this.uiRenderer.showTyping(false);
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
