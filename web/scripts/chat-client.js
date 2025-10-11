/**
 * Fractalic Chat Client - Main class that coordinates all modules
 */

import { FileBrowser } from './file-browser.js?v=6';
import { UIRenderer } from './ui-rendering.js?v=6';
import { StreamClient } from './stream-client.js?v=6';
import { createSVGIcon, formatTime } from './utils.js?v=6';

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

    handleStreamEvent(data, receivedAtIso) {
        const receivedIso = receivedAtIso || new Date().toISOString();
        const now = formatTime(receivedIso);
        const sessionId = data.session_id || data.execution_id;

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
                    const placeholder = bubbleRefs.responseContent.querySelector('[data-placeholder="true"]');
                    if (placeholder) {
                        placeholder.remove();
                    }

                    const rendered = this.uiRenderer.renderMarkdownish(data.content);
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message-content system-message';
                    messageDiv.style.cssText = 'margin-bottom: 12px;';
                    messageDiv.innerHTML = rendered;
                    bubbleRefs.responseContent.appendChild(messageDiv);

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

                    if (parentExecId && this.executionBubbles.has(parentExecId)) {
                        const parentBubble = this.executionBubbles.get(parentExecId);
                        const childMeta = {
                            executionId: nestedExecId,
                            blockId: blockId || null,
                            currentLocation: 'response',
                            savedParent: bubbleRefs.bubble.parentNode || parentBubble.responseContent,
                            savedNextSibling: bubbleRefs.bubble.nextSibling || null,
                            placeholderEl: null
                        };
                        parentBubble.childBubbles.push(childMeta);

                        if (parentBubble.inspectPanel && parentBubble.inspectPanel.classList.contains('active')) {
                            this.uiRenderer.moveChildBubblesToInspect(parentBubble, nestedExecId);
                        }
                    }
                }
                break;
            }

            case 'workflow_complete': {
                const nestedExecId = data.nested_execution_id || data.execution_id;

                if (nestedExecId && this.executionBubbles.has(nestedExecId)) {
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

                    if (data.return_content) {
                        const existingReturnContent = bubbleRefs.responseContent.querySelector('.workflow-return-content');
                        if (!existingReturnContent) {
                            const returnBlock = document.createElement('div');
                            returnBlock.className = 'workflow-return-content';
                            returnBlock.style.cssText = `
                                margin: 12px 0 0;
                                font-size: 13px;
                                color: inherit;
                                line-height: 1.5;
                            `;
                            returnBlock.innerHTML = this.uiRenderer.renderMarkdownish(data.return_content);
                            bubbleRefs.responseContent.appendChild(returnBlock);
                        }
                    }
                }
                break;
            }

            case 'workflow_error': {
                const nestedExecId = data.nested_execution_id || data.execution_id;

                if (nestedExecId && this.executionBubbles.has(nestedExecId)) {
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
                }
                break;
            }

            case 'execution_start': {
                const execId = data.execution_id;
                if (!this.executionBubbles.has(execId)) {
                    const filePath = data.target || data.file_path || (this.selectedFile || '');
                    const bubbleRefs = this.uiRenderer.createExecutionBubble(execId, filePath, now);
                    this.executionBubbles.set(execId, bubbleRefs);
                }
                break;
            }

            case 'execution_complete': {
                const execId = data.execution_id;
                if (execId && this.executionBubbles.has(execId)) {
                    const bubbleRefs = this.executionBubbles.get(execId);
                    const checkIcon = createSVGIcon('checkCircle', 16, '#83d69d');
                    const filePath = data.target || data.file_path || '';
                    bubbleRefs.title.innerHTML = `
                        ${checkIcon}
                        <span>Completed: ${filePath}</span>
                        <span style="font-size: 11px; color: #83d69d; margin-left: 8px;">✓ Done</span>
                    `;
                    bubbleRefs.bubble.style.borderColor = '#83d69d';
                }
                break;
            }

            case 'execution_error': {
                const execId = data.execution_id;
                if (execId && this.executionBubbles.has(execId)) {
                    const bubbleRefs = this.executionBubbles.get(execId);
                    const errorIcon = createSVGIcon('error', 16, '#d33f3f');
                    const filePath = data.target || data.file_path || '';
                    bubbleRefs.title.innerHTML = `
                        ${errorIcon}
                        <span>Error: ${filePath}</span>
                        <span style="font-size: 11px; color: #d33f3f; margin-left: 8px;">✗ Failed</span>
                    `;
                    bubbleRefs.bubble.style.borderColor = '#d33f3f';
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
                    const bubbleRefs = this.executionBubbles.get(execId);
                    this.uiRenderer.renderASTBlocks(data.blocks || [], data.operation, bubbleRefs.astContainer, bubbleRefs);
                }
                break;
            }

            case 'tool_call': {
                const execId = data.execution_id;
                if (execId && this.executionBubbles.has(execId)) {
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

            case 'token_usage': {
                const execId = data.execution_id;
                const blockId = data.block_id;
                const isSummary = data.is_summary === true;

                if (execId && this.executionBubbles.has(execId)) {
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

                    if (blockId) {
                        bubbleRefs.blockTokens.set(blockId, {
                            input: inputTokens,
                            output: outputTokens,
                            inputCost: inputCost,
                            outputCost: outputCost,
                            toolUsageCost: toolUsageCost,
                            totalCost: responseCost
                        });

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
                    const bubbleRefs = this.executionBubbles.get(execId);
                    bubbleRefs.activeBlockId = data.block_id;
                    this.uiRenderer.highlightActiveBlock(data.block_id, execId);
                }
                break;
            }

            case 'keepalive':
                break;

            default:
                console.log('Unknown event type:', data.type, data);
                break;
        }

        if (data.return_content && data.type !== 'workflow_complete' && data.type !== 'workflow_error') {
            console.log('[DEBUG] Received event with return_content field (unexpected!)');
            const execId = data.execution_id;
            if (execId && this.executionBubbles.has(execId)) {
                const bubbleRefs = this.executionBubbles.get(execId);
                const rendered = this.uiRenderer.renderMarkdownish(data.return_content);
                const resultDiv = document.createElement('div');
                resultDiv.className = 'message-content final-result';
                resultDiv.style.cssText = 'margin-bottom: 12px;';
                resultDiv.innerHTML = rendered;
                bubbleRefs.responseContent.appendChild(resultDiv);
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
