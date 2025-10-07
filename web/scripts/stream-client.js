/**
 * Stream Client Module - handles HTTP streaming with NDJSON
 */

import { createSVGIcon, formatTime } from './utils.js';

export class StreamClient {
    constructor(client) {
        this.client = client;
    }

    async fetchAndStreamChat(payload) {
        try {
            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) {
                this.client.uiRenderer.showTyping(false);
                this.client.uiRenderer.addMessage('error', 'Ошибка запуска потока', formatTime(new Date().toISOString()));
                return;
            }
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                let idx;
                while ((idx = buffer.indexOf('\n')) >= 0) {
                    const line = buffer.slice(0, idx).trim();
                    buffer = buffer.slice(idx + 1);
                    if (!line) continue;
                    this.handleStreamLine(line);
                }
            }
        } catch (e) {
            console.error('Stream error', e);
            this.client.uiRenderer.addMessage('error', 'Ошибка потока: ' + e.message, formatTime(new Date().toISOString()));
        } finally {
            this.client.uiRenderer.showTyping(false);
        }
    }

    handleStreamLine(line) {
        try {
            const data = JSON.parse(line);
            const now = formatTime(new Date().toISOString());
            const sessionId = data.session_id || data.execution_id; // Fallback to execution_id if session_id not present

            // DEBUG: Log ALL incoming events
            console.log('📥 Event:', data.type,
                data.role ? `(role: ${data.role})` : '',
                data.return_content ? '(has return_content)' : ''
            );

            // Dispatch events by type using switch/case for reliability
            switch (data.type) {
                case 'chat_message': {
                    const isUserMessage = data.role === 'user';
                    const isAssistantMessage = data.role === 'assistant';
                    const execId = data.execution_id;

                    // Check if it's a system message (not real dialogue)
                    const isSystemMessage = data.content && (
                        data.content.includes('Фракталик запущен') ||
                        data.content.includes('Фракталик завершил') ||
                        data.content.startsWith('⚙️') ||
                        data.content.startsWith('✅') ||
                        data.content.startsWith('❌')
                    );

                    // Store ALL user messages and ALL assistant responses, except system messages
                    const shouldStoreInHistory = (isUserMessage || isAssistantMessage) && !isSystemMessage;

                    // DEBUG: Log assistant messages to understand routing
                    if (isAssistantMessage) {
                        console.log('🔍 Assistant message:',
                            '\n  execId:', execId,
                            '\n  hasExecBubble:', execId && this.client.executionBubbles.has(execId),
                            '\n  isSystemMessage:', isSystemMessage,
                            '\n  eventId:', data.event_id,
                            '\n  contentPreview:', data.content ? data.content.substring(0, 100) : 'no content'
                        );
                    }

                    // Route assistant messages to execution bubble if available
                    if (isAssistantMessage && execId && this.client.executionBubbles.has(execId)) {
                        const bubbleRefs = this.client.executionBubbles.get(execId);

                        // Clear "Waiting for response..." placeholder on first message
                        const currentContent = bubbleRefs.responseContent.textContent;
                        if (currentContent.includes('Waiting for response')) {
                            bubbleRefs.responseContent.innerHTML = '';
                        }

                        // Show system messages (⚙️, ✅, ❌) directly
                        // Note: return_content is handled separately below
                        const rendered = this.client.uiRenderer.renderMarkdownish(data.content);
                        const messageDiv = document.createElement('div');
                        messageDiv.className = 'message-content system-message';
                        messageDiv.style.cssText = 'margin-bottom: 12px;';
                        messageDiv.innerHTML = rendered;
                        bubbleRefs.responseContent.appendChild(messageDiv);

                        // Store in history
                        if (shouldStoreInHistory) {
                            this.client.conversation.push({ role: 'assistant', content: data.content });
                        }
                    } else {
                        // Regular message handling for user messages or messages without execution_id
                        this.client.uiRenderer.addMessage(data.role === 'user' ? 'user' : 'assistant', data.content, now, {
                            session_id: sessionId,
                            storeHistory: shouldStoreInHistory,
                            markdownish: isAssistantMessage
                        });
                    }
                    break;
                }

                case 'workflow_complete': {
                    // Workflow completion event (from @run operation / agent execution)
                    const execId = data.execution_id;
                    if (execId && this.client.executionBubbles.has(execId)) {
                        const bubbleRefs = this.client.executionBubbles.get(execId);
                        const filePath = data.target || '';
                        const fileName = filePath.split('/').pop() || filePath;

                        // Create a small notification in the response area
                        const notification = document.createElement('div');
                        notification.style.cssText = `
                            padding: 6px 10px;
                            margin: 4px 0;
                            border-left: 3px solid #83d69d;
                            background: rgba(131, 214, 157, 0.1);
                            font-size: 12px;
                            color: #83d69d;
                            border-radius: 3px;
                        `;
                        notification.textContent = `✓ Agent completed: ${fileName}`;

                        bubbleRefs.responseContent.appendChild(notification);

                        // If has return_content, show it
                        if (data.return_content) {
                            const returnBlock = document.createElement('div');
                            returnBlock.style.cssText = `
                                margin: 8px 0;
                                padding: 8px;
                                background: rgba(var(--accent-rgb), 0.05);
                                border: 1px solid rgba(var(--accent-rgb), 0.2);
                                border-radius: 4px;
                                font-size: 13px;
                            `;
                            returnBlock.textContent = data.return_content;
                            bubbleRefs.responseContent.appendChild(returnBlock);
                        }
                    }
                    break;
                }

                case 'workflow_error': {
                    // Workflow error event
                    const execId = data.execution_id;
                    if (execId && this.client.executionBubbles.has(execId)) {
                        const bubbleRefs = this.client.executionBubbles.get(execId);
                        const filePath = data.target || '';
                        const fileName = filePath.split('/').pop() || filePath;

                        const notification = document.createElement('div');
                        notification.style.cssText = `
                            padding: 6px 10px;
                            margin: 4px 0;
                            border-left: 3px solid #d33f3f;
                            background: rgba(211, 63, 63, 0.1);
                            font-size: 12px;
                            color: #d33f3f;
                            border-radius: 3px;
                        `;
                        notification.textContent = `✗ Agent failed: ${fileName}`;
                        bubbleRefs.responseContent.appendChild(notification);
                    }
                    break;
                }

                case 'execution_start': {
                    // Create execution bubble if doesn't exist
                    const execId = data.execution_id;
                    if (!this.client.executionBubbles.has(execId)) {
                        const filePath = data.target || data.file_path || (this.client.selectedFile || '');
                        const bubbleRefs = this.client.uiRenderer.createExecutionBubble(execId, filePath, now);
                        this.client.executionBubbles.set(execId, bubbleRefs);
                    }
                    break;
                }

                case 'execution_complete': {
                    // Update bubble status to completed
                    const execId = data.execution_id;
                    if (execId && this.client.executionBubbles.has(execId)) {
                        const bubbleRefs = this.client.executionBubbles.get(execId);

                        const checkIcon = createSVGIcon('checkCircle', 16, '#83d69d');
                        const title = bubbleRefs.title;
                        const filePath = data.target || data.file_path || '';
                        title.innerHTML = `
                            ${checkIcon}
                            <span>Completed: ${filePath}</span>
                            <span style="font-size: 11px; color: #83d69d; margin-left: 8px;">✓ Done</span>
                        `;
                        bubbleRefs.bubble.style.borderColor = '#83d69d';
                    }
                    break;
                }

                case 'execution_error': {
                    // Update bubble status to error
                    const execId = data.execution_id;
                    if (execId && this.client.executionBubbles.has(execId)) {
                        const bubbleRefs = this.client.executionBubbles.get(execId);
                        const errorIcon = createSVGIcon('error', 16, '#d33f3f');
                        const title = bubbleRefs.title;
                        const filePath = data.target || data.file_path || '';
                        title.innerHTML = `
                            ${errorIcon}
                            <span>Error: ${filePath}</span>
                            <span style="font-size: 11px; color: #d33f3f; margin-left: 8px;">✗ Failed</span>
                        `;
                        bubbleRefs.bubble.style.borderColor = '#d33f3f';
                    }
                    break;
                }

                case 'error':
                    this.client.uiRenderer.addMessage('error', data.message || 'Ошибка', now, { session_id: sessionId });
                    break;

                case 'termination':
                case 'stream_end':
                    // end of stream
                    this.client.uiRenderer.showTyping(false);
                    break;

                case 'ast_update': {
                    // Handle AST structure update - route to execution bubble
                    const execId = data.execution_id;
                    if (execId && this.client.executionBubbles.has(execId)) {
                        const bubbleRefs = this.client.executionBubbles.get(execId);
                        this.client.uiRenderer.renderASTBlocks(data.blocks || [], data.operation, bubbleRefs.astContainer, bubbleRefs);
                    }
                    break;
                }

                case 'tool_call': {
                    // Insert tool call as pending block after active AST block
                    const execId = data.execution_id;
                    if (execId && this.client.executionBubbles.has(execId)) {
                        const toolIcon = createSVGIcon('tool', 14, '#4A54F5');
                        const eventHtml = `
                            <div style="font-weight: 500; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                                ${toolIcon}
                                <span>Tool: <strong style="color: #afcaf5;">${data.tool_name}</strong></span>
                            </div>
                            <div style="font-size: 11px; color: #a0a8b2; margin-bottom: 4px; font-family: monospace;">ID: ${data.tool_call_id}</div>
                            <div style="font-family: 'Monaco', 'Menlo', monospace; font-size: 11px; white-space: pre-wrap; max-height: 150px; overflow-y: auto; background: #3c424a; padding: 8px; border-radius: 8px;">${data.arguments || '{}'}</div>
                        `;
                        // Use server timestamp (convert from seconds to milliseconds)
                        const timestamp = data.timestamp ? data.timestamp * 1000 : Date.now();
                        this.client.uiRenderer.addPendingBlockAfterActive(execId, eventHtml, timestamp);
                    }
                    break;
                }

                case 'tool_result': {
                    // Insert tool result as pending block after active AST block
                    const execId = data.execution_id;
                    if (execId && this.client.executionBubbles.has(execId)) {
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
                        // Use server timestamp (convert from seconds to milliseconds)
                        const timestamp = data.timestamp ? data.timestamp * 1000 : Date.now();
                        this.client.uiRenderer.addPendingBlockAfterActive(execId, eventHtml, timestamp);
                    }
                    break;
                }

                case 'token_usage': {
                    // Handle token usage event - route to execution bubble
                    const execId = data.execution_id;
                    const blockId = data.block_id;
                    const isSummary = data.is_summary === true;

                    if (execId && this.client.executionBubbles.has(execId)) {
                        const bubbleRefs = this.client.executionBubbles.get(execId);
                        const inputTokens = data.input_tokens || 0;
                        const outputTokens = data.output_tokens || 0;

                        // For summary events, SET totals (already cumulative from backend)
                        // For non-summary events, ACCUMULATE
                        if (isSummary) {
                            bubbleRefs.totalTokens.input = inputTokens;
                            bubbleRefs.totalTokens.output = outputTokens;
                        } else {
                            bubbleRefs.totalTokens.input += inputTokens;
                            bubbleRefs.totalTokens.output += outputTokens;
                        }

                        // Store block-level stats if block_id provided
                        if (blockId) {
                            bubbleRefs.blockTokens.set(blockId, {
                                input: inputTokens,
                                output: outputTokens
                            });

                            // Add inline token block in Inspect mode with timestamp for chronological ordering
                            const chartIcon = createSVGIcon('chart', 14, isSummary ? '#6a9955' : '#d7a558');
                            const inputStr = inputTokens.toLocaleString();
                            const outputStr = outputTokens.toLocaleString();
                            const totalStr = (inputTokens + outputTokens).toLocaleString();
                            const modelName = data.model || 'unknown';
                            const sourceFile = data.source_file || '';

                            const tokenEventHtml = `
                                <div style="font-weight: 500; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
                                    ${chartIcon}
                                    <span style="color: ${isSummary ? '#6a9955' : '#d7a558'};">${isSummary ? 'File Summary' : 'Tokens'}</span>
                                </div>
                                <div style="font-size: 11px; color: #a0a8b2; font-family: monospace;">
                                    ${isSummary && sourceFile ? `File: ${sourceFile}<br/>` : ''}
                                    ${isSummary ? 'Total ' : ''}${!isSummary ? `Model: ${modelName} | ` : ''}Input: ${inputStr} | ${isSummary ? 'Total ' : ''}Output: ${outputStr} | Total: ${totalStr}
                                </div>
                            `;

                            // Use server timestamp (convert from seconds to milliseconds)
                            const timestamp = data.timestamp ? data.timestamp * 1000 : Date.now();
                            this.client.uiRenderer.addPendingBlockAfterActive(execId, tokenEventHtml, timestamp, isSummary);
                        }

                        // Update header token counter
                        this.client.uiRenderer.updateTokenCounter(bubbleRefs);
                    } else {
                        // Fallback for events without execution bubble
                        this.client.uiRenderer.addTokenUsageMessage(data, now);
                    }
                    break;
                }

                case 'block_processing': {
                    // Handle block processing event - highlight active block in execution bubble's AST
                    const execId = data.execution_id;
                    if (execId && this.client.executionBubbles.has(execId)) {
                        const bubbleRefs = this.client.executionBubbles.get(execId);
                        bubbleRefs.activeBlockId = data.block_id;  // Track active block
                        this.client.uiRenderer.highlightActiveBlock(data.block_id, execId);
                    }
                    break;
                }

                case 'keepalive':
                    // ignore keepalive messages
                    break;

                default:
                    console.log('Unknown event type:', data.type, data);
                    break;
            }

            // Handle return_content for ANY event type (separate from event type dispatch)
            // Note: return_content is now emitted as regular CHAT_MESSAGE from fractalic.py:522
            // So this block may not be used anymore
            // Skip workflow events as they have their own dedicated handler above
            if (data.return_content && data.type !== 'workflow_complete' && data.type !== 'workflow_error') {
                console.log('[DEBUG] Received event with return_content field (unexpected!)');
                const execId = data.execution_id;
                if (execId && this.client.executionBubbles.has(execId)) {
                    const bubbleRefs = this.client.executionBubbles.get(execId);

                    const rendered = this.client.uiRenderer.renderMarkdownish(data.return_content);

                    // Add final result
                    const resultDiv = document.createElement('div');
                    resultDiv.className = 'message-content final-result';
                    resultDiv.style.cssText = 'margin-bottom: 12px;';
                    resultDiv.innerHTML = rendered;
                    bubbleRefs.responseContent.appendChild(resultDiv);
                } else {
                    // Fallback for messages without execution bubble
                    this.client.uiRenderer.addMessage('assistant', 'Ответ Fractalic:', now, {
                        returnContent: data.return_content,
                        returnVariant: data.return_variant,
                        session_id: sessionId,
                        storeHistory: false
                    });
                }
            }
        } catch (e) {
            console.error('Bad line ERROR:', e.message, e.stack);
            console.warn('Bad line data:', line);
        }
    }
}
