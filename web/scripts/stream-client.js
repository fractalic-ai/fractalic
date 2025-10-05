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

            if (data.type === 'chat_message') {
                const isUserMessage = data.role === 'user';
                const isAssistantMessage = data.role === 'assistant';
                const execId = data.execution_id;

                // Check if it's a system message (not real dialogue)
                const isSystemMessage = data.content && (
                    data.content.includes('Фракталик запущен') ||
                    data.content.includes('Фракталик завершил') ||
                    data.content.startsWith('⚙️') ||
                    data.content.startsWith('✅')
                );

                // Store ALL user messages and ALL assistant responses, except system messages
                const shouldStoreInHistory = (isUserMessage || isAssistantMessage) && !isSystemMessage;

                // Route assistant messages to execution bubble if available
                if (isAssistantMessage && execId && this.client.executionBubbles.has(execId)) {
                    const bubbleRefs = this.client.executionBubbles.get(execId);

                    // Check if this event was already processed (prevent duplicates)
                    const eventId = data.event_id;
                    if (eventId && bubbleRefs.processedEventIds.has(eventId)) {
                        // This event already processed, skip
                        return;
                    }

                    const rendered = this.client.uiRenderer.renderMarkdownish(data.content);

                    // Clear "Waiting for response..." placeholder on first message
                    const currentContent = bubbleRefs.responseContent.textContent;
                    if (currentContent.includes('Waiting for response')) {
                        bubbleRefs.responseContent.innerHTML = '';
                    }

                    // Append new message (accumulate, don't replace)
                    const messageDiv = document.createElement('div');
                    messageDiv.style.cssText = 'color: #ffffff; font-size: 13px; margin-bottom: 12px;';
                    messageDiv.innerHTML = rendered;
                    bubbleRefs.responseContent.appendChild(messageDiv);

                    // Mark this event as processed
                    if (eventId) {
                        bubbleRefs.processedEventIds.add(eventId);
                    }

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

                if (data.return_content) {
                    // Route @return result to execution bubble Response mode
                    const execId = data.execution_id;
                    if (execId && this.client.executionBubbles.has(execId)) {
                        const bubbleRefs = this.client.executionBubbles.get(execId);
                        const rendered = this.client.uiRenderer.renderMarkdownish(data.return_content);

                        // REPLACE all intermediate messages with final result
                        bubbleRefs.responseContent.innerHTML = `<div style="color: #ffffff; font-size: 13px;">${rendered}</div>`;
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
            } else if (data.type === 'execution') {
                if (data.phase === 'ack' || data.phase === 'start') {
                    // Create execution bubble if doesn't exist
                    const execId = data.execution_id;
                    if (!this.client.executionBubbles.has(execId)) {
                        const filePath = data.target || data.file_path || (this.client.selectedFile || '');
                        const bubbleRefs = this.client.uiRenderer.createExecutionBubble(execId, filePath, now);
                        this.client.executionBubbles.set(execId, bubbleRefs);
                    }
                } else if (data.phase === 'complete') {
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
                } else if (data.phase === 'error') {
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
                }
            } else if (data.type === 'error') {
                this.client.uiRenderer.addMessage('error', data.message || 'Ошибка', now, { session_id: sessionId });
            } else if (data.type === 'termination' || data.type === 'stream_end') {
                // end of stream
                this.client.uiRenderer.showTyping(false);
            } else if (data.type === 'ast_update') {
                // Handle AST structure update - route to execution bubble
                const execId = data.execution_id;
                if (execId && this.client.executionBubbles.has(execId)) {
                    const bubbleRefs = this.client.executionBubbles.get(execId);
                    this.client.uiRenderer.renderASTBlocks(data.blocks || [], data.operation, bubbleRefs.astContainer, bubbleRefs);
                }
            } else if (data.type === 'tool_call') {
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
                    this.client.uiRenderer.addPendingBlockAfterActive(execId, eventHtml);
                }
            } else if (data.type === 'tool_result') {
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
                    this.client.uiRenderer.addPendingBlockAfterActive(execId, eventHtml);
                }
            } else if (data.type === 'token_usage') {
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

                        // Add inline token block in Inspect mode
                        const chartIcon = createSVGIcon('chart', 14, '#d7a558');
                        const inputStr = inputTokens.toLocaleString();
                        const outputStr = outputTokens.toLocaleString();
                        const totalStr = (inputTokens + outputTokens).toLocaleString();
                        const modelName = data.model || 'unknown';

                        const tokenEventHtml = `
                            <div style="font-weight: 500; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
                                ${chartIcon}
                                <span style="color: #d7a558;">Tokens</span>
                            </div>
                            <div style="font-size: 11px; color: #a0a8b2; font-family: monospace;">
                                Model: ${modelName} | In: ${inputStr} | Out: ${outputStr} | Total: ${totalStr}
                            </div>
                        `;
                        this.client.uiRenderer.addPendingBlockAfterActive(execId, tokenEventHtml);
                    }

                    // Update header token counter
                    this.client.uiRenderer.updateTokenCounter(bubbleRefs);
                } else {
                    // Fallback for events without execution bubble
                    this.client.uiRenderer.addTokenUsageMessage(data, now);
                }
            } else if (data.type === 'block_processing') {
                // Handle block processing event - highlight active block in execution bubble's AST
                const execId = data.execution_id;
                if (execId && this.client.executionBubbles.has(execId)) {
                    const bubbleRefs = this.client.executionBubbles.get(execId);
                    bubbleRefs.activeBlockId = data.block_id;  // Track active block
                    this.client.uiRenderer.highlightActiveBlock(data.block_id, execId);
                }
            } else if (data.type === 'keepalive') {
                // ignore keepalive messages
            } else {
                console.log('Unknown event type:', data.type, data);
            }
        } catch (e) {
            console.error('Bad line ERROR:', e.message, e.stack);
            console.warn('Bad line data:', line);
        }
    }
}
