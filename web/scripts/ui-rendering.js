/**
 * UI Rendering Module - handles all DOM manipulation and rendering
 */

import { formatTime, escapeHtml } from './utils.js';

export class UIRenderer {
    constructor(client) {
        this.client = client;
        this.messagesContainer = client.messagesContainer;
        this.messageKeys = new Set();
        this.astBlocks = new Map();
        this.isFirstASTSnapshot = true;
        this.markedConfigured = false;
        // Track pending nested bubbles waiting for AST blocks to be created
        this.pendingNestedBubbles = new Map(); // blockId -> Array of {bubble, executionId}
        // Track nested bubbles that need to be preserved across AST re-renders (per execution)
        // Map: executionId -> Map(blockId -> Array of bubble elements)
        this.savedNestedBubbles = new Map();
    }

    // ========== MARKDOWN RENDERING ==========

    configureMarked() {
        if (this.markedConfigured || typeof marked === 'undefined') return;

        // Configure marked-highlight extension if available
        if (typeof markedHighlight !== 'undefined' && typeof hljs !== 'undefined') {
            marked.use(markedHighlight.markedHighlight({
                langPrefix: 'hljs language-',
                highlight(code, lang) {
                    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                    return hljs.highlight(code, { language }).value;
                }
            }));
        }

        marked.setOptions({
            breaks: true,  // Convert \n to <br>
            gfm: true,     // GitHub Flavored Markdown
            headerIds: false,
            mangle: false
        });

        this.markedConfigured = true;
    }

    renderMarkdown(text) {
        if (!text) return '';

        // Ensure marked.js is configured once
        this.configureMarked();

        if (typeof marked !== 'undefined') {
            try {
                const html = marked.parse(text);
                return html;
            } catch (err) {
                console.error('Marked.js parsing error:', err);
                // Fallback to escaped text
                return '<pre>' + text.replace(/&/g, '&amp;')
                                     .replace(/</g, '&lt;')
                                     .replace(/>/g, '&gt;') + '</pre>';
            }
        }

        // Fallback if marked.js is not loaded
        console.warn('marked.js not loaded, using fallback renderer');
        return '<pre>' + text.replace(/&/g, '&amp;')
                             .replace(/</g, '&lt;')
                             .replace(/>/g, '&gt;') + '</pre>';
    }

    renderMarkdownish(text) {
        return this.renderMarkdown(text);
    }

    // ========== MESSAGE RENDERING ==========

    addMessage(type, message, time, options = {}) {
        const keyParts = [type || '', time || '', message || ''];
        if (options.returnContent) keyParts.push(options.returnContent);
        const key = keyParts.join('|');
        if (this.messageKeys.has(key)) return; // skip duplicate
        this.messageKeys.add(key);

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        messageDiv.dataset.msgKey = key;

        // Store session_id in dataset for filtering/grouping
        if (options.session_id) {
            messageDiv.dataset.sessionId = options.session_id;
        }

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (options.markdownish) {
            contentDiv.innerHTML = this.renderMarkdownish(message || '');
        } else if (options.allowHtml) {
            contentDiv.innerHTML = options.allowHtml;
        } else {
            contentDiv.textContent = message || '';
        }

        if (options.returnContent && options.returnContent.trim()) {
            const returnBlock = document.createElement('pre');
            returnBlock.className = 'assistant-return-block';
            if (options.returnVariant === 'error') {
                returnBlock.classList.add('error');
            }
            returnBlock.textContent = options.returnContent;
            contentDiv.appendChild(returnBlock);
        }

        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = time;

        // Add session_id badge if present
        if (options.session_id) {
            const sessionBadge = document.createElement('span');
            sessionBadge.className = 'session-badge';
            sessionBadge.textContent = `Session: ${options.session_id.substring(0, 8)}`;
            sessionBadge.title = `Full session ID: ${options.session_id}`;
            timeDiv.appendChild(sessionBadge);
        }

        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);

        this.client.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        if (options.storeHistory !== false) {
            let historyContent = '';
            if (message && message.trim()) historyContent = message.trim();
            const returnText = options.returnContent && options.returnContent.trim();
            if (returnText) {
                historyContent = historyContent ? `${historyContent}\n\n${returnText}` : returnText;
            }
            this.appendConversationEntry(type, historyContent);
        }
    }

    addMessageWithCode(type, message, code, time) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        // Разделить сообщение и код
        const parts = message.split('```markdown');
        if (parts.length > 1) {
            const textPart = parts[0].trim();
            if (textPart) {
                const textNode = document.createTextNode(textPart);
                contentDiv.appendChild(textNode);
            }

            const codeDiv = document.createElement('div');
            codeDiv.className = 'code-block';
            codeDiv.textContent = code;
            contentDiv.appendChild(codeDiv);
        } else {
            contentDiv.textContent = message;
        }

        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = time;

        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);

        this.client.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    // ========== TOKEN USAGE RENDERING ==========

    addTokenUsageMessage(data, time) {
        const executionId = data.execution_id;
        const isSummary = data.is_summary === true;

        const modelName = data.model || 'unknown';
        const inputTokens = (data.input_tokens || 0).toLocaleString();
        const outputTokens = (data.output_tokens || 0).toLocaleString();
        const totalTokens = (data.input_tokens + data.output_tokens || 0).toLocaleString();

        // Extract cost information
        const responseCost = data.response_cost || 0;
        const inputCost = data.input_cost || 0;
        const outputCost = data.output_cost || 0;
        const toolUsageCost = data.tool_usage_cost || 0;

        let headerIcon = this.createSVGIcon('chart', 14, '#d7a558');
        let headerText = 'Token Usage';
        if (isSummary) {
            headerIcon = this.createSVGIcon('trendingUp', 14, '#6a9955');
            headerText = 'File Summary';
        }

        // Format cost display
        let costHtml = '';
        if (responseCost > 0) {
            if (isSummary) {
                // For summary, show total cost only (no breakdown since we aggregate multiple calls)
                costHtml = `<div style="margin-top: 4px; color: #10b981; font-weight: 500;">${isSummary ? 'File ' : ''}Cost: <strong>$${responseCost.toFixed(6)}</strong></div>`;
            } else {
                // For individual calls, show cost breakdown
                const inputCostStr = inputCost > 0 ? `$${inputCost.toFixed(6)}` : '$0';
                const outputCostStr = outputCost > 0 ? `$${outputCost.toFixed(6)}` : '$0';
                costHtml = `<div style="margin-top: 4px; color: #10b981; font-weight: 500;">Cost: ${inputCostStr} / ${outputCostStr}`;
                if (toolUsageCost > 0) {
                    costHtml += ` [tool: $${toolUsageCost.toFixed(6)}]`;
                }
                costHtml += ` = <strong>$${responseCost.toFixed(6)}</strong></div>`;
            }
        }

        // For summary, show file totals AND global session totals
        let summaryHtml = '';
        if (isSummary && data.total_input !== undefined) {
            const sessionInput = (data.total_input || 0).toLocaleString();
            const sessionOutput = (data.total_output || 0).toLocaleString();
            const sessionTotal = ((data.total_input || 0) + (data.total_output || 0)).toLocaleString();
            summaryHtml = `
                <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #ddd; font-weight: 500;">
                    <div>Session Total: ${sessionTotal} (${sessionInput}/${sessionOutput})</div>
                </div>
            `;
        }

        const tokenContent = `
            <div style="font-weight: 500; margin-bottom: 6px; display: flex; align-items: center; gap: 6px; color: #03073D;">
                ${headerIcon}
                <span>${headerText}</span>
            </div>
            <div style="font-size: 11px;">
                ${data.source_file && isSummary ? `<div style="margin-bottom: 4px; color: rgba(3, 7, 61, 0.6);">File: <strong style="color: #03073D;">${data.source_file}</strong></div>` : ''}
                ${!isSummary ? `<div style="margin-bottom: 4px; color: rgba(3, 7, 61, 0.6);">Model: <strong style="color: #03073D;">${modelName}</strong></div>` : ''}
                <div style="margin-bottom: 2px; color: rgba(3, 7, 61, 0.6);">${isSummary ? 'File ' : ''}Input: <strong style="color: #03073D;">${inputTokens}</strong></div>
                <div style="margin-bottom: 2px; color: rgba(3, 7, 61, 0.6);">${isSummary ? 'File ' : ''}Output: <strong style="color: #03073D;">${outputTokens}</strong></div>
                <div style="margin-top: 4px; color: #4A54F5; font-weight: 500;">${isSummary ? 'File ' : ''}Total: <strong>${totalTokens}</strong></div>
                ${costHtml}
                ${summaryHtml}
            </div>
        `;

        // Try to nest inside execution block
        const eventType = isSummary ? 'token_usage_summary' : 'token_usage';
        if (executionId && this.addNestedEventToExecution(executionId, eventType, tokenContent, time)) {
            // Successfully nested, skip creating separate message
        } else {
            // Fallback: create standalone message
            const tokenDiv = document.createElement('div');
            tokenDiv.className = `message token-usage${isSummary ? ' summary' : ''}`;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.innerHTML = tokenContent;

            const timeDiv = document.createElement('div');
            timeDiv.className = 'message-time';
            timeDiv.textContent = time;

            tokenDiv.appendChild(contentDiv);
            tokenDiv.appendChild(timeDiv);

            this.client.messagesContainer.appendChild(tokenDiv);
            this.scrollToBottom();
        }

        // Accumulate session statistics (only for non-summary events to avoid double counting)
        if (!isSummary) {
            if (!this.client.tokenStats) {
                this.client.tokenStats = { total_input: 0, total_output: 0, calls: 0 };
            }
            this.client.tokenStats.total_input += (data.input_tokens || 0);
            this.client.tokenStats.total_output += (data.output_tokens || 0);
            this.client.tokenStats.calls += 1;
            this.updateTokenStatsDisplay();
        }
    }

    updateTokenStatsDisplay() {
        let statsDiv = document.querySelector('.token-stats-summary');
        if (!statsDiv) {
            // Create stats display at the top of chat
            statsDiv = document.createElement('div');
            statsDiv.className = 'token-stats-summary';
            const chatHeader = document.querySelector('.chat-header');
            if (chatHeader) {
                chatHeader.after(statsDiv);
            }
        }

        if (this.client.tokenStats && this.client.tokenStats.calls > 0) {
            const totalInput = this.client.tokenStats.total_input.toLocaleString();
            const totalOutput = this.client.tokenStats.total_output.toLocaleString();
            const totalAll = (this.client.tokenStats.total_input + this.client.tokenStats.total_output).toLocaleString();
            statsDiv.innerHTML = `
                <div class="stats-content">
                    <span class="stats-title">Session Total:</span>
                    <span class="stats-value">↑${totalInput}</span>
                    <span class="stats-value">↓${totalOutput}</span>
                    <span class="stats-value total">∑${totalAll}</span>
                    <span class="stats-calls">(${this.client.tokenStats.calls} calls)</span>
                </div>
            `;
            statsDiv.style.display = 'block';
        }
    }

    // ========== EXECUTION BUBBLE RENDERING ==========

    createExecutionBubble(executionId, filePath, time, parentExecutionId = null, blockId = null) {
        // Create main bubble container
        const bubble = document.createElement('div');
        bubble.className = 'execution-bubble';
        bubble.setAttribute('data-execution-id', executionId);

        // Mark as nested if has parent
        if (parentExecutionId) {
            bubble.classList.add('nested-bubble');
            bubble.setAttribute('data-parent-execution-id', parentExecutionId);
            if (blockId) {
                bubble.setAttribute('data-parent-block-id', blockId);
            }
        }

        // Create header
        const header = document.createElement('div');
        header.className = 'bubble-header';

        // Create title with status icon
        const title = document.createElement('div');
        title.className = 'bubble-title';
        const spinnerIcon = this.createSVGIcon('spinner', 16, '#4A54F5');
        title.innerHTML = `
            ${spinnerIcon}
            <span>${filePath ? `Execution: ${filePath}` : 'Running...'}</span>
            <span style="font-size: 11px; color: #a0a8b2; margin-left: 8px;">${time}</span>
        `;

        // Create controls container (terminal button + token counter + tab switcher)
        const controlsContainer = document.createElement('div');
        controlsContainer.style.cssText = 'display: flex; align-items: center; gap: 12px;';

        // Token counter badge (for Response mode)
        const tokenCounter = document.createElement('div');
        tokenCounter.className = 'token-counter-badge';
        tokenCounter.style.cssText = `
            background: #25292e;
            border: 1px solid #525b67;
            border-radius: 8px;
            padding: 4px 8px;
            font-size: 11px;
            color: #a0a8b2;
            font-family: 'Monaco', monospace;
            display: none;
        `;
        tokenCounter.title = 'Token usage (input/output)';

        // Terminal button
        const terminalButton = document.createElement('button');
        terminalButton.className = 'terminal-icon-button';
        terminalButton.innerHTML = this.createSVGIcon('terminal', 16, '#a0a8b2');
        terminalButton.title = 'Открыть терминал';
        terminalButton.style.cssText = `
            background: #25292e;
            border: 1px solid #525b67;
            border-radius: 8px;
            cursor: pointer;
            padding: 6px 10px;
            display: flex;
            align-items: center;
            opacity: 0.8;
            transition: all 0.2s;
        `;
        terminalButton.onmouseover = () => {
            terminalButton.style.opacity = '1';
            terminalButton.style.borderColor = '#4a54f5';
        };
        terminalButton.onmouseout = () => {
            terminalButton.style.opacity = '0.8';
            terminalButton.style.borderColor = '#525b67';
        };
        terminalButton.onclick = (e) => {
            e.stopPropagation();
            const targetExecutionId = bubbleRefsRef?.rootExecutionId || executionId;
            this.client.fileBrowser.showTerminalViewer(filePath, executionId, targetExecutionId);
        };

        // Create tab switcher
        const tabSwitcher = document.createElement('div');
        tabSwitcher.className = 'bubble-tab-switcher';

        const responseTab = document.createElement('button');
        responseTab.className = 'bubble-tab active';
        responseTab.textContent = 'Response';
        responseTab.setAttribute('data-mode', 'response');

        const inspectTab = document.createElement('button');
        inspectTab.className = 'bubble-tab';
        inspectTab.textContent = 'Inspect';
        inspectTab.setAttribute('data-mode', 'inspect');

        tabSwitcher.appendChild(responseTab);
        tabSwitcher.appendChild(inspectTab);

        controlsContainer.appendChild(terminalButton);
        controlsContainer.appendChild(tokenCounter);
        controlsContainer.appendChild(tabSwitcher);

        header.appendChild(title);
        header.appendChild(controlsContainer);

        // Create response content area
        const responseContent = document.createElement('div');
        responseContent.className = 'bubble-response-content';
        responseContent.innerHTML = '<div data-placeholder="true" style="color: #a0a8b2; font-size: 13px;">Waiting for response...</div>';

        // Create inspect panel
        const inspectPanel = document.createElement('div');
        inspectPanel.className = 'bubble-inspect-panel';

        const astContainer = document.createElement('div');
        astContainer.className = 'bubble-ast-container';
        astContainer.setAttribute('data-execution-id', executionId);
        astContainer.innerHTML = '<div style="color: #a0a8b2; text-align: center;">No AST data yet</div>';

        const pendingOutput = document.createElement('div');
        pendingOutput.className = 'bubble-pending-output';
        pendingOutput.setAttribute('data-execution-id', executionId);

        inspectPanel.appendChild(astContainer);
        inspectPanel.appendChild(pendingOutput);

        // Add tab click handlers
        let bubbleRefsRef = null;
        responseTab.onclick = () => {
            responseTab.classList.add('active');
            inspectTab.classList.remove('active');
            responseContent.classList.remove('hidden');
            inspectPanel.classList.remove('active');
            if (bubbleRefsRef) {
                this.moveChildBubblesToResponse(bubbleRefsRef);
            }
        };

        inspectTab.onclick = () => {
            inspectTab.classList.add('active');
            responseTab.classList.remove('active');
            responseContent.classList.add('hidden');
            inspectPanel.classList.add('active');
            if (bubbleRefsRef) {
                this.moveChildBubblesToInspect(bubbleRefsRef);
            }
        };

        // Assemble bubble
        bubble.appendChild(header);
        bubble.appendChild(responseContent);
        bubble.appendChild(inspectPanel);

        // Add to DOM (nested or top-level)
        if (parentExecutionId && this.client.executionBubbles.has(parentExecutionId)) {
            const parentBubble = this.client.executionBubbles.get(parentExecutionId);

            // IMPORTANT: Nested bubbles appear in Response mode (as child execution)
            // They are full execution bubbles with their own Response/Inspect tabs
            // So we ALWAYS add them to parent's responseContent (not AST container)
            parentBubble.responseContent.appendChild(bubble);
            console.log(`[Nested Bubble] Added to parent response content (parent: ${parentExecutionId.substring(0, 8)}, blockId: ${blockId ? blockId.substring(0, 8) : 'none'})`);
        } else {
            // Top-level bubble: add to messages container
            this.messagesContainer.appendChild(bubble);
        }

        this.scrollToBottom();

        // Return references for later updates
        const parentBubbleRefs = parentExecutionId && this.client.executionBubbles.has(parentExecutionId)
            ? this.client.executionBubbles.get(parentExecutionId)
            : null;

        const rootExecutionId = parentBubbleRefs?.rootExecutionId || parentExecutionId || executionId;

        const bubbleRefs = {
            bubble,
            title,
            responseContent,
            astContainer,
            inspectPanel,
            pendingOutput,
            header,
            tokenCounter,  // Reference to token counter badge
            activeBlockId: null,  // Track currently processing block
            blockTokens: new Map(),  // blockId -> {input, output} token stats
            totalTokens: {input: 0, output: 0},  // Execution-level accumulator
            parentExecutionId,  // Track parent for hierarchical token aggregation
            childBubbles: [],  // Track child bubbles for aggregation and placement
            responseTab,
            inspectTab,
            rootExecutionId,
            executionId
        };
        bubbleRefsRef = bubbleRefs;
        return bubbleRefs;
    }

    removePendingNestedBubble(executionId) {
        if (!executionId) return;
        this.pendingNestedBubbles.forEach((entries, blockId) => {
            const filtered = entries.filter(entry => entry.executionId !== executionId);
            if (filtered.length === 0) {
                this.pendingNestedBubbles.delete(blockId);
            } else if (filtered.length !== entries.length) {
                this.pendingNestedBubbles.set(blockId, filtered);
            }
        });
    }

    moveChildBubblesToInspect(bubbleRefs, targetExecutionId = null) {
        if (!bubbleRefs?.childBubbles || bubbleRefs.childBubbles.length === 0) return;

        bubbleRefs.childBubbles.forEach(child => {
            if (!child || !child.executionId) return;
            if (targetExecutionId && child.executionId !== targetExecutionId) return;
            if (child.currentLocation === 'inspect') return;

            const childRefs = this.client.executionBubbles.get(child.executionId);
            if (!childRefs) return;
            const bubbleEl = childRefs.bubble;
            if (!bubbleEl) return;

            const currentParent = bubbleEl.parentNode || child.savedParent || bubbleRefs.responseContent;

            // Ensure we have a placeholder to preserve original position
            if (!child.placeholderEl || !child.placeholderEl.isConnected) {
                const placeholder = document.createElement('div');
                placeholder.className = 'execution-bubble-placeholder';
                placeholder.dataset.placeholderFor = child.executionId;
                placeholder.style.display = 'none';
                if (currentParent) {
                    currentParent.insertBefore(placeholder, bubbleEl.nextSibling);
                }
                child.placeholderEl = placeholder;
            } else if (child.placeholderEl.parentNode !== currentParent && currentParent) {
                currentParent.insertBefore(child.placeholderEl, bubbleEl.nextSibling);
            }

            child.savedParent = currentParent;
            child.savedNextSibling = bubbleEl.nextSibling || null;

            if (bubbleEl.parentNode) {
                bubbleEl.parentNode.removeChild(bubbleEl);
            }

            if (child.blockId) {
                const blockEl = bubbleRefs.astContainer.querySelector(`.ast-block[data-block-id="${child.blockId}"]`);
                if (blockEl) {
                    blockEl.insertAdjacentElement('afterend', bubbleEl);
                } else {
                    const pendingList = this.pendingNestedBubbles.get(child.blockId) || [];
                    if (!pendingList.some(entry => entry.executionId === child.executionId)) {
                        pendingList.push({ bubble: bubbleEl, executionId: child.executionId });
                    }
                    this.pendingNestedBubbles.set(child.blockId, pendingList);
                    bubbleRefs.pendingOutput.appendChild(bubbleEl);
                }
            } else {
                bubbleRefs.astContainer.appendChild(bubbleEl);
            }

            child.currentLocation = 'inspect';
        });
    }

    moveChildBubblesToResponse(bubbleRefs, targetExecutionId = null) {
        if (!bubbleRefs?.childBubbles || bubbleRefs.childBubbles.length === 0) return;

        bubbleRefs.childBubbles.forEach(child => {
            if (!child || !child.executionId) return;
            if (targetExecutionId && child.executionId !== targetExecutionId) return;
            if (child.currentLocation === 'response') return;

            const childRefs = this.client.executionBubbles.get(child.executionId);
            if (!childRefs) return;
            const bubbleEl = childRefs.bubble;
            if (!bubbleEl) return;

            if (bubbleEl.parentNode) {
                bubbleEl.parentNode.removeChild(bubbleEl);
            }

            this.removePendingNestedBubble(child.executionId);

            let inserted = false;
            const placeholder = child.placeholderEl;
            if (placeholder && placeholder.parentNode) {
                placeholder.parentNode.insertBefore(bubbleEl, placeholder);
                placeholder.remove();
                child.placeholderEl = null;
                inserted = true;
            }

            if (!inserted) {
                let targetParent = child.savedParent && child.savedParent.isConnected ? child.savedParent : bubbleRefs.responseContent;
                let nextSibling = child.savedNextSibling;
                if (nextSibling && nextSibling.parentNode !== targetParent) {
                    nextSibling = null;
                }

                if (targetParent) {
                    if (nextSibling) {
                        targetParent.insertBefore(bubbleEl, nextSibling);
                    } else {
                        targetParent.appendChild(bubbleEl);
                    }
                } else {
                    bubbleRefs.responseContent.appendChild(bubbleEl);
                }
            }

            child.currentLocation = 'response';
            child.savedParent = bubbleEl.parentNode;
            child.savedNextSibling = bubbleEl.nextSibling;
        });
    }

    // ========== AST RENDERING ==========

    renderASTBlocks(blocks, operationType, container, bubbleRefs = null) {
        const executionId = bubbleRefs?.bubble?.getAttribute('data-execution-id') || 'unknown';
        console.log(`[AST] renderASTBlocks called for execution: ${executionId.substring(0, 8)}`);
        console.log(`[AST] Blocks to render: [${blocks.map(b => b.id.substring(0, 8)).join(', ')}]`);

        // SAVE pending blocks before clearing - map them by parent block ID
        const pendingBlocksMap = new Map();

        // Save pending event blocks (token usage, etc.)
        const pendingBlocks = container.querySelectorAll('.pending-event-inline');
        console.log(`[AST] Found ${pendingBlocks.length} pending blocks to save`);
        pendingBlocks.forEach(block => {
            // Find parent AST block by going backwards through siblings
            let parent = block.previousElementSibling;
            while (parent) {
                if (parent.classList.contains('ast-block')) {
                    const parentBlockId = parent.dataset.blockId;
                    if (!pendingBlocksMap.has(parentBlockId)) {
                        pendingBlocksMap.set(parentBlockId, []);
                    }
                    // Clone the element to preserve it (including timestamp and styling)
                    const clonedBlock = block.cloneNode(true);
                    pendingBlocksMap.get(parentBlockId).push(clonedBlock);
                    console.log(`[AST] Saved pending block for parent: ${parentBlockId.substring(0, 8)} (ts: ${block.dataset.timestamp})`);
                    break;
                }
                // Skip over other pending blocks to find the AST block
                parent = parent.previousElementSibling;
            }
        });

        // SAVE nested bubbles before clearing - map them by parent block ID
        const nestedBubblesMap = new Map();
        const nestedBubbles = container.querySelectorAll('.execution-bubble.nested-bubble');
        console.log(`[AST] Found ${nestedBubbles.length} nested bubbles to save`);
        nestedBubbles.forEach(bubble => {
            const parentBlockId = bubble.getAttribute('data-parent-block-id');
            if (parentBlockId) {
                if (!nestedBubblesMap.has(parentBlockId)) {
                    nestedBubblesMap.set(parentBlockId, []);
                }
                // Remove from DOM but keep reference (we'll re-insert it after rebuild)
                nestedBubblesMap.get(parentBlockId).push(bubble);
                console.log(`[AST] Saved nested bubble for parent block: ${parentBlockId.substring(0, 8)} (exec: ${bubble.getAttribute('data-execution-id')?.substring(0, 8)})`);
            }
        });

        // Clear container if no blocks
        if (!blocks || blocks.length === 0) {
    container.innerHTML = '<div style="text-align: center; color: #a0a8b2; padding: 20px;">No AST blocks yet</div>';
    this.astBlocks.clear();
    return;
        }

        // Clear container to rebuild in correct order
        container.innerHTML = '';

        // Track which blocks already exist to mark as new or existing
        const existingBlockIds = new Set(this.astBlocks.keys());

        // Don't highlight anything on first snapshot (initial parse)
        const shouldHighlightNew = !this.isFirstASTSnapshot;

        // Rebuild blocks in correct order from the snapshot
        const newBlocksMap = new Map();
        blocks.forEach(block => {
    const blockId = block.id;
    const isNew = !existingBlockIds.has(blockId) && shouldHighlightNew;

    // Debug logging
    if (isNew) {
        console.log(`[AST] New block detected: ${blockId.substring(0, 8)} (${block.type}) - ${block.header || block.content_preview?.substring(0, 30)}`);
    }

    // Create or reuse block element
    let blockEl = this.astBlocks.get(blockId);
    if (!blockEl) {
        blockEl = document.createElement('div');
        blockEl.className = 'ast-block';
        blockEl.dataset.blockId = blockId;
    }

    // Update block content (pass bubbleRefs for token hints)
    this.updateASTBlockContent(blockEl, block, isNew, bubbleRefs);

    // Append in order
    container.appendChild(blockEl);
    newBlocksMap.set(blockId, blockEl);

    // CHECK if any nested bubbles are waiting for this AST block (deferred insertion)
    if (this.pendingNestedBubbles.has(blockId)) {
        const pendingBubbles = this.pendingNestedBubbles.get(blockId);
        console.log(`[AST] Moving ${pendingBubbles.length} pending nested bubbles to correct position after block ${blockId.substring(0, 8)}`);

        pendingBubbles.forEach(({bubble, executionId}) => {
            // Remove from temporary location and insert after AST block
            if (bubble.parentNode) {
                bubble.parentNode.removeChild(bubble);
            }
            blockEl.insertAdjacentElement('afterend', bubble);
            console.log(`[Nested Bubble] Moved bubble ${executionId.substring(0, 8)} to correct position`);
        });

        // Clear pending list for this block
        this.pendingNestedBubbles.delete(blockId);
    }

    // RESTORE nested bubbles that were saved before clearing (from previous render)
    if (nestedBubblesMap.has(blockId)) {
        const savedBubbles = nestedBubblesMap.get(blockId);
        console.log(`[AST] Restoring ${savedBubbles.length} nested bubbles for: ${blockId.substring(0, 8)}`);
        savedBubbles.forEach(bubble => {
            blockEl.insertAdjacentElement('afterend', bubble);
            // Update blockEl reference to keep inserting after the last bubble
            blockEl = bubble;
        });
    } else if (blocks.length > 0 && blocks.findIndex(b => b.id === blockId) === 0) {
        // Only log once per render cycle (on first block)
        console.log(`[AST] nestedBubblesMap has ${nestedBubblesMap.size} entries: [${Array.from(nestedBubblesMap.keys()).map(k => k.substring(0, 8)).join(', ')}]`);
    }

    // RESTORE pending blocks for this AST block in chronological order
    if (pendingBlocksMap.has(blockId)) {
        const savedPendingBlocks = pendingBlocksMap.get(blockId);

        // DEBUG: Log timestamps before sorting
        console.log(`[AST] Pending blocks BEFORE sort:`, savedPendingBlocks.map(b => ({
            ts: b.dataset.timestamp,
            isSummary: b.dataset.isSummary,
            preview: b.textContent.substring(0, 30)
        })));

        // Sort by timestamp to maintain chronological order
        savedPendingBlocks.sort((a, b) => {
            const tsA = parseInt(a.dataset.timestamp || '0');
            const tsB = parseInt(b.dataset.timestamp || '0');
            return tsA - tsB;
        });

        // DEBUG: Log timestamps after sorting
        console.log(`[AST] Pending blocks AFTER sort:`, savedPendingBlocks.map(b => ({
            ts: b.dataset.timestamp,
            isSummary: b.dataset.isSummary,
            preview: b.textContent.substring(0, 30)
        })));

        console.log(`[AST] Restoring ${savedPendingBlocks.length} pending blocks for: ${blockId.substring(0, 8)} (sorted by timestamp)`);
        // Insert pending blocks right after their parent AST block
        savedPendingBlocks.forEach(pendingBlock => {
            blockEl.insertAdjacentElement('afterend', pendingBlock);
            // Update blockEl reference to keep inserting after the last pending block
            blockEl = pendingBlock;
        });
    }
        });

        // Replace old map with new one
        this.astBlocks = newBlocksMap;

        // Mark that we've received the first snapshot
        this.isFirstASTSnapshot = false;
    }

    updateASTBlockContent(blockEl, block, isNew, bubbleRefs = null) {
        const blockId = block.id;

        // Get icon for block type
        const typeIcon = this.getBlockTypeIcon(block.type);

        // Check if block has token stats (for @llm operations)
        let tokenBadge = '';
        if (bubbleRefs && bubbleRefs.blockTokens && bubbleRefs.blockTokens.has(blockId)) {
    const tokenStats = bubbleRefs.blockTokens.get(blockId);
    const inputStr = tokenStats.input.toLocaleString();
    const outputStr = tokenStats.output.toLocaleString();
    tokenBadge = `<span style="
        font-size: 10px;
        color: #d7a558;
        background: rgba(215, 165, 88, 0.15);
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 6px;
        font-family: Monaco, monospace;
    " title="Token usage for this block">${inputStr}/${outputStr}</span>`;
        }

        // Build block HTML
        let html = `
    <div class="ast-block-type">${typeIcon} ${block.type}${tokenBadge}</div>
        `;

        if (block.header) {
    html += `<div class="ast-block-header">${escapeHtml(block.header)}</div>`;
        }

        // Create preview from full content (truncate to 100 chars)
        if (block.content) {
    let preview = block.content;
    if (preview.length > 100) {
        preview = preview.substring(0, 100) + '...';
    }
    html += `<div class="ast-block-preview">${escapeHtml(preview)}</div>`;
        }

        html += `<div class="ast-block-id">ID: ${escapeHtml(blockId.substring(0, 16))}</div>`;

        blockEl.innerHTML = html;

        // Store full content in data attribute for expansion
        if (block.content) {
    blockEl.dataset.fullContent = block.content;
        }

        // Add click handler to toggle expanded state
        blockEl.onclick = (e) => {
    e.stopPropagation();
    const isExpanding = !blockEl.classList.contains('expanded');
    blockEl.classList.toggle('expanded');

    // Switch between preview and full content
    const previewEl = blockEl.querySelector('.ast-block-preview');
    if (previewEl && blockEl.dataset.fullContent) {
        if (isExpanding) {
            // Show full content
            previewEl.textContent = blockEl.dataset.fullContent;
        } else {
            // Show truncated preview
            let preview = blockEl.dataset.fullContent;
            if (preview.length > 100) {
                preview = preview.substring(0, 100) + '...';
            }
            previewEl.textContent = preview;
        }
    }
        };

        // Remove 'new' class first to avoid duplicate animations
        blockEl.classList.remove('new');

        // Add 'new' class with animation for new blocks
        if (isNew) {
    // Force reflow to restart animation
    void blockEl.offsetWidth;
    blockEl.classList.add('new');
    setTimeout(() => {
        blockEl.classList.remove('new');
    }, 3000);
        }
    }

    getBlockTypeIcon(type) {
        const icons = {
    'heading': '📋',
    'operation': '⚙️'
        };
        return icons[type] || '📝';
    }

    highlightActiveBlock(blockId, executionId) {
        if (!blockId) return;

        // Remove 'active' class from all blocks in all bubbles
        this.astBlocks.forEach((blockEl) => {
    blockEl.classList.remove('active');
        });

        // Add 'active' class to the target block
        const targetBlock = this.astBlocks.get(blockId);
        if (targetBlock) {
    targetBlock.classList.add('active');

    // Scroll to the active block within its container
    targetBlock.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest'
    });

    // Auto-switch to Inspect mode if block is being processed
    if (executionId && this.client.executionBubbles.has(executionId)) {
        const bubbleRefs = this.client.executionBubbles.get(executionId);
        const inspectTab = bubbleRefs.header.querySelector('[data-mode="inspect"]');
        if (inspectTab && !inspectTab.classList.contains('active')) {
            inspectTab.click(); // Auto-open Inspect mode to show AST
        }
    }
        }
    }

    addPendingBlockAfterActive(executionId, content, timestamp = Date.now(), isSummary = false) {
        if (!executionId || !this.client.executionBubbles.has(executionId)) return;

        const bubbleRefs = this.client.executionBubbles.get(executionId);
        const activeBlockId = bubbleRefs.activeBlockId;

        if (!activeBlockId) {
    // No active block yet, add to pending output section
    bubbleRefs.pendingOutput.innerHTML += content;
    return;
        }

        // Find the active block element
        const activeBlockEl = this.astBlocks.get(activeBlockId);
        if (!activeBlockEl) {
    bubbleRefs.pendingOutput.innerHTML += content;
    return;
        }

        // Create pending block element
        const pendingBlock = document.createElement('div');
        pendingBlock.className = 'pending-event-inline';
        pendingBlock.dataset.timestamp = timestamp.toString();
        pendingBlock.dataset.isSummary = isSummary.toString();

        // Add special styling for summary blocks
        if (isSummary) {
            pendingBlock.style.cssText = `
                margin-top: 12px;
                padding: 12px;
                background: rgba(106, 153, 85, 0.1);
                border: 1px solid rgba(106, 153, 85, 0.3);
                border-radius: 8px;
                border-left: 3px solid #6a9955;
            `;
        }

        pendingBlock.innerHTML = content;

        // Insert after active block
        activeBlockEl.insertAdjacentElement('afterend', pendingBlock);
    }

    updateTokenCounter(bubbleRefs) {
        // Update token counter badge in bubble header
        if (!bubbleRefs || !bubbleRefs.tokenCounter) return;

        // Calculate own tokens (not including children)
        const ownInput = bubbleRefs.totalTokens.input;
        const ownOutput = bubbleRefs.totalTokens.output;
        const ownCost = bubbleRefs.totalTokens.cost || 0;

        // Calculate aggregated tokens (own + all children recursively)
        let aggregatedInput = ownInput;
        let aggregatedOutput = ownOutput;
        let aggregatedCost = ownCost;

        // Recursively sum child bubble tokens
        if (bubbleRefs.childBubbles && bubbleRefs.childBubbles.length > 0) {
            bubbleRefs.childBubbles.forEach(childMeta => {
                const childExecId = childMeta?.executionId;
                if (!childExecId) return;
                if (this.client.executionBubbles.has(childExecId)) {
                    const childBubble = this.client.executionBubbles.get(childExecId);
                    aggregatedInput += childBubble.totalTokens.input || 0;
                    aggregatedOutput += childBubble.totalTokens.output || 0;
                    aggregatedCost += childBubble.totalTokens.cost || 0;

                    // Recursively add grandchildren
                    if (childBubble.childBubbles && childBubble.childBubbles.length > 0) {
                        const grandchildStats = this.aggregateChildTokens(childBubble);
                        aggregatedInput += grandchildStats.input;
                        aggregatedOutput += grandchildStats.output;
                        aggregatedCost += grandchildStats.cost;
                    }
                }
            });
        }

        // If no tokens at all, keep hidden
        if (aggregatedInput === 0 && aggregatedOutput === 0) {
            bubbleRefs.tokenCounter.style.display = 'none';
            return;
        }

        // Format numbers (K for thousands)
        const formatNum = (num) => {
            if (num >= 1000) {
                return (num / 1000).toFixed(1) + 'K';
            }
            return num.toString();
        };

        const aggInputStr = formatNum(aggregatedInput);
        const aggOutputStr = formatNum(aggregatedOutput);

        // Build display text with aggregation info
        let displayText = `🎯 ${aggInputStr}/${aggOutputStr}`;
        let tooltipText = `Total Token usage - Input: ${aggregatedInput.toLocaleString()} / Output: ${aggregatedOutput.toLocaleString()}`;

        // Show breakdown if there are children
        if (bubbleRefs.childBubbles && bubbleRefs.childBubbles.length > 0) {
            const ownInputStr = formatNum(ownInput);
            const ownOutputStr = formatNum(ownOutput);
            const childInput = aggregatedInput - ownInput;
            const childOutput = aggregatedOutput - ownOutput;
            const childInputStr = formatNum(childInput);
            const childOutputStr = formatNum(childOutput);

            tooltipText += `\n\nOwn: ${ownInput.toLocaleString()}/${ownOutput.toLocaleString()}`;
            tooltipText += `\nChildren: ${childInput.toLocaleString()}/${childOutput.toLocaleString()}`;
        }

        if (aggregatedCost > 0) {
            displayText += ` 💰$${aggregatedCost.toFixed(6)}`;
            tooltipText += `\n\nTotal Cost: $${aggregatedCost.toFixed(6)}`;

            if (bubbleRefs.childBubbles && bubbleRefs.childBubbles.length > 0) {
                const childCost = aggregatedCost - ownCost;
                tooltipText += `\nOwn Cost: $${ownCost.toFixed(6)}`;
                tooltipText += `\nChildren Cost: $${childCost.toFixed(6)}`;
            }
        }

        bubbleRefs.tokenCounter.textContent = displayText;
        bubbleRefs.tokenCounter.style.display = 'block';
        bubbleRefs.tokenCounter.title = tooltipText;
    }

    aggregateChildTokens(bubbleRefs) {
        // Helper to recursively aggregate tokens from all descendants
        let input = 0;
        let output = 0;
        let cost = 0;

        if (bubbleRefs.childBubbles && bubbleRefs.childBubbles.length > 0) {
            bubbleRefs.childBubbles.forEach(childMeta => {
                const childExecId = childMeta?.executionId;
                if (!childExecId) return;
                if (this.client.executionBubbles.has(childExecId)) {
                    const childBubble = this.client.executionBubbles.get(childExecId);
                    input += childBubble.totalTokens.input || 0;
                    output += childBubble.totalTokens.output || 0;
                    cost += childBubble.totalTokens.cost || 0;

                    // Recursively add descendants
                    const descendantStats = this.aggregateChildTokens(childBubble);
                    input += descendantStats.input;
                    output += descendantStats.output;
                    cost += descendantStats.cost;
                }
            });
        }

        return { input, output, cost };
    }

    // ========== EXECUTION MESSAGE HANDLING ==========

    addExecutionRunningMessage(message, time, filePath, executionId) {
        console.log('DEBUG: addExecutionRunningMessage called with:', { message, time, filePath, executionId });

        // Check if execution block already exists (prevent duplicates)
        const existing = document.querySelector(`[data-execution-id="${executionId}"]`);
        if (existing) {
    console.log('DEBUG: Execution block already exists for', executionId);
    return;
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message execution expandable';
        messageDiv.setAttribute('data-execution-id', executionId);

        // Заголовок с статус-иконкой, файлом, терминалом и временем
        const headerDiv = document.createElement('div');
        headerDiv.className = 'message-header execution-header';
        headerDiv.style.display = 'flex';
        headerDiv.style.justifyContent = 'space-between';
        headerDiv.style.alignItems = 'center';
        headerDiv.style.marginBottom = '8px';
        headerDiv.style.fontSize = '13px';
        headerDiv.style.color = '#333';
        headerDiv.style.cursor = 'pointer';
        headerDiv.style.userSelect = 'none';

        const leftSection = document.createElement('div');
        leftSection.style.display = 'flex';
        leftSection.style.alignItems = 'center';
        leftSection.style.gap = '8px';

        // Status icon (spinner while running)
        const statusIcon = document.createElement('span');
        statusIcon.className = 'completion-status';
        statusIcon.innerHTML = this.createSVGIcon('spinner', 16, '#4A54F5');
        statusIcon.style.display = 'flex';
        statusIcon.style.alignItems = 'center';

        // Expand/collapse indicator
        const expandIcon = document.createElement('span');
        expandIcon.className = 'expand-icon';
        expandIcon.innerHTML = this.createSVGIcon('chevronDown', 14, 'rgba(3, 7, 61, 0.6)');
        expandIcon.style.transition = 'transform 0.2s';
        expandIcon.style.display = 'flex';
        expandIcon.style.alignItems = 'center';

        // File name and status text
        const titleSpan = document.createElement('span');
        titleSpan.className = 'exec-status-text';
        titleSpan.textContent = filePath ? `Выполнение: ${filePath}` : message;
        titleSpan.style.fontWeight = '500';
        titleSpan.style.color = '#ffffff';

        // Terminal icon button
        const terminalIcon = document.createElement('button');
        terminalIcon.className = 'terminal-icon-button';
        terminalIcon.innerHTML = this.createSVGIcon('terminal', 16, 'rgba(3, 7, 61, 0.6)');
        terminalIcon.title = 'Открыть терминал';
        terminalIcon.style.background = 'none';
        terminalIcon.style.border = 'none';
        terminalIcon.style.cursor = 'pointer';
        terminalIcon.style.padding = '4px 8px';
        terminalIcon.style.display = 'flex';
        terminalIcon.style.alignItems = 'center';
        terminalIcon.style.opacity = '0.7';
        terminalIcon.style.transition = 'opacity 0.2s';
        terminalIcon.onmouseover = () => terminalIcon.style.opacity = '1';
        terminalIcon.onmouseout = () => terminalIcon.style.opacity = '0.7';
        terminalIcon.onclick = (e) => {
    e.stopPropagation(); // Prevent collapse/expand
    this.client.fileBrowser.showTerminalViewer(filePath, executionId);
        };

        leftSection.appendChild(statusIcon);
        leftSection.appendChild(expandIcon);
        leftSection.appendChild(titleSpan);
        leftSection.appendChild(terminalIcon);

        const timeSpan = document.createElement('span');
        timeSpan.className = 'message-time';
        timeSpan.textContent = time;
        timeSpan.style.fontSize = '11px';
        timeSpan.style.color = '#a0a8b2';

        headerDiv.appendChild(leftSection);
        headerDiv.appendChild(timeSpan);

        // Preview section for collapsed state (shows @return result)
        const previewDiv = document.createElement('div');
        previewDiv.className = 'execution-preview';
        previewDiv.style.marginTop = '12px';
        previewDiv.style.padding = '12px 16px';
        previewDiv.style.background = '#25292e';
        previewDiv.style.borderRadius = '12px';
        previewDiv.style.fontSize = '13px';
        previewDiv.style.color = '#ffffff';
        previewDiv.style.display = 'none'; // Hidden until we have return content
        previewDiv.style.maxHeight = '80px';
        previewDiv.style.overflow = 'hidden';
        previewDiv.style.textOverflow = 'ellipsis';
        previewDiv.style.border = '1px solid rgba(3, 7, 61, 0.08)';

        // Collapsible body for nested events (tool calls, token usage)
        const bodyDiv = document.createElement('div');
        bodyDiv.className = 'message-body execution-body';
        bodyDiv.style.marginTop = '12px';
        bodyDiv.style.padding = '16px';
        bodyDiv.style.background = '#25292e';
        bodyDiv.style.borderRadius = '12px';
        bodyDiv.style.display = 'block'; // Start expanded

        // Container for nested events
        const eventsContainer = document.createElement('div');
        eventsContainer.className = 'execution-events';
        eventsContainer.setAttribute('data-execution-id', executionId);

        bodyDiv.appendChild(eventsContainer);

        // Toggle expand/collapse on header click
        headerDiv.onclick = () => {
    const isCollapsed = bodyDiv.style.display === 'none';
    bodyDiv.style.display = isCollapsed ? 'block' : 'none';
    previewDiv.style.display = isCollapsed ? 'none' : 'block';
    expandIcon.style.transform = isCollapsed ? 'rotate(0deg)' : 'rotate(-90deg)';
        };

        messageDiv.appendChild(headerDiv);
        messageDiv.appendChild(previewDiv);
        messageDiv.appendChild(bodyDiv);

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        console.log('DEBUG: addExecutionRunningMessage completed, message added to DOM');
    }

    // Helper function to append nested events to execution block
    addNestedEventToExecution(executionId, eventType, content, time) {
        const eventsContainer = document.querySelector(`.execution-events[data-execution-id="${executionId}"]`);
        if (!eventsContainer) {
    console.warn('No execution block found for', executionId, '- event will not be nested');
    return false;
        }

        const eventDiv = document.createElement('div');
        eventDiv.className = `nested-event ${eventType}`;
        eventDiv.style.marginBottom = '8px';
        eventDiv.style.padding = '8px 12px';
        eventDiv.style.borderRadius = '6px';
        eventDiv.style.fontSize = '12px';
        eventDiv.style.backgroundColor = '#f5f5f5';

        if (eventType === 'tool_call') {
    eventDiv.style.backgroundColor = '#f0f8ff';
    eventDiv.style.borderLeft = '3px solid #4A54F5';
        } else if (eventType === 'tool_result') {
    eventDiv.style.backgroundColor = '#f0fff4';
    eventDiv.style.borderLeft = '3px solid #10b981';
        } else if (eventType === 'token_usage') {
    eventDiv.style.backgroundColor = '#fffbeb';
    eventDiv.style.borderLeft = '3px solid #f59e0b';
        } else if (eventType === 'token_usage_summary') {
    eventDiv.style.backgroundColor = '#f0fff4';
    eventDiv.style.borderLeft = '3px solid #10b981';
    eventDiv.style.boxShadow = '0 2px 8px rgba(16, 185, 129, 0.15)';
        } else if (eventType === 'return_value') {
    eventDiv.style.backgroundColor = '#faf5ff';
    eventDiv.style.borderLeft = '3px solid #8b5cf6';
    eventDiv.style.boxShadow = '0 2px 8px rgba(139, 92, 246, 0.15)';
        } else if (eventType === 'error') {
    eventDiv.style.backgroundColor = '#fef2f2';
    eventDiv.style.borderLeft = '3px solid #ef4444';
    eventDiv.style.color = '#ef4444';
        }

        const contentDiv = document.createElement('div');
        contentDiv.className = 'nested-event-content';
        if (typeof content === 'string') {
    contentDiv.innerHTML = content;
        } else {
    contentDiv.appendChild(content);
        }

        const timeSpan = document.createElement('span');
        timeSpan.style.fontSize = '10px';
        timeSpan.style.color = '#999';
        timeSpan.style.marginTop = '4px';
        timeSpan.style.display = 'block';
        timeSpan.textContent = time;

        eventDiv.appendChild(contentDiv);
        eventDiv.appendChild(timeSpan);

        eventsContainer.appendChild(eventDiv);
        this.scrollToBottom();

        return true;
    }

    updateExecutionMessage(data) {
        const { execution_id, return_content, message, branch_name, file_path, duration_ms, status, error_message, recent_output, bytes_read } = data;
        const resolvedStatus = status || 'completed';
        const messageElement = document.querySelector(`[data-execution-id="${execution_id}"]`);
        if (!messageElement) {
    console.warn('Message element not found for execution_id (creating fallback):', execution_id);
    // Fallback: create a completed execution block so user still sees result
    this.addExecutionCompletedFallback({ execution_id, return_content, message, branch_name, file_path, duration_ms, status: resolvedStatus, error_message, recent_output, bytes_read });
    return;
        }

        // Update header with completion status
        const headerElement = messageElement.querySelector('.message-header');
        if (headerElement) {
    // Find and update status icon (replace spinner with checkmark/X)
    let statusIcon = headerElement.querySelector('.completion-status');
    if (statusIcon) {
        if (resolvedStatus === 'error') {
            statusIcon.innerHTML = this.createSVGIcon('x', 16, '#ef4444');
        } else {
            statusIcon.innerHTML = this.createSVGIcon('check', 16, '#10b981');
        }
    }

    // Find and update status text
    let statusSpan = headerElement.querySelector('.exec-status-text');
    if (statusSpan) {
        // Keep the file name but update status
        const currentText = statusSpan.textContent;
        const fileMatch = currentText.match(/для файла: (.+)$/);
        const fileName = fileMatch ? fileMatch[0] : (file_path ? `для файла: ${file_path}` : '');

        if (resolvedStatus === 'error') {
            if (typeof duration_ms === 'number') {
                const seconds = (duration_ms / 1000).toFixed(1);
                statusSpan.textContent = `Ошибка ${fileName} (${seconds}s)`;
            } else {
                statusSpan.textContent = `Ошибка ${fileName}`;
            }
            statusSpan.style.color = '#ef4444';
        } else {
            if (typeof duration_ms === 'number') {
                const seconds = (duration_ms / 1000).toFixed(1);
                statusSpan.textContent = `Завершено ${fileName} (${seconds}s)`;
            } else {
                statusSpan.textContent = `Завершено ${fileName}`;
            }
            statusSpan.style.color = '#10b981';
        }
    }
        }

        // Add Diff button icon next to terminal icon if completed successfully
        if (branch_name && file_path && resolvedStatus !== 'error') {
    const leftSection = headerElement.querySelector('div');
    if (leftSection && !leftSection.querySelector('.diff-icon-button')) {
        const diffIcon = document.createElement('button');
        diffIcon.className = 'diff-icon-button';
        diffIcon.innerHTML = this.createSVGIcon('document', 16, 'rgba(3, 7, 61, 0.6)');
        diffIcon.title = 'Просмотр Diff';
        diffIcon.style.background = 'none';
        diffIcon.style.border = 'none';
        diffIcon.style.cursor = 'pointer';
        diffIcon.style.padding = '4px 8px';
        diffIcon.style.display = 'flex';
        diffIcon.style.alignItems = 'center';
        diffIcon.style.opacity = '0.7';
        diffIcon.style.transition = 'opacity 0.2s';
        diffIcon.onmouseover = () => diffIcon.style.opacity = '1';
        diffIcon.onmouseout = () => diffIcon.style.opacity = '0.7';
        diffIcon.onclick = (e) => {
            e.stopPropagation(); // Prevent collapse/expand
            this.client.fileBrowser.showDiffViewer(file_path, branch_name);
        };

        // Insert before terminal button
        const terminalButton = leftSection.querySelector('.terminal-icon-button');
        if (terminalButton) {
            leftSection.insertBefore(diffIcon, terminalButton);
        } else {
            leftSection.appendChild(diffIcon);
        }
    }
        }

        // Add error details as nested event if error occurred
        if (resolvedStatus === 'error' && (error_message || recent_output)) {
    const esc = (str) => (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const base = error_message || message || 'Ошибка выполнения';
    let snippetHtml = '';
    if (recent_output) {
        snippetHtml = `<pre style="margin:8px 0 0; padding:8px; background:#fff5f5; border:1px solid #f44336; border-radius:6px; white-space:pre-wrap; font-size:13px; max-height:300px; overflow:auto;">${esc(recent_output)}</pre>`;
    }
    const meta = (typeof bytes_read === 'number') ? `<div style="font-size:11px; color:#999; margin-top:4px;">bytes: ${bytes_read}</div>` : '';
    const errorContent = `${esc(base)}${snippetHtml}${meta}`;

    this.addNestedEventToExecution(execution_id, 'error', errorContent, this.formatTime(new Date().toISOString()));
        }

        // Update preview with return_content (for collapsed state)
        if (return_content && return_content.trim()) {
    const previewDiv = messageElement.querySelector('.execution-preview');
    if (previewDiv) {
        const preview = return_content.length > 150 ? return_content.substring(0, 150) + '...' : return_content;
        previewDiv.textContent = preview;
        previewDiv.style.display = 'none'; // Show when collapsed
    }

    // Add full return content as nested event
    const returnIcon = this.createSVGIcon('arrowUp', 14, '#4a54f5');
    const renderedReturn = this.renderMarkdownish(return_content);
    const returnContent = `
        <div style="font-weight: 500; margin-bottom: 6px; display: flex; align-items: center; gap: 6px; color: #ffffff;">
            ${returnIcon}
            <span style="color: #4a54f5;">Результат выполнения</span>
        </div>
        <div style="font-family: 'Monaco', 'Menlo', monospace; font-size: 11px; color: #ffffff; white-space: pre-wrap; max-height: 400px; overflow-y: auto; margin: 0; background: #25292e; padding: 8px; border-radius: 12px; border: 1px solid #525b67;">${renderedReturn}</div>
    `;
    this.addNestedEventToExecution(execution_id, 'return_value', returnContent, this.formatTime(new Date().toISOString()));
        }

        console.log('Updated execution message:', execution_id);
    }

    // Fallback creator when completion arrives before running message was rendered
    addExecutionCompletedFallback({ execution_id, return_content, message, branch_name, file_path, duration_ms, status, error_message, recent_output, bytes_read }) {
        const resolvedStatus = status === 'error' ? 'error' : 'success';
        const resolvedMessage = message || 'Выполнение завершено';
        const wrapper = document.createElement('div');
        wrapper.className = 'message execution';
        wrapper.setAttribute('data-execution-id', execution_id);

        // Header
        const headerDiv = document.createElement('div');
        headerDiv.className = 'message-header';
        headerDiv.style.display = 'flex';
        headerDiv.style.justifyContent = 'space-between';
        headerDiv.style.alignItems = 'center';
        headerDiv.style.marginBottom = '8px';
        headerDiv.style.fontSize = '12px';
        headerDiv.style.color = '#666';

        const statusSpan = document.createElement('span');
        statusSpan.className = 'exec-status-text';
        if (resolvedStatus === 'error') {
    if (typeof duration_ms === 'number') {
        const seconds = (duration_ms / 1000).toFixed(1);
        statusSpan.textContent = `Ошибка (${seconds}s)`;
    } else {
        statusSpan.textContent = 'Ошибка';
    }
        } else {
    if (typeof duration_ms === 'number') {
        const seconds = (duration_ms / 1000).toFixed(1);
        statusSpan.textContent = `Готово (${seconds}s)`;
    } else {
        statusSpan.textContent = 'Готово';
    }
        }

        const timeSpan = document.createElement('span');
        timeSpan.className = 'message-time';
        timeSpan.textContent = this.formatTime(new Date().toISOString());

        // Add status icon
        const checkSpan = document.createElement('span');
        checkSpan.className = 'completion-status';
        if (resolvedStatus === 'error') {
    checkSpan.textContent = '❌';
    checkSpan.style.color = '#c62828';
        } else {
    checkSpan.textContent = '✅';
    checkSpan.style.color = '#2e8b57';
        }
        checkSpan.style.marginRight = '6px';
        headerDiv.appendChild(checkSpan);
        headerDiv.appendChild(statusSpan);
        headerDiv.appendChild(timeSpan);

        const bodyDiv = document.createElement('div');
        bodyDiv.className = 'message-body';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        const esc = (str) => (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        if (resolvedStatus === 'error') {
    const base = error_message || message || 'Ошибка выполнения';
    let snippetHtml = '';
    if (recent_output) {
        snippetHtml = `<pre style="margin:8px 0 0; padding:8px; background:#fff5f5; border:1px solid #f44336; border-radius:6px; white-space:pre-wrap; font-size:13px; max-height:300px; overflow:auto;">${esc(recent_output)}</pre>`;
    }
    const meta = (typeof bytes_read === 'number') ? `<div style="font-size:11px; color:#999; margin-top:4px;">bytes: ${bytes_read}</div>` : '';
    contentDiv.innerHTML = `${esc(base)}${snippetHtml}${meta}`;
        } else {
    contentDiv.textContent = resolvedMessage;
        }

        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        if (branch_name && file_path) {
    const diffButton = document.createElement('button');
    diffButton.className = 'action-button primary diff-button';
    diffButton.innerHTML = '📋 Просмотр Diff';
    diffButton.onclick = () => this.client.fileBrowser.showDiffViewer(file_path, branch_name);
    actionsDiv.appendChild(diffButton);
        }

        bodyDiv.appendChild(contentDiv);
        bodyDiv.appendChild(actionsDiv);
        wrapper.appendChild(headerDiv);
        wrapper.appendChild(bodyDiv);
        this.messagesContainer.appendChild(wrapper);
        this.scrollToBottom();
    }

    // ========== UTILITY METHODS ==========

    createSVGIcon(type, size = 16, color = 'currentColor') {
        const icons = {
    spinner: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="${color}" stroke-width="2" stroke-dasharray="15 85" stroke-linecap="round" class="spinner-circle"/></svg>`,
    check: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    x: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"><path d="M18 6L6 18M6 6l12 12" stroke="${color}" stroke-width="2" stroke-linecap="round"/></svg>`,
    terminal: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"><path d="M4 17l6-6-6-6M12 19h8" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    document: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    chevronDown: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"><path d="M6 9l6 6 6-6" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    tool: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    checkCircle: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="${color}" stroke-width="2"/><path d="M9 12l2 2 4-4" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    chart: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"><path d="M18 20V10M12 20V4M6 20v-6" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    trendingUp: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"><path d="M23 6l-9.5 9.5-5-5L1 18" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M17 6h6v6" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    arrowUp: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"><path d="M12 19V5M5 12l7-7 7 7" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    settings: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="3" stroke="${color}" stroke-width="2"/><path d="M12 1v6m0 6v6M5.6 5.6l4.2 4.2m4.2 4.2l4.2 4.2M1 12h6m6 0h6M5.6 18.4l4.2-4.2m4.2-4.2l4.2-4.2" stroke="${color}" stroke-width="2" stroke-linecap="round"/></svg>`
        };
        return icons[type] || '';
    }

    appendConversationEntry(type, content) {
        console.log('[DEBUG] appendConversationEntry called:', { type, contentLength: content?.length });
        if (!['user', 'assistant'].includes(type)) {
    console.log('[DEBUG] Skipping - type not user/assistant:', type);
    return;
        }
        const normalized = (content || '').trim();
        if (!normalized) {
    console.log('[DEBUG] Skipping - empty content');
    return;
        }
        const role = type === 'user' ? 'user' : 'assistant';
        const lastEntry = this.client.conversation[this.client.conversation.length - 1];
        if (lastEntry && lastEntry.role === role && lastEntry.content === normalized) {
    console.log('[DEBUG] Skipping - duplicate entry');
    return;
        }
        this.client.conversation.push({ role, content: normalized });
        console.log('[DEBUG] Added to conversation. Total entries:', this.client.conversation.length);
        if (this.client.conversation.length > 200) {
    this.client.conversation = this.client.conversation.slice(-200);
        }
    }

    showTyping(show) {
        this.client.typingIndicator.className = show ? 'typing-indicator show' : 'typing-indicator';
        this.client.sendButton.disabled = show;
        this.scrollToBottom();
    }

    scrollToBottom() {
        setTimeout(() => {
    this.client.messagesContainer.scrollTop = this.client.messagesContainer.scrollHeight;
        }, 100);
    }

    formatTime(isoString) {
        const date = new Date(isoString);
        return date.toLocaleTimeString('ru-RU', {
    hour: '2-digit',
    minute: '2-digit'
        });
    }
}
