/**
 * Terminal Viewer - manages terminal output buffers and display
 * Receives TERMINAL_OUTPUT events from stream and renders them
 */

export class TerminalViewer {
    constructor(domElements) {
        // DOM references (passed from chat-client.js)
        this.modal = domElements.modal;
        this.content = domElements.content;
        this.fileName = domElements.fileName;
        this.status = domElements.status;

        // State
        this.terminalLogs = new Map(); // executionId -> buffer
        this.utilsModule = null;
        this.currentViewExecutionId = null; // Track which execution is currently displayed
    }

    /**
     * Ensure terminal buffer exists for given execution ID
     */
    ensureBuffer(executionId) {
        if (!executionId) {
            return null;
        }
        if (!this.terminalLogs.has(executionId)) {
            this.terminalLogs.set(executionId, {
                seqMap: new Map(),
                order: [],
                fallback: [],
                sorted: true,
                dirty: false,
                cachedText: '',
                isComplete: false
            });
        }
        return this.terminalLogs.get(executionId);
    }

    /**
     * Append terminal chunk to buffer (called by chat-client when TERMINAL_OUTPUT event arrives)
     */
    handleTerminalOutput(event) {
        const executionId = event.execution_id;
        const data = event.data;
        const seq = event.seq;

        if (!executionId || !data) {
            return;
        }

        console.log(`[TerminalViewer] Received output for ${executionId.substring(0, 8)}, seq=${seq}, length=${data.length}`);

        const buffer = this.ensureBuffer(executionId);
        if (!buffer) return;

        const numericSeq = typeof seq === 'number' ? seq : Number(seq);
        if (Number.isFinite(numericSeq)) {
            if (buffer.seqMap.has(numericSeq)) {
                const existing = buffer.seqMap.get(numericSeq) || '';
                buffer.seqMap.set(numericSeq, existing + data);
            } else {
                buffer.seqMap.set(numericSeq, data);
                buffer.order.push(numericSeq);
                buffer.sorted = false;
            }
        } else {
            buffer.fallback.push(data);
        }
        buffer.dirty = true;

        // If this buffer is currently displayed, re-render
        if (this.currentViewExecutionId === executionId && this.modal.style.display === 'block') {
            this.renderBuffer(executionId);
        }
    }

    /**
     * Mark execution as complete (no more terminal output expected)
     */
    markComplete(executionId) {
        const buffer = this.terminalLogs.get(executionId);
        if (buffer) {
            buffer.isComplete = true;
            console.log(`[TerminalViewer] Marked ${executionId.substring(0, 8)} as complete`);

            // Update status if currently viewing this execution
            if (this.currentViewExecutionId === executionId && this.modal.style.display === 'block') {
                this.status.textContent = `Завершено`;
                this.status.className = 'terminal-status completed';
            }
        }
    }

    /**
     * Get full terminal buffer text
     */
    getBufferText(executionId) {
        if (!executionId || !this.terminalLogs.has(executionId)) {
            return '';
        }
        const buffer = this.terminalLogs.get(executionId);
        if (!buffer) {
            return '';
        }

        if (buffer.dirty) {
            if (!buffer.sorted) {
                buffer.order.sort((a, b) => a - b);
                buffer.sorted = true;
            }

            let combined = '';
            for (const seq of buffer.order) {
                combined += buffer.seqMap.get(seq) || '';
            }
            if (buffer.fallback.length) {
                combined += buffer.fallback.join('');
            }

            buffer.cachedText = combined;
            buffer.dirty = false;
        }

        return buffer.cachedText;
    }

    /**
     * Render buffer content to modal (with ANSI parsing)
     */
    async renderBuffer(executionId) {
        if (!this.utilsModule) {
            this.utilsModule = await import('./utils.js');
        }
        const parseAnsiToHtml = this.utilsModule.parseAnsiToHtml;

        const cachedRaw = this.getBufferText(executionId);
        this.content.innerHTML = cachedRaw ? parseAnsiToHtml(cachedRaw) : '<div style="color: #a0a8b2; padding: 20px;">No terminal output yet</div>';
        this.content.scrollTop = this.content.scrollHeight;
    }

    /**
     * Show terminal viewer and display buffer for given execution
     */
    async show(filePath, executionId, streamExecutionId = executionId) {
        this.currentViewExecutionId = streamExecutionId;

        // Load utils module if not loaded
        if (!this.utilsModule) {
            this.utilsModule = await import('./utils.js');
        }

        this.fileName.textContent = filePath;
        this.modal.style.display = 'block';

        // Check if buffer exists and render it
        const buffer = this.terminalLogs.get(streamExecutionId);
        if (buffer) {
            await this.renderBuffer(streamExecutionId);
            this.status.textContent = buffer.isComplete ? `Завершено` : `Выполняется: ${filePath}`;
            this.status.className = buffer.isComplete ? 'terminal-status completed' : 'terminal-status running';
        } else {
            this.content.innerHTML = '<div style="color: #a0a8b2; padding: 20px;">Waiting for terminal output...</div>';
            this.status.textContent = `Ожидание вывода...`;
            this.status.className = 'terminal-status running';
        }
    }

    /**
     * Close terminal viewer modal
     */
    close() {
        this.modal.style.display = 'none';
        this.currentViewExecutionId = null;
    }
}
