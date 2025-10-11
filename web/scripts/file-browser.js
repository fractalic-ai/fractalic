/**
 * File Browser Module - handles file selection and directory navigation
 */

export class FileBrowser {
    constructor(client) {
        this.client = client;
        this.currentBrowserPath = '.';
    }

    async openFileBrowser() {
        this.currentBrowserPath = '.';
        await this.loadDirectoryContents('.');
        this.client.fileBrowserModal.style.display = 'block';
    }

    closeFileBrowser() {
        this.client.fileBrowserModal.style.display = 'none';
    }

    async loadDirectoryContents(path) {
        try {
            const response = await fetch(`/list_directory/?path=${encodeURIComponent(path)}`);
            const items = await response.json();

            if (Array.isArray(items)) {
                this.currentBrowserPath = path || '.';
                this.client.currentPathDisplay.textContent = path === '' || path === '.' ? '/' : '/' + path;
                this.renderFileList(items);
            } else {
                console.error('Failed to load directory: Invalid response format');
                alert('Ошибка загрузки папки: неверный формат ответа');
            }
        } catch (error) {
            console.error('Error loading directory:', error);
            alert('Ошибка соединения');
        }
    }

    renderFileList(items) {
        this.client.fileList.innerHTML = '';

        items.forEach(item => {
            // Показывать только папки и .md файлы
            if (!item.is_dir && !item.name.toLowerCase().endsWith('.md')) {
                return;
            }

            const itemElement = document.createElement('div');
            const itemType = item.is_dir ? 'directory' : 'file';
            itemElement.className = `file-item ${itemType}`;

            if (item.is_dir) {
                itemElement.innerHTML = `
                    <span class="file-icon">📁</span>
                    <div class="file-info">
                        <div class="file-name">${item.name}</div>
                    </div>
                `;
                itemElement.addEventListener('click', () => {
                    this.loadDirectoryContents(item.path);
                });
            } else {
                itemElement.innerHTML = `
                    <span class="file-icon">📝</span>
                    <div class="file-info">
                        <div class="file-name">${item.name}</div>
                    </div>
                `;
                itemElement.addEventListener('click', () => {
                    this.selectFile(item.path, item.name);
                });
            }

            this.client.fileList.appendChild(itemElement);
        });
    }

    selectFile(path, name) {
        this.client.selectedFile = path;
        this.client.selectedFileDisplay.textContent = `📄 ${name}`;
        this.client.selectedFileDisplay.title = path; // tooltip with full path
        this.client.selectedFileDisplay.classList.add('has-file');
        this.client.clearSelectionButton.style.display = 'block';
        this.updateInputPlaceholder();
        this.closeFileBrowser();
        this.client.sendButton.disabled = false;
        this.client.chatInput.disabled = false;
    }

    clearFileSelection() {
        this.client.selectedFile = null;
        this.client.selectedFileDisplay.textContent = 'Файл не выбран';
        this.client.selectedFileDisplay.removeAttribute('title');
        this.client.selectedFileDisplay.classList.remove('has-file');
        this.client.clearSelectionButton.style.display = 'none';
        this.updateInputPlaceholder();
        this.client.sendButton.disabled = true;
        this.client.chatInput.disabled = true;
    }

    updateInputPlaceholder() {
        this.client.chatInput.placeholder = this.client.selectedFile
            ? 'Введите запрос и нажмите Отправить'
            : 'Сначала выберите markdown файл выше';
    }

    // Diff Viewer
    closeDiffViewer() {
        this.client.diffViewerModal.style.display = 'none';
    }

    async showDiffViewer(filePath, branchName) {
        try {
            const response = await fetch(`/api/chat/file-diff?filePath=${encodeURIComponent(filePath)}`);
            const data = await response.json();

            if (data.success) {
                this.client.diffContent.textContent = data.diff;
                this.client.diffViewerModal.style.display = 'block';
            } else {
                alert('Ошибка получения diff: ' + data.error);
            }
        } catch (error) {
            console.error('Error fetching diff:', error);
            alert('Ошибка соединения при получении diff');
        }
    }

    // Terminal Viewer
    closeTerminalViewer() {
        this.client.terminalViewerModal.style.display = 'none';
        // Остановить все активные стримы
        this.client.activeStreams.forEach((controller) => {
            controller.abort();
        });
        this.client.activeStreams.clear();
    }

    async showTerminalViewer(filePath, executionId, streamExecutionId = executionId, isExecutionCompleted = false, startOffsetRaw = 0, endOffsetRaw = null) {
        let parseAnsiToHtml;
        let escapeHtml;
        try {
            if (!this.client.terminalLogs) {
                this.client.terminalLogs = new Map();
            }
            let logEntry = this.client.terminalLogs.get(streamExecutionId);
            if (!logEntry || typeof logEntry !== 'object' || typeof logEntry.raw !== 'string') {
                const raw = typeof logEntry === 'string' ? logEntry : '';
                logEntry = { raw };
                this.client.terminalLogs.set(streamExecutionId, logEntry);
            }

            ({ parseAnsiToHtml, escapeHtml } = await import('./utils.js'));

            this.client.terminalFileName.textContent = filePath;
            const cachedRaw = logEntry.raw || '';
            const bubbleRefs = this.client.executionBubbles?.get(executionId);

            let startOffset = typeof startOffsetRaw === 'number' ? startOffsetRaw : 0;
            if (bubbleRefs && typeof bubbleRefs.terminalStartOffset === 'number') {
                startOffset = bubbleRefs.terminalStartOffset;
            }

            let endOffset = typeof endOffsetRaw === 'number' ? endOffsetRaw : null;
            if (bubbleRefs && typeof bubbleRefs.terminalEndOffset === 'number' && bubbleRefs.terminalEndOffset > 0) {
                endOffset = bubbleRefs.terminalEndOffset;
            }

            const marker = executionId !== streamExecutionId ? `nested_exec_id=${executionId}` : null;

            const updateOffsetsAfterData = () => {
                const logRaw = logEntry.raw || '';
                const previousStart = startOffset;

                if (marker) {
                    if (startOffset <= 0 || startOffset > logRaw.length) {
                        const markerIndex = logRaw.indexOf(marker);
                        if (markerIndex !== -1) {
                            startOffset = markerIndex;
                            if (bubbleRefs) {
                                bubbleRefs.terminalStartOffset = markerIndex;
                            }
                        }
                    }
                } else {
                    startOffset = 0;
                    if (bubbleRefs) {
                        bubbleRefs.terminalStartOffset = 0;
                    }
                }

                if (startOffset < 0) startOffset = 0;
                if (startOffset > logRaw.length) startOffset = logRaw.length;

                let resolvedEnd = endOffset;
                if (bubbleRefs && typeof bubbleRefs.terminalEndOffset === 'number' && bubbleRefs.terminalEndOffset > startOffset) {
                    resolvedEnd = bubbleRefs.terminalEndOffset;
                } else if (typeof endOffsetRaw === 'number' && endOffsetRaw > startOffset) {
                    resolvedEnd = endOffsetRaw;
                } else if (bubbleRefs?.isCompleted) {
                    resolvedEnd = logRaw.length;
                    if (bubbleRefs) {
                        bubbleRefs.terminalEndOffset = resolvedEnd;
                    }
                } else {
                    resolvedEnd = null;
                }

                if (resolvedEnd !== null && resolvedEnd < startOffset) {
                    resolvedEnd = null;
                }

                endOffset = resolvedEnd;
                return { startChanged: startOffset !== previousStart };
            };

            const initialOffsets = updateOffsetsAfterData();

            const baseSliceEnd = endOffset !== null ? Math.min(endOffset, cachedRaw.length) : undefined;
            let baseRaw = cachedRaw.slice(startOffset, baseSliceEnd);
            this.client.terminalContent.innerHTML = parseAnsiToHtml(baseRaw);
            this.client.terminalViewerModal.style.display = 'block';

            if (initialOffsets.startChanged) {
                // start offset adjusted by marker; refresh base view
                baseRaw = cachedRaw.slice(startOffset, baseSliceEnd);
                this.client.terminalContent.innerHTML = parseAnsiToHtml(baseRaw);
            }

            const hasFullData = endOffset !== null && cachedRaw.length >= endOffset;
            if (hasFullData) {
                this.client.terminalStatus.textContent = `Завершено: ${filePath}`;
                this.client.terminalStatus.className = 'terminal-status completed';
                return;
            }

            this.client.terminalStatus.textContent = 'Подключение к терминалу...';
            this.client.terminalStatus.className = 'terminal-status running';

            // Создаем AbortController для этого стрима
            const controller = new AbortController();
            this.client.activeStreams.set(executionId, controller);

            const response = await fetch(`/api/chat/terminal-stream/${encodeURIComponent(streamExecutionId)}`, {
                signal: controller.signal
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let displayedLength = baseRaw.length;

            this.client.terminalStatus.textContent = `Выполняется: ${filePath}`;

            while (true) {
                const { done, value } = await reader.read();

                if (done) {
                    this.client.terminalStatus.textContent = `Завершено: ${filePath}`;
                    this.client.terminalStatus.className = 'terminal-status completed';
                    this.client.terminalLogs.set(streamExecutionId, logEntry);
                    break;
                }

                const chunk = decoder.decode(value, { stream: true });
                if (!chunk) continue;

                logEntry.raw += chunk;
                this.client.terminalLogs.set(streamExecutionId, logEntry);

                const { startChanged } = updateOffsetsAfterData();

                const currentSliceEnd = endOffset !== null ? Math.min(endOffset, logEntry.raw.length) : undefined;
                const viewRaw = logEntry.raw.slice(startOffset, currentSliceEnd);

                if (startChanged) {
                    this.client.terminalContent.innerHTML = parseAnsiToHtml(viewRaw);
                    displayedLength = viewRaw.length;
                    this.client.terminalContent.scrollTop = this.client.terminalContent.scrollHeight;
                    if (endOffset !== null && logEntry.raw.length >= endOffset) {
                        this.client.terminalStatus.textContent = `Завершено: ${filePath}`;
                        this.client.terminalStatus.className = 'terminal-status completed';
                        this.client.terminalLogs.set(streamExecutionId, logEntry);
                        break;
                    }
                    continue;
                }

                const newPortion = viewRaw.slice(displayedLength);
                if (newPortion.length > 0) {
                    const htmlChunk = parseAnsiToHtml(newPortion);
                    this.client.terminalContent.innerHTML += htmlChunk;
                    displayedLength += newPortion.length;

                    this.client.terminalContent.scrollTop = this.client.terminalContent.scrollHeight;
                }

                if (endOffset !== null && logEntry.raw.length >= endOffset) {
                    this.client.terminalStatus.textContent = `Завершено: ${filePath}`;
                    this.client.terminalStatus.className = 'terminal-status completed';
                    this.client.terminalLogs.set(streamExecutionId, logEntry);
                    break;
                }
            }

        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Terminal stream aborted');
            } else {
                console.error('Error fetching terminal stream:', error);
                const safeEscape = escapeHtml || (await import('./utils.js')).escapeHtml;
                this.client.terminalContent.innerHTML += `\n<span class="ansi-red">[Ошибка соединения: ${safeEscape(error.message)}]</span>`;
                this.client.terminalStatus.textContent = 'Ошибка соединения';
                this.client.terminalStatus.className = 'terminal-status completed';
                const existingLog = this.client.terminalLogs.get(streamExecutionId) || { raw: '' };
                existingLog.raw += `\n[Ошибка соединения: ${error.message}]`;
                this.client.terminalLogs.set(streamExecutionId, existingLog);
            }
        } finally {
            // Убираем из активных стримов
            this.client.activeStreams.delete(executionId);
        }
    }
}
