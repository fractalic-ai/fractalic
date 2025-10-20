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
        if (this.client.selectedFileDisplay) {
            this.client.selectedFileDisplay.textContent = `📄 ${name}`;
            this.client.selectedFileDisplay.title = path; // tooltip with full path
            this.client.selectedFileDisplay.classList.add('has-file');
        }
        if (this.client.clearSelectionButton) {
            this.client.clearSelectionButton.style.display = 'block';
        }
        this.updateInputPlaceholder();
        this.closeFileBrowser();
        if (this.client.sendButton) {
            this.client.sendButton.disabled = false;
        }
        if (this.client.chatInput) {
            this.client.chatInput.disabled = false;
        }
    }

    clearFileSelection() {
        this.client.selectedFile = null;
        if (this.client.selectedFileDisplay) {
            this.client.selectedFileDisplay.textContent = 'Файл не выбран';
            this.client.selectedFileDisplay.removeAttribute('title');
            this.client.selectedFileDisplay.classList.remove('has-file');
        }
        if (this.client.clearSelectionButton) {
            this.client.clearSelectionButton.style.display = 'none';
        }
        this.updateInputPlaceholder();
        if (this.client.sendButton) {
            this.client.sendButton.disabled = true;
        }
        if (this.client.chatInput) {
            this.client.chatInput.disabled = true;
        }
    }

    updateInputPlaceholder() {
        if (this.client.chatInput) {
            this.client.chatInput.placeholder = this.client.selectedFile
                ? 'Введите запрос и нажмите Отправить'
                : 'Сначала выберите markdown файл выше';
        }
    }
}
