/**
 * Diff Viewer - displays git diff in modal
 * Uses existing HTML modal elements for display
 */

export class DiffViewer {
    constructor(domElements) {
        // DOM references (passed from chat-client.js)
        this.modal = domElements.modal;
        this.content = domElements.content;
    }

    /**
     * Show diff viewer with file diff
     */
    async show(filePath, branchName) {
        try {
            const response = await fetch(`/api/chat/file-diff?filePath=${encodeURIComponent(filePath)}`);
            const data = await response.json();

            if (data.success) {
                this.content.textContent = data.diff;
                this.modal.style.display = 'block';
            } else {
                alert('Ошибка получения diff: ' + data.error);
            }
        } catch (error) {
            console.error('Error fetching diff:', error);
            alert('Ошибка соединения при получении diff');
        }
    }

    /**
     * Close diff viewer modal
     */
    close() {
        this.modal.style.display = 'none';
    }
}
