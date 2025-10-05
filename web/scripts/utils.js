/**
 * Utility functions for Fractalic Chat Client
 */

/**
 * ANSI Code Parser for rendering colored text from Python Rich
 * @param {string} text - Text with ANSI codes
 * @returns {string} HTML with styled spans
 */
export function parseAnsiToHtml(text) {
    const ansiRegex = /\x1b\[([0-9;]*)m/g;
    let result = '';
    let lastIndex = 0;
    let currentClasses = [];

    const colorMap = {
        '30': 'ansi-black', '31': 'ansi-red', '32': 'ansi-green', '33': 'ansi-yellow',
        '34': 'ansi-blue', '35': 'ansi-magenta', '36': 'ansi-cyan', '37': 'ansi-white',
        '90': 'ansi-bright-black', '91': 'ansi-bright-red', '92': 'ansi-bright-green',
        '93': 'ansi-bright-yellow', '94': 'ansi-bright-blue', '95': 'ansi-bright-magenta',
        '96': 'ansi-bright-cyan', '97': 'ansi-bright-white',
        '40': 'ansi-bg-black', '41': 'ansi-bg-red', '42': 'ansi-bg-green', '43': 'ansi-bg-yellow',
        '44': 'ansi-bg-blue', '45': 'ansi-bg-magenta', '46': 'ansi-bg-cyan', '47': 'ansi-bg-white',
        '1': 'ansi-bold', '2': 'ansi-dim', '3': 'ansi-italic', '4': 'ansi-underline'
    };

    let match;
    while ((match = ansiRegex.exec(text)) !== null) {
        // Add text before ANSI code
        if (match.index > lastIndex) {
            const textSegment = text.substring(lastIndex, match.index);
            if (currentClasses.length > 0) {
                result += `<span class="${currentClasses.join(' ')}">${escapeHtml(textSegment)}</span>`;
            } else {
                result += escapeHtml(textSegment);
            }
        }

        // Process ANSI code
        const codes = match[1].split(';');
        for (const code of codes) {
            if (code === '0' || code === '') {
                // Reset
                currentClasses = [];
            } else if (colorMap[code]) {
                // Remove conflicting classes
                if (code.startsWith('3') || code.startsWith('9')) {
                    currentClasses = currentClasses.filter(c => !c.startsWith('ansi-') || c.startsWith('ansi-bg-') || c.startsWith('ansi-bold') || c.startsWith('ansi-dim') || c.startsWith('ansi-italic') || c.startsWith('ansi-underline'));
                }
                if (code.startsWith('4') && code.length === 2) {
                    currentClasses = currentClasses.filter(c => !c.startsWith('ansi-bg-'));
                }
                currentClasses.push(colorMap[code]);
            }
        }

        lastIndex = ansiRegex.lastIndex;
    }

    // Add remaining text
    if (lastIndex < text.length) {
        const textSegment = text.substring(lastIndex);
        if (currentClasses.length > 0) {
            result += `<span class="${currentClasses.join(' ')}">${escapeHtml(textSegment)}</span>`;
        } else {
            result += escapeHtml(textSegment);
        }
    }

    return result;
}

/**
 * Escape HTML special characters
 * @param {string} text - Text to escape
 * @returns {string} Escaped HTML
 */
export function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Format ISO timestamp to HH:MM
 * @param {string} isoString - ISO 8601 timestamp
 * @returns {string} Formatted time
 */
export function formatTime(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return '';
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    return `${hours}:${minutes}`;
}

/**
 * Format file size in bytes to human readable string
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted size
 */
export function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Create SVG icon element
 * @param {string} type - Icon type (check, error, tool, etc)
 * @param {number} size - Icon size in pixels
 * @param {string} color - Icon color
 * @returns {string} SVG HTML string
 */
export function createSVGIcon(type, size = 16, color = 'currentColor') {
    const icons = {
        'check': `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>`,
        'checkCircle': `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`,
        'x': `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`,
        'error': `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`,
        'tool': `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path></svg>`,
        'document': `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>`,
        'terminal': `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2"><polyline points="4 17 10 11 4 5"></polyline><line x1="12" y1="19" x2="20" y2="19"></line></svg>`,
        'chart': `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>`,
        'spinner': `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2" class="spinner"><circle cx="12" cy="12" r="10" opacity="0.25"></circle><path d="M12 2a10 10 0 0 1 10 10" opacity="0.75" class="spinner-circle"></path></svg>`
    };

    return icons[type] || icons['check'];
}
