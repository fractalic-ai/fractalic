/**
 * Main entry point - initializes Fractalic Chat Client
 */

import { FractalicChatClient } from './chat-client.js?v=16';

// Initialize chat client when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new FractalicChatClient();
});
