/**
 * Message List Component
 *
 * Displays a list of messages that can be updated over time.
 * Useful for showing logs, notifications, or structured data.
 */

import { BaseComponent } from './base-component.js?v=10';

export class MessageListComponent extends BaseComponent {
    constructor(componentId, options) {
        super(componentId, options);

        // Initialize state
        this.setState({
            messages: [],
            title: 'Messages',
            maxMessages: 100 // Limit to prevent memory issues
        }, false); // Don't render yet
    }

    /**
     * Component manifest declaring behavior
     */
    static getManifest() {
        return {
            events: ['message_added', 'messages_batch', 'message_list_update', 'msg_info', 'msg_success', 'msg_warning', 'msg_error', 'msg_debug'],
            singleton: false, // Can have multiple instances per execution
            mountPoint: 'chat', // Show inline with chat messages
            updateStrategy: 'append', // Add new messages to existing ones
            description: 'Displays a dynamically updating list of messages'
        };
    }

    /**
     * Handle incoming events
     */
    handleEvent(eventType, eventData) {
        console.log(`[MessageList] Received event: ${eventType}`, eventData);

        switch (eventType) {
            case 'message_added':
                // Single message event
                this._addMessage({
                    text: eventData.message || eventData.text || '',
                    type: eventData.type || 'info',
                    timestamp: Date.now()
                });
                break;

            case 'messages_batch':
                // Batch of messages
                if (Array.isArray(eventData.messages)) {
                    eventData.messages.forEach(msg => {
                        this._addMessage({
                            text: msg.text || msg.message || '',
                            type: msg.type || 'info',
                            timestamp: Date.now()
                        });
                    });
                }
                break;

            case 'message_list_update':
                // Generic list update with flexible structure
                if (eventData.messages) {
                    if (Array.isArray(eventData.messages)) {
                        eventData.messages.forEach(msg => this._addMessage(msg));
                    } else {
                        this._addMessage(eventData.messages);
                    }
                }
                if (eventData.title) {
                    this.setState({ title: eventData.title }, false);
                }
                if (eventData.clear) {
                    this.setState({ messages: [] }, false);
                }
                this.render();
                break;

            case 'msg_info':
            case 'msg_success':
            case 'msg_warning':
            case 'msg_error':
            case 'msg_debug':
                // Direct message events with type as event name
                // Extract actual type by removing 'msg_' prefix
                const messageType = eventType.replace('msg_', '');
                this._addMessage({
                    text: eventData.message || eventData.text || '',
                    type: messageType, // 'info', 'success', 'warning', 'error', 'debug'
                    timestamp: Date.now()
                });
                break;

            default:
                console.warn(`[MessageList] Unknown event type: ${eventType}`);
        }
    }

    /**
     * Add message to list
     * @private
     */
    _addMessage(messageData) {
        const currentMessages = this.getState().messages;
        const maxMessages = this.getState().maxMessages;

        // Add new message
        currentMessages.push({
            text: messageData.text || messageData.message || '',
            type: messageData.type || 'info',
            timestamp: messageData.timestamp || Date.now()
        });

        // Trim if exceeds max
        if (currentMessages.length > maxMessages) {
            currentMessages.shift(); // Remove oldest
        }

        this.setState({ messages: currentMessages }, true); // Trigger render
    }

    /**
     * Mount component and create initial structure
     */
    onMount() {
        this.container.classList.add('message-list-component');
        this.render();
    }

    /**
     * Render message list HTML
     */
    render() {
        const { messages, title } = this.getState();

        // Clear container
        this.container.innerHTML = '';

        // Create header
        const header = this.createElement('div', {
            classes: ['message-list-header'],
            innerHTML: `<h4>${title}</h4><span class="message-count">${messages.length} message(s)</span>`
        });
        this.container.appendChild(header);

        // Create message container
        const messageContainer = this.createElement('div', {
            classes: ['message-list-container']
        });

        // Add messages
        messages.forEach((message, index) => {
            const messageEl = this._createMessageElement(message, index);
            messageContainer.appendChild(messageEl);
        });

        this.container.appendChild(messageContainer);

        // Show empty state if no messages
        if (messages.length === 0) {
            const emptyState = this.createElement('div', {
                classes: ['message-list-empty'],
                textContent: 'No messages yet.'
            });
            messageContainer.appendChild(emptyState);
        }

        // Auto-scroll to bottom
        messageContainer.scrollTop = messageContainer.scrollHeight;
    }

    /**
     * Create message element
     * @private
     */
    _createMessageElement(message, index) {
        const messageEl = this.createElement('div', {
            classes: ['message-item', `message-type-${message.type}`]
        });

        // Message icon based on type
        const icon = this._getIconForType(message.type);
        const iconEl = this.createElement('span', {
            classes: ['message-icon'],
            innerHTML: icon
        });
        messageEl.appendChild(iconEl);

        // Message content
        const contentEl = this.createElement('div', {
            classes: ['message-content']
        });

        const textEl = this.createElement('div', {
            classes: ['message-text'],
            textContent: message.text
        });
        contentEl.appendChild(textEl);

        // Timestamp
        const timeEl = this.createElement('div', {
            classes: ['message-timestamp'],
            textContent: this._formatTimestamp(message.timestamp)
        });
        contentEl.appendChild(timeEl);

        messageEl.appendChild(contentEl);

        return messageEl;
    }

    /**
     * Get icon for message type
     * @private
     */
    _getIconForType(type) {
        const icons = {
            info: 'ℹ️',
            success: '✅',
            warning: '⚠️',
            error: '❌',
            debug: '🔍'
        };
        return icons[type] || icons.info;
    }

    /**
     * Format timestamp
     * @private
     */
    _formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString();
    }
}
