/**
 * Base Component Class
 *
 * Abstract base class for all UI components in the Fractalic component system.
 * Provides lifecycle hooks, event handling, and DOM management.
 */

export class BaseComponent {
    /**
     * @param {string} componentId - Unique identifier for this component instance
     * @param {Object} options - Configuration options
     * @param {string} options.mountPoint - 'chat' or 'artifacts'
     * @param {string} options.executionId - Execution context ID
     */
    constructor(componentId, options = {}) {
        this.componentId = componentId;
        this.mountPoint = options.mountPoint || 'chat';
        this.executionId = options.executionId || null;

        // DOM element that contains this component
        this.container = null;

        // Internal state storage
        this.state = {};

        // Event listeners registry for cleanup
        this.eventListeners = [];
    }

    /**
     * Create and mount the component to the DOM
     * Override this method to create custom initial HTML structure
     *
     * @returns {HTMLElement} The created container element
     */
    mount() {
        if (this.container) {
            console.warn(`[BaseComponent] Component ${this.componentId} already mounted`);
            return this.container;
        }

        // Create container element
        this.container = document.createElement('div');
        this.container.classList.add('fractalic-component');
        this.container.setAttribute('data-component-id', this.componentId);
        this.container.setAttribute('data-component-type', this.constructor.name);

        // Add control buttons if in artifacts panel
        if (this.mountPoint === 'artifacts') {
            this._addControlButtons();
            this._setupDragAndDrop();
        }

        // Call lifecycle hook
        this.onMount();

        console.log(`[BaseComponent] Mounted ${this.constructor.name} with id ${this.componentId}`);
        return this.container;
    }

    /**
     * Unmount and cleanup the component
     */
    unmount() {
        // Call lifecycle hook
        this.onUnmount();

        // Remove all event listeners
        this.eventListeners.forEach(({ element, event, handler }) => {
            element.removeEventListener(event, handler);
        });
        this.eventListeners = [];

        // Remove from DOM
        if (this.container && this.container.parentNode) {
            this.container.parentNode.removeChild(this.container);
        }

        this.container = null;
        console.log(`[BaseComponent] Unmounted ${this.constructor.name} with id ${this.componentId}`);
    }

    /**
     * Handle incoming event data
     * Override this method in subclasses to process specific events
     *
     * @param {string} eventType - Type of the event
     * @param {Object} eventData - Event payload data
     */
    handleEvent(eventType, eventData) {
        console.log(`[BaseComponent] ${this.constructor.name} received event: ${eventType}`, eventData);
        // Override in subclass
    }

    /**
     * Update component state and optionally trigger re-render
     *
     * @param {Object} newState - State updates to merge
     * @param {boolean} shouldRender - Whether to call render() after update (default: true)
     */
    setState(newState, shouldRender = true) {
        this.state = { ...this.state, ...newState };
        if (shouldRender) {
            this.render();
        }
    }

    /**
     * Get current component state
     *
     * @returns {Object} Current state object
     */
    getState() {
        return { ...this.state };
    }

    /**
     * Render/update the component's HTML
     * Override this method to define how your component displays data
     */
    render() {
        // Override in subclass
        console.log(`[BaseComponent] ${this.constructor.name}.render() called (override this method)`);
    }

    /**
     * Lifecycle hook: Called after component is mounted
     * Override to perform initialization logic
     */
    onMount() {
        // Override in subclass if needed
    }

    /**
     * Lifecycle hook: Called before component is unmounted
     * Override to perform cleanup logic
     */
    onUnmount() {
        // Override in subclass if needed
    }

    /**
     * Helper: Add event listener and track it for cleanup
     *
     * @param {HTMLElement} element - DOM element to attach listener to
     * @param {string} event - Event type (e.g., 'click')
     * @param {Function} handler - Event handler function
     */
    addEventListener(element, event, handler) {
        element.addEventListener(event, handler);
        this.eventListeners.push({ element, event, handler });
    }

    /**
     * Helper: Create HTML element with classes and attributes
     *
     * @param {string} tag - HTML tag name
     * @param {Object} options - Configuration options
     * @param {Array<string>} options.classes - CSS classes to add
     * @param {Object} options.attributes - Attributes to set
     * @param {string} options.textContent - Text content
     * @param {string} options.innerHTML - HTML content (careful with XSS!)
     * @returns {HTMLElement} Created element
     */
    createElement(tag, options = {}) {
        const element = document.createElement(tag);

        if (options.classes) {
            element.classList.add(...options.classes);
        }

        if (options.attributes) {
            Object.entries(options.attributes).forEach(([key, value]) => {
                element.setAttribute(key, value);
            });
        }

        if (options.textContent) {
            element.textContent = options.textContent;
        }

        if (options.innerHTML) {
            element.innerHTML = options.innerHTML;
        }

        return element;
    }

    /**
     * Add close and drag buttons to component
     * @private
     */
    _addControlButtons() {
        // Drag handle (⋮⋮)
        const dragHandle = this.createElement('div', {
            classes: ['component-drag-handle'],
            innerHTML: '⋮⋮',
            attributes: {
                'title': 'Drag to reorder',
                'draggable': 'false'
            }
        });
        this.container.appendChild(dragHandle);

        // Close button
        const closeBtn = this.createElement('button', {
            classes: ['component-close-btn'],
            innerHTML: '×',
            attributes: {
                'title': 'Close component'
            }
        });

        this.addEventListener(closeBtn, 'click', (e) => {
            e.stopPropagation();
            this._handleClose();
        });

        this.container.appendChild(closeBtn);
    }

    /**
     * Setup drag and drop functionality
     * @private
     */
    _setupDragAndDrop() {
        this.container.setAttribute('draggable', 'true');

        this.addEventListener(this.container, 'dragstart', (e) => {
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/html', this.componentId);
            this.container.classList.add('dragging');
        });

        this.addEventListener(this.container, 'dragend', (e) => {
            this.container.classList.remove('dragging');
        });

        this.addEventListener(this.container, 'dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';

            const draggingComponent = document.querySelector('.dragging');
            if (draggingComponent && draggingComponent !== this.container) {
                const rect = this.container.getBoundingClientRect();
                const midpoint = rect.left + rect.width / 2;

                if (e.clientX < midpoint) {
                    this.container.parentNode.insertBefore(draggingComponent, this.container);
                } else {
                    this.container.parentNode.insertBefore(draggingComponent, this.container.nextSibling);
                }
            }
        });
    }

    /**
     * Handle component close
     * @private
     */
    _handleClose() {
        // Animate out
        this.container.style.opacity = '0';
        this.container.style.transform = 'scale(0.95)';

        setTimeout(() => {
            this.unmount();
        }, 200);
    }

    /**
     * Get the manifest for this component class
     * Must be implemented by subclasses
     *
     * @returns {Object} Component manifest
     * @static
     */
    static getManifest() {
        throw new Error('Component must implement static getManifest() method');
    }
}
