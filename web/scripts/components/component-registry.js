/**
 * Component Registry
 *
 * Manages registration and retrieval of UI components with their manifests.
 * Components declare which events they handle and how they should be instantiated.
 */

export class ComponentRegistry {
    constructor() {
        // Map of component names to their class constructors
        this.components = new Map();

        // Map of event types to component names that handle them
        this.eventHandlers = new Map();
    }

    /**
     * Register a component class with its manifest
     *
     * @param {string} name - Unique component identifier
     * @param {Class} componentClass - Component class constructor
     * @param {Object} manifest - Component manifest defining behavior
     * @param {Array<string>} manifest.events - Event types this component handles
     * @param {boolean} manifest.singleton - If true, only one instance exists (default: false)
     * @param {string} manifest.mountPoint - Where to mount: 'chat' or 'artifacts' (default: 'chat')
     * @param {string} manifest.updateStrategy - 'replace' or 'append' (default: 'replace')
     *
     * @example
     * registry.register('image-gallery', ImageGalleryComponent, {
     *   events: ['image_generated', 'images_batch'],
     *   singleton: true,
     *   mountPoint: 'artifacts',
     *   updateStrategy: 'append'
     * });
     */
    register(name, componentClass, manifest) {
        // Validate manifest
        if (!manifest.events || !Array.isArray(manifest.events)) {
            throw new Error(`Component ${name} must declare events array in manifest`);
        }

        // Store component with its manifest
        this.components.set(name, {
            name,
            componentClass,
            manifest: {
                events: manifest.events,
                singleton: manifest.singleton || false,
                mountPoint: manifest.mountPoint || 'chat',
                updateStrategy: manifest.updateStrategy || 'replace',
                description: manifest.description || ''
            }
        });

        // Map events to this component
        manifest.events.forEach(eventType => {
            if (!this.eventHandlers.has(eventType)) {
                this.eventHandlers.set(eventType, []);
            }
            this.eventHandlers.get(eventType).push(name);
        });

        console.log(`[ComponentRegistry] Registered component: ${name}`, manifest);
    }

    /**
     * Get component info by name
     *
     * @param {string} name - Component name
     * @returns {Object|null} Component registration info or null
     */
    getComponent(name) {
        return this.components.get(name) || null;
    }

    /**
     * Find components that handle a specific event type
     *
     * @param {string} eventType - Event type string
     * @returns {Array<Object>} Array of component registration objects
     */
    getComponentsForEvent(eventType) {
        const componentNames = this.eventHandlers.get(eventType) || [];
        return componentNames
            .map(name => this.components.get(name))
            .filter(comp => comp !== undefined);
    }

    /**
     * Check if any component handles this event type
     *
     * @param {string} eventType - Event type string
     * @returns {boolean} True if at least one component handles this event
     */
    hasHandlerForEvent(eventType) {
        return this.eventHandlers.has(eventType) &&
               this.eventHandlers.get(eventType).length > 0;
    }

    /**
     * Get all registered components
     *
     * @returns {Array<Object>} Array of all component registrations
     */
    getAllComponents() {
        return Array.from(this.components.values());
    }

    /**
     * Unregister a component (useful for hot reloading during development)
     *
     * @param {string} name - Component name to unregister
     */
    unregister(name) {
        const component = this.components.get(name);
        if (!component) return;

        // Remove from event handlers map
        component.manifest.events.forEach(eventType => {
            const handlers = this.eventHandlers.get(eventType) || [];
            const filtered = handlers.filter(h => h !== name);
            if (filtered.length > 0) {
                this.eventHandlers.set(eventType, filtered);
            } else {
                this.eventHandlers.delete(eventType);
            }
        });

        // Remove component itself
        this.components.delete(name);
        console.log(`[ComponentRegistry] Unregistered component: ${name}`);
    }
}

// Create and export singleton instance
export const componentRegistry = new ComponentRegistry();
