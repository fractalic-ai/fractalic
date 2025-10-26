/**
 * Component Router
 *
 * Manages component lifecycle and routes events to appropriate component instances.
 * Handles automatic component instantiation based on incoming events.
 */

import { componentRegistry } from './component-registry.js?v=10';

export class ComponentRouter {
    constructor() {
        // Map of execution_id to Map of componentId to component instances
        // Structure: Map<executionId, Map<componentId, componentInstance>>
        this.componentsByExecution = new Map();

        // Map of component names to singleton instances (only for singleton components)
        // Structure: Map<componentName, componentInstance>
        this.singletons = new Map();

        // Reference to DOM mount points
        this.mountPoints = {
            chat: null
        };

        // Reference to GridStackManager for creating dynamic widget tiles
        this.gridManager = null;
    }

    /**
     * Initialize router with DOM mount points and GridStackManager
     *
     * @param {Object} options - Configuration options
     * @param {Object} options.mountPoints - DOM elements for mounting
     * @param {HTMLElement} options.mountPoints.chat - Chat container element
     * @param {GridStackManager} options.gridManager - GridStackManager instance for creating dynamic widgets
     */
    initialize({ mountPoints, gridManager = null }) {
        this.mountPoints = mountPoints;
        this.gridManager = gridManager;
        console.log('[ComponentRouter] Initialized', { mountPoints, hasGridManager: !!gridManager });
    }

    /**
     * Route an event to appropriate components
     * This is the main entry point called by chat-client.js
     *
     * @param {string} eventType - Event type string
     * @param {Object} eventData - Event payload
     * @param {string} executionId - Execution context ID
     */
    routeEvent(eventType, eventData, executionId) {
        // Check if event has explicit 'to' parameter for targeting specific component
        if (eventData.to) {
            this._routeToSpecificComponent(eventType, eventData, executionId);
            return;
        }

        // Find components that handle this event type
        const componentInfos = componentRegistry.getComponentsForEvent(eventType);

        if (componentInfos.length === 0) {
            // No components registered for this event, skip silently
            return;
        }

        console.log(`[ComponentRouter] Routing event ${eventType} to ${componentInfos.length} component(s)`);

        componentInfos.forEach(componentInfo => {
            this._routeToComponent(componentInfo, eventType, eventData, executionId);
        });
    }

    /**
     * Route event to explicitly specified component using 'to' parameter
     * Format: "component-name:component-id" or just "component-id"
     *
     * @private
     */
    _routeToSpecificComponent(eventType, eventData, executionId) {
        const targetId = eventData.to;

        console.log(`[ComponentRouter] Routing event ${eventType} to specific target: ${targetId}`);

        // Check if it's a singleton first
        const singletonInstance = Array.from(this.singletons.entries()).find(([name, instance]) => {
            return instance.componentId === targetId || name === targetId;
        });

        if (singletonInstance) {
            const [name, instance] = singletonInstance;
            console.log(`[ComponentRouter] Found singleton target: ${name}`);
            instance.handleEvent(eventType, eventData);
            return;
        }

        // Search in execution-scoped components
        const executionComponents = this.componentsByExecution.get(executionId);
        if (executionComponents) {
            const targetInstance = executionComponents.get(targetId);
            if (targetInstance) {
                console.log(`[ComponentRouter] Found execution-scoped target: ${targetId}`);
                targetInstance.handleEvent(eventType, eventData);
                return;
            }
        }

        // If target format is "component-name:instance-id", try to parse and create if needed
        if (targetId.includes(':')) {
            const [componentName, instanceSuffix] = targetId.split(':');
            const componentInfo = componentRegistry.getComponent(componentName);

            if (componentInfo) {
                console.log(`[ComponentRouter] Creating new instance for target: ${targetId}`);
                const { componentClass, manifest } = componentInfo;
                const instance = this._getOrCreateInstance(
                    componentName,
                    componentClass,
                    targetId,
                    manifest,
                    executionId
                );
                instance.handleEvent(eventType, eventData);
                return;
            }
        }

        console.warn(`[ComponentRouter] Target component not found: ${targetId}`);
    }

    /**
     * Route event to a specific component (internal)
     *
     * @private
     */
    _routeToComponent(componentInfo, eventType, eventData, executionId) {
        const { name, componentClass, manifest } = componentInfo;

        let componentInstance;

        if (manifest.singleton) {
            // Singleton: reuse or create single instance
            componentInstance = this._getSingletonInstance(name, componentClass, manifest, executionId);
        } else {
            // Non-singleton: create instance per execution context
            const componentId = this._generateComponentId(name, executionId);
            componentInstance = this._getOrCreateInstance(
                name,
                componentClass,
                componentId,
                manifest,
                executionId
            );
        }

        // Route event to the component instance
        componentInstance.handleEvent(eventType, eventData);
    }

    /**
     * Get or create a singleton component instance
     *
     * @private
     */
    _getSingletonInstance(name, componentClass, manifest, executionId) {
        if (this.singletons.has(name)) {
            return this.singletons.get(name);
        }

        // Create new singleton instance
        const componentId = `${name}-singleton`;
        const instance = new componentClass(componentId, {
            mountPoint: manifest.mountPoint,
            executionId: executionId
        });

        // Mount to DOM
        const containerElement = instance.mount();
        this._mountToDom(containerElement, manifest.mountPoint);

        // Store singleton
        this.singletons.set(name, instance);

        console.log(`[ComponentRouter] Created singleton instance: ${name}`);
        return instance;
    }

    /**
     * Get or create a component instance for execution context
     *
     * @private
     */
    _getOrCreateInstance(name, componentClass, componentId, manifest, executionId) {
        // Ensure execution context map exists
        if (!this.componentsByExecution.has(executionId)) {
            this.componentsByExecution.set(executionId, new Map());
        }

        const executionComponents = this.componentsByExecution.get(executionId);

        // Check if instance already exists
        if (executionComponents.has(componentId)) {
            return executionComponents.get(componentId);
        }

        // Create new instance
        const instance = new componentClass(componentId, {
            mountPoint: manifest.mountPoint,
            executionId: executionId
        });

        // Mount to DOM
        const containerElement = instance.mount();
        this._mountToDom(containerElement, manifest.mountPoint);

        // Store instance
        executionComponents.set(componentId, instance);

        console.log(`[ComponentRouter] Created component instance: ${componentId} for execution ${executionId}`);
        return instance;
    }

    /**
     * Mount component container to appropriate DOM location
     * For 'artifacts' mountPoint, creates a GridStack widget tile
     * For 'chat' mountPoint, adds directly to chat container
     *
     * @private
     */
    _mountToDom(containerElement, mountPoint) {
        // Handle legacy 'artifacts' mountPoint by creating grid widget
        if (mountPoint === 'artifacts' || mountPoint === 'dynamic') {
            if (this.gridManager) {
                // Extract component type from container classes
                const componentType = containerElement.className.split(' ').find(c => c.endsWith('-component'))?.replace('-component', '') || 'component';

                this.gridManager.addDynamicWidget(componentType, containerElement, {
                    w: 4,  // 33% width (4 out of 12 columns)
                    h: 6   // Height in grid units
                });
                console.log(`[ComponentRouter] Mounted ${componentType} as grid widget`);
            } else {
                console.warn(`[ComponentRouter] GridManager not available, cannot create widget for ${mountPoint}`);
            }
            return;
        }

        // For 'chat' mountPoint, append directly
        const targetElement = this.mountPoints[mountPoint];

        if (!targetElement) {
            console.error(`[ComponentRouter] Mount point "${mountPoint}" not found`);
            return;
        }

        targetElement.appendChild(containerElement);
        console.log(`[ComponentRouter] Mounted component to ${mountPoint}`);
    }

    /**
     * Generate unique component ID
     *
     * @private
     */
    _generateComponentId(componentName, executionId) {
        // Don't include timestamp - we want to reuse the same component instance
        // for the same execution context
        return `${componentName}-${executionId}`;
    }

    /**
     * Get all component instances for a specific execution
     *
     * @param {string} executionId - Execution ID
     * @returns {Array<Object>} Array of component instances
     */
    getComponentsForExecution(executionId) {
        const executionComponents = this.componentsByExecution.get(executionId);
        if (!executionComponents) return [];

        return Array.from(executionComponents.values());
    }

    /**
     * Cleanup components for a specific execution
     * Useful when an execution completes
     *
     * @param {string} executionId - Execution ID to cleanup
     */
    cleanupExecution(executionId) {
        const executionComponents = this.componentsByExecution.get(executionId);
        if (!executionComponents) return;

        // Unmount all components for this execution
        executionComponents.forEach((instance, componentId) => {
            instance.unmount();
            console.log(`[ComponentRouter] Cleaned up component: ${componentId}`);
        });

        // Remove execution from map
        this.componentsByExecution.delete(executionId);
        console.log(`[ComponentRouter] Cleaned up execution: ${executionId}`);
    }

    /**
     * Cleanup a singleton component
     *
     * @param {string} componentName - Name of singleton component
     */
    cleanupSingleton(componentName) {
        const instance = this.singletons.get(componentName);
        if (!instance) return;

        instance.unmount();
        this.singletons.delete(componentName);
        console.log(`[ComponentRouter] Cleaned up singleton: ${componentName}`);
    }

    /**
     * Cleanup all components (useful for reset/reload)
     */
    cleanupAll() {
        // Cleanup all execution-scoped components
        this.componentsByExecution.forEach((executionComponents, executionId) => {
            executionComponents.forEach(instance => instance.unmount());
        });
        this.componentsByExecution.clear();

        // Cleanup all singletons
        this.singletons.forEach(instance => instance.unmount());
        this.singletons.clear();

        console.log('[ComponentRouter] Cleaned up all components');
    }

    /**
     * Get statistics about active components (useful for debugging)
     *
     * @returns {Object} Statistics object
     */
    getStats() {
        const executionCount = this.componentsByExecution.size;
        let totalComponents = 0;

        this.componentsByExecution.forEach(executionComponents => {
            totalComponents += executionComponents.size;
        });

        const singletonCount = this.singletons.size;

        return {
            executions: executionCount,
            components: totalComponents,
            singletons: singletonCount,
            total: totalComponents + singletonCount
        };
    }
}

// Create and export singleton instance
export const componentRouter = new ComponentRouter();
