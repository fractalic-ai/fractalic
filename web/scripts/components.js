/**
 * Components Module - Central export point for component system
 * This file re-exports all component system modules to simplify imports
 */

// Re-export registry
export { ComponentRegistry, componentRegistry } from './components/component-registry.js?v=10';

// Re-export router
export { ComponentRouter, componentRouter } from './components/component-router.js?v=10';

// Re-export base component
export { BaseComponent } from './components/base-component.js?v=10';

// Re-export concrete components
export { ImageGalleryComponent } from './components/image-gallery.js?v=16';
export { MessageListComponent } from './components/message-list.js?v=10';
