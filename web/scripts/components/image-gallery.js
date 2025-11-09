/**
 * Image Gallery Component
 *
 * Displays a gallery of images that can grow dynamically as new images are added.
 * Supports both single images and batch updates.
 */

import { BaseComponent } from './base-component.js?v=10';

export class ImageGalleryComponent extends BaseComponent {
    constructor(componentId, options) {
        super(componentId, options);

        // Initialize state with empty images array
        this.setState({
            images: [],
            title: 'Image Gallery'
        }, false); // Don't render yet
    }

    /**
     * Component manifest declaring behavior
     */
    static getManifest() {
        return {
            events: ['image_generated', 'images_batch', 'image_gallery_update'],
            singleton: true, // Only one gallery instance
            mountPoint: 'artifacts', // Show in artifacts panel
            updateStrategy: 'append', // Add new images to existing ones
            description: 'Displays a dynamically growing gallery of images'
        };
    }

    /**
     * Handle incoming events
     */
    handleEvent(eventType, eventData) {
        console.log(`[ImageGallery] Received event: ${eventType}`, eventData);

        switch (eventType) {
            case 'image_generated':
                // Single image event - supports both url and local_path
                this._addImage({
                    url: eventData.url || eventData.image_url,
                    local_path: eventData.local_path,
                    execution_id: eventData.execution_id,
                    workspace_path: eventData._workspace_path,
                    caption: eventData.caption || eventData.message || '',
                    timestamp: Date.now()
                });
                break;

            case 'images_batch':
                // Batch of images - supports both url and local_path for each image
                if (Array.isArray(eventData.images)) {
                    eventData.images.forEach(img => {
                        this._addImage({
                            url: img.url,
                            local_path: img.local_path,
                            execution_id: eventData.execution_id,
                            workspace_path: eventData._workspace_path,
                            caption: img.caption || '',
                            timestamp: Date.now()
                        });
                    });
                }
                break;

            case 'image_gallery_update':
                // Generic gallery update with flexible structure
                if (eventData.images) {
                    if (Array.isArray(eventData.images)) {
                        eventData.images.forEach(img => this._addImage(img));
                    } else {
                        this._addImage(eventData.images);
                    }
                }
                if (eventData.title) {
                    this.setState({ title: eventData.title }, false);
                }
                this.render();
                break;

            default:
                console.warn(`[ImageGallery] Unknown event type: ${eventType}`);
        }
    }

    /**
     * Resolve image URL from either url or local_path
     * @private
     */
    _resolveImageUrl(imageData) {
        // Priority: local_path > url
        if (imageData.local_path) {
            // Convert local path to server endpoint URL
            const params = new URLSearchParams({
                path: imageData.local_path
            });

            // If workspace_path is provided, use it for resolution
            if (imageData.workspace_path) {
                params.append('workspace_path', imageData.workspace_path);
            } else if (imageData.execution_id) {
                // Fallback to execution_id for backward compatibility
                params.append('execution_id', imageData.execution_id);
            }

            return `/serve_local_image/?${params.toString()}`;
        }
        return imageData.url || imageData.image_url;
    }

    /**
     * Add image to gallery
     * @private
     */
    _addImage(imageData) {
        const currentImages = this.getState().images;
        currentImages.push({
            url: this._resolveImageUrl(imageData),
            caption: imageData.caption || '',
            timestamp: imageData.timestamp || Date.now()
        });
        this.setState({ images: currentImages }, true); // Trigger render
    }

    /**
     * Mount component and create initial structure
     */
    onMount() {
        this.container.classList.add('image-gallery-component');
        this.render();
    }

    /**
     * Render gallery HTML
     */
    render() {
        const { images, title } = this.getState();

        // Clear container
        this.container.innerHTML = '';

        // Create header (simple design matching artifacts-header)
        const header = this.createElement('div', {
            classes: ['gallery-header', 'grid-widget-header']
        });

        // Title
        const headerTitle = this.createElement('h3', {
            textContent: title
        });

        // Actions section
        const actions = this.createElement('div', {
            classes: ['gallery-header-actions']
        });

        const imageCount = this.createElement('span', {
            classes: ['image-count'],
            textContent: `${images.length} image${images.length !== 1 ? 's' : ''}`
        });

        const closeBtn = this.createElement('button', {
            classes: ['gallery-close-btn'],
            innerHTML: '×',
            attributes: {
                title: 'Close gallery'
            }
        });

        // Close button handler
        this.addEventListener(closeBtn, 'click', (e) => {
            e.stopPropagation(); // Prevent drag initiation
            this._closeGallery();
        });

        actions.appendChild(imageCount);
        actions.appendChild(closeBtn);

        header.appendChild(headerTitle);
        header.appendChild(actions);
        this.container.appendChild(header);

        // Create scrollable content wrapper
        const contentWrapper = this.createElement('div', {
            classes: ['gallery-content-wrapper']
        });

        // Show empty state if no images
        if (images.length === 0) {
            const emptyState = this.createElement('div', {
                classes: ['gallery-empty'],
                innerHTML: `
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                        <circle cx="8.5" cy="8.5" r="1.5"/>
                        <polyline points="21 15 16 10 5 21"/>
                    </svg>
                    <p>No images yet</p>
                    <span>Images will appear here as they are generated</span>
                `
            });
            contentWrapper.appendChild(emptyState);
        } else {
            // Create gallery grid
            const grid = this.createElement('div', {
                classes: ['gallery-grid']
            });

            // Add images
            images.forEach((image, index) => {
                const imageCard = this._createImageCard(image, index);
                grid.appendChild(imageCard);
            });

            contentWrapper.appendChild(grid);
        }

        this.container.appendChild(contentWrapper);
    }

    /**
     * Close gallery (remove from GridStack)
     * @private
     */
    _closeGallery() {
        // Emit unmount event to cleanup
        if (typeof this.onUnmount === 'function') {
            this.onUnmount();
        }

        // Remove widget from GridStack if available
        const widgetEl = this.container.closest('.grid-stack-item');
        if (widgetEl && window.gridManager) {
            const grid = window.gridManager.getGrid();
            if (grid) {
                grid.removeWidget(widgetEl);
            }
        }
    }

    /**
     * Create image card element
     * @private
     */
    _createImageCard(image, index) {
        const card = this.createElement('div', {
            classes: ['gallery-image-card']
        });

        // Image element
        const img = this.createElement('img', {
            classes: ['gallery-image'],
            attributes: {
                src: image.url,
                alt: image.caption || `Image ${index + 1}`,
                loading: 'lazy'
            }
        });

        // Error handling for failed image loads
        img.addEventListener('error', () => {
            img.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect width="200" height="200" fill="%23ddd"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em" fill="%23999"%3EImage not found%3C/text%3E%3C/svg%3E';
        });

        card.appendChild(img);

        // Caption if present
        if (image.caption) {
            const caption = this.createElement('div', {
                classes: ['gallery-caption'],
                textContent: image.caption
            });
            card.appendChild(caption);
        }

        // Click to enlarge
        this.addEventListener(card, 'click', () => {
            this._enlargeImage(image.url, image.caption);
        });

        return card;
    }

    /**
     * Show enlarged image in modal with download option
     * @private
     */
    _enlargeImage(url, caption) {
        // Create modal overlay
        const modal = this.createElement('div', {
            classes: ['image-modal']
        });

        const modalContent = this.createElement('div', {
            classes: ['image-modal-content']
        });

        // Modal header with actions
        const modalHeader = this.createElement('div', {
            classes: ['image-modal-header']
        });

        // Download button
        const downloadBtn = this.createElement('a', {
            classes: ['image-modal-download'],
            innerHTML: `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7 10 12 15 17 10"/>
                    <line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
                <span>Download</span>
            `,
            attributes: {
                href: url,
                download: caption || 'image',
                title: 'Download image'
            }
        });

        // Close button
        const closeBtn = this.createElement('button', {
            classes: ['image-modal-close'],
            innerHTML: '×',
            attributes: {
                title: 'Close (Esc)'
            }
        });

        modalHeader.appendChild(downloadBtn);
        modalHeader.appendChild(closeBtn);

        // Image container with loading state
        const imgContainer = this.createElement('div', {
            classes: ['image-modal-img-container']
        });

        const img = this.createElement('img', {
            classes: ['image-modal-img'],
            attributes: {
                src: url,
                alt: caption || 'Enlarged image',
                loading: 'eager'
            }
        });

        // Loading spinner
        const spinner = this.createElement('div', {
            classes: ['image-modal-spinner'],
            innerHTML: `
                <svg width="40" height="40" viewBox="0 0 50 50" stroke="currentColor">
                    <circle cx="25" cy="25" r="20" fill="none" stroke-width="4" opacity="0.2"/>
                    <circle cx="25" cy="25" r="20" fill="none" stroke-width="4" stroke-dasharray="80" stroke-dashoffset="60">
                        <animateTransform attributeName="transform" type="rotate" from="0 25 25" to="360 25 25" dur="1s" repeatCount="indefinite"/>
                    </circle>
                </svg>
            `
        });

        imgContainer.appendChild(spinner);
        imgContainer.appendChild(img);

        // Remove spinner when image loads
        img.addEventListener('load', () => {
            spinner.remove();
            img.classList.add('loaded');
        });

        img.addEventListener('error', () => {
            spinner.innerHTML = `
                <svg width="40" height="40" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                <p>Failed to load image</p>
            `;
            spinner.classList.add('error');
        });

        modalContent.appendChild(modalHeader);
        modalContent.appendChild(imgContainer);

        // Caption if present
        if (caption) {
            const captionDiv = this.createElement('div', {
                classes: ['image-modal-caption'],
                textContent: caption
            });
            modalContent.appendChild(captionDiv);
        }

        modal.appendChild(modalContent);
        document.body.appendChild(modal);

        // Add animation class after mount
        requestAnimationFrame(() => {
            modal.classList.add('visible');
        });

        // Close handlers
        const closeModal = () => {
            modal.classList.remove('visible');
            setTimeout(() => {
                if (modal.parentNode) {
                    document.body.removeChild(modal);
                }
            }, 200); // Match CSS transition duration
        };

        closeBtn.addEventListener('click', closeModal);
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeModal();
        });

        // Keyboard shortcut (Esc to close)
        const handleKeydown = (e) => {
            if (e.key === 'Escape') {
                closeModal();
                document.removeEventListener('keydown', handleKeydown);
            }
        };
        document.addEventListener('keydown', handleKeydown);
    }
}
