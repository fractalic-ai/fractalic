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

        // Create header (with drag handle class for GridStack)
        const header = this.createElement('div', {
            classes: ['gallery-header', 'grid-widget-header'],
            innerHTML: `<h3>${title}</h3><span class="image-count">${images.length} image(s)</span>`
        });
        this.container.appendChild(header);

        // Create gallery grid
        const grid = this.createElement('div', {
            classes: ['gallery-grid']
        });

        // Add images
        images.forEach((image, index) => {
            const imageCard = this._createImageCard(image, index);
            grid.appendChild(imageCard);
        });

        this.container.appendChild(grid);

        // Show empty state if no images
        if (images.length === 0) {
            const emptyState = this.createElement('div', {
                classes: ['gallery-empty'],
                textContent: 'No images yet. Images will appear here as they are generated.'
            });
            this.container.appendChild(emptyState);
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
     * Show enlarged image in modal
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

        const img = this.createElement('img', {
            attributes: { src: url, alt: caption || 'Enlarged image' }
        });

        const closeBtn = this.createElement('button', {
            classes: ['image-modal-close'],
            textContent: '×'
        });

        modalContent.appendChild(closeBtn);
        modalContent.appendChild(img);

        if (caption) {
            const captionDiv = this.createElement('div', {
                classes: ['image-modal-caption'],
                textContent: caption
            });
            modalContent.appendChild(captionDiv);
        }

        modal.appendChild(modalContent);
        document.body.appendChild(modal);

        // Close handlers
        const closeModal = () => {
            document.body.removeChild(modal);
        };

        closeBtn.addEventListener('click', closeModal);
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeModal();
        });
    }
}
