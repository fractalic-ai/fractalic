/**
 * GridStack Manager - управление динамическим grid layout
 * Отвечает за создание, размещение и персистентность плиток
 */

export class GridStackManager {
    constructor() {
        this.grid = null;
        this.widgets = new Map();  // executionId -> widget element
        this.nextRightColumn = 8;  // Следующая позиция для execution widgets (начинаем справа)
    }

    /**
     * Инициализация GridStack с базовым layout
     */
    init(containerSelector = '.grid-stack') {
        const container = document.querySelector(containerSelector);
        if (!container) {
            console.error('GridStack container not found:', containerSelector);
            return false;
        }

        // Инициализация GridStack
        this.grid = GridStack.init({
            column: 12,           // 12-колоночная сетка
            margin: 8,            // Отступ между плитками
            float: false,         // Не разрешать float вверх (строгий порядок)
            acceptWidgets: true,  // Разрешить drag из внешних источников
            disableOneColumnMode: true,  // Отключить авто-переключение в 1 колонку
            animate: true,        // Анимация перемещения
            removeTimeout: 100,   // Задержка перед удалением
            handle: '.grid-widget-header', // Перетаскивание только за заголовок
            handleClass: 'grid-widget-header' // Класс для drag handle
        }, containerSelector);

        // Подписаться на события изменения grid
        this.grid.on('change', (event, items) => {
            this.saveLayout();
        });

        this.grid.on('removed', (event, items) => {
            // Обновить карту widgets при удалении
            items.forEach(item => {
                const id = item.id || item.el?.id;
                if (id) {
                    this.widgets.delete(id);
                }
            });
            this.saveLayout();
        });

        console.log('[GridStackManager] Initialized with', this.grid.getColumn(), 'columns');
        return true;
    }

    /**
     * Создать базовые статические плитки (история, чат)
     */
    createBaseWidgets() {
        // Вычислить высоту плиток на основе высоты viewport
        // Используем 100% высоты viewport
        const viewportHeight = window.innerHeight;
        const cellHeight = this.grid.getCellHeight();
        const optimalHeight = Math.floor(viewportHeight / cellHeight);

        console.log(`[GridStackManager] Viewport height: ${viewportHeight}px, Cell height: ${cellHeight}px, Optimal h: ${optimalHeight}`);

        // Проверяем, есть ли сохранённый layout
        const savedLayout = localStorage.getItem('fractalic-grid-layout');
        let historyWidget = null;
        let chatWidget = null;

        if (savedLayout) {
            try {
                const layout = JSON.parse(savedLayout);
                historyWidget = layout.find(item => item.id === 'history-widget');
                chatWidget = layout.find(item => item.id === 'chat-widget');
                console.log('[GridStackManager] Found saved layout, will restore positions');
            } catch (e) {
                console.error('[GridStackManager] Failed to parse saved layout:', e);
            }
        }

        // Создать плитку истории (слева 20%)
        const historyContent = document.querySelector('.sidebar');
        if (historyContent) {
            this.grid.addWidget({
                x: historyWidget?.x ?? 0,
                y: historyWidget?.y ?? 0,
                w: historyWidget?.w ?? 2,
                h: historyWidget?.h ?? optimalHeight,
                id: 'history-widget',
                content: historyContent.outerHTML,
                noResize: false,
                noMove: false
            });
            historyContent.remove();  // Удалить из исходного места
        }

        // Создать плитку чата (центр 60%)
        const chatContent = document.querySelector('.main-content');
        if (chatContent) {
            this.grid.addWidget({
                x: chatWidget?.x ?? 2,
                y: chatWidget?.y ?? 0,
                w: chatWidget?.w ?? 6,
                h: chatWidget?.h ?? optimalHeight,
                id: 'chat-widget',
                content: chatContent.outerHTML,
                noResize: false,
                noMove: false
            });
            chatContent.remove();  // Удалить из исходного места
        }

        console.log('[GridStackManager] Base widgets created');
        this.saveLayout();
    }

    /**
     * Добавить execution bubble как новую плитку справа
     * @param {string} executionId - ID execution
     * @param {HTMLElement} bubbleElement - DOM элемент bubble
     * @returns {HTMLElement} grid widget element
     */
    addExecutionWidget(executionId, bubbleElement) {
        if (!this.grid) {
            console.warn('[GridStackManager] Grid not initialized, cannot add execution widget');
            return null;
        }

        // Создать wrapper для bubble
        const wrapper = document.createElement('div');
        wrapper.className = 'execution-widget-content';
        wrapper.appendChild(bubbleElement);

        // Добавить widget в grid (всегда справа, 20% ширины = ~2.4 колонки)
        const widgetEl = this.grid.addWidget({
            x: this.nextRightColumn,
            y: 0,  // GridStack сам найдёт подходящее Y
            w: 4,  // ~33% ширины (4 из 12 колонок)
            h: 6,  // Начальная высота
            id: `execution-${executionId}`,
            content: wrapper,
            noResize: false,
            noMove: false,
            autoPosition: true  // Автоматически найти лучшую позицию
        });

        // Сохранить ссылку
        this.widgets.set(executionId, widgetEl);

        // Добавить CSS класс для специфичной стилизации
        widgetEl.classList.add('execution-widget');
        widgetEl.setAttribute('data-execution-id', executionId);

        // Анимация появления
        widgetEl.classList.add('new-widget');
        setTimeout(() => {
            widgetEl.classList.remove('new-widget');
        }, 300);

        console.log(`[GridStackManager] Added execution widget: ${executionId.substring(0, 8)}`);
        this.saveLayout();

        return widgetEl;
    }

    /**
     * Добавить динамический компонент (галерея, диаграммы и т.д.)
     * @param {string} type - тип компонента
     * @param {HTMLElement} content - DOM элемент контента
     * @param {Object} options - дополнительные опции (ширина, высота)
     * @returns {HTMLElement} grid widget element
     */
    addDynamicWidget(type, content, options = {}) {
        if (!this.grid) {
            console.warn('[GridStackManager] Grid not initialized, cannot add dynamic widget');
            return null;
        }

        const widgetId = `${type}-${Date.now()}`;

        // Create widget without content first (GridStack will create the structure)
        const widgetEl = this.grid.addWidget({
            x: options.x !== undefined ? options.x : 8,
            y: options.y !== undefined ? options.y : 0,
            w: options.w || 4,  // По умолчанию 33% ширины
            h: options.h || 4,
            id: widgetId,
            noResize: false,
            noMove: false,
            autoPosition: true
        });

        // Find the content area and append our content element
        const contentArea = widgetEl.querySelector('.grid-stack-item-content');
        if (contentArea && content) {
            contentArea.appendChild(content);
        }

        widgetEl.classList.add('dynamic-widget', `${type}-widget`);
        widgetEl.setAttribute('data-widget-type', type);

        // Анимация появления
        widgetEl.classList.add('new-widget');
        setTimeout(() => {
            widgetEl.classList.remove('new-widget');
        }, 300);

        console.log(`[GridStackManager] Added dynamic widget: ${type}`);
        this.saveLayout();

        return widgetEl;
    }

    /**
     * Удалить widget по execution ID
     * @param {string} executionId
     */
    removeExecutionWidget(executionId) {
        const widgetEl = this.widgets.get(executionId);
        if (widgetEl && this.grid) {
            this.grid.removeWidget(widgetEl);
            this.widgets.delete(executionId);
            console.log(`[GridStackManager] Removed execution widget: ${executionId.substring(0, 8)}`);
        }
    }

    /**
     * Получить widget element по execution ID
     * @param {string} executionId
     * @returns {HTMLElement|null}
     */
    getExecutionWidget(executionId) {
        return this.widgets.get(executionId) || null;
    }

    /**
     * Сохранить текущий layout в localStorage
     */
    saveLayout() {
        if (!this.grid) return;

        try {
            const layout = this.grid.save(false);  // save без content (только позиции)
            localStorage.setItem('fractalic-grid-layout', JSON.stringify(layout));
            console.log('[GridStackManager] Layout saved to localStorage');
        } catch (error) {
            console.error('[GridStackManager] Failed to save layout:', error);
        }
    }


    /**
     * Очистить сохранённый layout
     */
    clearLayout() {
        localStorage.removeItem('fractalic-grid-layout');
        console.log('[GridStackManager] Layout cleared from localStorage');
    }

    /**
     * Получить текущий grid instance
     * @returns {GridStack}
     */
    getGrid() {
        return this.grid;
    }

    /**
     * Уничтожить grid (cleanup)
     */
    destroy() {
        if (this.grid) {
            this.grid.destroy(false);  // Не удалять DOM элементы
            this.grid = null;
            this.widgets.clear();
            console.log('[GridStackManager] Grid destroyed');
        }
    }
}
