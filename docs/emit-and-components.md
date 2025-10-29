---
title: @emit Operation & Component System
description: Event-driven UI components with @emit operation for real-time frontend interactions
outline: deep
---

# @emit Operation & Component System

The @emit operation enables Fractalic workflows to send custom events to the web frontend, triggering dynamic UI components that render in real-time. This creates an event-driven architecture where backend workflows can display images, messages, charts, or any custom visualization without modifying core Fractalic code.

## Quick Start

```markdown
@emit
event: msg_success
data:
  message: "Task completed successfully!"
prompt: "✅ Sent success message to frontend"
```

This operation sends a `msg_success` event which automatically triggers the MessageList component in the chat interface.

## Internal TOC
- [What is @emit?](#what-is-emit)
- [Core Concepts](#core-concepts)
- [Event Flow Architecture](#event-flow-architecture)
- [The @emit Operation](#the-emit-operation)
- [Component System](#component-system)
- [Event Routing](#event-routing)
- [Component Lifecycle](#component-lifecycle)
- [Built-in Components](#built-in-components)
- [Creating Custom Components](#creating-custom-components)
- [Advanced Patterns](#advanced-patterns)
- [Troubleshooting](#troubleshooting)

## What is @emit?

@emit is a Fractalic operation that sends custom events from backend workflows to the frontend web interface. Unlike @llm or @shell which produce text in the workflow document, @emit creates UI interactions visible only in the web interface.

**Key Characteristics:**
- **Non-blocking**: Executes instantly without waiting for frontend response
- **Optional rendering**: Events only display when using the web UI (ignored in CLI mode)
- **Event-driven**: Frontend components automatically respond to matching event types
- **Zero configuration**: No manual component initialization needed

## Core Concepts

| Concept | What | Why It Matters | Example |
| ------- | ---- | -------------- | ------- |
| **Event Type** | String identifier for event category | Routes events to appropriate components | `msg_info`, `image_generated` |
| **Event Data** | JSON payload with event-specific fields | Contains the actual content to display | `{message: "Hello", type: "info"}` |
| **Component Registry** | Central catalog of available UI components | Enables automatic component instantiation | Registers MessageList, ImageGallery |
| **Component Router** | Routing engine that delivers events to components | Handles automatic and explicit targeting | Routes `msg_*` events to MessageList |
| **Mount Points** | DOM locations where components render | Organizes UI layout (chat area vs sidebar) | `chat`, `artifacts` |
| **Singleton vs Non-Singleton** | Component instantiation strategy | Controls whether components are shared or per-execution | ImageGallery (singleton) vs MessageList (per-execution) |
| **Explicit Targeting** | Using `to` parameter to route to specific component | Enables multiple instances of same component type | `to: "message-list:notifications"` |

## Event Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (Python)                             │
├─────────────────────────────────────────────────────────────────┤
│  Workflow.md                                                    │
│  ┌─────────────┐                                                │
│  │   @emit     │ → emit_op.py → HTTP POST to /events           │
│  │ event: ...  │                                                │
│  │ data: ...   │                                                │
│  └─────────────┘                                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              UI Server (FastAPI) - Port 8000                    │
├─────────────────────────────────────────────────────────────────┤
│  Event Queue (FIFO)                                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ { event: "msg_success", data: {...}, execution_id: "..." }│  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│                             ▼                                   │
│  NDJSON Stream Generator                                        │
│  data: {"event":"msg_success","data":{...}}\n                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                Frontend (JavaScript)                            │
├─────────────────────────────────────────────────────────────────┤
│  stream-client.js                                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Receive NDJSON stream                                  │  │
│  │ 2. Parse JSON lines                                       │  │
│  │ 3. Extract event type and data                            │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│  chat-client.js                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ handleEvent(eventType, eventData, executionId)            │  │
│  │   - Check if system event (chat_message, error, etc.)     │  │
│  │   - If not system event → Route to ComponentRouter        │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│  component-router.js                                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ routeEvent(eventType, eventData, executionId)             │  │
│  │   1. Check for explicit 'to' parameter                    │  │
│  │   2. Otherwise: Query ComponentRegistry for handlers      │  │
│  │   3. Get or create component instance                     │  │
│  │   4. Call component.handleEvent()                         │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│  Component Instance (message-list.js, image-gallery.js, etc.)  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ handleEvent(eventType, eventData)                         │  │
│  │   - Update component state                                │  │
│  │   - Call render()                                         │  │
│  │   - Update DOM                                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│  Browser DOM                                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Chat Area          │  Artifacts Panel                     │  │
│  │ ┌────────────┐     │  ┌──────────────┐                   │  │
│  │ │ MessageList│     │  │ ImageGallery │                   │  │
│  │ │ - Info msg │     │  │ [img] [img]  │                   │  │
│  │ │ - Success  │     │  │ [img] [img]  │                   │  │
│  │ └────────────┘     │  └──────────────┘                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Flow Steps

1. **Backend Emission**: `@emit` operation in workflow.md triggers `emit_op.py`
2. **HTTP Transport**: Event POSTed to UI Server `/events` endpoint with execution ID
3. **Queue Management**: Event added to FIFO queue keyed by execution ID
4. **Stream Generation**: NDJSON stream delivers events to connected frontend clients
5. **Frontend Reception**: `stream-client.js` parses NDJSON and extracts events
6. **Event Delegation**: `chat-client.js` routes custom events to ComponentRouter
7. **Component Routing**: Router finds or creates appropriate component instance
8. **Event Handling**: Component processes event data and updates its state
9. **DOM Rendering**: Component re-renders and updates visible UI

## The @emit Operation

### Syntax

```yaml
@emit
event: <event_type>
data:
  <key>: <value>
  [<key>: <value>]
  [to: <target_component>]
prompt: <optional_confirmation_message>
```

### Parameters

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `event` | Yes | string | Event type identifier (e.g., `msg_success`, `image_generated`) |
| `data` | Yes | object | Event payload with component-specific fields |
| `data.to` | No | string | Explicit component target (format: `component-name:instance-id`) |
| `prompt` | No | string | Confirmation message shown in workflow output (not sent to frontend) |

**Important:** The `to` parameter goes **inside** the `data` object, not as a top-level field.

### Basic Examples

#### Single Event
```markdown
@emit
event: msg_info
data:
  message: "Processing started..."
prompt: "✅ Sent info message"
```

#### Event with Explicit Target
```markdown
@emit
event: msg_success
data:
  message: "Backup completed"
  to: "message-list:system-logs"
prompt: "✅ Logged to system-logs component"
```

#### Batch Event
```markdown
@emit
event: images_batch
data:
  images:
    - url: https://example.com/img1.jpg
      caption: First image
    - url: https://example.com/img2.jpg
      caption: Second image
prompt: "✅ Sent batch of 2 images"
```

### Event Naming Conventions

**Recommended Patterns:**
- `msg_<type>`: Messages for MessageList component (`msg_info`, `msg_success`, `msg_warning`, `msg_error`, `msg_debug`)
- `<noun>_<verb>`: Action-oriented events (`image_generated`, `chart_updated`, `data_loaded`)
- `<noun>_batch`: Batch operations (`images_batch`, `messages_batch`)

**Namespace Separation:**
Custom event types (like `msg_*`) are intentionally designed to NOT conflict with system events. System events include:
- `chat_message`, `chat_started`, `chat_completed`
- `error`, `warning`, `debug`
- `tool_call`, `tool_result`
- `execution_started`, `execution_completed`

If you use event types like `msg_error` instead of `error`, they are routed to components rather than system error handlers, allowing you to create custom UI for errors.

## Component System

### Component Structure

All components inherit from `BaseComponent` (defined in `web/scripts/components/base-component.js`) which provides:

- **State Management**: `setState()`, `getState()` with automatic re-rendering
- **DOM Utilities**: `createElement()`, `mount()`, `unmount()`
- **Lifecycle Hooks**: `onMount()`, `onUnmount()`, `render()`
- **Event Handling**: `handleEvent()` method

### Component Manifest

Each component class must define a static `getManifest()` method:

```javascript
static getManifest() {
    return {
        events: ['msg_info', 'msg_success'],  // Array of event types to handle
        singleton: false,                      // true = one instance globally, false = per-execution
        mountPoint: 'chat',                    // 'chat' or 'artifacts'
        updateStrategy: 'append',              // 'append', 'replace', or custom
        description: 'Displays messages'       // Human-readable description
    };
}
```

**Manifest Fields:**

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `events` | Yes | array | Event type strings this component handles |
| `singleton` | Yes | boolean | If true, only one instance exists system-wide |
| `mountPoint` | Yes | string | Where to render: `'chat'` (inline) or `'artifacts'` (sidebar) |
| `updateStrategy` | No | string | How component handles multiple events (informational) |
| `description` | No | string | Human-readable component description |

### Component Registration

Components are registered in `web/scripts/components/component-registry.js`:

```javascript
import { MessageListComponent } from './message-list.js?v=10';
import { ImageGalleryComponent } from './image-gallery.js?v=10';

export class ComponentRegistry {
    constructor() {
        this.components = new Map();
        this._registerComponents();
    }

    _registerComponents() {
        this.register('message-list', MessageListComponent);
        this.register('image-gallery', ImageGalleryComponent);
    }

    register(name, componentClass) {
        const manifest = componentClass.getManifest();
        this.components.set(name, {
            name,
            componentClass,
            manifest
        });
        console.log(`[ComponentRegistry] Registered: ${name}`);
    }
}
```

## Event Routing

### Automatic Routing

When no `to` parameter is specified, events are automatically routed to all components registered for that event type:

```javascript
// In component-router.js
routeEvent(eventType, eventData, executionId) {
    // Find components that handle this event type
    const componentInfos = componentRegistry.getComponentsForEvent(eventType);

    componentInfos.forEach(componentInfo => {
        this._routeToComponent(componentInfo, eventType, eventData, executionId);
    });
}
```

**Example:**
```markdown
@emit
event: msg_success
data:
  message: "All tests passed"
```

This automatically finds MessageList component (registered for `msg_success`), creates or reuses an instance for the current execution, and delivers the event.

### Explicit Routing with `to` Parameter

Use the `to` parameter to target a specific component instance:

**Format:** `"component-name:instance-id"`

```markdown
@emit
event: msg_info
data:
  message: "Custom notification"
  to: "message-list:notifications"
prompt: "✅ Sent to custom notifications instance"
```

**Routing Logic:**
1. Check if `to` matches an existing singleton component
2. Check if `to` matches an existing execution-scoped component
3. If `to` format is `"component-name:instance-id"`, create new instance with that ID
4. Warn if no component found

**Key Insight:** Using `to` with format `"component-name:custom-id"` allows you to create multiple instances of the same component type within one execution, each maintaining separate state.

## Component Lifecycle

### Instance Creation

**Singleton Components (like ImageGallery):**
- Created once on first event
- Shared across all executions
- ID format: `{component-name}-singleton`
- Persists until page reload

**Non-Singleton Components (like MessageList):**
- Created once per execution context
- ID format: `{component-name}-{execution-id}`
- **Critical:** NO timestamp in ID to ensure reuse
- Cleaned up when execution completes

**Custom Targeted Components:**
- Created when `to` parameter specifies custom ID
- ID format: `{component-name}:{custom-id}`
- Persists for execution lifetime
- Allows multiple isolated instances

### Reuse Strategy

The component router ensures instance reuse through consistent ID generation:

```javascript
// web/scripts/components/component-router.js:240-244
_generateComponentId(componentName, executionId) {
    // Don't include timestamp - we want to reuse the same component instance
    // for the same execution context
    return `${componentName}-${executionId}`;
}
```

**Before Fix (BAD):**
```javascript
return `${componentName}-${executionId}-${Date.now()}`;
// Result: message-list-abc123-1760461122867
// Problem: New instance for EVERY event
```

**After Fix (GOOD):**
```javascript
return `${componentName}-${executionId}`;
// Result: message-list-abc123
// Benefit: Same instance receives ALL events for this execution
```

### Cleanup

Components can be cleaned up via:

```javascript
// Clean up all components for specific execution
componentRouter.cleanupExecution(executionId);

// Clean up a singleton
componentRouter.cleanupSingleton('image-gallery');

// Clean up everything
componentRouter.cleanupAll();
```

Currently cleanup happens:
- On page reload (automatic)
- When user explicitly triggers cleanup (future feature)
- When execution completes (future implementation)

## Built-in Components

### MessageList Component

**Location:** `web/scripts/components/message-list.js`

**Purpose:** Display timestamped messages with type indicators (info, success, warning, error, debug)

**Manifest:**
```javascript
static getManifest() {
    return {
        events: ['message_added', 'messages_batch', 'message_list_update',
                 'msg_info', 'msg_success', 'msg_warning', 'msg_error', 'msg_debug'],
        singleton: false,
        mountPoint: 'chat',
        updateStrategy: 'append',
        description: 'Displays a dynamically updating list of messages'
    };
}
```

**Event Handlers:**

| Event | Data Fields | Behavior |
|-------|-------------|----------|
| `msg_info` | `message` or `text` | Adds info message with ℹ️ icon |
| `msg_success` | `message` or `text` | Adds success message with ✅ icon |
| `msg_warning` | `message` or `text` | Adds warning message with ⚠️ icon |
| `msg_error` | `message` or `text` | Adds error message with ❌ icon |
| `msg_debug` | `message` or `text` | Adds debug message with 🔍 icon |
| `message_added` | `message`, `type` | Generic message event with explicit type |
| `messages_batch` | `messages` array | Batch add multiple messages |
| `message_list_update` | `messages`, `title`, `clear` | Update entire list |

**Example Usage:**
```markdown
@emit
event: msg_info
data:
  message: "Starting data processing..."
prompt: "✅ Info message sent"

@emit
event: msg_success
data:
  message: "Processing completed: 1,000 records"
prompt: "✅ Success message sent"

@emit
event: msg_error
data:
  message: "Failed to connect to database"
prompt: "✅ Error message sent"
```

**Visual Output:**
```
Messages                                            4 message(s)
─────────────────────────────────────────────────────────────
ℹ️  Starting data processing...                    10:30:15 AM
✅  Processing completed: 1,000 records             10:30:45 AM
❌  Failed to connect to database                   10:31:02 AM
```

### ImageGallery Component

**Location:** `web/scripts/components/image-gallery.js`

**Purpose:** Display grid of images with captions in artifacts panel

**Manifest:**
```javascript
static getManifest() {
    return {
        events: ['image_generated', 'images_batch', 'image_gallery_update'],
        singleton: true,  // One gallery for entire application
        mountPoint: 'artifacts',
        updateStrategy: 'append',
        description: 'Displays a gallery of images'
    };
}
```

**Event Handlers:**

| Event | Data Fields | Behavior |
|-------|-------------|----------|
| `image_generated` | `url` or `local_path`, `caption` | Add single image to gallery (use `url` for web images, `local_path` for local files) |
| `images_batch` | `images` array | Add multiple images at once (each image can have `url` or `local_path`) |
| `image_gallery_update` | `title`, `images`, `clear` | Update gallery title and/or images |

**Example Usage:**
```markdown
@emit
event: image_generated
data:
  url: https://picsum.photos/400/300?random=1
  caption: Mountain Landscape
prompt: "✅ Image 1 added to gallery"

@emit
event: images_batch
data:
  images:
    - url: https://picsum.photos/400/300?random=2
      caption: Ocean Sunset
    - url: https://picsum.photos/400/300?random=3
      caption: Forest Path
prompt: "✅ Added batch of 2 images"

@emit
event: image_gallery_update
data:
  title: "Trip Photos 2024"
  images:
    - url: https://picsum.photos/400/300?random=4
      caption: Final destination
prompt: "✅ Updated gallery with custom title"
```

**Local File Examples** (for locally-run Fractalic):
```markdown
@emit
event: image_generated
data:
  local_path: outputs/chart.png
  caption: Generated Sales Chart
prompt: "✅ Local image added to gallery"

@emit
event: images_batch
data:
  images:
    - local_path: outputs/diagram1.png
      caption: System Architecture
    - local_path: outputs/diagram2.png
      caption: Data Flow
prompt: "✅ Added batch of local images"

# Mix of local and web images
@emit
event: images_batch
data:
  images:
    - url: https://example.com/logo.png
      caption: Company Logo
    - local_path: outputs/generated_report.png
      caption: Monthly Report
prompt: "✅ Mixed local and web images"
```

> **Note:** Use `local_path` (relative to repository root) when Fractalic runs locally. The path is automatically converted to `/serve_image/?path=...` endpoint. Use `url` for web-hosted images.

**Visual Output:**
```
┌─────────────────────────────────────────┐
│      Image Gallery - Trip Photos 2024   │  ← Artifacts Panel
├─────────────────────────────────────────┤
│  ┌──────┐  ┌──────┐  ┌──────┐          │
│  │ img1 │  │ img2 │  │ img3 │          │
│  └──────┘  └──────┘  └──────┘          │
│  Mountain   Ocean     Forest            │
│  Landscape  Sunset    Path              │
│                                         │
│  ┌──────┐                               │
│  │ img4 │                               │
│  └──────┘                               │
│  Final                                  │
│  destination                            │
└─────────────────────────────────────────┘
```

## Creating Custom Components

### Step 1: Create Component Class

Create a new file in `web/scripts/components/`:

```javascript
// web/scripts/components/my-chart.js
import { BaseComponent } from './base-component.js?v=10';

export class MyChartComponent extends BaseComponent {
    constructor(componentId, options) {
        super(componentId, options);

        // Initialize component state
        this.setState({
            dataPoints: [],
            title: 'Chart'
        }, false); // false = don't render yet
    }

    /**
     * Component manifest - REQUIRED
     */
    static getManifest() {
        return {
            events: ['chart_data', 'chart_update'],
            singleton: false,
            mountPoint: 'artifacts',
            updateStrategy: 'replace',
            description: 'Displays a data chart'
        };
    }

    /**
     * Handle incoming events - REQUIRED
     */
    handleEvent(eventType, eventData) {
        console.log(`[MyChart] Received event: ${eventType}`, eventData);

        switch (eventType) {
            case 'chart_data':
                this._addDataPoint(eventData);
                break;
            case 'chart_update':
                this._updateChart(eventData);
                break;
            default:
                console.warn(`[MyChart] Unknown event: ${eventType}`);
        }
    }

    /**
     * Called when component is mounted to DOM
     */
    onMount() {
        this.container.classList.add('my-chart-component');
        this.render();
    }

    /**
     * Render component UI
     */
    render() {
        const { dataPoints, title } = this.getState();

        this.container.innerHTML = '';

        // Create header
        const header = this.createElement('h3', {
            textContent: title
        });
        this.container.appendChild(header);

        // Render data points
        const chartContainer = this.createElement('div', {
            classes: ['chart-container']
        });

        dataPoints.forEach(point => {
            const bar = this.createElement('div', {
                classes: ['chart-bar'],
                innerHTML: `<span>${point.label}</span><span>${point.value}</span>`
            });
            chartContainer.appendChild(bar);
        });

        this.container.appendChild(chartContainer);
    }

    // Private helper methods
    _addDataPoint(eventData) {
        const currentData = this.getState().dataPoints;
        currentData.push({
            label: eventData.label || 'Unknown',
            value: eventData.value || 0
        });
        this.setState({ dataPoints: currentData }, true); // true = trigger render
    }

    _updateChart(eventData) {
        if (eventData.title) {
            this.setState({ title: eventData.title }, false);
        }
        if (eventData.data) {
            this.setState({ dataPoints: eventData.data }, false);
        }
        this.render();
    }
}
```

### Step 2: Register Component

Edit `web/scripts/components/component-registry.js`:

```javascript
import { MyChartComponent } from './my-chart.js?v=10';

export class ComponentRegistry {
    // ...

    _registerComponents() {
        this.register('message-list', MessageListComponent);
        this.register('image-gallery', ImageGalleryComponent);
        this.register('my-chart', MyChartComponent);  // ← Add this line
    }
}
```

### Step 3: Add CSS Styling (Optional)

Edit `web/styles/components.css`:

```css
.my-chart-component {
    background: #ffffff;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.my-chart-component h3 {
    margin: 0 0 16px 0;
    font-size: 18px;
    color: #333;
}

.chart-container {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.chart-bar {
    display: flex;
    justify-content: space-between;
    padding: 8px 12px;
    background: #f5f5f5;
    border-radius: 4px;
    font-size: 14px;
}
```

### Step 4: Use in Workflow

```markdown
@emit
event: chart_data
data:
  label: "Q1 Sales"
  value: 125000
prompt: "✅ Added Q1 data"

@emit
event: chart_data
data:
  label: "Q2 Sales"
  value: 148000
prompt: "✅ Added Q2 data"

@emit
event: chart_update
data:
  title: "2024 Sales Performance"
prompt: "✅ Updated chart title"
```

## Advanced Patterns

### Pattern 1: Multiple Component Instances

Create separate instances using the `to` parameter:

```markdown
# System Logs Component
@emit
event: msg_info
data:
  message: "System initialized"
  to: "message-list:system-logs"
prompt: "✅ System log"

# User Activity Component
@emit
event: msg_info
data:
  message: "User John logged in"
  to: "message-list:user-activity"
prompt: "✅ User activity log"

# Another system log (goes to first instance)
@emit
event: msg_success
data:
  message: "All services running"
  to: "message-list:system-logs"
prompt: "✅ System log"
```

**Result:** Two separate MessageList components, each with its own messages.

### Pattern 2: Progressive Updates

Update a component incrementally as workflow progresses:

```markdown
@emit
event: msg_info
data:
  message: "Starting database backup..."
prompt: "✅ Step 1"

@shell
prompt: "pg_dump mydb > backup.sql"

@emit
event: msg_success
data:
  message: "Database backup completed (2.3 GB)"
prompt: "✅ Step 2"

@emit
event: msg_info
data:
  message: "Uploading to S3..."
prompt: "✅ Step 3"

@shell
prompt: "aws s3 cp backup.sql s3://my-backups/"

@emit
event: msg_success
data:
  message: "Upload completed to S3"
prompt: "✅ Step 4"
```

All messages appear in the same MessageList component, creating a log of the entire workflow.

### Pattern 3: Conditional Event Emission

Use @llm with decision logic to emit events conditionally:

```markdown
@shell
prompt: "pytest tests/ -q"
use-header: "# Test Results {id=test-results}"

@llm
prompt: |
  Analyze the test results above. If tests passed, emit a success event.
  If tests failed, emit an error event with failing test details.

  Use this format:

  @emit
  event: msg_success  # or msg_error
  data:
    message: "Your analysis here"
block: test-results
```

The LLM decides which event to emit based on test results.

### Pattern 4: Image Gallery with Generated Images

Combine @llm with image generation tools and @emit:

```markdown
@llm
prompt: |
  Generate 3 different landscape images using DALL-E tool.
  For each image, emit an event:

  @emit
  event: image_generated
  data:
    url: <image_url_from_tool>
    caption: <descriptive_caption>
tools: dalle_image_generation
```

As images are generated, they appear in the ImageGallery component in real-time.

### Pattern 5: Batch Operations

Send multiple related items at once:

```markdown
@emit
event: messages_batch
data:
  messages:
    - text: "Connected to database"
      type: info
    - text: "Loaded 1,500 records"
      type: success
    - text: "Processing started"
      type: info
prompt: "✅ Sent batch of 3 messages"
```

More efficient than individual events when you have multiple items ready.

## Troubleshooting

### Problem: Component Not Appearing

**Symptoms:**
- Event sent with @emit but no UI component visible
- No errors in browser console

**Possible Causes & Solutions:**

1. **Component not registered**
   - Check `web/scripts/components/component-registry.js`
   - Ensure `this.register('component-name', ComponentClass)` is called

2. **Event type mismatch**
   - Check component's manifest `events` array
   - Event type in @emit must exactly match registered event
   - Example: `msg_info` ≠ `message_info`

3. **Wrong mount point**
   - Check component manifest's `mountPoint` field
   - Ensure mount point exists in DOM (`chat` or `artifacts`)

4. **Using CLI mode**
   - @emit only works in web UI mode
   - CLI execution ignores @emit operations

**Debug Steps:**
```javascript
// In browser console:
console.log(componentRegistry.components); // List all registered components
console.log(componentRouter.getStats());    // See active component instances
```

### Problem: Multiple Component Instances Created

**Symptoms:**
- Each event creates a new component instead of updating existing one
- Multiple "Messages" blocks appear instead of one

**Cause:**
Component ID includes timestamp, creating unique ID for each event.

**Solution:**
Check `component-router.js:240-244` - ensure `_generateComponentId()` does NOT include `Date.now()`:

```javascript
// WRONG - includes timestamp
_generateComponentId(componentName, executionId) {
    return `${componentName}-${executionId}-${Date.now()}`;
}

// CORRECT - reuses same ID for execution
_generateComponentId(componentName, executionId) {
    return `${componentName}-${executionId}`;
}
```

### Problem: Events Going to Wrong Component

**Symptoms:**
- Messages appearing in wrong component
- Expected component not receiving events

**Possible Causes:**

1. **Multiple components registered for same event**
   - Check all component manifests for overlapping `events` arrays
   - Use explicit `to` parameter to target specific component

2. **Incorrect `to` parameter format**
   - Format must be `"component-name:instance-id"`
   - Check for typos in component name

3. **Targeting non-existent component**
   - Console shows `[ComponentRouter] Target component not found: ...`
   - Verify component is registered
   - Use automatic routing (omit `to`) to create default instance

**Fix:**
Use explicit targeting to control routing:
```markdown
@emit
event: msg_info
data:
  message: "Goes to custom instance"
  to: "message-list:custom-logs"
```

### Problem: Component State Not Updating

**Symptoms:**
- Component receives events (visible in console logs)
- UI doesn't update with new data

**Possible Causes:**

1. **Not calling `setState()` with render flag**
   ```javascript
   // WRONG - no render
   this.state.data.push(newItem);

   // WRONG - render flag false
   this.setState({ data: newData }, false);

   // CORRECT - triggers render
   this.setState({ data: newData }, true);
   ```

2. **Not calling `render()` manually**
   ```javascript
   handleEvent(eventType, eventData) {
       this.state.title = eventData.title;
       // Missing: this.render();
   }
   ```

3. **DOM not updated in `render()`**
   - Check that `render()` actually modifies `this.container`
   - Ensure elements are appended/replaced correctly

**Fix:**
Always use proper state management:
```javascript
handleEvent(eventType, eventData) {
    const currentData = this.getState().data;
    currentData.push(eventData.newItem);
    this.setState({ data: currentData }, true); // true = render
}
```

### Problem: Console Errors

**Common Errors:**

1. **"Cannot read property 'addEventListener' of null"**
   - Mount point doesn't exist in DOM
   - Check `mountPoint` in manifest matches DOM elements

2. **"componentClass.getManifest is not a function"**
   - Component class missing static `getManifest()` method
   - Add manifest to component class

3. **"TypeError: instance.handleEvent is not a function"**
   - Component missing `handleEvent()` method
   - Implement event handler in component class

4. **Import errors: "Failed to load module script"**
   - Check import paths have correct version query param: `?v=10`
   - Verify file exists at specified path

## Testing Your Components

### Test File Reference

See complete working examples in:
**`/tutorials/newchat-test/test_components.md`**

This file demonstrates:
- ✅ Single event emission (Test 1-2)
- ✅ Batch events (Test 2)
- ✅ Multiple message types (Test 3-4)
- ✅ Custom component targeting with `to` (Test 6)
- ✅ Gallery title updates (Test 5)

### Manual Testing Checklist

1. **Start dev server:**
   ```bash
   ./test_chat.sh --dev
   ```

2. **Open browser to** `http://localhost:8000/chat`

3. **Load test file** from file browser (left sidebar)

4. **Check console for logs:**
   - `[ComponentRouter] Routing event...`
   - `[ComponentRouter] Created component instance: ...`
   - `[MessageList] Received event: ...`
   - No red errors

5. **Verify visual output:**
   - Components appear in correct mount points
   - Data displays correctly
   - Multiple events update same component (not create new ones)

6. **Test explicit targeting:**
   - Emit events with `to` parameter
   - Verify separate instances created
   - Verify second event to same target reuses instance

### Automated Testing (Future)

Component system is designed for testing via:
- Selenium/Playwright for UI testing
- Jest for JavaScript unit tests
- Integration tests via Python test harness

## Cross References

### Related Documentation
- [Operations Reference](operations-reference.md) - All Fractalic operations
- [UI Server & API](ui-server-api.md) - Backend API details
- [Core Concepts](core-concepts.md) - AST and execution model

### Key Implementation Files

**Backend:**
- `core/operations/emit_op.py` - @emit operation implementation
- `core/ui_server/server.py` - Event queue and NDJSON streaming

**Frontend:**
- `web/scripts/stream-client.js` - HTTP streaming and NDJSON parsing
- `web/scripts/chat-client.js` - Event delegation to ComponentRouter
- `web/scripts/components/component-router.js` - Event routing and lifecycle
- `web/scripts/components/component-registry.js` - Component registration
- `web/scripts/components/base-component.js` - Base class for all components
- `web/scripts/components/message-list.js` - MessageList implementation
- `web/scripts/components/image-gallery.js` - ImageGallery implementation

**Styling:**
- `web/styles/components.css` - Component CSS styles

**Examples:**
- `tutorials/newchat-test/test_components.md` - Complete test suite

---

★ Insight ─────────────────────────────────────────────────────────
**Event-Driven Architecture Benefits:**
1. **Decoupling**: Backend workflows don't need to know about UI implementation
2. **Extensibility**: Add new components without modifying core Fractalic code
3. **Real-time Updates**: Components react instantly to workflow progress
4. **Multiple UIs**: Same events could drive web UI, mobile app, or desktop app
───────────────────────────────────────────────────────────────────

## Next Steps

1. **Try the examples** - Run `test_components.md` to see the system in action
2. **Create a custom component** - Follow the creation guide above
3. **Build an interactive workflow** - Combine @emit with @llm for dynamic UIs
4. **Explore integrations** - Use @emit with MCP tools for external data visualization

For questions or issues, check the [troubleshooting](#troubleshooting) section or open an issue on GitHub.
