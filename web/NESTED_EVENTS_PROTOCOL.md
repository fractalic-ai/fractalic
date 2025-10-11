# Nested Execution Events Protocol

Этот документ описывает протокол событий для вложенных (nested) executions в Fractalic Chat UI.

## Структура execution IDs

- **Parent execution ID**: `dc11ec32-8e81-4b7d-b888-5e66c95e4b19`
- **Nested execution ID**: `{parent_id}_{unique_node_id}`
  - Пример: `dc11ec32-8e81-4b7d-b888-5e66c95e4b19_6f54719d-a6d`

## Routing механизм

События роутятся в нужный bubble на основе `execution_id`:

```javascript
// В emit_event() (Python)
if FRACTALIC_NESTED_EXECUTION_ID is set:
    event.execution_id = FRACTALIC_NESTED_EXECUTION_ID  // для routing
    event.parent_execution_id = FRACTALIC_EXECUTION_ID  // для связи
else:
    event.execution_id = FRACTALIC_EXECUTION_ID
```

## Полная цепочка событий

### Parent Execution (test-nested-execution.md)

```
execution_id: "dc11ec32-8e81-4b7d-b888-5e66c95e4b19"
```

1. **execution_start** → создать parent bubble
2. **chat_message** (role: system, content: "⚙️ Фракталик запущен...")
3. **ast_update** (operation: parse) → AST после парсинга
4. **ast_update** (operation: param_inject) → AST после params
5. **block_processing** (block_id: <first_run_block>) → подсветка @run
6. **workflow_start** → **СОЗДАТЬ NESTED BUBBLE #1**
   ```json
   {
     "type": "workflow_start",
     "execution_id": "dc11ec32...",  // parent (для совместимости)
     "nested_execution_id": "dc11ec32..._6f54719d-a6d",
     "parent_execution_id": "dc11ec32...",
     "node_id": "6f54719d-a6d",
     "block_id": "facd35aa",
     "target": "child-module.md"
   }
   ```

### First Nested Execution (child-module.md, run 1)

```
execution_id: "dc11ec32-8e81-4b7d-b888-5e66c95e4b19_6f54719d-a6d"
```

7. **ast_update** (operation: parse) → РОУТ в nested bubble #1
8. **ast_update** (operation: param_inject) → РОУТ в nested bubble #1
9. **block_processing** (block: @llm) → РОУТ в nested bubble #1
10. **token_usage** × 2 → РОУТ в nested bubble #1
11. **ast_update** (operation: llm) → РОУТ в nested bubble #1
12. **block_processing** (block: @return) → РОУТ в nested bubble #1
13. **ast_update** (operation: return) → РОУТ в nested bubble #1
14. **workflow_complete** → **ЗАВЕРШИТЬ NESTED BUBBLE #1**
    ```json
    {
      "type": "workflow_complete",
      "execution_id": "dc11ec32..._6f54719d-a6d",
      "parent_execution_id": "dc11ec32...",
      "nested_execution_id": "dc11ec32..._6f54719d-a6d",
      "node_id": "6f54719d-a6d",
      "target": "child-module.md",
      "return_content": "# LLM response block\n```\nNumbers: 15 and 27...",
      "explicit_return": true
    }
    ```

### Return to Parent + Second Nested

15. **block_processing** (block: second_run) → РОУТ в parent
16. **workflow_start** → **СОЗДАТЬ NESTED BUBBLE #2**
    ```json
    {
      "nested_execution_id": "dc11ec32..._7e7f0718-7f7",
      "parent_execution_id": "dc11ec32...",
      ...
    }
    ```

### Second Nested Execution (child-module.md, run 2)

```
execution_id: "dc11ec32-8e81-4b7d-b888-5e66c95e4b19_7e7f0718-7f7"
```

17-25. (аналогично first nested: ast_update, block_processing, workflow_complete)

### Final Parent Events

26. **block_processing** (block: parent_llm)
27. **token_usage** × 2 → parent bubble
28. **ast_update** (operation: llm) → parent bubble
29. **block_processing** (block: @return)
30. **workflow_complete** → parent завершён с return_content
31. **chat_message** (role: system, "✅ Фракталик завершил...")
32. **execution_complete** → финальное завершение

## UI Implementation Requirements

### 1. Event Handlers

```javascript
switch (data.type) {
  case 'workflow_start':
    // СОЗДАТЬ nested bubble
    const nestedId = data.nested_execution_id;
    const parentId = data.parent_execution_id;
    const blockId = data.block_id;

    // Find parent bubble
    const parentBubble = executionBubbles.get(parentId);

    // Create nested bubble inside parent's responseContent
    const nestedBubble = createExecutionBubble(nestedId, data.target, timestamp);

    // Insert at specific block position if blockId provided
    insertNestedBubble(parentBubble, nestedBubble, blockId);

    // Track relationship
    executionBubbles.set(nestedId, nestedBubble);
    break;

  case 'workflow_complete':
    // ЗАВЕРШИТЬ nested bubble
    const bubbleId = data.execution_id || data.nested_execution_id;
    const bubble = executionBubbles.get(bubbleId);

    // Update status
    bubble.title.innerHTML = `✓ Completed: ${data.target}`;
    bubble.bubble.style.borderColor = '#83d69d';

    // Show return_content if present
    if (data.return_content) {
      const rendered = renderMarkdownish(data.return_content);
      bubble.responseContent.appendChild(createReturnBlock(rendered));
    }
    break;

  case 'ast_update':
  case 'block_processing':
  case 'token_usage':
    // РОУТИНГ в правильный bubble
    const execId = data.execution_id;
    const targetBubble = executionBubbles.get(execId);

    // Handle event in target bubble
    if (targetBubble) {
      handleEventInBubble(data, targetBubble);
    }
    break;
}
```

### 2. Placeholder Removal

**ПРОБЛЕМА**: "Waiting for response..." должен удаляться ТОЛЬКО когда приходит первый контент для ЭТОГО bubble.

**РЕШЕНИЕ**: Проверять execution_id события и удалять placeholder только из соответствующего bubble:

```javascript
// В chat_message handler
if (isAssistantMessage && execId && executionBubbles.has(execId)) {
  const bubbleRefs = executionBubbles.get(execId);

  // Remove placeholder ONLY from THIS bubble
  const placeholder = bubbleRefs.responseContent.querySelector('[data-placeholder="true"]');
  if (placeholder) {
    placeholder.remove();
  }

  // Add content
  bubbleRefs.responseContent.appendChild(contentDiv);
}
```

### 3. Nested Bubble Insertion

Nested bubbles должны вставляться:
- **Внутри parent bubble's responseContent**
- **После конкретного block_id** (если указан)
- **С визуальными отступами** для иерархии

```javascript
function insertNestedBubble(parentBubble, nestedBubble, blockId) {
  // Add nested class for styling
  nestedBubble.bubble.classList.add('nested-bubble');

  // Insert at block position or append
  if (blockId) {
    const blockElement = parentBubble.responseContent.querySelector(`[data-block-id="${blockId}"]`);
    if (blockElement) {
      blockElement.after(nestedBubble.bubble);
      return;
    }
  }

  // Fallback: append to responseContent
  parentBubble.responseContent.appendChild(nestedBubble.bubble);
}
```

## Testing Protocol

### 1. Backend Event Check

```bash
# Run workflow and check events are emitted
python fractalic.py tutorials/newchat-test/test-nested-execution.md \
  --param_input_user_request "UserRequest" \
  --param_input_user_request_value "test"

# Verify:
# - WORKFLOW_START events have correct nested_execution_id
# - Subsequent events have matching execution_id
# - WORKFLOW_COMPLETE has return_content
```

### 2. Frontend Event Flow

```javascript
// Log ALL events to verify routing
console.log('Event:', data.type, 'execution_id:', data.execution_id);

// Verify:
// - Parent events route to parent bubble
// - Nested events route to nested bubbles
// - No placeholder remains after content arrives
```

### 3. Visual Verification

- [ ] Parent bubble created with correct title
- [ ] First nested bubble created inside parent
- [ ] Second nested bubble created inside parent
- [ ] Each nested bubble shows its own content
- [ ] No "Waiting for response..." visible in final state
- [ ] Inspect mode works for all bubbles
- [ ] Token counters accurate for each level

## Next Steps

1. ✅ Проанализировать backend events
2. ⏭ Обновить stream-client.js с правильным routing
3. ⏭ Обновить ui-rendering.js для nested bubble creation
4. ⏭ Тестировать через chat UI
5. ⏭ Проверить все режимы (Response/Inspect) для всех уровней
