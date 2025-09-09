# Fractalic Language Specification

## Introduction

Fractalic is a system that allows you to create executable workflows directly within Markdown documents. By using a specific syntax, you can structure your documents into meaningful blocks and define operations that Fractalic will execute. These operations interact with the document's content, external tools, and control the workflow's progression. Fractalic essentially brings your Markdown documents to life, enabling automation, content generation, and complex task execution.

This specification explains the language elements, concepts, and behavior of Fractalic workflows.

## Core Concepts

Understanding these core ideas is key to writing Fractalic workflows:

### The Document & The AST (Abstract Syntax Tree)

*   **Document:** Your workflow starts as a standard Markdown file (`.md`). This file contains your text, headings, and special Fractalic Operation Blocks.
*   **AST (Abstract Syntax Tree):** When Fractalic runs your document, it creates a structured, in-memory representation called the AST. This isn't just a static copy; it's the *dynamic version* of your document during execution.
    *   The AST is made of **Nodes**, each representing a significant part of your document, like a Heading or an Operation.
    *   Each Node holds information about its type, its text content, any parameters (if it's an Operation), its **role** (typically "user" for original content or "assistant" for generated content), and its position relative to other Nodes.
    *   Operations actively **modify the AST**. They can add new blocks, change content, or remove sections. The final state of the AST after the workflow finishes is the result.

### Blocks: The Building Units

Blocks are the fundamental pieces Fractalic recognizes in your Markdown.

*   **Heading Blocks:**
    *   **Syntax:** Created using standard Markdown headings (`#`, `##`, etc.).
    *   **Purpose:** Structure your document logically. They act as named sections within the AST.
    *   **Identification:** You can give a heading a unique **Block ID** using the `{id=your-unique-id}` syntax at the end of the line. This ID allows operations to target this specific section.
    *   **Content:** A Heading Block contains the text that follows it, up until the next block (Heading or Operation) of the same or a higher level.
*   **Operation Blocks:**
    *   **Syntax:** Defined by a line starting with `@operation_name`, followed by configuration parameters written in YAML format on the subsequent lines.
    *   **Purpose:** These blocks define the actions your workflow performs – the "verbs" of Fractalic (e.g., `@import` to load data, `@llm` to call an AI).
    *   **Parameters:** YAML key-value pairs configure how the operation behaves. The parameter section ends at the first blank line after it starts.
    *   **Role:** Drive the workflow by interacting with the AST, files, external services, etc., based on their function and your provided parameters.

### Block IDs and Block Paths: Addressing Content

Operations need a way to refer to specific parts of the document (AST).

*   **Block ID:**
    *   **Syntax:** Assigned via `{id=your-unique-id}` on a Heading Block line.
    *   **Purpose:** Gives a stable, unique name to a section, making it targetable by operations (e.g., in `to` or `block` parameters). The ID should be descriptive (e.g., `introduction`, `results-summary`). IDs should start with a letter and contain only letters, numbers, hyphens, or underscores.
*   **Block Path:**
    *   **Syntax:** A string used in parameters (like `block`, `to`) to specify one or more target blocks.
    *   **Formats:**
        *   `my-block-id`: Targets the single block with this ID.
        *   `section-a/sub-section-b`: Targets block `sub-section-b` specifically within the context of `section-a`. Useful for organization.
        *   `data-inputs/*`: Targets *all* blocks directly nested under the `data-inputs` block. The `/*` acts as a wildcard for direct children.
    *   **Purpose:** Allows operations to precisely specify which part(s) of the AST they should read from or write to.
    *   **Resolution:** When resolving a Block Path like `parent/child`, Fractalic primarily looks for a block with the ID `child` that is nested under the block with the ID `parent` according to the document structure (heading levels). The `/*` wildcard specifically targets the *direct children* of the preceding block ID.
    *   **Empty Blocks:** If a Block Path references a valid block ID, but that block contains no text content, operations will generally treat it as providing an empty string. It typically does not cause an error, but provides no content for context building (e.g., in `@llm` or `@run`).

### Parameters and Operation Configuration

You control how operations work using parameters.

*   **YAML Format:** Parameters are provided using standard YAML syntax within the Operation Block (lines after `@operation_name`, before the next blank line). This allows for simple key-value settings, lists, and structured configurations. For parameters accepting text (like `prompt`), standard YAML multiline strings (`|` for literal block, `>` for folded block) can be used for readability.
    ```yaml
    @shell
    prompt: |
      echo "Starting multi-step process..."
      ./run_script.sh --input data.csv
      echo "Process complete."
    ```
*   **Operation Awareness:** Each Fractalic operation knows which parameters it expects, their types (text, true/false, number, list), whether they are required or optional, and any default values it can use if you don't provide a parameter. If you provide incorrect parameters (wrong type, missing required ones, unknown ones), Fractalic will report an error.
*   **Special Parameter Interpretation:** For some parameters (like those specifying file locations or block references), Fractalic applies specific interpretation rules. For example, a string like `path/to/file.md` given to a `file` parameter will be understood as a file path, and `results/*` given to a `block` parameter will be understood as a reference to the children of the `results` block. **Note:** Relative file paths provided in parameters are typically resolved relative to the location of the Fractalic document currently being executed.

### State and Execution Context

*   **State:** The "state" of your workflow at any moment is the current content and structure of its **AST**. Operations read and modify this state.
*   **Execution Context:** This is the environment the workflow runs in, primarily defined by the current AST state.
    *   **Sub-Workflows (`@run`):** When you use `@run` to execute another Fractalic file, a *new, separate execution context* is created for that sub-workflow. It gets its own AST based on its source file.
    *   **Passing Context (`@run`):** You can pass information (using the `prompt` or `block` parameters of `@run`) from the calling workflow into the sub-workflow. This information typically becomes the initial content or part of the initial content in the sub-workflow's AST.
    *   **Isolation:** Changes made to the sub-workflow's AST generally don't affect the caller's AST directly. Only the *final state* of the sub-workflow's AST (what it looks like when it finishes or uses `@return`) is passed back and merged into the caller's AST.

## Execution Model: How Workflows Run

Fractalic follows these conceptual steps to run your document:

1.  **Parsing:**
    *   Fractalic reads your Markdown document.
    *   It identifies the structure, recognizing Heading Blocks (lines starting with `#`) and Operation Blocks (lines starting with `@`), paying attention to blank lines as delimiters.
    *   For each Operation Block, it reads the associated YAML parameters.
    *   It checks if the parameters provided for each operation are valid according to what that operation expects (correct names, types, required parameters present). It applies default values for optional parameters you didn't specify. It interprets parameters needing special handling (like file paths or block references) correctly.
    *   It builds the initial **AST** – the structured, in-memory representation of your document, ready for execution.
2.  **Execution Loop:**
    *   Fractalic starts processing the AST from the first Node.
    *   It moves through the Nodes in sequence.
    *   When it encounters an **Operation Node**:
        *   It executes the logic for that specific operation (e.g., `@import` reads a file, `@shell` runs a command).
        *   The operation uses the parameters you provided to configure its behavior.
        *   **AST Modification:** If the operation produces output (like text from a file, a response from an AI, or command output), it modifies the *current AST* based on its `to` and `mode` parameters:
            *   **Target (`to`):** Specifies *which* block(s) in the AST should receive the results. If unspecified, results are typically placed relative to the operation block itself.
            *   **Mode (`mode`):** Specifies *how* to integrate results: `append` (add after), `prepend` (add before), or `replace` (overwrite).
        *   **Flow Control:** Operations like `@goto` can change the *next* Node to be processed, altering the sequential flow. `@return` stops the current workflow level.
    *   If the node is a Heading or simple content, Fractalic generally moves to the next Node.
3.  **Termination:** The workflow stops when it reaches the end of the AST or encounters a `@return` operation in the top-level document.
4.  **Result:** The final content and structure of the AST after execution constitute the workflow's outcome. This might be implicitly saved, printed, or passed to another process.

## Operation Specifications

Here's a detailed look at each standard Fractalic operation:

### `@import`

*   **Purpose:** To bring content from another Markdown file (or specific sections of it) into your current workflow's document structure (AST). This is fundamental for reusing content, templates, or data across workflows.
*   **Parameters:**
    | Parameter  | Required | Type   | Description                                                                                                                              | Default  |
    | :--------- | :------- | :----- | :--------------------------------------------------------------------------------------------------------------------------------------- | :------- |
    | `file`     | Yes      | String | The path to the source Markdown file (e.g., `../shared/template.md`). Relative paths resolved from the current file's location.          | -        |
    | `block`    | No       | String | A Block Path specifying which block(s) to import from the source file (e.g., `setup-instructions` or `results/*`). Imports the entire file if omitted. | -        |
    | `mode`     | No       | String | How to merge the imported content into the target location: `append` (add after), `prepend` (add before), `replace` (overwrite).       | `append` |
    | `to`       | No       | String | A Block Path specifying the target block in the *current* document where the imported content should be placed. If omitted, content is placed relative to the `@import` operation itself. | -        |
    | `run-once` | No       | Boolean| If `true`, this specific `@import` executes only the first time it's encountered *during a single workflow execution pass*. It will run again if the entire workflow is restarted. | `false`  |
*   **Example:**
    ```yaml
    # Import the 'introduction' section from another file and append it here
    @import
    file: ../common/definitions.md
    block: introduction
    mode: append
    ```
*   **Execution Flow:**
    1.  **Find Source File:** Fractalic locates the file specified by the `file` parameter, resolving relative paths from the current document's directory. If the file cannot be found, it stops with an error.
    2.  **Load Source Content:** It reads the *entire* source file and builds a temporary AST representing that file's structure and content.
    3.  **Select Content:**
        *   If the `block` parameter *was not* provided, the entire AST of the source file is prepared for insertion.
        *   If the `block` parameter *was* provided, Fractalic uses the specified Block Path to find the corresponding block(s) within the temporary source AST. If the referenced block(s) don't exist in the source file, it stops with an error. Only the selected block(s) and their content are prepared for insertion.
    4.  **Identify Target Location:** Fractalic determines where in the *current* workflow's AST the imported content should go.
        *   If the `to` parameter was provided, it finds the block(s) matching that Block Path in the current AST. This is the target.
        *   If `to` was *not* provided, the target location is implicitly determined relative to the `@import` block itself (typically immediately following it).
    5.  **Insert Content:** The selected content (either the whole source file AST or the specific block AST fragment) is merged into the current workflow's AST at the target location using the specified `mode` (`append`, `prepend`, or `replace`). The current workflow's AST is now updated with the imported content.
    6.  Fractalic proceeds to the next operation.

### `@llm`

*   **Purpose:** To interact with a configured Large Language Model (LLM). It constructs a prompt based on your parameters, sends it to the LLM, and inserts the model's response back into the workflow's AST.
*   **Parameters:**
    | Parameter     | Required | Type                    | Description                                                                                                                               | Default  |
    | :------------ | :------- | :---------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :------- |
    | `prompt`      | Maybe    | String                  | Literal text to be included in the prompt sent to the LLM. Supports YAML multiline syntax.                                                 | -        |
    | `block`       | Maybe    | String or List[String]  | Block Path(s) specifying existing blocks in the current AST whose content should be included in the prompt.                                 | -        |
    | `media`       | No       | List[String]            | Paths to media files (like images) to be included with the prompt (requires a model supporting multimodal input).                         | -        |
    | `save-to-file`| No       | String                  | File path where the raw LLM response text will be saved (overwrites existing file). Relative paths resolved from the current file's location. The response *header* (if any) is not saved here.       | -        |
    | `use-header`  | No       | String                  | A Markdown header to place before the LLM response block in the AST. Can include `{id=new-id}` to assign an ID. Use the special value `"none"` (case-insensitive) to omit the header entirely. | -        |
    | `mode`        | No       | String                  | How to merge the LLM response block into the target location: `append`, `prepend`, `replace`.                                             | `append` |
    | `to`          | No       | String                  | Block Path specifying the target block in the current AST where the response block should be placed. If omitted, placed relative to `@llm`. | -        |
    | `provider`    | No       | String                  | Specify an LLM provider (e.g., "openai", "anthropic") to use, overriding the default configuration.                                       | -        |
    | `model`       | No       | String                  | Specify a specific LLM model (e.g., "gpt-4", "claude-3-sonnet") to use, overriding the default.                                             | -        |
    | `temperature` | No       | Number (0-1)            | Controls the randomness of the LLM response (higher value means more random). Overrides the default.                                      | -        |
    | `run-once`    | No       | Boolean                 | If `true`, this specific `@llm` call will only happen the first time Fractalic encounters it *during a single workflow execution pass*.    | `false`  |
    | `stop-sequences` | No | List[String] | List of strings where the model should stop generation (for Anthropic models it maps to `stop_sequences` parameter). | - |
    | `tools` | No | String/Array | Specify which tools to use: 'none' for no tools (default), 'all' for all tools, an array of specific tool names, or an array mixing tool names and '@mcpname' patterns to include all tools from specific MCP servers (e.g., ['tool1', '@playwright-mcp', 'tool2']). When set to 'none', streaming mode is automatically enabled. | "none" |
    | `tools-turns-max` | No | Integer | Maximum number of tool calls allowed for this @llm operation. If set, overrides the default or global tool call limit for this operation only. | - |
    | `context` | No | String (auto\|none) | Controls implicit preceding-context injection when only `prompt` is provided. `auto` (default) includes preceding blocks/headings; `none` disables adding any preceding context so only the literal `prompt` (and any explicit `block` content if supplied) is sent. | `auto` |
*   **Constraint:** You must provide *at least one* of `prompt` or `block`.
*   **Example:**
    ```yaml
    # Analyze the 'raw-data' block and put results under 'analysis' header
    @llm
    prompt: |
      Analyze the following data for anomalies:
    block: raw-data
    use-header: "## LLM Analysis {id=llm-analysis}"
    to: analysis-results
    mode: replace
    ```
*   **Execution Flow:**
    1.  **Validate Input:** Checks that either `prompt` or `block` (or both) is provided.
    2.  **Assemble Prompt Content:** Determines the content to send to the LLM based on the `prompt` and `block` parameters:
        *   **`block` only:** Retrieves the content of the specified block(s) from the current AST. This content *alone* forms the prompt context provided to the LLM.
        *   **`prompt` only:** Retrieves the content of *all blocks preceding the `@llm` operation* in the current AST. This accumulated preceding content is provided as context, followed by the literal `prompt` text.
        *   **`prompt` only + `context: none`:** Skips implicit preceding content; only the literal `prompt` text is sent.
        *   **Both `block` and `prompt`:** Retrieves the content of the specified `block`(s) from the current AST. This specific block content is provided as context, followed by the literal `prompt` text.
    3.  **Prepare API Call:** Identifies the LLM provider, model, and temperature (using defaults or the provided overrides). Includes any specified `media` file references.
    4.  **Call LLM:** Sends the assembled prompt content (and media, if applicable) to the designated LLM API.
    5.  **Receive Response:** Gets the text response back from the LLM.
    6.  **Save Response (Optional):** If `save-to-file` was provided, writes the raw LLM response text to the specified file.
    7.  **Prepare Response Block:** Creates a new block (or blocks, if the response is structured) containing the LLM's response text.
        *   If `use-header` was provided (and isn't `"none"`), adds the specified Markdown header before the response content. If the header includes `{id=...}`, the new response block gets that ID.
    8.  **Identify Target Location:** Determines where in the current workflow's AST the new response block should go, based on the `to` parameter or defaulting to a location relative to the `@llm` block.
    9.  **Insert Response Block:** Merges the prepared response block into the current workflow's AST at the target location using the specified `mode`.
    10. Fractalic proceeds to the next operation.

### `@run`

*   **Purpose:** To execute another Fractalic Markdown file as a self-contained sub-workflow. Allows breaking down complex tasks into modular, reusable components. Can pass input to the sub-workflow and merge its results back into the main workflow.
*   **Parameters:**
    | Parameter  | Required | Type                    | Description                                                                                                                                                       | Default  |
    | :--------- | :------- | :---------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------- |
    | `file`     | Yes      | String                  | The path to the Fractalic Markdown file to execute (e.g., `subtasks/data-processing.md`). Relative paths resolved from the current file's location.              | -        |
    | `prompt`   | No       | String                  | Literal text to provide as input context to the sub-workflow. Supports YAML multiline syntax.                                                                     | -        |
    | `block`    | No       | String or List[String]  | Block Path(s) specifying blocks from the *current* AST whose content should be provided as input context to the sub-workflow.                                        | -        |
    | `use-header`| No      | String                  | If input is provided via `prompt` or `block`, this Markdown header will be prepended to that input content *before* it's passed to the sub-workflow. Use `"none"` (case-insensitive) to omit. | -        |
    | `mode`     | No       | String                  | How to merge the *final results* (the sub-workflow's final AST state) back into the *calling* workflow's AST: `append`, `prepend`, `replace`.                     | `append` |
    | `to`       | No       | String                  | Block Path specifying the target block in the *calling* workflow's AST where the sub-workflow's results should be placed. If omitted, placed relative to the `@run` block. | -        |
    | `run-once` | No       | Boolean                 | If `true`, this specific `@run` execution will only happen the first time Fractalic encounters it *during a single workflow execution pass*.                      | `false`  |
*   **Example:**
    ```yaml
    # Run a sub-workflow using data from 'input-data' block
    @run
    file: ./sub_process.md
    block: input-data
    to: processed-results
    mode: append
    ```
*   **Execution Flow:**
    1.  **Locate Sub-Workflow File:** Finds the Markdown file specified by `file`, resolving relative paths from the current document's directory. Fails if not found.
    2.  **Assemble Input Context:** Prepares the input data to be passed to the sub-workflow:
        *   Retrieves content from the current AST for any Block Paths specified in `block`.
        *   If only `prompt` is provided: Retrieves content of *all blocks preceding the `@run` operation* in the current AST as context, followed by the literal `prompt` text.
        *   If only `block` is provided: Uses only the content of the specified block(s) as context.
        *   If both are provided: Uses the content of the specified `block`(s) as context, followed by the literal `prompt` text.
        *   If no input context was assembled (e.g., `block` refers to a non-existent block, or `prompt` is empty), the sub-workflow still executes, but with an empty initial context.
        *   If any input context was assembled, *and* `use-header` was provided (and isn't `"none"`), prepends the specified header to the input content.
    3.  **Create Sub-Workflow Context:** Initializes a new, isolated execution environment for the sub-workflow.
    4.  **Execute Sub-Workflow:** Starts the standard Fractalic execution process (Parse -> Execute Loop -> Terminate) on the target `file`.
        *   The assembled input context (from step 2) is made available to the sub-workflow, typically appearing as the initial content in its AST (perhaps under a standard header like `# Input Parameters`).
    5.  **Receive Results:** When the sub-workflow finishes (reaches its end or hits a `@return`), its final AST state is captured as the result.
    6.  **Identify Target Location (Caller):** Back in the calling workflow, Fractalic determines where the sub-workflow's results should be inserted into the *caller's* AST, based on the `to` parameter or defaulting relative to the `@run` block.
    7.  **Insert Results:** Merges the entire final AST received from the sub-workflow into the caller's AST at the target location, using the specified `mode`.
    8.  The calling workflow proceeds to the next operation.

### `@shell`

*   **Purpose:** Executes a command or script in the system's shell (like Bash, Zsh, or Windows CMD/PowerShell) and captures its standard output, inserting it back into the workflow's AST. Useful for running external tools, scripts, or system commands.
*   **Parameters:**
    | Parameter  | Required | Type   | Description                                                                                                                                      | Default                          |
    | :--------- | :------- | :----- | :----------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------- |
    | `prompt`   | Yes      | String | The shell command(s) to execute. Can be a single line or a multi-line script block (use YAML multiline syntax `|` or `>`).                      | -                                |
    | `use-header`| No      | String | A Markdown header for the block containing the command's output. Supports `{id=new-id}`. Use the special value `"none"` (case-insensitive) to omit. Defaults if not specified. | `# OS Shell Tool response block` |
    | `mode`     | No       | String | How to merge the command output block into the target location: `append`, `prepend`, `replace`.                                                    | `append`                         |
    | `to`       | No       | String | Block Path specifying the target block in the current AST where the output block should be placed. If omitted, placed relative to the `@shell` block. | -                                |
    | `run-once` | No       | Boolean| If `true`, this specific command execution will only happen the first time Fractalic encounters it *during a single workflow execution pass*.       | `false`                          |
*   **Example:**
    ```yaml
    # List files and append the output to the 'file-listing' block
    @shell
    prompt: ls -la
    to: file-listing
    mode: append
    use-header: "## Directory Contents"
    ```
*   **Execution Flow:**
    1.  **Get Command:** Retrieves the command string(s) from the `prompt` parameter.
    2.  **Execute Command:** Executes the command using the operating system's default shell. Fractalic waits for the command to complete.
    3.  **Capture Output:** Captures the text printed by the command to its standard output (stdout). Handling of standard error (stderr) might vary; typically, only stdout is captured for insertion.
    4.  **Prepare Output Block:** Creates a new block containing the captured standard output text.
        *   If `use-header` is *not* `"none"`, prepends the specified header (or the default header if `use-header` wasn't provided) to the output content. Assigns ID if specified in the header.
    5.  **Identify Target Location:** Determines where in the current workflow's AST the new output block should go, based on the `to` parameter or defaulting relative to the `@shell` block.
    6.  **Insert Output Block:** Merges the prepared output block into the current workflow's AST at the target location using the specified `mode`.
    7.  Fractalic proceeds to the next operation.

### `@return`

*   **Purpose:** To explicitly stop the execution of the *current* workflow level and optionally specify exactly what content should be passed back as the result. Primarily used within sub-workflows called by `@run` to control the data returned to the caller. If used in the top-level document, it simply stops the entire workflow.
*   **Parameters:**
    | Parameter  | Required | Type                    | Description                                                                                                                       | Default |
    | :--------- | :------- | :---------------------- | :-------------------------------------------------------------------------------------------------------------------------------- | :------ |
    | `prompt`   | Maybe    | String                  | Literal text content to be returned as the result of this workflow level. Supports YAML multiline syntax.                           | -       |
    | `block`    | Maybe    | String or List[String]  | Block Path(s) specifying blocks from the *current* AST whose content should be packaged together and returned as the result.        | -       |
    | `use-header`| No      | String                  | If returning literal `prompt` text, this optional header will be prepended to it in the returned result. Use the special value `"none"` (case-insensitive) to omit. | -       |
*   **Constraint:** You must provide *at least one* of `prompt` or `block`.
*   **Example:**
    ```yaml
    # Return the content of the 'final-summary' block
    @return
    block: final-summary
    ```
*   **Execution Flow:**
    1.  **Validate Input:** Checks that either `prompt` or `block` (or both) is provided.
    2.  **Assemble Return Content:** Determines the content to be returned:
        *   **`prompt` only:** Creates a minimal AST fragment containing the literal `prompt` text. If `use-header` was provided (and isn't `"none"`), prepends the header. (*Note: The underlying schema description suggests preceding blocks might also be included implicitly in this case, similar to `@llm`/`@run`. This specific behavior should be verified if critical.*)
        *   **`block` only:** Finds the specified block(s) in the current AST using the Block Path(s). Extracts these blocks and their content into a new, temporary AST fragment. If multiple blocks are specified, they are appended sequentially in the fragment. If any specified block is not found, it stops with an error. Handles empty blocks gracefully (includes them as empty content).
        *   **Both `prompt` and `block`:** Based on the schema description, finds the specified `block`(s) and extracts their content, then appends the literal `prompt` text to form the returned AST fragment. Content from `block` comes first.
    3.  **Terminate Execution:** Immediately stops processing any further operations *at the current workflow level*.
    4.  **Return Result:** The assembled AST fragment (from step 2) becomes the return value of this workflow level.
        *   If this was a sub-workflow called by `@run`, this returned AST fragment is passed back to the `@run` operation in the calling workflow for merging.
        *   If this was the top-level workflow, execution ends, and this fragment represents the final result.

### `@goto`

*   **Purpose:** To unconditionally change the execution flow *within the current document*, jumping directly to a specified Heading Block ID. Allows for creating loops or conditional jumps (though explicit conditional logic isn't built-in, `goto` can be used to skip sections).
*   **Parameters:**
    | Parameter  | Required | Type   | Description                                                                                                   | Default |
    | :--------- | :------- | :----- | :------------------------------------------------------------------------------------------------------------ | :------ |
    | `block`    | Yes      | String | The **Block ID** (must be a simple ID, no `/` or `*`) of the target Heading Block to jump execution to.        | -       |
    | `run-once` | No       | Boolean| If `true`, this specific `@goto` jump will only happen the first time Fractalic encounters it *during a single workflow execution pass*. | `false` |
*   **Example:**
    ```yaml
    # Jump back to the 'processing-loop' block
    @goto
    block: processing-loop
    ```
*   **Execution Flow:**
    1.  **Get Target ID:** Retrieves the target Block ID from the `block` parameter.
    2.  **Find Target Node:** Locates the Heading Block Node within the *current* AST that has the matching ID. If no block with that ID exists, it stops with an error.
    3.  **Jump Execution:** Modifies the execution runner's internal pointer. Instead of proceeding to the node immediately following the `@goto` block, the runner will process the located target Node next.
    4.  Execution continues sequentially *from the target node*.
    *   **Caution:** It is easy to create infinite loops using `@goto`. Ensure your logic provides an eventual exit path or use `run-once: true` where appropriate.

## Error Handling

Workflows can encounter errors at different stages:

*   **Parsing Errors:** Issues with the Markdown structure or invalid YAML syntax within Operation Blocks.
*   **Validation Errors:** Providing incorrect parameters to operations (missing required fields, wrong data types, unknown parameters) or using invalid formats for Block IDs or Block Paths.
*   **Runtime Errors:** Trying to access files that don't exist (`@import`, `@run`), referencing Block IDs or Paths that aren't found in the relevant AST (`BlockNotFoundError`), or using an operation name that isn't defined (`@unknown-op`).
*   **Execution Errors:** Problems occurring during an operation's execution, such as errors from external LLM APIs, shell commands failing, or file system permission issues.

When an error occurs, Fractalic typically stops execution and provides a message indicating the nature and location of the problem.

## Glossary

*   **AST (Abstract Syntax Tree):** The structured, in-memory representation of a Fractalic document during execution. Operations read from and modify the AST.
*   **Node:** An individual element within the AST, typically representing a Heading Block or an Operation Block from the source Markdown. Holds content, parameters, and metadata like `role`.
*   **Block:** A fundamental structural unit in the source Markdown, either a Heading Block or an Operation Block.
*   **Heading Block:** A standard Markdown heading (`#`, `##`, etc.), potentially with an ID (`{id=...}`), used for structure and targeting.
*   **Operation Block:** A block starting with `@operation_name` followed by YAML parameters, defining an action to be executed.
*   **Block ID:** A unique identifier (e.g., `results-summary`) assigned to a Heading Block using `{id=...}`, used for targeting operations.
*   **Block Path:** A string used in parameters to reference blocks, potentially using `/` for nesting or `/*` for targeting children (e.g., `section-a/data-input`, `results/*`).
*   **Operation:** An action defined by an Operation Block (e.g., `@import`, `@llm`).
*   **Parameter:** A configuration setting (key-value pair) provided in YAML within an Operation Block to control its behavior.
*   **Role:** Metadata on an AST Node, typically "user" for original content or input, and "assistant" for content generated by operations. Can influence context building for certain operations like `@llm`.
*   **State:** The current content and structure of the AST at any point during workflow execution.
*   **Context (Execution Context):** The environment in which an operation runs, primarily consisting of the current AST state. For `@run`, a new, isolated context is created for the sub-workflow.
*   **Mode:** A parameter (`append`, `prepend`, `replace`) determining how operation results are merged into the target location in the AST.
