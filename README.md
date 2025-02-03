# Fractalic

Program AI in plain language (any language). That's it.

## Vision 🚀

Modern AI workflows shouldn’t be harder than they already are. We have powerful LLMs, yet we still end up wrestling with Python scripts and tangled spaghetti-like node-based editors. **Fractalic** aims to fix this by letting you build AI systems as naturally as writing a simple doc.

## What is Fractalic? ✨

Fractalic combines Markdown and YAML to create agentic AI systems using straightforward, human-readable documents. It lets you grow context step by step, control AI knowledge precisely, and orchestrate complex workflows through simple document structure and syntax.

## Key Features

🧬 **Structured Knowledge & Precision** - Use Markdown heading blocks to form a semantic tree. Reference specific nodes or branches with a simple, path-like syntax.

🧠 **Dynamic Knowledge context** - Each operation can modify specific nodes and branches, allowing your system to evolve dynamically.
    
🤖 **Agentic ready** - The module system (`@run`) isolates execution contexts. It passes parameters and returns results as semantic blocks, enabling both specialized agents and reusable workflows with clear inputs and outputs.

🪄 **Runtime Instruction Generation** Generate instructions dynamically during execution, or delegate this task to the LLM. This enables conditional workflows and supports autonomous agent behavior.
    
🤝 **Multi-Model Collaboration** - You can explicitly specify LLM provider, model, and parameters (e.g., temperature) for each call.
    
🖥️ **Shell Integration** - Execute CLI tools and scripts (Python, curl, Docker, Git, etc.) and automatically update the knowledge context with the results.
    
📝 **Transparent Versioning** - Automatically track every context change and decision using Git-native version control. 
    
📒 **User Interface** - A notebook-like UI provides straightforward parameter selection and operation management.

## Quick 101
When a Markdown file is executed (either directly or called as an agent/module), the interpreter creates a context tree in memory. This tree consists of two types of blocks:

1. Knowledge blocks: These correspond to Markdown header sections AND their associated content (all text/content under that header until the next header or operation). 
2. Operation blocks: These are YAML instructions starting with a custom @operation name. They receive inputs, prompts, and block references as parameters.

Block identifiers can be:
- Explicitly defined in headers using {id=block-name}
- Automatically generated by converting header text to kebab-case (lowercase, no spaces or special characters)

Operations can modify the context, return results, and generate new operations within the current execution context. The images referenced show a comparison between an original Markdown file and its resulting context (.ctx) file.

![alt text](<docs/images/slide_01.png>)

In this example, we define text blocks and run LLM generations sequentially. Key points:

1. First operation combines all previous blocks with the prompt (default behavior) using global settings (Claude-3.5-Sonnet)

2. Results are stored in context under "# LLM Response block" (ID: llm-response-block). Headers and nesting levels can be customized via use-header parameter

3. Second LLM call (DeepSeek R-1 via OpenAI-like API) can be restricted to original task block only, preventing access to previous results. This enables independent model comparison or evaluation

4. The blocks parameter provides access to all context blocks, including those appearing later in the file, as the entire file loads into context during execution

5. Temperature can be adjusted per operation (global default is 0.0, recommended for prototyping workflows and processes)

1. Operations
@llm: AI-powered content generation

@shell: Execute system commands

@import: Include external files or blocks

@run: Execute sub-documents

@goto: Conditional flow control


## Quick intro 1
![alt text](<docs/images/slide_02.png>)

## Quick intro 2
![alt text](<docs/images/slide_03.png>)




# fractalic updating in progress
Hello, repo updating is in progress. Please wait for a while.

# Requirements
Important: please ensure you have Git installed on your system.
Git dependecy would be removed in future releases. Sessions would be stored on .zip or .tar.gz files.

# Installation (Docker)
As for now, the best way to use Fraxtalic is to install both interpreter with backend server and UI frontend and run it as odcker container, if you dont need docker - please skip that step.

```bash
git clone https://github.com/fractalic-ai/fractalic.git && \
git clone https://github.com/fractalic-ai/fractalic-ui.git  && \
cd fractalic && \
cp -a docker/. .. && \
cd .. && \
docker build -t fractalic-app . && \
docker run -d \
  -p 8000:8000 \
  -p 3000:3000 \
  --name fractalic-app \
  fractalic-app
```
Now UI should be avaliable on http://localhost:3000
Please be aware to connect local folder with .md files to persist changes
Same action is required for settings toml


# Installation (Local)

1. Backend installation
```bash
git clone https://github.com/fractalic-ai/fractalic.git && \
cd fractalic && \
python3 -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt 
```
2. Run backend
```bash
./run_server.sh
```

3. Frontend installation
```bash
git clone https://github.com/fractalic-ai/fractalic-ui.git  && \
cd fractalic-ui && \
npm install 
```

4. Run frontend
```bash
npm run dev
```



# Running fractalic backend server
Required for UI to work. Please run the following command in the terminal.
```bash
./run_server.sh
```

# Settings
First time you run the UI, settings.toml would be created required for parser (at least while working from UI, if you are using it headless from CLI - you can use script CLI params). You should select default provider and enter env keys for external providers (repicate, tavily and etc).