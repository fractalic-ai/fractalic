# Agent Goals
Okay, your task is to tell a funny one sentence punchline about 
the Tree of Thoughts.

## Joke Generation Instructions
- Print only the joke, with no comments.

@llm
prompt: Hello, Claude Sonnet, please provide your punchline.
temperature: 0.7
use-header: "# Joke 1"

@llm
prompt: Hello, ChatGPT, please provide your punchline.
block: agent-goals/*
temperature: 0.7
provider: openai
model: gpt-4o
use-header: "# Joke 2"

@llm
prompt: Rate punchlines from 1st place to 2nd, explain result in one brief sentence
block:
    - joke-1
    - joke-2
provider: groq
model: qwen-2.5-32b
temperature: 0.9



