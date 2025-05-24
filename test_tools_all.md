# Test Tools All

This is a test to verify that `tools: all` parameter works correctly with LLM operations.

@llm
prompt: |
  Please use the fractalic_shell tool to list the current directory contents. 
  Just run a simple command like "ls" or "dir" depending on the system.
tools: all
