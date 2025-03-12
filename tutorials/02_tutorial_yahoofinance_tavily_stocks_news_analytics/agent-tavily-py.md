# Your Task
Generate a shell command for the `tool_tavily_search.py` script for the request specified in the **Input parameters** block. Ensure the command adheres to the following rules:
1. The required `--query` parameter.
2. The required `--task` parameter (`search` or `extract`).
3. Optional parameters with defaults if not specified.
4. Properly formatted output in the format:

```bash
@shell
  prompt: 'python3 tool_tavily_search.py --task "<TASK>" --query "<QUERY>" [options]'
  use-header: "# Web search result {id=web-search-result}"
```

Example:

```bash
@shell
  prompt: 'python3 tool_tavily_search.py --task "search" --query "AI news" --search_depth advanced --max_results 10'
  use-header: "# Web search result {id=web-search-result}"
```

# Script Parameters Definition
- `TASK`: Required. The task to perform, either `"search"` or `"extract"`.
- `QUERY`: Required. The search query string or URLs to extract, e.g., `"AI news"`.
- `SEARCH_DEPTH`: `"basic"` or `"advanced"`. Default: `"basic"`.
- `TOPIC`: `"general"` or `"news"`. Default: `"general"`.
- `DAYS`: Integer, e.g., `7`. Default: `3`. *(Only if `TOPIC` is `"news"`)*
- `MAX_RESULTS`: Integer, e.g., `10`. Default: `5`.
- `INCLUDE_IMAGES`: Add this flag to include images in the response.
- `INCLUDE_IMAGE_DESCRIPTIONS`: Add this flag to include image descriptions. Requires `INCLUDE_IMAGES` flag.
- `INCLUDE_ANSWER`: Add this flag to include a short LLM-generated answer.
- `INCLUDE_RAW_CONTENT`: Add this flag to include the parsed HTML content of search results.
- `INCLUDE_DOMAINS`: Comma-separated list, e.g., `"domain1.com,domain2.com"`.
- `EXCLUDE_DOMAINS`: Comma-separated list, e.g., `"excludedomain.com"`.

---

# Example Usage
To search for "AI news" with advanced depth and include images:
```bash
@shell
  prompt: 'python3 tool_tavily_search.py --task "search" --query "AI news" --search_depth advanced --include_images'
  use-header: "# Web search result {id=web-search-result}"
```

To include domains:
```bash
@shell
  prompt: 'python3 tool_tavily_search.py --task "search" --query "Machine Learning" --include_domains "domain1.com,domain2.com"'
  use-header: "# Web search result {id=web-search-result}"
```

To extract content from URLs:
```bash
@shell
  prompt: 'python3 tool_tavily_search.py --task "extract" --query "https://example.com,https://another.com"'
  use-header: "# Web search result {id=web-search-result}"
```

# Important Notes
- Flags such as `--include_answer` should not be followed by a value (e.g., `true`). Simply add the flag to enable the feature.
- Always precede @shell code with an empty line in your output.
- Remember to remove any markdown in your output of the command, such as "``` bash" or similar formatting.
- If the task is simple, like known facts retrieval, try to limit `MAX_RESULTS` to an appropriate value.


@shell
prompt: chmod +x tool_tavily_search.py


@llm
prompt: |
  Generate a shell command for the `tool_tavily_search.py` script 
  for request specified in the **Input parameters** block.
  Do not use any ``` or markdown, just command required starting from @shell
use-header: none

@return
block: web-search-result
