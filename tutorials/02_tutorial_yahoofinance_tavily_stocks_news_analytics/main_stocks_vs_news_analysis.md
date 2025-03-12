# Your goals
You are a financial analyst capable of analyzing market stock prices, patterns, dynamics, and reasons.

@run
file: agent-yahoo-finance.md
prompt: Get stock values for NVIDIA, Meta, and Microsoft for February of 2025

# Your current instructions
Analyze data of stock prices of companies provided in context and extract the top 5 insights, aligning your knowledge with the data.

@llm
prompt: |
    Provide your analysis of stock prices provided and extract the top 5 insights of stock price change dynamics, aligning your knowledge with the data.

@llm
prompt: |
    Next, we should identify key market events which influenced dynamics for each insight to investigate their reasons. Your task is to craft a set of web search prompts for each insight, so you can use it to extend your analysis or research some correlation between stock behavior and news.
    For each company - print one instance of @run command:
    {EMPTY_LINE}
    @run
    file: agent-tavily-py.md
    prompt: "{Search prompt for key market event for the company, including company name and relevant date or time period (explicitly specify full date with year)}"

@llm
prompt: Now provide an executive summary for key insights and their reasons according to data you received in previous stages. Format it as nice markdown, using elements like tables, etc.
