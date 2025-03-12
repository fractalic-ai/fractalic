# Your goals
Your goal is to process input requirments from InputParameters block and output them as custom YAML operation with provided format and parameters.

# How to prcoess input data
- In your request can be request to get historical stock values for one or few companies. 
- If few companies are presented in request: for each company you should output one instance of YAML @shell command, so for example if user requests five companies, you should print five sequential instances of 
YAML operation in one output

# Stocks Data

@llm
prompt: |
  Please output shell command for getting required company stock price
  + using as example bash processing for running pyton script like this (example for NVidia)
  + Important: replace "ticker" value with proper ticker in script
  + Important: assign to start_date and end_date specific dates in python code if they are presented as part of request from InputParameters 
  + Important: replace "days" with integer value of days if its possible to get this information from InputParameters request, that would search stock values from current date minus "days" by default. By default "days" is equal to 30 days
  + Important: in YAML operation replace {companyName} with name of company we are getting stocks values and {ticker} with proper ticjer value
  + Very important!: "ALWAYS replace {emptyline} macro with empty line in your output\""
  + For each company - print one instace of @shell command and dont print any other not required output:
  {emptyline}
  @shell
  use-header: "## {companyName} stock values"
  mode: append
  to: stocks-data/*
  prompt: |
    python3 -c '
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    ticker = "{ticker}"
    data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    print(data.to_string(index=True))
    '
use-header: none

@return
block: stocks-data/*