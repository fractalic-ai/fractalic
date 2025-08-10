# Valid Fractalic Syntax Test

This file contains valid Fractalic syntax that should pass all linting checks.

## Introduction

This is a properly structured Fractalic document for testing.

## Data Processing

@import
file: ../shared/data.md
mode: append

## LLM Analysis

@llm
prompt: |
  Analyze the imported data and provide insights.
  Focus on key trends and patterns.
use-header: "# Analysis Results"
mode: replace
to: analysis-results

## Shell Command

@shell
prompt: echo "Processing complete"
use-header: "# Processing Status"

## Analysis Results {id=analysis-results}

This section will be populated by the LLM operation.

## Conditional Flow

@llm
prompt: Check if additional processing is needed
tools: none
use-header: none

@goto
block: additional-processing

## Additional Processing {id=additional-processing}

Additional steps would go here.

## Final Results

@return
block: analysis-results/*
use-header: "# Final Report"
