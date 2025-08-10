# YAML Edge Cases Test

This file tests various YAML edge cases that the linter should handle properly.

## Empty Parameters

@llm
prompt: Simple prompt with minimal params

## Multiline YAML

@llm
prompt: |
  This is a multiline prompt
  with multiple lines
  that should work correctly
use-header: |
  # Complex Header
  with multiline content
mode: append

## Complex Block References

@import
file: ../data/complex.md
block: section-a/subsection-b/*
mode: replace
to: target-section/*

## Array Parameters

@llm
block:
  - data-section
  - analysis-section/*
  - results/summary
prompt: Combine all these sections
tools: all

## Nested YAML Structure

@run
file: ./processor.md
block: input-data
mode: append
to: processed-results
run-once: true

## Boolean and Numeric Values

@llm
prompt: Test numeric and boolean values
temperature: 0.7
tools-turns-max: 5
header-auto-align: true
run-once: false

## Special Characters in Strings

@shell
prompt: |
  echo "Testing special chars: @#$%^&*()[]{}|;:,.<>?/~`"
  echo 'Single quotes with "double quotes" inside'
use-header: "# Shell Output with Special: @#$%"

## Empty Blocks

@llm
prompt: Test with empty target
to: empty-section

## Empty Section {id=empty-section}

## Quoted Values

@llm
prompt: "This is a quoted prompt"
use-header: "# Quoted Header"
mode: "append"
