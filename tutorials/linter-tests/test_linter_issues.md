# Test file with linting issues

This is a test document to verify the linter functionality.

## Valid Section

This section should pass validation.

@llm
prompt: This is a valid LLM operation
use-header: "# LLM Response"

## Problem Section

@unknown_operation
some-param: value
another-param: test

## YAML Issue Section

@llm
prompt: |
  This is a multiline prompt
  that should work fine
invalid-param: this-should-fail

empty line within YAML

mode: append

## Orphaned Content Issue

@shell
prompt: echo "test"

This content appears after operation without proper block structure.

## Missing Blank Line Issue

@llm
prompt: test
@llm
prompt: another operation without blank line separation
