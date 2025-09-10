#!/bin/bash
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"
echo "Running VitePress..."
exec ./node_modules/.bin/vitepress dev
