#!/bin/bash

# Fractalic Documentation Development Script

set -e

echo "🚀 Setting up Fractalic Documentation"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

echo "📦 Installing dependencies..."
npm install

echo "🔧 Starting development server..."
echo "📚 Documentation will be available at http://localhost:5173"
echo "Press Ctrl+C to stop the server"

npm run dev
