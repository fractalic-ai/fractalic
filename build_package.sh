#!/bin/bash
# Build and publish script for Fractalic PyPI package

set -e

echo "🔨 Building Fractalic package..."

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build the package
python -m build

echo "✅ Package built successfully!"
echo "📦 Package files:"
ls -la dist/

echo ""
echo "To publish to PyPI:"
echo "  Test PyPI: python -m twine upload --repository testpypi dist/*"  
echo "  PyPI:      python -m twine upload dist/*"
echo ""
echo "To install locally:"
echo "  pip install dist/fractalic-*.whl"
