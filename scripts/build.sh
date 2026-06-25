#!/bin/bash
set -e

echo "Building distribution packages..."

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source and wheel distributions
python -m build --sdist --wheel

echo "Build completed successfully!"
echo "Statek distribution packages are in dist/"

echo "Installing packages..."
pip install dist/*.whl
echo "Packages installed successfully!"
