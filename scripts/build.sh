#!/bin/bash
set -e

echo "Building distribution packages..."

# Clean previous builds
rm -rf dist/ build/ *.egg-info StatekWebUI/dist/ StatekWebUI/build/ StatekWebUI/*.egg-info

# Build source and wheel distributions
python -m build --sdist --wheel
python -m build StatekWebUI --sdist --wheel

echo "Build completed successfully!"
echo "Statek distribution packages are in dist/"
echo "StatekWebUI distribution packages are in StatekWebUI/dist/"

echo "Installing packages..."
pip install dist/*.whl
pip install StatekWebUI/dist/*.whl
echo "Packages installed successfully!"
