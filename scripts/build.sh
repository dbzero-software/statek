#!/bin/bash
set -e

echo "Building distribution packages..."

# Clean previous builds
rm -rf dist/ build/ *.egg-info web_ui/dist/ web_ui/build/ web_ui/*.egg-info

# Build source and wheel distributions
python -m build --sdist --wheel
python -m build web_ui --sdist --wheel

echo "Build completed successfully!"
echo "Statek distribution packages are in dist/"
echo "StatekWebUI distribution packages are in web_ui/dist/"

echo "Installing packages..."
pip install dist/*.whl
pip install web_ui/dist/*.whl
echo "Packages installed successfully!"
