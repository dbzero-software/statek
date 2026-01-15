#!/bin/bash
set -e

echo "Building sdist package..."

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source distribution
python -m build --sdist

echo "Build completed successfully!"
echo "Distribution packages are in dist/"
