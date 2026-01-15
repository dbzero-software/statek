#!/bin/bash
set -e

echo "Running linters..."

# Run pylint for code quality checks
echo "Running pylint..."
pylint statek/ tests/

echo "All lint checks passed!"
