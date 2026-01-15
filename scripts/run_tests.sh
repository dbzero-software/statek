#!/bin/bash
set -e

echo "Running tests..."

# Run pytest with coverage
pytest tests/ -v

echo "All tests passed!"
