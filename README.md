# Stateful Temporal LLM Orchestrator

A Python package for stateful temporal LLM orchestration.

## Installation

### From Source

```bash
git clone <repository-url>
cd statek
pip install -e .
```

## Development

### Setup Development Environment

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

Install development dependencies:

```bash
pip install -r requirements-dev.txt
# or
pip install -e ".[dev]"
```

### Building the Package

Build source distribution:

```bash
./scripts/build.sh
```

Or manually:

```bash
python -m build --sdist
python -m build --wheel
```

### Running Tests

```bash
pytest
```

### Code Quality

Run linters:

```bash
./scripts/run_lint.sh
```

This will run `pylint` for code quality checks.

### Docker

Build and run the package in Docker:

```bash
docker build -t statek .
docker run -it statek
```

## CI/CD

### Continuous Integration

The project uses GitHub Actions for CI:
- Runs tests on Python 3.8, 3.9, 3.10, 3.11, and 3.12
- Executes linting checks
- Builds and validates the package

### Deployment

Deployment to the custom PyPI repository is triggered on version tags:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The package will be automatically built and published to the configured repository.

## Project Structure

```
statek/
├── .github/
│   └── workflows/        # CI/CD workflows
├── scripts/              # Build and utility scripts
├── statek/               # Main package code
├── tests/                # Test files
├── Dockerfile            # Docker build configuration
├── pyproject.toml        # Package configuration
├── requirements.txt      # Runtime dependencies
├── requirements-dev.txt  # Development dependencies
└── README.md            # This file
```