FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir build

# Copy package files
COPY pyproject.toml README.md ./
COPY statek/ ./statek/
COPY tests/ ./tests/

# Build the package
RUN python -m build --sdist

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy built distribution
COPY --from=builder /app/dist/*.tar.gz /app/dist/

# Install the package
RUN pip install --no-cache-dir /app/dist/*.tar.gz

