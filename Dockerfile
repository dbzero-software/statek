FROM python:3.14-slim as builder
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

WORKDIR /app

# Copy package files
COPY pyproject.toml README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md MANIFEST.in ./
COPY statek/ ./statek/
COPY tests/ ./tests/

# Build the package
RUN uv build --sdist

# Final stage
FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

WORKDIR /app

# Copy built distribution
COPY --from=builder /app/dist/*.tar.gz /app/dist/

# Install the package
RUN uv pip install --system --no-cache /app/dist/*.tar.gz

