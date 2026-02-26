# Use a lightweight Python base image
FROM python:3.12-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (modern Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application
COPY . .

# Final sync to include the project itself
RUN uv sync --frozen --no-dev

# Set the entry point to run with uv
ENTRYPOINT ["uv", "run"]

# Default command (can be overridden)
CMD ["python", "generate.py"]
