# ============================================
# Review Analyzer - Docker Image
# Multi-stage build for optimized production image
# ============================================

# Build stage for dependencies
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-deps --wheel-dir /wheels -r requirements.txt


# ============================================
# Production stage
# ============================================
FROM python:3.11-slim as production

# Labels
LABEL maintainer="BCG GeoAnalytics Team" \
      description="Review Analyzer - Discover, collect, and classify reviews" \
      version="2.1.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    # Streamlit configuration
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_BASE=light

# Create non-root user for security
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser app.py /app/
COPY --chown=appuser:appuser requirements.txt /app/

# Create data directories with proper permissions
RUN mkdir -p /app/data/00_config \
             /app/data/01_raw \
             /app/data/02_interim \
             /app/data/03_processed \
             /app/data/04_analysis \
             /app/data/99_archive \
             /app/logs \
    && chown -R appuser:appuser /app

# Copy config files (templates, city data)
COPY --chown=appuser:appuser data/00_config/ /app/data/00_config/

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command - run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

