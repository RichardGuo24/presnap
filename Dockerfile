FROM python:3.12-slim

# libgomp1 is scikit-learn's OpenMP runtime; the wheel needs it at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install deps first so the layer caches unless requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY presnap/ ./presnap/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Default: run the test suite. Compose overrides this to run the smoke pipeline.
CMD ["python", "-m", "pytest", "tests/", "-q"]
