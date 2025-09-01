# Dockerfile
FROM python:3.11-slim

# System deps: ffmpeg for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . .

# Defaults (override in Render if needed)
ENV WHISPER_MODEL=tiny \
    WHISPER_COMPUTE=int8 \
    PYTHONUNBUFFERED=1

# Bind to $PORT on Render; 8000 locally
CMD ["bash", "-lc", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
