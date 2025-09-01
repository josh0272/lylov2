# Dockerfile
FROM python:3.11-slim

# System dependencies (libstdc++ required for faster-whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 git && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Hugging Face default port
ENV PORT=7860 \
    WHISPER_MODEL=base \
    WHISPER_COMPUTE=int8

# Launch FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
