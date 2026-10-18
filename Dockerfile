FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ser_server.py .

# Pre-download model (optional, can be slow on first run)
# RUN python -c "from transformers import pipeline; pipeline('audio-classification', model='Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition')"

EXPOSE 8000

CMD ["uvicorn", "ser_server:app", "--host", "0.0.0.0", "--port", "8000"]
