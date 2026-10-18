FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download models to avoid cold start delays
# Lajavaness SER model
RUN python -c "from transformers import pipeline; pipeline('audio-classification', model='Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition', device=-1)" || echo "SER model download may fail offline"

# Whisper model (small for French, CPU-friendly)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')" || echo "Whisper download skipped"

# Note: Chatterbox Multilingual model is downloaded on first use via from_pretrained

EXPOSE 8000

CMD ["uvicorn", "pipeline_server:app", "--host", "0.0.0.0", "--port", "8000"]
