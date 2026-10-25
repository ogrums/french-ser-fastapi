FROM python:3.11-slim

ARG TTS_ENGINE=chatterbox
ENV TTS_ENGINE=${TTS_ENGINE}

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy both requirement files
COPY requirements-chatterbox.txt requirements-pocket.txt ./

# Install the right set according to build arg
RUN if [ "$TTS_ENGINE" = "pocket" ]; then \
        pip install --no-cache-dir -r requirements-pocket.txt; \
    else \
        pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"; \
        pip install --no-cache-dir -r requirements-chatterbox.txt; \
    fi

COPY . .

# Pre-download common models
RUN python -c "from transformers import pipeline; pipeline('audio-classification', model='Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition', device=-1)" || echo "SER download skipped"
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')" || echo "Whisper download skipped"

EXPOSE 8000

CMD ["uvicorn", "websocket_server:app", "--host", "0.0.0.0", "--port", "8000"]
