import pytest
from unittest.mock import MagicMock
import io
import sys

# Mock heavy dependencies before importing the app
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False

mock_pipeline_fn = MagicMock()
mock_classifier = MagicMock()
mock_classifier.return_value = [
    {"label": "Neutral", "score": 0.75},
    {"label": "Tension", "score": 0.15},
    {"label": "Relaxed", "score": 0.10},
]
mock_pipeline_fn.return_value = mock_classifier

mock_whisper_cls = MagicMock()
mock_whisper_inst = MagicMock()
seg = MagicMock()
seg.text = "Bonjour comment ça va"
mock_whisper_inst.transcribe.return_value = ([seg], MagicMock(language="fr"))
mock_whisper_cls.return_value = mock_whisper_inst

sys.modules["torch"] = mock_torch
sys.modules["transformers"] = MagicMock(pipeline=mock_pipeline_fn)
sys.modules["faster_whisper"] = MagicMock(WhisperModel=mock_whisper_cls)
sys.modules["chatterbox"] = MagicMock()
sys.modules["chatterbox.mtl_tts"] = MagicMock()
sys.modules["torchaudio"] = MagicMock()

import pipeline_server
pipeline_server.ser_classifier = mock_classifier
pipeline_server.whisper_model = mock_whisper_inst

from fastapi.testclient import TestClient
client = TestClient(pipeline_server.app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert "ser" in data["endpoints"]
    assert "tts" in data["endpoints"]

def test_emotions():
    audio = io.BytesIO(b"RIFF" + b"\x00"*200)
    files = {"file": ("test.wav", audio, "audio/wav")}
    r = client.post("/v1/audio/emotions", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["top_emotion"]["label"] == "Neutral"
    assert data["top_emotion"]["score"] == 0.75

def test_transcription():
    audio = io.BytesIO(b"RIFF" + b"\x00"*200)
    files = {"file": ("test.wav", audio, "audio/wav")}
    r = client.post("/v1/audio/transcriptions", files=files, data={"language": "fr"})
    assert r.status_code == 200
    data = r.json()
    assert "Bonjour" in data["text"]
    assert data["language"] == "fr"

def test_full_analyze():
    audio = io.BytesIO(b"RIFF" + b"\x00"*200)
    files = {"file": ("test.wav", audio, "audio/wav")}
    r = client.post("/v1/audio/analyze", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["emotion"]["label"] == "Neutral"
    assert "transcription" in data
