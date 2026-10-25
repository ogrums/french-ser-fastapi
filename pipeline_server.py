from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from transformers import pipeline
from faster_whisper import WhisperModel
import tempfile
import os
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Robot Emotion Pipeline API",
    description="SER (Lajavaness) + ASR (Whisper) + TTS émotionnel (Chatterbox Multilingual)",
    version="1.0.0"
)

# --- Chargement des modèles ---
SER_MODEL = "Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition"
WHISPER_SIZE = "small"  # bon compromis FR / vitesse

logger.info("Chargement SER...")
ser_classifier = pipeline("audio-classification", model=SER_MODEL, device=-1)

logger.info("Chargement Whisper...")
whisper_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

# Chatterbox chargé à la demande (plus lourd)
chatterbox_model = None

def get_chatterbox():
    global chatterbox_model
    if chatterbox_model is None:
        logger.info("Chargement Chatterbox Multilingual...")
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    return chatterbox_model

@app.get("/")
async def root():
    return {
        "endpoints": {
            "ser": "POST /v1/audio/emotions",
            "asr": "POST /v1/audio/transcriptions",
            "full_analyze": "POST /v1/audio/analyze",
            "tts": "POST /v1/audio/speech"
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/audio/emotions")
async def emotions(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        results = ser_classifier(path)
        top = max(results, key=lambda x: x["score"])
        return {
            "top_emotion": {"label": top["label"], "score": round(top["score"], 4)},
            "emotions": [{"label": r["label"], "score": round(r["score"], 4)} for r in results]
        }
    finally:
        os.unlink(path)

@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), language: str = Form("fr")):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        segments, info = whisper_model.transcribe(path, language=language)
        text = " ".join([seg.text.strip() for seg in segments])
        return {"text": text, "language": info.language}
    finally:
        os.unlink(path)

@app.post("/v1/audio/analyze")
async def full_analyze(file: UploadFile = File(...)):
    """Analyse complète : émotion + transcription"""
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(content)
        path = tmp.name
    try:
        # Émotion
        ser_results = ser_classifier(path)
        top = max(ser_results, key=lambda x: x["score"])
        # Transcription
        segments, info = whisper_model.transcribe(path, language="fr")
        text = " ".join([seg.text.strip() for seg in segments])
        return {
            "emotion": {"label": top["label"], "score": round(top["score"], 4)},
            "transcription": text,
            "emotions_all": [{"label": r["label"], "score": round(r["score"], 4)} for r in ser_results]
        }
    finally:
        os.unlink(path)

@app.post("/v1/audio/speech")
async def speech(
    text: str = Form(...),
    language: str = Form("fr"),
    exaggeration: float = Form(0.5)  # 0.25=calme, 0.7+=émotion forte
):
    """TTS avec contrôle d'émotion via exaggeration"""
    model = get_chatterbox()
    wav = model.generate(text, language_id=language, exaggeration=exaggeration)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        import torchaudio as ta
        ta.save(tmp.name, wav, model.sr)
        out_path = tmp.name
    
    return FileResponse(out_path, media_type="audio/wav", filename="speech.wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
