from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="French Speech Emotion Recognition API",
    description="Mini serveur basé sur Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition",
    version="1.0.0"
)

MODEL_NAME = "Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition"

logger.info(f"Chargement du modèle {MODEL_NAME}...")
try:
    classifier = pipeline(
        "audio-classification",
        model=MODEL_NAME,
        device=-1  # CPU par défaut (mettre 0 pour GPU)
    )
    logger.info("Modèle chargé avec succès")
except Exception as e:
    logger.error(f"Erreur au chargement du modèle: {e}")
    raise

@app.get("/")
async def root():
    return {
        "message": "French SER API is running",
        "model": MODEL_NAME,
        "classes": ["Pleased", "Relaxed", "Neutral", "Sad", "Tension"],
        "endpoint": "POST /v1/audio/emotions"
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/audio/emotions")
async def analyze_emotion(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni")

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"Analyse du fichier: {file.filename}")
        results = classifier(tmp_path)

        emotions = [
            {"label": r["label"], "score": round(r["score"], 4)}
            for r in results
        ]
        top = max(results, key=lambda x: x["score"])

        return {
            "model": MODEL_NAME,
            "filename": file.filename,
            "top_emotion": {"label": top["label"], "score": round(top["score"], 4)},
            "emotions": emotions
        }
    except Exception as e:
        logger.error(f"Erreur d'inférence: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
