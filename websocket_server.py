from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import pipeline
from faster_whisper import WhisperModel
import torch
import torchaudio as ta
import tempfile
import os
import logging
import json
import os as os_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Robot Emotion Pipeline - WebSocket")

# Toggle : "pocket" (streaming CPU) ou "chatterbox" (défaut)
TTS_ENGINE = os_env.getenv("TTS_ENGINE", "chatterbox").lower()
logger.info(f"TTS engine : {TTS_ENGINE}")

# --- Modèles ---
SER_MODEL = "Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition"
WHISPER_SIZE = "small"

logger.info("Chargement SER...")
ser_classifier = pipeline("audio-classification", model=SER_MODEL, device=-1)

logger.info("Chargement Whisper...")
whisper_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

chatterbox_model = None
pocket_model = None
pocket_voice = None

def get_chatterbox():
    global chatterbox_model
    if chatterbox_model is None:
        logger.info("Chargement Chatterbox Multilingual...")
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    return chatterbox_model

def get_pocket():
    global pocket_model, pocket_voice
    if pocket_model is None:
        logger.info("Chargement Pocket TTS (français)...")
        from pocket_tts import TTSModel
        pocket_model = TTSModel.load_model(language="french")
        # Voix française par défaut
        pocket_voice = pocket_model.get_state_for_audio_prompt("estelle")
    return pocket_model, pocket_voice

@app.get("/")
async def root():
    return {
        "websocket": "ws://host:8000/ws/robot",
        "tts_engine": TTS_ENGINE,
        "description": "Persistent connection. Set TTS_ENGINE=pocket for streaming Pocket TTS."
    }

@app.get("/health")
async def health():
    return {"status": "ok", "tts_engine": TTS_ENGINE}

@app.websocket("/ws/robot")
async def robot_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("Robot connecté")
    try:
        while True:
            data = await websocket.receive()
            if "bytes" not in data:
                continue
            audio_bytes = data["bytes"]

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                audio_path = tmp.name

            try:
                # Émotion + transcription
                ser_results = ser_classifier(audio_path)
                top_emotion = max(ser_results, key=lambda x: x["score"])

                segments, info = whisper_model.transcribe(audio_path, language="fr")
                text = " ".join([seg.text.strip() for seg in segments])

                await websocket.send_json({
                    "type": "analysis",
                    "emotion": {"label": top_emotion["label"], "score": round(top_emotion["score"], 4)},
                    "transcription": text,
                    "emotions_all": [{"label": r["label"], "score": round(r["score"], 4)} for r in ser_results]
                })

                if not text.strip():
                    await websocket.send_json({"type": "tts_done"})
                    continue

                # --- TTS ---
                if TTS_ENGINE == "pocket":
                    model, voice = get_pocket()
                    await websocket.send_json({"type": "tts_start", "sample_rate": 24000, "format": "pcm_s16le"})

                    # Streaming chunks
                    for chunk in model.generate_audio_stream(voice, text):
                        # Convert tensor to PCM 16-bit bytes
                        pcm = (chunk.numpy() * 32767).astype("int16").tobytes()
                        await websocket.send_bytes(pcm)

                    await websocket.send_json({"type": "tts_done"})

                else:
                    # Chatterbox (non-streaming)
                    model = get_chatterbox()
                    exaggeration = 0.65 if top_emotion["label"] in ["anger", "joy", "surprise"] else 0.4
                    wav = model.generate(text, language_id="fr", exaggeration=exaggeration)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out:
                        ta.save(out.name, wav, model.sr)
                        with open(out.name, "rb") as f:
                            audio_out = f.read()
                        os.unlink(out.name)

                    await websocket.send_bytes(audio_out)
                    await websocket.send_json({"type": "tts_done"})

            finally:
                os.unlink(audio_path)

    except WebSocketDisconnect:
        logger.info("Robot déconnecté")
    except Exception as e:
        logger.error(f"Erreur WebSocket: {e}")
        await websocket.close(code=1011)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
