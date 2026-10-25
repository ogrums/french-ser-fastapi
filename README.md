# French SER + Pipeline FastAPI (Robot)

Mini serveur FastAPI pour reconnaissance d'émotions (français) + pipeline complet (SER + ASR + TTS) destiné à un petit robot.

## Endpoints HTTP classiques

- `POST /v1/audio/emotions` — émotion uniquement
- `POST /v1/audio/transcriptions` — transcription
- `POST /v1/audio/analyze` — émotion + transcription
- `POST /v1/audio/speech` — TTS (Chatterbox, paramètre `exaggeration`)

## WebSocket (recommandé pour le robot)

**Endpoint** : `ws://host:8000/ws/robot`

1. Le robot envoie un WAV (bytes).
2. Le serveur répond avec un JSON :
   ```json
   {
     "type": "analysis",
     "emotion": {"label": "joy", "score": 0.87},
     "transcription": "Bonjour, je vais bien",
     "emotions_all": [...]
   }
   ```
3. Puis envoie l'audio TTS (bytes) + un JSON `{"type": "tts_done"}`.

Connexion persistante → faible latence, faible overhead mémoire/CPU pour le robot.

## Démarrage

```bash
docker compose up --build
# ou localement :
# uvicorn websocket_server:app --host 0.0.0.0 --port 8000
```

## Modèles

- SER : Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition
- ASR : faster-whisper `small` (CPU)
- TTS : Chatterbox Multilingual (chargé à la demande)

## Ressources (petit robot)

Les modèles sont chargés **une seule fois**. La connexion WebSocket elle-même est très légère. Adapté pour garder la connexion ouverte en permanence.
