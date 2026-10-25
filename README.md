# French SER + Pipeline FastAPI (Robot)

Mini serveur FastAPI pour reconnaissance d'émotions (français) + pipeline complet (SER + ASR + TTS) destiné à un petit robot.

## Toggle TTS

Variable d'environnement :

- `TTS_ENGINE=chatterbox` (défaut) — TTS complet, contrôle d'émotion via exaggeration
- `TTS_ENGINE=pocket` — Pocket TTS (CPU, streaming, français, voix "estelle")

Exemple :
```bash
TTS_ENGINE=pocket docker compose up --build
```

## WebSocket

**Endpoint** : `ws://host:8000/ws/robot`

### Protocole

1. Client envoie un WAV (bytes).
2. Serveur envoie JSON `analysis` (émotion + transcription).
3. Si `TTS_ENGINE=pocket` :
   - JSON `tts_start` (sample_rate, format)
   - Plusieurs messages **bytes** (chunks PCM 16-bit)
   - JSON `tts_done`
4. Si Chatterbox :
   - Un seul message **bytes** (WAV complet)
   - JSON `tts_done`

### Client de test

```bash
pip install websockets
python client_ws.py --wav test.wav
```

## Démarrage

```bash
docker compose up --build
```

## Modèles

- SER : Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition
- ASR : faster-whisper `small`
- TTS : Chatterbox **ou** Pocket TTS (toggle)
