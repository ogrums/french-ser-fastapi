# French SER + Pipeline FastAPI (Robot)

Mini serveur FastAPI pour reconnaissance d'émotions (français) + pipeline complet (SER + ASR + TTS).

## Deux moteurs TTS (séparés à cause des conflits numpy)

| Moteur | Fichier | Numpy | Usage |
|--------|---------|-------|-------|
| **Chatterbox** | `requirements-chatterbox.txt` | < 2 | Contrôle d'émotion (`exaggeration`) |
| **Pocket TTS** | `requirements-pocket.txt` | ≥ 2 | Streaming CPU, français |

## Build Docker

```bash
# Chatterbox (défaut)
docker compose up --build

# Pocket TTS
docker compose build --build-arg TTS_ENGINE=pocket
# puis dans docker-compose.yml change TTS_ENGINE=pocket
docker compose up
```

Ou en une commande :

```bash
docker compose build --build-arg TTS_ENGINE=pocket && \
docker compose up -e TTS_ENGINE=pocket
```

## WebSocket

`ws://host:8000/ws/robot`

- Envoie un WAV (bytes)
- Reçoit JSON `analysis`
- Puis audio (WAV complet ou chunks PCM selon le moteur)
- Puis JSON `tts_done`

## Client de test

```bash
pip install websockets
python client_ws.py --wav test.wav
```
