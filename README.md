# Robot Emotion Pipeline API

Serveur FastAPI pour un petit robot capable de :
1. **Reconnaître l'émotion** dans la voix (modèle Lajavaness français)
2. **Transcrire** l'audio (Whisper)
3. **Générer une réponse vocale émotionnelle** (Chatterbox Multilingual)

## Modèles utilisés

| Tâche | Modèle | Taille | Notes |
|-------|--------|--------|-------|
| SER (émotion) | `Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition` | ~300M | Français réel, 5 classes |
| ASR | Whisper `small` (faster-whisper) | ~500M | Bon en français, CPU ok |
| TTS émotionnel | Chatterbox Multilingual | ~500M | 23 langues, contrôle `exaggeration` |

Les modèles SER et Whisper sont **pré-téléchargés** dans le Docker. Chatterbox se télécharge au premier appel TTS.

## Classes d'émotions (SER)

- Pleased, Relaxed, Neutral, Sad, Tension

## Lancement

### Docker (recommandé)

```bash
git clone https://github.com/ogrums/french-ser-fastapi.git
cd french-ser-fastapi
docker compose up --build
```

Le build télécharge les modèles (peut prendre plusieurs minutes la première fois).

### Local

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python pipeline_server.py
```

## Endpoints

### 1. Émotion seule
```bash
curl -X POST http://localhost:8000/v1/audio/emotions -F "file=@audio.wav"
```

### 2. Transcription
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions -F "file=@audio.wav" -F "language=fr"
```

### 3. Analyse complète (émotion + texte)
```bash
curl -X POST http://localhost:8000/v1/audio/analyze -F "file=@audio.wav"
```

### 4. TTS avec émotion
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -F "text=Bonjour, je suis content de te voir !" \
  -F "language=fr" \
  -F "exaggeration=0.7" \
  --output reponse.wav
```

`exaggeration` : 0.25 = calme/neutre, 0.5 = normal, 0.7+ = plus expressif/dramatique.

## Pipeline robot typique

1. Robot enregistre 3-10s d'audio via micro
2. `POST /v1/audio/analyze` → obtient émotion + transcription
3. LLM externe génère une réponse adaptée à l'émotion
4. `POST /v1/audio/speech` avec le texte + `exaggeration` selon l'émotion détectée
5. Robot joue le fichier audio

## Notes matérielles

- CPU suffit pour SER + Whisper small.
- TTS Chatterbox est plus agréable avec un GPU (CUDA).
- Audio recommandé : 16 kHz mono WAV.

## Licence

Modèles sous leurs licences respectives (Apache 2.0 pour Lajavaness, etc.). Code fourni tel quel.
