# French Speech Emotion Recognition API

Mini serveur FastAPI pour la reconnaissance d'émotions dans la voix française, basé sur le modèle [Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition](https://huggingface.co/Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition).

Conçu pour un petit robot qui écoute via micro et envoie un fichier audio pour analyse émotionnelle.

## Classes d'émotions

- Pleased
- Relaxed
- Neutral
- Sad
- Tension

## Installation locale

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python ser_server.py
```

## Docker

```bash
docker compose up --build
```

L'API sera disponible sur `http://localhost:8000`.

## Utilisation

```bash
curl -X POST "http://localhost:8000/v1/audio/emotions" \
  -F "file=@audio.wav"
```

### Exemple de réponse

```json
{
  "model": "Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition",
  "filename": "audio.wav",
  "top_emotion": {
    "label": "Neutral",
    "score": 0.72
  },
  "emotions": [
    {"label": "Neutral", "score": 0.72},
    {"label": "Tension", "score": 0.15}
  ]
}
```

## Notes

- Préfère les fichiers audio à 16 kHz mono.
- Tourne sur CPU (modèle ~300M params).
- Pour un robot : enregistre un clip de 3-10 secondes et envoie-le à l'endpoint.

## Licence

Le modèle Lajavaness est sous Apache 2.0. Ce code est fourni tel quel.
