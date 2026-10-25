#!/usr/bin/env python3
"""
Client de test pour le WebSocket du robot.
Usage :
  python client_ws.py --wav test.wav
  python client_ws.py --wav test.wav --url ws://localhost:8000/ws/robot
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Installe websockets : pip install websockets")
    sys.exit(1)


async def test_websocket(url: str, wav_path: str):
    wav_file = Path(wav_path)
    if not wav_file.exists():
        print(f"Fichier introuvable : {wav_path}")
        return

    audio_bytes = wav_file.read_bytes()
    print(f"Connexion à {url} ...")
    print(f"Envoi de {len(audio_bytes)} octets ({wav_file.name})")

    async with websockets.connect(url) as ws:
        # Envoi de l'audio
        await ws.send(audio_bytes)

        # Réception des messages
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                out_path = "response_tts.wav"
                Path(out_path).write_bytes(msg)
                print(f"→ Audio TTS reçu et sauvegardé : {out_path} ({len(msg)} octets)")
            else:
                try:
                    data = json.loads(msg)
                    print(f"→ JSON : {json.dumps(data, ensure_ascii=False, indent=2)}")
                    if data.get("type") == "tts_done":
                        print("Terminé.")
                        break
                except json.JSONDecodeError:
                    print(f"→ Message texte : {msg}")


def main():
    parser = argparse.ArgumentParser(description="Test client WebSocket robot")
    parser.add_argument("--wav", required=True, help="Chemin vers le fichier WAV à envoyer")
    parser.add_argument("--url", default="ws://localhost:8000/ws/robot", help="URL du WebSocket")
    args = parser.parse_args()

    asyncio.run(test_websocket(args.url, args.wav))


if __name__ == "__main__":
    main()
