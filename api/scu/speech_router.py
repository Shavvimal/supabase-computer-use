from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import os
import random

router = APIRouter()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "eeo6VQB8twAd1zWSnjeB"

class TextInput(BaseModel):
    text: str

@router.post("/text-to-ant")
async def text_to_speech(input: TextInput):
    text = input.text
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")

    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API key not configured.")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
    }
    payload = {
        "text": text,
        # Include additional parameters if needed
        # "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.75,
            "similarity_boost": 0.75
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with ElevenLabs API: {str(e)}")

    return StreamingResponse(response.iter_content(chunk_size=8192), media_type="audio/mpeg")

@router.post("/angry-ant")
async def angry_ant():
    """
    curl -X POST "http://localhost:8000/angry-ant" -H "Content-Type: application/json" --output output_audio.mp3
    """
    phrases = ["Hmmmm... Alright, here we go again... for the third time today, let me explain this. Slowly. Ughhhh.",
    "Ahem... I swear I've answered this before, but sure, let's walk through it. Again. Mmmhmm.",
    "Have you checked the documentation? Pfft... It's almost like I wrote it for this exact question. Sighhhh.",
    "Ughhhh... Oh, another ticket about this? Shocker. Hmmmmmm.",
    "Mmmph... I'm starting to think I need to tattoo this solution on my forehead. Ughhh.",
    "Hnnnngh... Not to sound repetitive, but this is literally the same thing I said earlier. Hmmmm.",
    "Ughhhh... Did you even read the email I sent, or was it just for decoration? Hmmmph.",
    "Haaaaaah... Let me just paste my response from 20 minutes ago and 20 minutes before that. Mmmhmm.",
    "Hmmmm... Sure, I'll fix your problem. Again. Because apparently, I'm a wizard. Ughhhhh.",
    "Hnnnngh... Alright, let's break it down. Step one: Listen. Step two: Remember. Hmmmph.",
    "Ughhhhh... Let me explain this one more time, and I'll try to use smaller words. Hmmmm.",
    "Haaaaah... You know what? Let's pretend this is the first time I'm seeing this issue. Mmmhmm.",
    "Hmmmph... I'm having the strangest feeling of deja vu right now. How peculiar.",
    "Ughhhhh... Fascinating. This looks exactly like the problem we solved yesterday. And the day before.",
    "Hnnnngh... Well, well, well. If it isn't my favorite recurring question. Again."]
    # choose a random phrase from the list
    text = random.choice(phrases)

    ## Stream back the audio
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")

    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API key not configured.")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
    }
    payload = {
        "text": text,
        # Include additional parameters if needed
        # "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.75,
            "similarity_boost": 0.75
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with ElevenLabs API: {str(e)}")

    return StreamingResponse(response.iter_content(chunk_size=8192), media_type="audio/mpeg")



