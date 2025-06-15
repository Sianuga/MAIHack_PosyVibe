import uuid, pathlib, asyncio, base64, json, httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# Import your existing music generation logic
from musicgen_utils import generate_and_render_music

# ----------------- CONFIGURATION -----------------
OUTPUT_DIR = pathlib.Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# URLs for the data pipeline architecture
MAIN_BACKEND_WEBHOOK = "http://localhost:8000/webhooks/aiModel"
MUSIC_SERVICE_BASE_URL = "http://localhost:8002"

# ----------------- PYDANTIC MODELS -----------------
# Model for your original "/generate" endpoint
class PromptBody(BaseModel):
    prompt: str

# Models for the webhook data pipeline
class DataPayload(BaseModel):
    data: Dict[str, Any]
    timestamp: Optional[datetime] = None

class AIResponse(BaseModel):
    processed_data: Dict[str, Any]
    request_id: str

# ----------------- FASTAPI APP -----------------
app = FastAPI(
    title="Integrated AI-Music Generator",
    description="Generates music from direct prompts or via a data pipeline webhook.",
    version="1.1"
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- HELPERS -----------------
async def run_sync(func, *args, **kw):
    """Runs a synchronous function in a separate thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kw))

# ----------------- NEW WEBHOOK LOGIC -----------------
async def generate_and_notify(payload: DataPayload):
    """
    This is the background task for the webhook.
    1. Creates a music prompt from sensor data.
    2. Runs your music generation function.
    3. Notifies the main backend with a URL to the finished track.
    """
    # 1. Synthesize a music prompt from the incoming data

    #CHANGE TEMP TO EMOTIONS-------------------
    temp = payload.data.get("temperature", 25.0)
    prompt = ""
    if temp >= 28.0:
        prompt = "Energize: a fast, 160 bpm electronic track with a driving beat."
    elif temp <= 22.0:
        prompt = "Make Calmer: a slow, 70 bpm ambient, soothing track."
    else:
        prompt = "Make Happier: an upbeat, 120 bpm pop track."

    print(f"AI Service: Generated prompt from data: '{prompt}'")

    # 2. Generate music file using your existing logic
    uid = uuid.uuid4().hex
    midi_path = (OUTPUT_DIR / f"{uid}.mid").resolve()
    wav_path = (OUTPUT_DIR / f"{uid}.wav").resolve()

    try:
        await run_sync(generate_and_render_music, prompt, str(midi_path), str(wav_path))
        print(f"AI Service: Music generated successfully: {wav_path.name}")

        # 3. Prepare the response payload for the main backend
        download_url = f"{MUSIC_SERVICE_BASE_URL}/download/{wav_path.name}"
        processed_data = {
            "message": "Music generated from sensor data.",
            "music_url": download_url,
            "original_prompt": prompt,
            "source_data": payload.data
        }
        response_payload = AIResponse(processed_data=processed_data, request_id=uid)

        # 4. Notify the main backend by calling its webhook
        async with httpx.AsyncClient() as client:
            await client.post(MAIN_BACKEND_WEBHOOK, json=response_payload.dict(), timeout=10.0)
        print(f"AI Service: Notified main backend with URL: {download_url}")

    except Exception as e:
        print(f"AI Service ERROR: Failed during music generation or notification: {e}")


@app.post("/process")
async def process_data_from_webhook(payload: DataPayload, background_tasks: BackgroundTasks):
    """
    NEW: This is the endpoint the main backend will call.
    It receives sensor data and starts the generation task in the background.
    """
    print(f"AI Service: Received data from main backend: {payload.data}")
    background_tasks.add_task(generate_and_notify, payload)
    return {"status": "music_generation_task_accepted"}


# ----------------- YOUR ORIGINAL ENDPOINTS (preserved for direct testing) -----------------

@app.post("/generate")
async def generate_music_direct(body: PromptBody):
    """For direct testing: POST a prompt and get a WAV file back."""
    prompt_text = body.prompt.strip()
    if not prompt_text:
        raise HTTPException(400, "Prompt cannot be empty")

    uid = uuid.uuid4().hex
    midi_path = OUTPUT_DIR / f"{uid}.mid"
    wav_path = OUTPUT_DIR / f"{uid}.wav"

    try:
        await run_sync(generate_and_render_music, prompt_text, str(midi_path), str(wav_path))
    except Exception as e:
        raise HTTPException(500, str(e))

    return FileResponse(str(wav_path), media_type="audio/wav", filename="track.wav")


@app.websocket("/ws")
async def music_socket_direct(ws: WebSocket):
    """For direct testing: Connect via WebSocket to generate music."""
    await ws.accept()
    # ... (Your original WebSocket logic is preserved here) ...
    try:
        while True:
            data = await ws.receive_text()
            try:
                body = json.loads(data)
                prompt = body.get("prompt", "").strip()
            except json.JSONDecodeError:
                prompt = data.strip()

            if not prompt: continue
            uid = uuid.uuid4().hex
            midi_path, wav_path = OUTPUT_DIR/f"{uid}.mid", OUTPUT_DIR/f"{uid}.wav"
            await ws.send_text("STATUS: generating")
            try:
                await run_sync(generate_and_render_music, prompt, str(midi_path), str(wav_path))
            except Exception as e:
                await ws.send_text(f"ERROR: {e}")
                continue
            await ws.send_text("STATUS: complete")
            await ws.send_text(f"URL:/download/{wav_path.name}")
    except WebSocketDisconnect:
        pass


@app.get("/download/{file}")
def download(file: str):
    """ESSENTIAL: Serves the generated audio file for both direct and webhook use."""
    fp = OUTPUT_DIR / file
    if not fp.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(fp), media_type="audio/wav", filename=file)

@app.delete("/cleanup")
async def cleanup():
    # ... (Your cleanup logic is preserved) ...
    removed = 0
    for f in OUTPUT_DIR.glob("*"):
        try: f.unlink(); removed += 1
        except Exception: pass
    return {"deleted": removed}