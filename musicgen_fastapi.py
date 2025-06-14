import uuid, pathlib, asyncio, base64, json
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
from musicgen_utils import generate_and_render_music

# ───────── config ─────────
OUTPUT_DIR = pathlib.Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ───────── request schema ─────────
class PromptBody(BaseModel):
    prompt: str

# ───────── app ─────────
app = FastAPI(
    title="AI-Music Generator",
    description="Generate calmer / happier / energize tracks and get a WAV.",
    version="1.0"
)

# (Optional) CORS for browser tests
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ───────── thread-helper ─────────
async def run_sync(func, *args, **kw):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kw))

# ───────── REST route ─────────
@app.post("/generate")
async def generate_music(body: PromptBody):
    prompt_text = body.prompt.strip()
    if not prompt_text:
        raise HTTPException(400, "Prompt cannot be empty")

    uid        = uuid.uuid4().hex
    midi_path  = OUTPUT_DIR / f"{uid}.mid"
    wav_path   = OUTPUT_DIR / f"{uid}.wav"

    try:
        await run_sync(generate_and_render_music,
                       prompt_text, str(midi_path), str(wav_path))
    except Exception as e:
        raise HTTPException(500, str(e))

    return FileResponse(str(wav_path), media_type="audio/wav",
                        filename="track.wav", headers={"Cache-Control": "no-store"})

# ───────── cleanup helper ─────────
@app.delete("/cleanup")
async def cleanup():
    removed = 0
    for f in OUTPUT_DIR.glob("*"):
        try:
            f.unlink()
            removed += 1
        except Exception:
            pass
    return {"deleted": removed}

# ───────── WebSocket route ─────────
@app.websocket("/ws")
async def music_socket(ws: WebSocket):
    await ws.accept()
    try:
        while True:                       # keep socket alive
            data = await ws.receive_text()
            try:
                body        = json.loads(data)
                prompt      = body.get("prompt", "").strip()
                send_bytes  = bool(body.get("binary", False))
            except json.JSONDecodeError:
                prompt, send_bytes = data.strip(), False

            if not prompt:
                await ws.send_text("ERROR: empty prompt")
                continue

            uid        = uuid.uuid4().hex
            midi_path  = OUTPUT_DIR / f"{uid}.mid"
            wav_path   = OUTPUT_DIR / f"{uid}.wav"

            await ws.send_text("STATUS: generating")
            try:
                await run_sync(generate_and_render_music,
                              prompt, str(midi_path), str(wav_path))
            except Exception as e:
                await ws.send_text(f"ERROR: {e}")
                continue

            await ws.send_text("STATUS: complete")

            if send_bytes:
                await ws.send_text("TYPE: binary.b64")
                with open(wav_path, "rb") as f:
                    while chunk := f.read(16384):
                        await ws.send_text(base64.b64encode(chunk).decode())
                await ws.send_text("END")
            else:
                await ws.send_text(f"URL:/download/{wav_path.name}")
    except WebSocketDisconnect:
        pass

# ───────── tiny download helper ─────────
@app.get("/download/{file}")
def download(file: str):
    fp = OUTPUT_DIR / file
    if not fp.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(str(fp), media_type="audio/wav", filename=file)
