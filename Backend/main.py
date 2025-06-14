from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
import uuid
import httpx
import json
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import random # <-- IMPORT RANDOM

origins = [
    "http://localhost:3000",
    "http://localhost",
    "http://127.0.0.1:3000",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataPayload(BaseModel):
    data: Dict[str, Any]
    timestamp: Optional[datetime] = None

class AIResponse(BaseModel):
    processed_data: Dict[str, Any]
    request_id: str

segments = {}
ai_webhook_url: Optional[str] = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        print(f"Broadcasting to {len(self.active_connections)} client(s): {message}")
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

async def forward_request(url: str, json_data: dict):
    """Asynchronously sends a POST request to the AI model."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=json_data, timeout=60.0)
            response.raise_for_status()
            print(f"Successfully forwarded data to AI at {url}.")
    except httpx.RequestError as e:
        print(f"Error forwarding request to AI: {e}")

@app.get("/")
def root():
    return {"status": "running", "segments": len(segments)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """The endpoint the React frontend will connect to."""
    await manager.connect(websocket)
    print("Frontend client connected via WebSocket.")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Frontend client disconnected.")

# --- NEW ENDPOINT FOR FRONTEND TRIGGER ---
@app.post("/trigger-generation")
async def trigger_generation(background_tasks: BackgroundTasks):
    """A simple endpoint for the frontend to kick off the music generation pipeline."""
    if not ai_webhook_url:
        raise HTTPException(status_code=503, detail="AI Service is not configured on the backend.")

    temp = round(random.uniform(18.0, 32.0), 2)
    payload_dict = {
        "data": { "sensor_id": f"frontend_trigger_{uuid.uuid4().hex[:6]}", "temperature": temp },
        "timestamp": datetime.now().isoformat()
    }
    print(f"Triggering generation with synthetic data: {payload_dict['data']}")
    
    # Use the existing background task logic to forward the request
    background_tasks.add_task(forward_request, url=ai_webhook_url, json_data=payload_dict)
    
    return {"status": "music_generation_triggered", "sent_data": payload_dict['data']}


@app.post("/dataSegment/start")
def start_data_segment():
    # This endpoint is no longer used in the main flow but is kept for compatibility/testing.
    segment_id = str(uuid.uuid4())
    segments[segment_id] = { "status": "running", "started_at": datetime.now().isoformat() }
    return {"segment_id": segment_id, "status": "started"}

@app.post("/dataSegment/stop")
def stop_data_segment(segment_id: str):
    if segment_id not in segments: raise HTTPException(status_code=404, detail="Segment not found")
    segments[segment_id]["status"] = "stopped"
    return {"segment_id": segment_id, "status": "stopped"}

@app.post("/webhooks/dataSegment")
async def data_segment_webhook(payload: DataPayload, background_tasks: BackgroundTasks):
    """Receives data and forwards it to the AI model webhook."""
    print(f"Received data payload: {payload.data}")
    if ai_webhook_url:
        background_tasks.add_task(forward_request, url=ai_webhook_url, json_data=payload.dict())
        return {"status": "received_and_forwarding_to_ai"}
    else:
        return {"status": "received_but_not_forwarded (AI URL not configured)"}

@app.post("/webhooks/aiModel")
async def ai_model_webhook(response: AIResponse):
    """Receives processed data from the AI and BROADCASTS it."""
    print(f"Received AI response. Broadcasting to frontend clients...")
    await manager.broadcast(response.json())
    return {"status": "received_from_ai_and_broadcasted"}

@app.post("/config/ai")
def configure_ai_webhook(webhook_url: str):
    global ai_webhook_url
    ai_webhook_url = webhook_url
    return {"status": "configured", "webhook_url": webhook_url}

@app.get("/status")
def get_status():
    return {
        "segments": segments,
        "ai_webhook_url": ai_webhook_url,
        "connected_clients": len(manager.active_connections)
    }