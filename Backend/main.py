from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
import uuid
import httpx
import json
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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

        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()


async def forward_request(url: str, json_data: dict):
    """Asynchronously sends a POST request to the AI model."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=json_data, timeout=10.0)
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

@app.post("/dataSegment/start")

def start_data_segment():
    segment_id = str(uuid.uuid4())
    segments[segment_id] = { "status": "running", "started_at": datetime.now().isoformat() }
    return {"segment_id": segment_id, "status": "started"}
@app.post("/dataSegment/stop")
def stop_data_segment(segment_id: str):
    if segment_id not in segments: raise HTTPException(status_code=404, detail="Segment not found")
    segments[segment_id]["status"] = "stopped"
    return {"segment_id": segment_id, "status": "stopped"}
@app.post("/dataSegment/connect")
def connect_data_segment(segment_id: str):
    if segment_id not in segments: segments[segment_id] = {}
    segments[segment_id]["status"] = "connected"
    return {"segment_id": segment_id, "status": "connected"}
@app.post("/dataSegment/disconnect")
def disconnect_data_segment(segment_id: str):
    if segment_id not in segments: raise HTTPException(status_code=404, detail="Segment not found")
    segments[segment_id]["status"] = "disconnected"
    return {"segment_id": segment_id, "status": "disconnected"}
# ---

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
    """
    MODIFIED: Receives processed data from the AI and BROADCASTS it to all
    connected frontend clients via WebSocket.
    """
    print(f"Received AI response. Broadcasting to frontend clients...")
    await manager.broadcast(response.json()) # Broadcast the JSON data
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