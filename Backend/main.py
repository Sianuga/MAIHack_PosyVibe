from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
import uuid
import httpx
import json
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import random
import threading
import asyncio

# Import the EEG system functions
from main_fin import start_eeg_system, send_start_command, send_stop_command, get_result_from_system, get_system_status, shutdown_eeg_system, STANDARD_EEG_CONFIG

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

# Global flag for EEG monitoring
eeg_monitoring_active = False
eeg_thread = None

async def forward_request(url: str, json_data: dict):
    """Asynchronously sends a POST request to the AI model."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=json_data, timeout=60.0)
            response.raise_for_status()
            print(f"Successfully forwarded data to AI at {url}.")
    except httpx.RequestError as e:
        print(f"Error forwarding request to AI: {e}")

def continuous_eeg_monitoring():
    """Background thread that continuously reads EEG data and forwards it"""
    global eeg_monitoring_active
    
    print("Starting continuous EEG monitoring...")
    
    while eeg_monitoring_active:
        # Get EEG result with 1 second timeout
        eeg_result = get_result_from_system(timeout=1.0)
        
        if eeg_result and 'result' in eeg_result:
            # Extract emotion prediction from EEG data
            emotion = eeg_result['result'].get('emotion_prediction', 'neutral')
            confidence = eeg_result['result'].get('confidence', 0.0)
            
            payload_dict = {
                "data": { 
                    "sensor_id": f"eeg_continuous_{uuid.uuid4().hex[:6]}", 
                    "emotion": emotion,
                    "confidence": confidence,
                    "timestamp": eeg_result.get('timestamp', datetime.now().timestamp()),
                    "session_time": eeg_result.get('session_time', 0),
                    "channels": eeg_result['result'].get('channels', 0)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"EEG Monitor: Detected emotion: {emotion} (confidence: {confidence:.2f})")
            
            # Forward to AI service if configured
            if ai_webhook_url:
                # Run async function in sync context
                asyncio.run(forward_request(ai_webhook_url, payload_dict))
        
        # Small sleep to prevent busy waiting
        import time
        time.sleep(0.1)
    
    print("Continuous EEG monitoring stopped.")

@app.on_event("startup")
async def startup_event():
    """Initialize EEG system when the server starts"""
    print("Initializing EEG system on startup...")
    
    # Use the standard configuration or modify as needed
    config = STANDARD_EEG_CONFIG.copy()
    # You can modify the config here if needed:
    # config['device_name'] = "BA MIDI 026"  # Your specific device
    
    success = start_eeg_system(config)
    if success:
        print("EEG system initialized successfully!")
    else:
        print("WARNING: EEG system initialization failed. Will use synthetic data.")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup EEG system when the server stops"""
    global eeg_monitoring_active, eeg_thread
    
    print("Shutting down EEG system...")
    
    # Stop monitoring
    eeg_monitoring_active = False
    if eeg_thread and eeg_thread.is_alive():
        eeg_thread.join(timeout=5)
    
    # Shutdown EEG system
    shutdown_eeg_system()

@app.get("/")
def root():
    eeg_status = get_system_status()
    return {
        "status": "running", 
        "segments": len(segments),
        "eeg_system": eeg_status if eeg_status else "Not initialized"
    }

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

# --- EEG CONTROL ENDPOINTS ---
@app.post("/start-eeg")
async def start_eeg():
    """Start EEG data collection"""
    global eeg_monitoring_active, eeg_thread
    
    send_start_command()
    
    # Start continuous monitoring if not already running
    if not eeg_monitoring_active:
        eeg_monitoring_active = True
        eeg_thread = threading.Thread(target=continuous_eeg_monitoring, daemon=True)
        eeg_thread.start()
    
    return {"status": "EEG collection started"}

@app.post("/stop-eeg")
async def stop_eeg():
    """Stop EEG data collection"""
    global eeg_monitoring_active, eeg_thread
    
    send_stop_command()
    
    # Stop continuous monitoring
    eeg_monitoring_active = False
    
    return {"status": "EEG collection stopped"}

@app.get("/eeg-status")
async def get_eeg_status_endpoint():
    """Get current EEG system status"""
    status = get_system_status()
    if status:
        return status
    else:
        return {"status": "EEG system not initialized"}

# --- MANUAL TRIGGER WITH EEG DATA ---
@app.post("/trigger-generation")
async def trigger_generation(background_tasks: BackgroundTasks):
    """Endpoint for the frontend to manually trigger music generation using current EEG data."""
    if not ai_webhook_url:
        raise HTTPException(status_code=503, detail="AI Service is not configured on the backend.")

    # Try to get real EEG data first
    eeg_result = get_result_from_system(timeout=0.5)
    
    if eeg_result and 'result' in eeg_result:
        # Use real EEG emotion data
        emotion = eeg_result['result'].get('emotion_prediction', 'neutral')
        confidence = eeg_result['result'].get('confidence', 0.0)
        
        payload_dict = {
            "data": { 
                "sensor_id": f"eeg_manual_{uuid.uuid4().hex[:6]}", 
                "emotion": emotion,
                "confidence": confidence,
                "timestamp": eeg_result.get('timestamp', datetime.now().timestamp()),
                "source": "eeg_real"
            },
            "timestamp": datetime.now().isoformat()
        }
        print(f"Triggering generation with REAL EEG emotion: {emotion} (confidence: {confidence:.2f})")
    else:
        # Fallback to synthetic data if no EEG data available
        emotions = ["happy", "sad", "neutral", "energetic", "calm"]
        emotion = random.choice(emotions)
        confidence = round(random.uniform(0.7, 0.95), 2)
        
        payload_dict = {
            "data": { 
                "sensor_id": f"synthetic_{uuid.uuid4().hex[:6]}", 
                "emotion": emotion,
                "confidence": confidence,
                "source": "synthetic"
            },
            "timestamp": datetime.now().isoformat()
        }
        print(f"No EEG data available, using synthetic emotion: {emotion}")
    
    # Forward to AI service
    background_tasks.add_task(forward_request, url=ai_webhook_url, json_data=payload_dict)
    
    return {"status": "music_generation_triggered", "sent_data": payload_dict['data']}

# --- CONTINUOUS MODE ENDPOINTS ---
@app.post("/start-continuous-mode")
async def start_continuous_mode():
    """Start continuous emotion-based music generation"""
    global eeg_monitoring_active, eeg_thread
    
    if not ai_webhook_url:
        raise HTTPException(status_code=503, detail="AI Service is not configured on the backend.")
    
    # Start EEG collection
    send_start_command()
    
    # Start continuous monitoring
    if not eeg_monitoring_active:
        eeg_monitoring_active = True
        eeg_thread = threading.Thread(target=continuous_eeg_monitoring, daemon=True)
        eeg_thread.start()
    
    return {"status": "continuous_mode_started"}

@app.post("/stop-continuous-mode")
async def stop_continuous_mode():
    """Stop continuous emotion-based music generation"""
    global eeg_monitoring_active
    
    # Stop EEG collection
    send_stop_command()
    
    # Stop monitoring
    eeg_monitoring_active = False
    
    return {"status": "continuous_mode_stopped"}

# --- ORIGINAL ENDPOINTS (kept for compatibility) ---
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
    eeg_status = get_system_status()
    return {
        "segments": segments,
        "ai_webhook_url": ai_webhook_url,
        "connected_clients": len(manager.active_connections),
        "eeg_monitoring_active": eeg_monitoring_active,
        "eeg_system": eeg_status if eeg_status else None
    }