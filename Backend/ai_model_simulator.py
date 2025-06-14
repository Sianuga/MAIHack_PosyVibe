from fastapi import FastAPI, BackgroundTasks
import httpx
import uuid
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

# --- Configuration ---

MAIN_BACKEND_URL = "http://localhost:8000"
AI_MODEL_WEBHOOK_ENDPOINT = "/webhooks/aiModel"

app = FastAPI()


class DataPayload(BaseModel):
    data: Dict[str, Any]
    timestamp: Optional[datetime] = None

class AIResponse(BaseModel):
    processed_data: Dict[str, Any]
    request_id: str


async def forward_to_backend(response: AIResponse):
    """Forwards the processed data back to the main backend."""
    target_url = f"{MAIN_BACKEND_URL}{AI_MODEL_WEBHOOK_ENDPOINT}"
    try:
        async with httpx.AsyncClient() as client:
            await client.post(target_url, json=response.dict(), timeout=10.0)
        print(f"Successfully sent processed data back to {target_url}")
    except httpx.RequestError as e:
        print(f"Error sending processed data to backend: {e}")


@app.post("/process")
async def process_data(payload: DataPayload, background_tasks: BackgroundTasks):
    """
    This endpoint simulates an AI model.
    It receives data, processes it, and sends it back to the main API.
    """
    print(f"AI model received data: {payload.data}")

    # --- Simulate AI Processing ---
    # For this simulation, we'll just add 1 to every integer or float value.
    processed_data = {}
    for key, value in payload.data.items():
        if isinstance(value, (int, float)):
            processed_data[key] = value + 1
        else:
            processed_data[key] = value
    # -----------------------------

    print(f"AI model processed data: {processed_data}")


    ai_response = AIResponse(
        processed_data=processed_data,
        request_id=str(uuid.uuid4())
    )


    background_tasks.add_task(forward_to_backend, ai_response)

    return {"status": "processing_started", "original_data": payload.data}

@app.get("/")
def root():
    return {"status": "AI Model Simulator is running"}