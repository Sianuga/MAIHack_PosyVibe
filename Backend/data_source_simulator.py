import requests
import time
import random
import sys
from datetime import datetime

# --- Configuration ---
MAIN_BACKEND_URL = "http://localhost:8000"
AI_MODEL_URL = "http://localhost:8001/process"
SIMULATION_DURATION_SECONDS = 30
SEND_INTERVAL_SECONDS = 2

def main():
    print("--- Starting Data Pipeline Simulation ---")
    segment_id = None
    try:

        print("\n[Step 1] Configuring AI webhook...")
        requests.post(f"{MAIN_BACKEND_URL}/config/ai", params={"webhook_url": AI_MODEL_URL}).raise_for_status()
        print(f" -> AI webhook configured to: {AI_MODEL_URL}")



        print("\n[Step 2] Starting a new data segment...")
        response = requests.post(f"{MAIN_BACKEND_URL}/dataSegment/start")
        response.raise_for_status()
        segment_id = response.json()["segment_id"]
        print(f" -> Segment started with ID: {segment_id}")


        print(f"\n[Step 3] Sending data for {SIMULATION_DURATION_SECONDS} seconds...")
        start_time = time.time()
        while time.time() - start_time < SIMULATION_DURATION_SECONDS:
            payload = { "data": { "sensor_id": f"sensor_{random.randint(1, 5)}", "temperature": round(random.uniform(20.0, 30.0), 2), "reading_count": random.randint(100, 1000) } }
            print(f"\nSending: {payload['data']}")
            res = requests.post(f"{MAIN_BACKEND_URL}/webhooks/dataSegment", json=payload)
            print(f" -> Backend responded with status: {res.status_code}")
            time.sleep(SEND_INTERVAL_SECONDS)

    except requests.exceptions.RequestException as e:
        print(f"\nERROR: Could not connect to the backend at {MAIN_BACKEND_URL}", file=sys.stderr)
        print("Please ensure the main.py server is running.", file=sys.stderr)
        sys.exit(1)
    finally:
        if segment_id:
            print(f"\n[Step 4] Stopping data segment {segment_id}...")
            requests.post(f"{MAIN_BACKEND_URL}/dataSegment/stop", params={"segment_id": segment_id})
            print(" -> Segment stopped.")

    print("\n--- Simulation Finished ---")

if __name__ == "__main__":
    main()