import requests
import time
import random
import sys
from datetime import datetime

# --- Configuration ---
MAIN_BACKEND_URL = "http://localhost:8000"
AI_MODEL_URL = "http://localhost:8002/process"

def main():
    print("--- Starting Single Music Generation Trigger ---")
    
    try:
        print("\n[Step 1] Configuring AI webhook in the main backend...")
        requests.post(f"{MAIN_BACKEND_URL}/config/ai", params={"webhook_url": AI_MODEL_URL}).raise_for_status()
        print(f" -> AI webhook configured to: {AI_MODEL_URL}")

        print("\n[Step 2] Sending a single data payload to trigger music generation...")
        temp = round(random.uniform(18.0, 32.0), 2)
        payload = { "data": { "sensor_id": f"sensor_cli_test", "temperature": temp } }
        
        print(f" -> Sending data (temp: {temp}°C)...")
        # NOTE: We can now use the new trigger endpoint or the old webhook endpoint for testing.
        # Let's use the old one to ensure it still works.
        res = requests.post(f"{MAIN_BACKEND_URL}/webhooks/dataSegment", json=payload)
        res.raise_for_status()
        
        print(f" -> Backend responded with status: {res.status_code}. Music generation started.")
        print(" -> Check the React app for the result.")

    except requests.exceptions.RequestException as e:
        print(f"\nERROR: Could not connect to the backend.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        print("Please ensure the main.py and musicgen_fastapi.py servers are running.", file=sys.stderr)
        sys.exit(1)

    print("\n--- Simulation Finished ---")

if __name__ == "__main__":
    main()