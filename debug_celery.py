import sys
import os
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

print("Attempting to import app.tasks...")
try:
    from app.tasks import download_youtube_task, orchestrate_pipeline_task
    print("Successfully imported app.tasks")
except Exception as e:
    print(f"Failed to import app.tasks: {e}")
    sys.exit(1)

print("Attempting to connect to Redis via Celery...")
try:
    from app.celery_app import celery_app
    with celery_app.connection() as conn:
        print(f"Connected to Redis: {conn.as_uri()}")
        conn.ensure_connection()
        print("Connection verified.")
except Exception as e:
    print(f"Failed to connect to Redis: {e}")
    sys.exit(1)

print("Debug check passed.")
