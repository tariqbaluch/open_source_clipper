from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_TMP_DIR = STORAGE_DIR / "tmp"
VIDEOS_DIR = STORAGE_DIR / "videos"

for p in (STORAGE_DIR, UPLOAD_TMP_DIR, VIDEOS_DIR):
    p.mkdir(parents=True, exist_ok=True)
