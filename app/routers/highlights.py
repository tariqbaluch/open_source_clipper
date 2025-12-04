from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..paths import STORAGE_DIR

router = APIRouter(prefix="/highlights", tags=["highlights"])


@router.get("/{job_id}")
def get_highlights(job_id: str):
    job_dir = STORAGE_DIR / "pipeline" / job_id
    highlights_path = job_dir / "highlights.json"
    if not highlights_path.exists():
        raise HTTPException(status_code=404, detail="highlights not found for this job_id")

    try:
        import json

        data = json.loads(highlights_path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="failed to read highlights.json")

    # Ensure we always return job_id along with the stored payload
    payload = {"job_id": job_id}
    if isinstance(data, dict):
        payload.update(data)
    else:
        payload["clips"] = data

    return JSONResponse(payload)


@router.get("/{job_id}/transcript")
def get_transcript(job_id: str):
    """Return transcript segments for a given job_id."""
    job_dir = STORAGE_DIR / "pipeline" / job_id
    transcript_path = job_dir / "transcript.json"
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="transcript not found for this job_id")

    try:
        import json

        segments = json.loads(transcript_path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="failed to read transcript.json")

    return JSONResponse({"job_id": job_id, "segments": segments})


@router.delete("/{job_id}/clips/{clip_id}")
def delete_clip(job_id: str, clip_id: str):
    """Delete a single clip from highlights.json for the given job_id.

    - Does NOT renumber remaining clip IDs.
    - Persists updated highlights.json back to disk.
    """
    job_dir = STORAGE_DIR / "pipeline" / job_id
    highlights_path = job_dir / "highlights.json"
    if not highlights_path.exists():
        raise HTTPException(status_code=404, detail="highlights not found for this job_id")

    try:
        import json

        raw = json.loads(highlights_path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="failed to read highlights.json")

    # Normalize structure: expect a dict with a "clips" list.
    if isinstance(raw, dict):
        clips = raw.get("clips") or []
    else:
        clips = raw
        raw = {"clips": clips}

    # Filter out the target clip
    original_len = len(clips)
    remaining = [c for c in clips if str(c.get("id")) != clip_id]

    if len(remaining) == original_len:
        # Nothing removed -> clip_id not found
        raise HTTPException(status_code=404, detail="clip_id not found for this job_id")

    raw["clips"] = remaining

    try:
        highlights_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        raise HTTPException(status_code=500, detail="failed to write updated highlights.json")

    payload = {"job_id": job_id, "deleted_clip_id": clip_id}
    payload.update(raw)
    return JSONResponse(payload)
