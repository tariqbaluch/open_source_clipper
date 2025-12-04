from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Form, HTTPException

from ..tasks import process_video_task, orchestrate_pipeline_task

router = APIRouter(tags=["pipeline"])  # endpoints: /process, /pipeline


@router.post("/process")
def process_video(path: str = Form(...)):
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="file not found")
    job = process_video_task.delay(path)
    return {"job_id": job.id}


@router.post("/pipeline")
def run_pipeline(
    path: str = Form(...),
    chunk_seconds: int = Form(30),
    overlap_seconds: int = Form(2),
    model_size: Optional[str] = Form("base"),
    prompt: Optional[str] = Form(None),
    duration_preset: Optional[str] = Form(None),
    timeframe_percent: Optional[int] = Form(None),
):
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="file not found")
    job = orchestrate_pipeline_task.delay(
        path=path,
        chunk_seconds=int(chunk_seconds),
        overlap_seconds=int(overlap_seconds),
        model_size=model_size or "base",
        prompt=prompt,
        duration_preset=duration_preset,
        timeframe_percent=timeframe_percent,
    )
    return {"job_id": job.id}
