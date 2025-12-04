from typing import Optional
from fastapi import APIRouter, Form

from ..tasks import download_youtube_task, orchestrate_pipeline_task

router = APIRouter(prefix="/download", tags=["youtube"])


@router.post("/youtube")
def download_youtube(
    url: str = Form(...),
    filename: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    duration_preset: Optional[str] = Form(None),
    timeframe_percent: Optional[int] = Form(None),
):
    # Chain: download_youtube_task -> orchestrate_pipeline_task
    # download_youtube_task returns the local video path string, which becomes the
    # "path" argument for orchestrate_pipeline_task.
    sig = download_youtube_task.s(url=url, filename=filename) | orchestrate_pipeline_task.s(
        chunk_seconds=30,
        overlap_seconds=2,
        model_size="base",
        prompt=prompt,
        duration_preset=duration_preset,
        timeframe_percent=timeframe_percent,
    )
    job = sig.apply_async()
    return {"job_id": job.id}
