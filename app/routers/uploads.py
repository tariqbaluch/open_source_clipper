import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from ..paths import UPLOAD_TMP_DIR, VIDEOS_DIR

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.post("/init")
def upload_init(filename: str = Form(...), total_chunks: int = Form(...)):
    upload_id = str(uuid.uuid4())
    tmp_dir = UPLOAD_TMP_DIR / upload_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    meta = {"filename": filename, "total_chunks": int(total_chunks)}
    (tmp_dir / "meta.txt").write_text(f"{meta}", encoding="utf-8")
    return {"upload_id": upload_id}


@router.post("/chunk")
async def upload_chunk(upload_id: str = Form(...), index: int = Form(...), chunk: UploadFile = File(...)):
    tmp_dir = UPLOAD_TMP_DIR / upload_id
    if not tmp_dir.exists():
        raise HTTPException(status_code=404, detail="upload_id not found")
    target = tmp_dir / f"chunk_{int(index):06d}"
    with open(target, "wb") as f:
        while True:
            data = await chunk.read(1024 * 1024)
            if not data:
                break
            f.write(data)
    return {"ok": True}


@router.post("/complete")
def upload_complete(upload_id: str = Form(...)):
    tmp_dir = UPLOAD_TMP_DIR / upload_id
    if not tmp_dir.exists():
        raise HTTPException(status_code=404, detail="upload_id not found")
    # read meta
    try:
        meta_text = (tmp_dir / "meta.txt").read_text(encoding="utf-8")
        meta = eval(meta_text)  # trusted file from our server
    except Exception:
        raise HTTPException(status_code=400, detail="invalid upload meta")

    filename = os.path.basename(meta.get("filename") or f"uploaded_{upload_id}.mp4")
    out_path = VIDEOS_DIR / filename

    # assemble
    chunk_files = sorted(tmp_dir.glob("chunk_*"))
    if not chunk_files:
        raise HTTPException(status_code=400, detail="no chunks uploaded")
    with open(out_path, "wb") as w:
        for cf in chunk_files:
            with open(cf, "rb") as r:
                shutil.copyfileobj(r, w)

    # cleanup tmp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"video_path": str(out_path)}
