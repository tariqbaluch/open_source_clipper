# Testing Checklist & File Summary

## Files Created

### Helper Scripts
1. ✅ `extract_audio.py` - Extract audio from video
2. ✅ `convert_transcript.py` - Convert transcript format
3. ✅ `pipeline_orchestrator.py` - Run full pipeline (steps 2-5)
4. ✅ `batch_speaker_center.py` - Apply speaker centering to all clips
5. ✅ `apply_karaoke.py` - Add karaoke subtitles to clips
6. ✅ `test_pipeline.py` - Check if everything is set up correctly

### Documentation
7. ✅ `PIPELINE_GUIDE.md` - Full pipeline documentation
8. ✅ `TESTING_CHECKLIST.md` - This file

## Quick Start Testing

### Step 0: Verify Setup
```bash
python test_pipeline.py
```
This checks if all dependencies and files are available.

### Step 1: Get a Test Video

**Option A - Download from YouTube:**
```bash
python youtube_downloader.py
# Enter a YouTube URL (e.g., a short video for testing)
# Output: downloads/your_video.mp4
```

**Option B - Use Existing Video:**
- Copy your video to `input.mp4` in the backend folder
- Or use one of your existing videos: `downloads/mashkaf.mp4` or `downloads/love me like u do.mp4`

### Step 2: Run Full Pipeline (Recommended)
```bash
# First, set your OpenAI API key in 2e_Highlight_Crop_Video.py line 13
# Then run:
python pipeline_orchestrator.py downloads/mashkaf.mp4
```

This automatically runs:
- Audio extraction
- Transcription
- Transcript conversion
- Highlight generation

### Step 3: Test Individual Steps (Alternative)

If you want to test each step separately:

```bash
# 1. Extract audio
python extract_audio.py downloads/mashkaf.mp4

# 2. Transcribe
python vad_fast_transcribe3.py

# 3. Convert transcript
python convert_transcript.py aligned_transcript.json

# 4. Generate highlights (make sure input.mp4 exists)
# Copy your video to input.mp4 first:
copy downloads\mashkaf.mp4 input.mp4
python 2e_Highlight_Crop_Video.py
```

### Step 4: Apply Post-Processing (Optional)

```bash
# Apply speaker centering to all clips
python batch_speaker_center.py

# Apply karaoke subtitles
python apply_karaoke.py
```

## Expected Output Files

After running the pipeline, you should have:

```
backend/
├── vocals.wav                          # Extracted audio
├── aligned_transcript.json            # Word-level transcript
├── transcript.json                     # Formatted transcript
├── highlights.json                     # Highlight metadata
├── highlights/                         # Generated clips
│   ├── Hook_highlight_clip_001.mp4
│   ├── Tip_highlight_clip_002.mp4
│   └── ...
├── tts_ready_transcript.txt            # TTS-ready text
└── word_pacing.csv                     # Word timing data
```

## Testing Order

1. ✅ **Setup Check**: `python test_pipeline.py`
2. ✅ **Get Video**: Use existing or download new one
3. ✅ **Full Pipeline**: `python pipeline_orchestrator.py <video>`
4. ✅ **Verify Outputs**: Check `highlights/` folder and `highlights.json`
5. ✅ **Optional Post-Processing**: Speaker centering and karaoke

## Common Issues

### Issue: "No module named 'openai'"
**Fix**: `pip install openai`

### Issue: "FFmpeg not found"
**Fix**: Install FFmpeg and add to PATH

### Issue: "OpenAI API key not set"
**Fix**: Edit `2e_Highlight_Crop_Video.py` line 13, or set environment variable:
```bash
set OPENAI_API_KEY=your-key-here
```

### Issue: "transcript.json not found"
**Fix**: Run `python convert_transcript.py aligned_transcript.json` first

### Issue: "input.mp4 not found"
**Fix**: Copy your video to `input.mp4` or update line 298 in `2e_Highlight_Crop_Video.py`

## Next Steps After Testing

Once testing is successful:
1. Set up Celery + Redis for queue processing
2. Create API endpoints for web interface
3. Implement chunked video upload
4. Add aspect ratio export (1:1, 16:9, 9:16)
5. Connect to mobile app backend

